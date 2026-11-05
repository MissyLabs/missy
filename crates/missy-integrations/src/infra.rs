use crate::{config::*, transport::Connection};
use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum NomadRead {
    List,
    Status { job: String },
    Nodes,
    Node { node: String },
    Placement { job: String },
    Allocations { job: String },
    Results { job: String },
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum NomadMutate {
    Deploy { job: String },
    Stop { job: String },
    Plan { job: String },
    Scale { grant: String },
}
pub(crate) struct Nomad {
    pub conn: Connection,
    pub cfg: NomadConfig,
}
impl Nomad {
    pub fn validate(cfg: &NomadConfig) -> Result<()> {
        identifier(&cfg.namespace)?;
        for job in &cfg.jobs {
            identifier(job)?;
        }
        for node in &cfg.nodes {
            identifier(node)?;
        }
        for (name, scale) in &cfg.scales {
            identifier(name)?;
            identifier(&scale.group)?;
            if !cfg.jobs.contains(&scale.job) || scale.count > 10000 {
                bail!("Nomad scale grant invalid");
            }
            if !cfg.deployments.get(&scale.job).is_some_and(|job| {
                job["TaskGroups"]
                    .as_array()
                    .is_some_and(|groups| groups.iter().any(|g| g["Name"] == scale.group))
            }) {
                bail!("Nomad scale group must occur in pinned job");
            }
        }
        for (name, graph) in &cfg.deployments {
            if !cfg.jobs.contains(name)
                || !graph.is_object()
                || graph["ID"] != *name
                || graph["Namespace"] != cfg.namespace
            {
                bail!("Nomad deployment must match exact job allowlist and configured namespace");
            }
            if serde_json::to_vec(graph)?.len() > 262_144 {
                bail!("Nomad deployment exceeds size bound");
            }
        }
        Ok(())
    }
    fn check(&self, job: &str) -> Result<()> {
        identifier(job)?;
        if !self.cfg.jobs.contains(job) {
            bail!("Nomad job denied by exact allowlist");
        }
        Ok(())
    }
    pub async fn read(&self, a: NomadRead) -> Result<Value> {
        match a {
            NomadRead::Nodes => {
                let data = self.conn.json("GET", "/v1/nodes", None, true).await?;
                let rows = data
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Nomad nodes"))?;
                Ok(Value::Array(
                    rows.iter()
                        .filter(|n| {
                            n["ID"]
                                .as_str()
                                .is_some_and(|id| self.cfg.nodes.contains(id))
                        })
                        .map(|n| {
                            select(
                                n,
                                &[
                                    "ID",
                                    "Name",
                                    "Datacenter",
                                    "Status",
                                    "SchedulingEligibility",
                                    "Drain",
                                    "NodeClass",
                                    "NodePool",
                                ],
                            )
                        })
                        .collect(),
                ))
            }
            NomadRead::Node { node } => {
                if !self.cfg.nodes.contains(&node) {
                    bail!("Nomad node denied");
                }
                let data = self
                    .conn
                    .json("GET", &format!("/v1/node/{node}"), None, true)
                    .await?;
                if data["ID"] != node {
                    bail!("Nomad node mismatch");
                }
                Ok(select(
                    &data,
                    &[
                        "ID",
                        "Name",
                        "Datacenter",
                        "Status",
                        "SchedulingEligibility",
                        "Drain",
                        "NodeClass",
                        "NodePool",
                        "NodeResources",
                        "ReservedResources",
                    ],
                ))
            }
            NomadRead::Placement { job } => {
                self.check(&job)?;
                let graph = self
                    .cfg
                    .deployments
                    .get(&job)
                    .ok_or_else(|| anyhow!("no pinned job"))?;
                let plan = self
                    .conn
                    .json(
                        "POST",
                        &format!("/v1/job/{job}/plan?namespace={}", self.cfg.namespace),
                        Some(json!({"Job":graph,"Diff":true})),
                        true,
                    )
                    .await?;
                // Nomad itself evaluates constraints/resources; do not invent a placement from host names.
                Ok(select(
                    &plan,
                    &[
                        "JobModifyIndex",
                        "CreatedEvals",
                        "FailedTGAllocs",
                        "NextPeriodicLaunch",
                        "Warnings",
                    ],
                ))
            }
            NomadRead::Allocations { job } | NomadRead::Results { job } => {
                self.check(&job)?;
                let data = self
                    .conn
                    .json(
                        "GET",
                        &format!(
                            "/v1/job/{job}/allocations?namespace={}&all=true",
                            self.cfg.namespace
                        ),
                        None,
                        true,
                    )
                    .await?;
                let rows = data
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Nomad allocations"))?;
                Ok(Value::Array(
                    rows.iter()
                        .filter(|a| a["JobID"] == job && a["Namespace"] == self.cfg.namespace)
                        .map(|a| {
                            select(
                                a,
                                &[
                                    "ID",
                                    "JobID",
                                    "Namespace",
                                    "NodeID",
                                    "TaskGroup",
                                    "ClientStatus",
                                    "ClientDescription",
                                    "DesiredStatus",
                                    "TaskStates",
                                    "CreateTime",
                                    "ModifyTime",
                                ],
                            )
                        })
                        .collect(),
                ))
            }
            NomadRead::List => {
                let data = self
                    .conn
                    .json(
                        "GET",
                        &format!("/v1/jobs?namespace={}", self.cfg.namespace),
                        None,
                        true,
                    )
                    .await?;
                let list = data
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Nomad job list"))?;
                Ok(Value::Array(
                    list.iter()
                        .filter(|j| j["ID"].as_str().is_some_and(|s| self.cfg.jobs.contains(s)))
                        .map(job_summary)
                        .collect(),
                ))
            }
            NomadRead::Status { job } => {
                self.check(&job)?;
                let data = self
                    .conn
                    .json(
                        "GET",
                        &format!("/v1/job/{job}?namespace={}", self.cfg.namespace),
                        None,
                        true,
                    )
                    .await?;
                if data["ID"] != job {
                    bail!("Nomad response job mismatch");
                }
                Ok(job_summary(&data))
            }
        }
    }
    pub async fn mutate(&self, a: NomadMutate) -> Result<Value> {
        match a {
            NomadMutate::Plan { job } => self.read(NomadRead::Placement { job }).await,
            NomadMutate::Scale { grant } => {
                let s = self
                    .cfg
                    .scales
                    .get(&grant)
                    .ok_or_else(|| anyhow!("Nomad scale grant denied"))?;
                let r = self.conn.json("POST", &format!("/v1/job/{}/scale?namespace={}", s.job, self.cfg.namespace), Some(json!({"Count":s.count,"Target":{"Job":s.job,"Group":s.group,"Namespace":self.cfg.namespace},"Reason":"operator-reviewed scale grant"})), true).await?;
                Ok(select(
                    &r,
                    &["EvalID", "EvalCreateIndex", "JobModifyIndex", "Warnings"],
                ))
            }
            NomadMutate::Deploy { job } => {
                self.check(&job)?;
                let graph = self
                    .cfg
                    .deployments
                    .get(&job)
                    .ok_or_else(|| anyhow!("no approved Nomad deployment"))?;
                let r = self
                    .conn
                    .json(
                        "POST",
                        &format!("/v1/jobs?namespace={}", self.cfg.namespace),
                        Some(json!({"Job":graph})),
                        true,
                    )
                    .await?;
                Ok(select(
                    &r,
                    &["EvalID", "EvalCreateIndex", "JobModifyIndex", "Warnings"],
                ))
            }
            NomadMutate::Stop { job } => {
                self.check(&job)?;
                // Never purge history or target a different namespace.
                let r = self
                    .conn
                    .json(
                        "DELETE",
                        &format!("/v1/job/{job}?namespace={}&purge=false", self.cfg.namespace),
                        None,
                        true,
                    )
                    .await?;
                Ok(select(&r, &["EvalID", "EvalCreateIndex", "JobModifyIndex"]))
            }
        }
    }
}
fn job_summary(v: &Value) -> Value {
    // No TaskGroups, Env, templates, Vault stanzas or deployment payloads.
    select(
        v,
        &[
            "ID",
            "Name",
            "Namespace",
            "Type",
            "Status",
            "StatusDescription",
            "Stop",
            "Version",
            "JobModifyIndex",
            "SubmitTime",
        ],
    )
}
pub(crate) fn select(v: &Value, fields: &[&str]) -> Value {
    let mut out = serde_json::Map::new();
    for f in fields {
        if let Some(v) = v.get(*f) {
            out.insert((*f).into(), v.clone());
        }
    }
    Value::Object(out)
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum IncusRead {
    List,
    Status { instance: String },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IncusMutate {
    pub instance: String,
    pub action: InstanceAction,
}
pub(crate) struct Incus {
    pub conn: Connection,
    pub cfg: IncusConfig,
}
impl Incus {
    pub fn validate(cfg: &IncusConfig) -> Result<()> {
        identifier(&cfg.project)?;
        for instance in &cfg.instances {
            identifier(instance)?;
        }
        Ok(())
    }
    pub(crate) fn check(&self, instance: &str) -> Result<()> {
        identifier(instance)?;
        if !self.cfg.instances.contains(instance) {
            bail!("Incus instance denied by exact allowlist");
        }
        Ok(())
    }
    pub(crate) async fn query(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let r = self.conn.json(method, path, body, false).await?;
        if r["type"] == "error" || r["error_code"].as_u64().is_some_and(|code| code != 0) {
            bail!("Incus rejected operation");
        }
        if r["type"] != "sync" && r["type"] != "async" {
            bail!("invalid Incus response envelope");
        }
        Ok(r)
    }
    pub async fn read(&self, a: IncusRead) -> Result<Value> {
        match a {
            IncusRead::List => {
                let r = self
                    .query(
                        "GET",
                        &format!("/1.0/instances?project={}&recursion=1", self.cfg.project),
                        None,
                    )
                    .await?;
                let list = r["metadata"]
                    .as_array()
                    .ok_or_else(|| anyhow!("invalid Incus instance list"))?;
                Ok(Value::Array(
                    list.iter()
                        .filter(|i| {
                            i["name"]
                                .as_str()
                                .is_some_and(|s| self.cfg.instances.contains(s))
                        })
                        .map(|i| {
                            select(
                                i,
                                &[
                                    "name",
                                    "status",
                                    "status_code",
                                    "type",
                                    "architecture",
                                    "location",
                                ],
                            )
                        })
                        .collect(),
                ))
            }
            IncusRead::Status { instance } => {
                self.check(&instance)?;
                let r = self
                    .query(
                        "GET",
                        &format!(
                            "/1.0/instances/{instance}/state?project={}",
                            self.cfg.project
                        ),
                        None,
                    )
                    .await?;
                Ok(select(
                    &r["metadata"],
                    &[
                        "status",
                        "status_code",
                        "pid",
                        "processes",
                        "cpu",
                        "memory",
                        "disk",
                        "network",
                    ],
                ))
            }
        }
    }
    pub async fn mutate(&self, a: IncusMutate) -> Result<Value> {
        self.check(&a.instance)?;
        if !self.cfg.actions.contains(&a.action) {
            bail!("Incus lifecycle action denied");
        }
        let r = self
            .query(
                "PUT",
                &format!(
                    "/1.0/instances/{}/state?project={}",
                    a.instance, self.cfg.project
                ),
                Some(json!({"action":a.action.wire(),"timeout":30,"force":false,"stateful":false})),
            )
            .await?;
        // Operation URLs are metadata only, never followed as an arbitrary API path.
        Ok(json!({"accepted":true,"operation":r["operation"],"status":r["status"]}))
    }
}
