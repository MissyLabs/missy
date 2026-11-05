use crate::{config::*, transport::Connection};
use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::Mutex,
    time::{Duration, Instant},
};

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
    Deploy { job: String, plan: String },
    Stop { job: String, plan: String },
    PlanStop { job: String },
    Plan { job: String },
    Scale { grant: String, plan: String },
    PlanScale { grant: String },
    Restart { job: String, plan: String },
    PlanRestart { job: String },
}
struct ServicePlan {
    action: String,
    job: String,
    graph: Value,
    index: u64,
    created: Instant,
}
pub(crate) struct Nomad {
    pub conn: Connection,
    pub cfg: NomadConfig,
    plans: Mutex<BTreeMap<String, ServicePlan>>,
    attempted: Mutex<std::collections::BTreeSet<(String, u64)>>,
}
impl Nomad {
    pub fn new(conn: Connection, cfg: NomadConfig) -> Self {
        Self {
            conn,
            cfg,
            plans: Mutex::new(BTreeMap::new()),
            attempted: Mutex::new(Default::default()),
        }
    }
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
        for (id, grant) in &cfg.services {
            identifier(id)?;
            identifier(&grant.owner)?;
            let graph = cfg
                .deployments
                .get(id)
                .ok_or_else(|| anyhow!("service needs exact pinned deployment"))?;
            if graph["Type"] != "service"
                || graph["Meta"]["missy_service_owner"] != grant.owner
                || grant.rollout_overlap == 0
                || grant.rollout_overlap > 100
                || cfg.nodes.is_empty()
                || graph.get("Periodic").is_some_and(|v| !v.is_null())
                || graph.get("ParameterizedJob").is_some_and(|v| !v.is_null())
            {
                bail!("invalid owned service grant or missing rollout capacity reservation");
            }
            service_resources(graph)?;
            for update in std::iter::once(&graph["Update"]).chain(
                graph["TaskGroups"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|g| &g["Update"]),
            ) {
                for field in ["Canary", "MaxParallel"] {
                    if update.get(field).is_some_and(|v| !v.is_null())
                        && update[field]
                            .as_u64()
                            .is_none_or(|v| v > u64::from(grant.rollout_overlap))
                    {
                        bail!("rollout strategy exceeds reviewed overlap reservation");
                    }
                }
            }
            for group in graph["TaskGroups"].as_array().unwrap() {
                for task in group["Tasks"].as_array().unwrap() {
                    if task["Env"].get("MISSY_SERVICE_RESTART_ID").is_some() {
                        bail!("reserved restart environment key in operator graph");
                    }
                }
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
            NomadRead::Results { job } => {
                let allocations = Box::pin(self.read(NomadRead::Allocations { job })).await?;
                Ok(
                    json!({"kind":"allocation_metadata","stdout_available":false,"allocations":allocations,"note":"Allocation status is not stdout. Use the owned batch workload result resource for bounded output."}),
                )
            }
            NomadRead::Allocations { job } => {
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
            NomadMutate::Plan { job } => self.plan_service(&job, "deploy", None).await,
            NomadMutate::PlanScale { grant } => {
                let s = self
                    .cfg
                    .scales
                    .get(&grant)
                    .ok_or_else(|| anyhow!("Nomad scale grant denied"))?;
                self.plan_service(&s.job, &format!("scale:{grant}"), Some(s))
                    .await
            }
            NomadMutate::PlanRestart { job } => self.plan_service(&job, "restart", None).await,
            NomadMutate::PlanStop { job } => self.plan_service(&job, "stop", None).await,
            NomadMutate::Deploy { job, plan } => self.submit_service(&job, "deploy", &plan).await,
            NomadMutate::Scale { grant, plan } => {
                let s = self
                    .cfg
                    .scales
                    .get(&grant)
                    .ok_or_else(|| anyhow!("Nomad scale grant denied"))?;
                self.submit_service(&s.job, &format!("scale:{grant}"), &plan)
                    .await
            }
            NomadMutate::Restart { job, plan } => self.submit_service(&job, "restart", &plan).await,
            NomadMutate::Stop { job, plan } => self.submit_service(&job, "stop", &plan).await,
        }
    }
    fn service_grant(&self, job: &str) -> Result<&NomadService> {
        self.check(job)?;
        self.cfg
            .services
            .get(job)
            .ok_or_else(|| anyhow!("owned SERVICE grant required"))
    }
    async fn current(&self, job: &str) -> Result<(Value, u64)> {
        let grant = self.service_grant(job)?;
        // List distinguishes a genuinely absent ID from transport/auth/JSON failures.
        let rows = self
            .conn
            .json(
                "GET",
                &format!("/v1/jobs?namespace={}", self.cfg.namespace),
                None,
                true,
            )
            .await?;
        let rows = rows
            .as_array()
            .ok_or_else(|| anyhow!("invalid Nomad jobs"))?;
        if !rows.iter().any(|v| v["ID"] == job) {
            if grant.expected_modify_index != 0 {
                bail!("pinned existing service disappeared");
            }
            return Ok((Value::Null, 0));
        }
        let graph = self
            .conn
            .json(
                "GET",
                &format!("/v1/job/{job}?namespace={}", self.cfg.namespace),
                None,
                true,
            )
            .await?;
        if graph["ID"] != job
            || graph["Namespace"] != self.cfg.namespace
            || graph["Type"] != "service"
            || graph["Meta"]["missy_service_owner"] != grant.owner
        {
            bail!("foreign service ownership or namespace mismatch");
        }
        let index = graph["JobModifyIndex"]
            .as_u64()
            .filter(|v| *v > 0)
            .ok_or_else(|| anyhow!("missing service modify index"))?;
        if index != grant.expected_modify_index {
            bail!("service optimistic index conflict; operator re-review required");
        }
        Ok((graph, index))
    }
    async fn plan_service(
        &self,
        job: &str,
        action: &str,
        scale: Option<&NomadScale>,
    ) -> Result<Value> {
        let grant = self.service_grant(job)?;
        if self
            .attempted
            .lock()
            .map_err(|_| anyhow!("service attempt lock"))?
            .contains(&(job.into(), grant.expected_modify_index))
        {
            bail!("service index already attempted; reconcile and reconfigure before new plan");
        }
        if action == "restart" && !grant.allow_restart {
            bail!("service restart grant denied");
        }
        if action == "stop" && !grant.allow_stop {
            bail!("service stop grant denied");
        }
        let (current, index) = self.current(job).await?;
        if (scale.is_some() || action == "restart" || action == "stop") && index == 0 {
            bail!("scale/restart/stop requires existing owned service");
        }
        let token = uuid::Uuid::new_v4().to_string();
        let mut graph = self.cfg.deployments[job].clone();
        // Tighten, never replace, the operator's constraints. Capacity discovery is
        // not permission for the scheduler to place on unreviewed nodes.
        let node_pattern = format!(
            "^({})$",
            self.cfg
                .nodes
                .iter()
                .map(|id| id.replace('.', "\\."))
                .collect::<Vec<_>>()
                .join("|")
        );
        if graph.get("Constraints").is_none_or(Value::is_null) {
            graph["Constraints"] = json!([]);
        }
        graph["Constraints"]
            .as_array_mut()
            .ok_or_else(|| anyhow!("invalid pinned constraints"))?
            .push(json!({"LTarget":"${node.unique.id}","Operand":"regexp","RTarget":node_pattern}));
        if let Some(s) = scale {
            let groups = graph["TaskGroups"]
                .as_array_mut()
                .ok_or_else(|| anyhow!("invalid groups"))?;
            for group in groups {
                if group["Name"] == s.group {
                    group["Count"] = json!(s.count);
                }
            }
        }
        if action == "restart" {
            for group in graph["TaskGroups"].as_array_mut().unwrap() {
                for task in group["Tasks"].as_array_mut().unwrap() {
                    if task.get("Env").is_none_or(Value::is_null) {
                        task["Env"] = json!({});
                    }
                    task["Env"]["MISSY_SERVICE_RESTART_ID"] = json!(token);
                }
            }
        }
        if action == "stop" {
            graph = current.clone();
            graph["Stop"] = json!(true);
        }
        let capacity = if action == "stop" {
            Value::Null
        } else {
            self.capacity(&graph, &current, grant.rollout_overlap)
                .await?
        };
        let dry = self
            .conn
            .json(
                "POST",
                &format!("/v1/job/{job}/plan?namespace={}", self.cfg.namespace),
                Some(json!({"Job":graph,"Diff":true})),
                true,
            )
            .await?;
        if dry["JobModifyIndex"].as_u64() != Some(index) {
            bail!("plan raced service index");
        }
        if !dry["FailedTGAllocs"].is_null()
            && dry["FailedTGAllocs"]
                .as_object()
                .is_none_or(|m| !m.is_empty())
        {
            bail!("Nomad dry-run placement rejected service");
        }
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| anyhow!("service plan lock"))?;
        plans.retain(|_, p| p.created.elapsed() < Duration::from_secs(300));
        if plans.len() >= 128 {
            bail!("service plan storage full");
        }
        plans.insert(
            token.clone(),
            ServicePlan {
                action: action.into(),
                job: job.into(),
                graph,
                index,
                created: Instant::now(),
            },
        );
        Ok(
            json!({"plan":token,"operation":action,"job":job,"namespace":self.cfg.namespace,"job_modify_index":index,"expires_in_seconds":300,"capacity":capacity,"placement":select(&dry,&["JobModifyIndex","Warnings","FailedTGAllocs"]),"state":"planned_not_submitted"}),
        )
    }
    async fn submit_service(&self, job: &str, action: &str, token: &str) -> Result<Value> {
        self.service_grant(job)?;
        // Remove before any await. Concurrent submissions, rejection and unknown outcomes
        // all consume the plan; there is no automatic mutation replay.
        let plan = self
            .plans
            .lock()
            .map_err(|_| anyhow!("service plan lock"))?
            .remove(token)
            .ok_or_else(|| anyhow!("unknown or consumed service plan"))?;
        if plan.job != job
            || plan.action != action
            || plan.created.elapsed() >= Duration::from_secs(300)
        {
            bail!("service plan binding/expiry mismatch");
        }
        let (current, index) = self.current(job).await?;
        if index != plan.index {
            bail!("service index changed since review");
        }
        if action != "stop" {
            self.capacity(
                &plan.graph,
                &current,
                self.service_grant(job)?.rollout_overlap,
            )
            .await?;
        }
        self.register(plan.graph, index).await
    }
    async fn capacity(&self, graph: &Value, _current: &Value, overlap: u32) -> Result<Value> {
        let groups = service_resources(graph)?;
        let nodes = self.conn.json("GET", "/v1/nodes", None, true).await?;
        let nodes = nodes
            .as_array()
            .ok_or_else(|| anyhow!("invalid capacity node list"))?;
        let mut free = Vec::new();
        for id in &self.cfg.nodes {
            if !nodes.iter().any(|v| v["ID"] == *id) {
                continue;
            }
            let node = self
                .conn
                .json("GET", &format!("/v1/node/{id}"), None, true)
                .await?;
            if node["ID"] != *id {
                bail!("capacity node mismatch");
            }
            if node["Status"] != "ready"
                || node["SchedulingEligibility"] != "eligible"
                || node["Drain"] == true
                || !node["DrainStrategy"].is_null()
            {
                continue;
            }
            if graph["Datacenters"]
                .as_array()
                .is_none_or(|ds| !ds.iter().any(|d| d == &node["Datacenter"]))
            {
                continue;
            }
            if graph["NodePool"]
                .as_str()
                .is_some_and(|pool| pool != node["NodePool"].as_str().unwrap_or("default"))
            {
                continue;
            }
            let mut available = node_resources(&node["NodeResources"])?;
            available.subtract(node_resources(&node["ReservedResources"])?);
            let allocations = self
                .conn
                .json("GET", &format!("/v1/node/{id}/allocations"), None, true)
                .await?;
            for alloc in allocations
                .as_array()
                .ok_or_else(|| anyhow!("invalid node allocations"))?
            {
                if alloc["NodeID"] != *id {
                    bail!("node allocation identity mismatch");
                }
                if ["complete", "failed"].contains(&alloc["ClientStatus"].as_str().unwrap_or("")) {
                    continue;
                }
                let used = if alloc["Resources"].is_object() {
                    let mut used = task_resources(&alloc["Resources"])?;
                    used.disk = required_u64(&alloc["Resources"], "DiskMB")?;
                    used
                } else {
                    let mut used = Resources::default();
                    let tasks = alloc["AllocatedResources"]["Tasks"]
                        .as_object()
                        .ok_or_else(|| anyhow!("allocation capacity missing"))?;
                    for task in tasks.values() {
                        let memory = required_u64(&task["Memory"], "MemoryMB")?;
                        let max = if task["Memory"]
                            .get("MemoryMaxMB")
                            .is_some_and(|v| !v.is_null())
                        {
                            required_u64(&task["Memory"], "MemoryMaxMB")?
                        } else {
                            0
                        };
                        used = used.add(Resources {
                            cpu: required_u64(&task["Cpu"], "CpuShares")?,
                            memory: memory.max(max),
                            disk: 0,
                        })?;
                    }
                    used.disk = required_u64(&alloc["AllocatedResources"]["Shared"], "DiskMB")?;
                    used
                };
                available.subtract(used);
            }
            free.push(available);
        }
        // Conservative full replacement: existing allocations remain charged while
        // all desired replicas plus overlap must fit. Never assume they'll terminate.
        let mut demands = Vec::new();
        for (_, count, resource) in groups {
            for _ in 0..count
                .checked_add(overlap)
                .ok_or_else(|| anyhow!("replica overflow"))?
            {
                demands.push(resource);
            }
        }
        demands.sort_by_key(|r| std::cmp::Reverse((r.memory, r.cpu, r.disk)));
        for demand in &demands {
            let node = free.iter_mut().find(|r| r.fits(*demand)).ok_or_else(|| {
                anyhow!("service capacity shortfall including full replacement and rollout overlap")
            })?;
            node.subtract(*demand);
        }
        Ok(
            json!({"strategy":"full_replacement_plus_overlap_existing_allocations_charged","eligible_nodes":free.len(),"reserved_allocations":demands.len(),"rollout_overlap_per_group":overlap}),
        )
    }
    async fn register(&self, graph: Value, index: u64) -> Result<Value> {
        let id = graph["ID"]
            .as_str()
            .ok_or_else(|| anyhow!("missing submission ID"))?;
        if !self
            .attempted
            .lock()
            .map_err(|_| anyhow!("service attempt lock"))?
            .insert((id.into(), index))
        {
            bail!("service index already attempted; no mutation retry");
        }
        let r = self.conn.json("POST", &format!("/v1/jobs?namespace={}", self.cfg.namespace), Some(json!({"Job":graph,"EnforceIndex":true,"JobModifyIndex":index})), true).await
            .map_err(|_| anyhow!("Nomad submission rejected or outcome unknown; plan consumed, reconcile remote state before new review; no retry"))?;
        if r["JobModifyIndex"].as_u64().is_none_or(|v| v <= index)
            || r["EvalID"].as_str().is_none_or(str::is_empty)
        {
            bail!("Nomad submission outcome unknown; reconcile, do not retry");
        }
        Ok(
            json!({"state":"submitted_not_yet_verified","submission":select(&r,&["EvalID","EvalCreateIndex","JobModifyIndex","Warnings"])}),
        )
    }
}
#[derive(Clone, Copy, Default)]
struct Resources {
    cpu: u64,
    memory: u64,
    disk: u64,
}
impl Resources {
    fn add(self, b: Self) -> Result<Self> {
        Ok(Self {
            cpu: self
                .cpu
                .checked_add(b.cpu)
                .ok_or_else(|| anyhow!("resource overflow"))?,
            memory: self
                .memory
                .checked_add(b.memory)
                .ok_or_else(|| anyhow!("resource overflow"))?,
            disk: self
                .disk
                .checked_add(b.disk)
                .ok_or_else(|| anyhow!("resource overflow"))?,
        })
    }
    fn subtract(&mut self, b: Self) {
        self.cpu = self.cpu.saturating_sub(b.cpu);
        self.memory = self.memory.saturating_sub(b.memory);
        self.disk = self.disk.saturating_sub(b.disk);
    }
    fn fits(self, b: Self) -> bool {
        self.cpu >= b.cpu && self.memory >= b.memory && self.disk >= b.disk
    }
}
fn required_u64(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| anyhow!("missing/invalid capacity resource {key}"))
}
fn node_resources(v: &Value) -> Result<Resources> {
    Ok(Resources {
        cpu: required_u64(&v["Cpu"], "CpuShares")?,
        memory: required_u64(&v["Memory"], "MemoryMB")?,
        disk: required_u64(&v["Disk"], "DiskMB")?,
    })
}
fn task_resources(v: &Value) -> Result<Resources> {
    if v["Cores"].as_u64().unwrap_or(0) > 0
        || v["Devices"].as_array().is_some_and(|a| !a.is_empty())
    {
        bail!("dedicated cores/devices require unsupported capacity accounting");
    }
    let memory = required_u64(v, "MemoryMB")?;
    let memory_max = if v.get("MemoryMaxMB").is_some_and(|v| !v.is_null()) {
        required_u64(v, "MemoryMaxMB")?
    } else {
        0
    };
    let disk = if v.get("DiskMB").is_some_and(|v| !v.is_null()) {
        required_u64(v, "DiskMB")?
    } else {
        0
    };
    Ok(Resources {
        cpu: required_u64(v, "CPU")?,
        memory: memory.max(memory_max),
        disk,
    })
}
fn service_resources(graph: &Value) -> Result<Vec<(String, u32, Resources)>> {
    let groups = graph["TaskGroups"]
        .as_array()
        .filter(|g| !g.is_empty() && g.len() <= 32)
        .ok_or_else(|| anyhow!("service requires 1..32 groups"))?;
    if graph["Datacenters"].as_array().is_none_or(|d| d.is_empty()) {
        bail!("service needs exact datacenters");
    }
    let mut out = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for group in groups {
        let name = group["Name"]
            .as_str()
            .ok_or_else(|| anyhow!("missing group name"))?;
        identifier(name)?;
        if !names.insert(name) {
            bail!("duplicate service group");
        }
        let count = required_u64(group, "Count")?;
        if count > 10000 {
            bail!("service replica bound");
        }
        let mut resources = Resources::default();
        let tasks = group["Tasks"]
            .as_array()
            .filter(|t| !t.is_empty() && t.len() <= 64)
            .ok_or_else(|| anyhow!("service requires bounded tasks"))?;
        for task in tasks {
            if task
                .get("Env")
                .is_some_and(|v| !v.is_null() && !v.is_object())
            {
                bail!("invalid pinned task environment");
            }
            let r = task_resources(&task["Resources"])?;
            if r.cpu == 0 || r.memory == 0 {
                bail!("service needs positive CPU/memory");
            }
            resources = resources.add(r)?;
        }
        let ephemeral = if group["EphemeralDisk"].is_null() {
            300
        } else {
            required_u64(&group["EphemeralDisk"], "SizeMB")?
        };
        resources.disk = resources
            .disk
            .checked_add(ephemeral)
            .ok_or_else(|| anyhow!("disk overflow"))?;
        out.push((name.into(), count as u32, resources));
    }
    Ok(out)
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
