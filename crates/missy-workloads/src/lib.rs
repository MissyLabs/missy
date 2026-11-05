//! Durable, scoped finite Nomad work. No provider, shell, Python or ambient
//! account import. Embedders MUST apply the outer exact durable approval gate.
mod config;
mod store;
use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
pub use config::{WorkloadTemplate, WorkloadsConfig};
use config::{digest, ident, render};
use missy_core::{
    PolicyConfig, Scope, ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor,
};
use missy_policy::{HttpResponse, PolicyHttpClient};
use missy_schedule_text::ScheduleSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Duration};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkloadData {
    pub workload_template: String,
    pub parameters: BTreeMap<String, String>,
    pub input_artifacts: BTreeMap<String, String>,
    pub output_artifacts: BTreeMap<String, String>,
    pub idempotency_key: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct Task {
    id: String,
    scope: Scope,
    data: WorkloadData,
    config_hash: String,
    spec: Value,
    spec_hash: String,
    state: String,
    created_at: DateTime<Utc>,
    submitted_at: Option<DateTime<Utc>>,
    evaluation: Option<String>,
    #[serde(default)]
    result: Option<Value>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDefinition {
    pub name: String,
    pub workload_template: String,
    pub workload_version: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, String>,
    #[serde(default)]
    pub input_artifacts: BTreeMap<String, String>,
    pub output_prefix: String,
    #[serde(default)]
    pub warmup_runs: usize,
    pub repetitions: usize,
    pub parallelism: usize,
    pub idempotency_key: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct Benchmark {
    id: String,
    scope: Scope,
    definition: BenchmarkDefinition,
    config_hash: String,
    runs: Vec<String>,
    started: bool,
    cancelled: bool,
    comparison: Option<Value>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Schedule {
    id: String,
    scope: Scope,
    data: WorkloadData,
    spec: ScheduleSpec,
    next: Option<DateTime<Utc>>,
    max_runs: u32,
    fired: u32,
    paused: bool,
    removed: bool,
    config_hash: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct Occurrence {
    id: String,
    scope: Scope,
    schedule_id: String,
    scheduled_at: DateTime<Utc>,
    state: String,
    task_id: String,
    error: Option<String>,
}

/// One process owns its state directory lock; calls are serialized to make
/// idempotency reservation and external mutation ordering atomic in that owner.
pub struct WorkloadTools {
    config: WorkloadsConfig,
    http: PolicyHttpClient,
    journal: store::Journal,
    fingerprint: String,
    serial: tokio::sync::Mutex<()>,
}
impl WorkloadTools {
    pub fn open(
        policy: PolicyConfig,
        config: WorkloadsConfig,
        state_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        config.validate()?;
        let fingerprint = digest(&(&policy, &config))?;
        let http = PolicyHttpClient::new(policy)?
            .with_timeout(Duration::from_secs(config.request_timeout_seconds))?;
        let journal = store::Journal::open(state_dir.as_ref(), config.max_records)?;
        Ok(Self {
            config,
            http,
            journal,
            fingerprint,
            serial: tokio::sync::Mutex::new(()),
        })
    }
    fn template(&self, name: &str) -> Result<&WorkloadTemplate> {
        self.config
            .templates
            .get(name)
            .ok_or_else(|| anyhow!("workload template not authorized"))
    }
    fn token(&self) -> Result<Vec<(String, String)>> {
        if let Some(e) = &self.config.token_env {
            let t =
                std::env::var(e).map_err(|_| anyhow!("dedicated Nomad credential unavailable"))?;
            ensure!(
                !t.is_empty() && t.len() <= 512 && t.bytes().all(|b| b.is_ascii_graphic()),
                "invalid dedicated Nomad credential"
            );
            Ok(vec![("X-Nomad-Token".into(), t)])
        } else {
            Ok(vec![])
        }
    }
    async fn request(
        &self,
        cancel: &CancellationToken,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<HttpResponse> {
        ensure!(
            !cancel.is_cancelled(),
            "workload operation cancelled before dispatch"
        );
        let mut h = self.token()?;
        let b = body.map(|v| serde_json::to_vec(&v)).transpose()?;
        if b.is_some() {
            h.push(("Content-Type".into(), "application/json".into()));
        }
        let url = format!("{}{path}", self.config.endpoint);
        tokio::select! {biased;_=cancel.cancelled()=>bail!("workload operation cancelled; remote mutation may be uncertain"),r=self.http.request(method,&url,h,b)=>r.map_err(|_|anyhow!("policy-bound Nomad request failed; inspect durable state before retry"))}
    }
    async fn json(
        &self,
        cancel: &CancellationToken,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let r = self.request(cancel, method, path, body).await?;
        ensure!(
            (200..300).contains(&r.status),
            "Nomad request rejected with status {}",
            r.status
        );
        serde_json::from_slice(&r.body).map_err(|_| anyhow!("invalid bounded Nomad JSON"))
    }
    fn q(&self) -> String {
        format!("namespace={}", self.config.namespace)
    }
    fn validate_data(&self, d: &WorkloadData) -> Result<()> {
        let t = self.template(&d.workload_template)?;
        ensure!(d.idempotency_key.len() <= 256, "idempotency key too long");
        render(
            t,
            &d.parameters,
            &d.input_artifacts,
            &d.output_artifacts,
            "missy-validation",
            &self.config,
            "validation",
        )?;
        Ok(())
    }
    async fn discover(
        &self,
        cancel: &CancellationToken,
        spec: &Value,
        parallelism: usize,
    ) -> Result<Value> {
        let nodes = self.json(cancel, "GET", "/v1/nodes", None).await?;
        let rows = nodes
            .as_array()
            .ok_or_else(|| anyhow!("invalid node discovery"))?;
        ensure!(rows.len() <= 10000, "node discovery too large");
        let cpu = spec["TaskGroups"][0]["Tasks"][0]["Resources"]["CPU"]
            .as_u64()
            .ok_or_else(|| anyhow!("invalid CPU demand"))?;
        let mem = spec["TaskGroups"][0]["Tasks"][0]["Resources"]["MemoryMB"]
            .as_u64()
            .ok_or_else(|| anyhow!("invalid memory demand"))?;
        let disk = spec["TaskGroups"][0]["EphemeralDisk"]["SizeMB"]
            .as_u64()
            .unwrap_or(300);
        let mut slots = 0u64;
        let mut observed = Vec::new();
        for n in rows {
            let Some(id) = n["ID"]
                .as_str()
                .filter(|id| self.config.allowed_node_ids.contains(*id))
            else {
                continue;
            };
            if n["Status"] != "ready"
                || n["SchedulingEligibility"] != "eligible"
                || n["Drain"] == true
            {
                continue;
            }
            ident(id)?;
            let node = self
                .json(cancel, "GET", &format!("/v1/node/{id}"), None)
                .await?;
            ensure!(
                node["ID"] == id
                    && node["Status"] == "ready"
                    && node["SchedulingEligibility"] == "eligible"
                    && node["Drain"] != true,
                "authorized node changed eligibility"
            );
            let total_cpu = node["NodeResources"]["Cpu"]["CpuShares"]
                .as_u64()
                .or_else(|| node["Resources"]["CPU"].as_u64())
                .unwrap_or(0);
            let total_mem = node["NodeResources"]["Memory"]["MemoryMB"]
                .as_u64()
                .or_else(|| node["Resources"]["MemoryMB"].as_u64())
                .unwrap_or(0);
            let reserved_cpu = node["ReservedResources"]["Cpu"]["CpuShares"]
                .as_u64()
                .unwrap_or(0);
            let reserved_mem = node["ReservedResources"]["Memory"]["MemoryMB"]
                .as_u64()
                .unwrap_or(0);
            let total_disk = node["NodeResources"]["Disk"]["DiskMB"]
                .as_u64()
                .or_else(|| node["Resources"]["DiskMB"].as_u64())
                .unwrap_or(0);
            let mut used_disk = node["ReservedResources"]["Disk"]["DiskMB"]
                .as_u64()
                .unwrap_or(0);
            let allocations = self
                .json(cancel, "GET", &format!("/v1/node/{id}/allocations"), None)
                .await?;
            let allocations = allocations
                .as_array()
                .ok_or_else(|| anyhow!("invalid node allocations"))?;
            ensure!(allocations.len() <= 10000, "node allocation bound exceeded");
            let mut used_cpu = reserved_cpu;
            let mut used_mem = reserved_mem;
            for a in allocations {
                if !["complete", "failed", "lost"]
                    .contains(&a["ClientStatus"].as_str().unwrap_or(""))
                {
                    used_disk = used_disk.saturating_add(
                        a["AllocatedResources"]["Shared"]["DiskMB"]
                            .as_u64()
                            .unwrap_or(total_disk),
                    );
                    let resources = &a["AllocatedResources"]["Tasks"];
                    if let Some(ts) = resources.as_object() {
                        for t in ts.values() {
                            used_cpu = used_cpu.saturating_add(
                                t["Cpu"]["CpuShares"].as_u64().unwrap_or(total_cpu),
                            );
                            used_mem = used_mem.saturating_add(
                                t["Memory"]["MemoryMB"].as_u64().unwrap_or(total_mem),
                            );
                        }
                    } else {
                        used_cpu = total_cpu;
                        used_mem = total_mem;
                    }
                }
            }
            let node_slots = (total_cpu.saturating_sub(used_cpu) / cpu)
                .min(total_mem.saturating_sub(used_mem) / mem)
                .min(total_disk.saturating_sub(used_disk) / disk);
            slots = slots.saturating_add(node_slots);
            observed.push(json!({"node_id":id,"name":node["Name"],"datacenter":node["Datacenter"],"architecture":node["Attributes"]["cpu.arch"],"total_cpu":total_cpu,"total_memory_mb":total_mem,"available_slots":node_slots}));
        }
        ensure!(
            slots >= parallelism as u64,
            "fresh authorized capacity cannot fit requested parallelism"
        );
        Ok(json!({"available_slots":slots,"nodes":observed,"observed_at":Utc::now()}))
    }
    async fn remote(&self, cancel: &CancellationToken, id: &str) -> Result<Option<Value>> {
        ident(id)?;
        let r = self
            .request(cancel, "GET", &format!("/v1/job/{id}?{}", self.q()), None)
            .await?;
        if r.status == 404 {
            return Ok(None);
        }
        ensure!(
            (200..300).contains(&r.status),
            "Nomad job inspection unavailable"
        );
        let v: Value =
            serde_json::from_slice(&r.body).map_err(|_| anyhow!("invalid job inspection JSON"))?;
        Ok(Some(v))
    }
    fn assert_owned(&self, t: &Task, remote: &Value) -> Result<()> {
        ensure!(
            t.config_hash == self.fingerprint,
            "workload configuration changed; explicit reconciliation required"
        );
        ensure!(
            remote["ID"] == t.id
                && remote["Namespace"] == self.config.namespace
                && remote["Meta"]["missy_owner"] == self.config.owner
                && remote["Meta"]["missy_scope"] == digest(&t.scope)?
                && remote["Meta"]["missy_job"] == t.id,
            "remote workload ownership mismatch"
        );
        ensure!(
            remote["TaskGroups"][0]["Tasks"][0]["Config"]
                == t.spec["TaskGroups"][0]["Tasks"][0]["Config"],
            "remote executable/container configuration changed"
        );
        ensure!(
            remote["TaskGroups"][0]["Tasks"][0]["Env"]
                == t.spec["TaskGroups"][0]["Tasks"][0]["Env"],
            "remote environment changed"
        );
        for key in ["Volumes", "Networks", "Services", "Devices"] {
            ensure!(
                remote["TaskGroups"][0][key].is_null()
                    || remote["TaskGroups"][0][key]
                        .as_array()
                        .is_some_and(Vec::is_empty)
                    || remote["TaskGroups"][0][key]
                        .as_object()
                        .is_some_and(serde_json::Map::is_empty),
                "remote resource/binding additions denied"
            );
        }
        for key in [
            "Templates",
            "Vault",
            "Artifacts",
            "CSIPluginConfig",
            "VolumeMounts",
        ] {
            ensure!(
                remote["TaskGroups"][0]["Tasks"][0][key].is_null()
                    || remote["TaskGroups"][0]["Tasks"][0][key]
                        .as_array()
                        .is_some_and(Vec::is_empty),
                "remote task authority changed"
            );
        }
        ensure!(
            project(remote, &t.spec)? == t.spec,
            "remote workload spec changed; mutation denied"
        );
        Ok(())
    }
    fn task(&self, scope: &Scope, id: &str) -> Result<Task> {
        ident(id)?;
        self.journal
            .get("task", id, scope)?
            .ok_or_else(|| anyhow!("workload not found in authenticated scope"))
    }
    fn save(&self, t: &Task) -> Result<()> {
        self.journal.put("task", &t.id, &t.scope, t)
    }
    async fn plan(&self, ctx: &ToolContext, data: WorkloadData) -> Result<Task> {
        self.validate_data(&data)?;
        let key = if data.idempotency_key.is_empty() {
            format!("turn:{}", ctx.turn_id)
        } else {
            data.idempotency_key.clone()
        };
        let id = format!("missy-work-{}", &digest(&(&ctx.scope, &key))?[..32]);
        if let Some(mut t) = self.journal.get::<Task>("task", &id, &ctx.scope)? {
            ensure!(
                digest(&t.data)? == digest(&data)? && t.config_hash == self.fingerprint,
                "idempotency key/config conflict"
            );
            if t.state == "planning" {
                self.finish_plan(ctx, &mut t).await?;
            }
            return Ok(t);
        }
        let spec = render(
            self.template(&data.workload_template)?,
            &data.parameters,
            &data.input_artifacts,
            &data.output_artifacts,
            &id,
            &self.config,
            &digest(&ctx.scope)?,
        )?;
        let spec_hash = digest(&spec)?;
        // Reserve before any protocol call, including planning. Failed/partial
        // planning is inspectable and never mistaken for a submitted workload.
        let mut t = Task {
            id,
            scope: ctx.scope.clone(),
            data,
            config_hash: self.fingerprint.clone(),
            spec,
            spec_hash,
            state: "planning".into(),
            created_at: Utc::now(),
            submitted_at: None,
            evaluation: None,
            result: None,
        };
        self.save(&t)?;
        self.finish_plan(ctx, &mut t).await?;
        Ok(t)
    }
    async fn finish_plan(&self, ctx: &ToolContext, t: &mut Task) -> Result<()> {
        self.discover(&ctx.cancellation, &t.spec, 1).await?;
        ensure!(
            self.remote(&ctx.cancellation, &t.id).await?.is_none(),
            "job ID already exists remotely; no adoption"
        );
        let p = self
            .json(
                &ctx.cancellation,
                "POST",
                &format!("/v1/job/{}/plan?{}", t.id, self.q()),
                Some(json!({"Job":t.spec,"Diff":true})),
            )
            .await?;
        ensure!(
            p["JobModifyIndex"].as_u64() == Some(0)
                && p["FailedTGAllocs"].as_object().is_none_or(|m| m.is_empty()),
            "Nomad plan conflict or insufficient placement"
        );
        t.state = "planned".into();
        self.save(t)?;
        Ok(())
    }
    async fn submit(&self, ctx: &ToolContext, id: &str) -> Result<Task> {
        let mut t = self.task(&ctx.scope, id)?;
        if [
            "submitted",
            "running",
            "completed",
            "completed_unverified",
            "failed",
            "cancelled",
            "timed_out",
            "result_validation_failed",
        ]
        .contains(&t.state.as_str())
        {
            return Ok(t);
        }
        ensure!(
            t.state == "planned" && t.config_hash == self.fingerprint,
            "workload not safely submit-ready; reconcile without replay"
        );
        self.discover(&ctx.cancellation, &t.spec, 1).await?;
        ensure!(
            self.remote(&ctx.cancellation, id).await?.is_none(),
            "job exists after planning; fresh reconciliation required"
        );
        let plan = self
            .json(
                &ctx.cancellation,
                "POST",
                &format!("/v1/job/{id}/plan?{}", self.q()),
                Some(json!({"Job":t.spec,"Diff":true})),
            )
            .await?;
        ensure!(
            plan["JobModifyIndex"].as_u64() == Some(0)
                && plan["FailedTGAllocs"]
                    .as_object()
                    .is_none_or(|m| m.is_empty()),
            "fresh plan conflict"
        );
        t.state = "submission_unknown".into();
        t.submitted_at = Some(Utc::now());
        self.save(&t)?;
        let r = self
            .json(
                &ctx.cancellation,
                "POST",
                &format!("/v1/jobs?{}", self.q()),
                Some(json!({"Job":t.spec,"EnforceIndex":true,"JobModifyIndex":0})),
            )
            .await?;
        let eval = r["EvalID"]
            .as_str()
            .filter(|s| ident(s).is_ok())
            .ok_or_else(|| anyhow!("Nomad submission acknowledgement missing; outcome unknown"))?;
        t.state = "submitted".into();
        t.evaluation = Some(eval.into());
        self.save(&t)?;
        Ok(t)
    }
    async fn observe(&self, ctx: &ToolContext, id: &str, collect: bool) -> Result<Value> {
        let mut t = self.task(&ctx.scope, id)?;
        let remote = self.remote(&ctx.cancellation, id).await?.ok_or_else(|| {
            anyhow!("owned workload missing remotely; never recreate automatically")
        })?;
        self.assert_owned(&t, &remote)?;
        let raw = self
            .json(
                &ctx.cancellation,
                "GET",
                &format!("/v1/job/{id}/allocations?{}&all=true", self.q()),
                None,
            )
            .await?;
        let allocations = raw
            .as_array()
            .ok_or_else(|| anyhow!("invalid workload allocations"))?;
        ensure!(allocations.len() <= 64, "workload allocations exceed bound");
        let mut summaries = Vec::new();
        let mut logs = Vec::new();
        let mut conditions = Vec::new();
        let mut all_complete = !allocations.is_empty();
        let mut failed = false;
        let mut result = None;
        for a in allocations {
            ensure!(
                a["JobID"] == id && a["Namespace"] == self.config.namespace,
                "foreign allocation returned"
            );
            let aid = a["ID"]
                .as_str()
                .ok_or_else(|| anyhow!("allocation ID missing"))?;
            ident(aid)?;
            let node = a["NodeID"]
                .as_str()
                .filter(|n| self.config.allowed_node_ids.contains(*n))
                .ok_or_else(|| anyhow!("allocation on unauthorized node"))?;
            let detail = self
                .json(&ctx.cancellation, "GET", &format!("/v1/node/{node}"), None)
                .await?;
            ensure!(detail["ID"] == node, "node identity mismatch");
            conditions.push(json!({"node_id":node,"architecture":detail["Attributes"]["cpu.arch"],"resources":detail["NodeResources"],"node_class":detail["NodeClass"]}));
            let s = a["ClientStatus"].as_str().unwrap_or("unknown");
            all_complete &= s == "complete";
            failed |= ["failed", "lost"].contains(&s);
            if let Some(ts) = a["TaskStates"].as_object() {
                for task in ts.values() {
                    failed |= task["Failed"] == true;
                }
            }
            summaries
                .push(json!({"id":aid,"node_id":node,"status":s,"task_states":a["TaskStates"]}));
            if collect {
                let mut entry = json!({"allocation_id":aid});
                for stream in ["stdout", "stderr"] {
                    let r=self.request(&ctx.cancellation,"GET",&format!("/v1/client/fs/logs/{aid}?{}&task=workload&type={stream}&plain=true&origin=start&offset=0&follow=false",self.q()),None).await?;
                    ensure!(
                        (200..300).contains(&r.status) && r.body.len() <= self.config.max_log_bytes,
                        "workload log unavailable or exceeds configured bound"
                    );
                    let text = String::from_utf8(r.body)
                        .map_err(|_| anyhow!("workload log is not UTF-8"))?;
                    if stream == "stdout" {
                        for line in text.lines() {
                            if let Some(v) = line.strip_prefix("MISSY_RESULT_JSON=")
                                && let Ok(v) = serde_json::from_str::<Value>(v)
                                && v.is_object()
                            {
                                result = Some(v);
                            }
                        }
                    }
                    entry[stream] = json!(missy_security::redact_output(&text));
                }
                logs.push(entry);
            }
        }
        let template = self.template(&t.data.workload_template)?;
        let timed_out = t
            .submitted_at
            .is_some_and(|s| (Utc::now() - s).num_seconds() > i64::from(template.max_run_seconds));
        let mut artifact_receipts = Vec::new();
        let mut validation_errors = Vec::new();
        if collect && all_complete && !failed {
            if !template.expected_output.is_empty()
                && !logs.iter().any(|l| {
                    l["stdout"]
                        .as_str()
                        .is_some_and(|s| s.contains(&template.expected_output))
                })
            {
                validation_errors.push("expected output marker missing");
            }
            for (k, kind) in &template.result_fields {
                if !result
                    .as_ref()
                    .and_then(|v| v.get(k))
                    .is_some_and(|v| matches_type(v, kind))
                {
                    validation_errors.push("structured result schema mismatch");
                }
            }
            for (k, url) in &t.data.output_artifacts {
                if result
                    .as_ref()
                    .and_then(|v| v["outputs"].get(k))
                    .and_then(Value::as_str)
                    != Some(url)
                {
                    validation_errors.push("declared output mismatch");
                    continue;
                }
                self.config.artifact(url)?;
                let r = tokio::select! {_=ctx.cancellation.cancelled()=>bail!("artifact collection cancelled"),r=self.http.get(url)=>r.map_err(|_|anyhow!("approved artifact retrieval failed"))?};
                ensure!(
                    (200..300).contains(&r.status)
                        && r.body.len() <= self.config.max_artifact_bytes,
                    "artifact unavailable or too large"
                );
                use sha2::{Digest, Sha256};
                artifact_receipts.push(json!({"name":k,"url":url,"bytes":r.body.len(),"sha256":hex::encode(Sha256::digest(&r.body))}));
            }
        }
        let prior_cancelled = ["cancelled", "cancel_unknown"].contains(&t.state.as_str());
        let state = if prior_cancelled {
            if remote["Stop"] == true {
                "cancelled"
            } else {
                "cancel_unknown"
            }
        } else if failed {
            "failed"
        } else if all_complete {
            if collect {
                if validation_errors.is_empty() {
                    "completed"
                } else {
                    "result_validation_failed"
                }
            } else {
                "completed_unverified"
            }
        } else if timed_out {
            "timed_out"
        } else if summaries.iter().any(|s| s["status"] == "running") {
            "running"
        } else {
            "submitted"
        };
        let result = result.map(|r| {
            serde_json::from_str::<Value>(&missy_security::redact_output(&r.to_string()))
                .unwrap_or(json!({"redacted":true}))
        });
        let output = json!({"task_id":id,"state":state,"complete":state=="completed","allocations":summaries,"logs":logs,"structured_result":result,"validation_errors":validation_errors,"artifacts":artifact_receipts,"node_conditions":conditions,"provenance":{"template":t.data.workload_template,"parameters":t.data.parameters,"input_artifacts":t.data.input_artifacts,"spec_sha256":t.spec_hash,"evaluation":t.evaluation,"submitted_at":t.submitted_at},"remote_stop":remote["Stop"]});
        t.state = state.into();
        if collect {
            t.result = Some(output.clone());
        }
        self.save(&t)?;
        Ok(output)
    }
    async fn cancel_task(&self, ctx: &ToolContext, id: &str) -> Result<Task> {
        let mut t = self.task(&ctx.scope, id)?;
        if t.state == "cancelled" {
            return Ok(t);
        }
        if ["planning", "planned"].contains(&t.state.as_str()) {
            t.state = "cancelled".into();
            self.save(&t)?;
            return Ok(t);
        }
        let remote = self
            .remote(&ctx.cancellation, id)
            .await?
            .ok_or_else(|| anyhow!("cannot cancel missing remote workload"))?;
        self.assert_owned(&t, &remote)?;
        if remote["Stop"] == true {
            t.state = "cancelled".into();
            self.save(&t)?;
            return Ok(t);
        }
        ensure!(
            t.state != "cancel_unknown",
            "previous cancel uncertain; reconcile, never blindly repeat"
        );
        t.state = "cancel_unknown".into();
        self.save(&t)?;
        self.json(
            &ctx.cancellation,
            "DELETE",
            &format!("/v1/job/{id}?{}&purge=false", self.q()),
            None,
        )
        .await?;
        t.state = "cancelled".into();
        self.save(&t)?;
        Ok(t)
    }
    async fn reconcile(&self, ctx: &ToolContext) -> Result<Value> {
        let tasks = self.journal.list::<Task>("task", Some(&ctx.scope))?;
        let mut out = Vec::new();
        for mut t in tasks {
            let state = match self.remote(&ctx.cancellation, &t.id).await {
                Ok(Some(remote)) => {
                    if self.assert_owned(&t, &remote).is_err() {
                        "ownership_or_config_mismatch"
                    } else if t.state == "cancel_unknown" && remote["Stop"] == true {
                        t.state = "cancelled".into();
                        self.save(&t)?;
                        "cancelled"
                    } else if t.state == "submission_unknown" {
                        t.state = "submitted".into();
                        self.save(&t)?;
                        "submitted_reconciled"
                    } else {
                        "owned_unchanged"
                    }
                }
                Ok(None) => "missing_no_replay",
                Err(_) => "unavailable_no_replay",
            };
            out.push(json!({"task_id":t.id,"state":state,"journal_state":t.state}));
        }
        Ok(json!({"jobs":out,"observed_at":Utc::now(),"mutations_replayed":false}))
    }
}
fn project(actual: &Value, expected: &Value) -> Result<Value> {
    Ok(match expected {
        Value::Object(e) => {
            let a = actual
                .as_object()
                .ok_or_else(|| anyhow!("remote object shape changed"))?;
            let mut m = serde_json::Map::new();
            for (k, v) in e {
                m.insert(
                    k.clone(),
                    project(
                        a.get(k).ok_or_else(|| anyhow!("remote spec key absent"))?,
                        v,
                    )?,
                );
            }
            Value::Object(m)
        }
        Value::Array(e) => {
            let a = actual
                .as_array()
                .ok_or_else(|| anyhow!("remote array shape changed"))?;
            ensure!(a.len() == e.len(), "remote spec array changed");
            Value::Array(
                a.iter()
                    .zip(e)
                    .map(|(a, e)| project(a, e))
                    .collect::<Result<_>>()?,
            )
        }
        _ => actual.clone(),
    })
}
fn matches_type(v: &Value, kind: &str) -> bool {
    match kind {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "boolean" => v.is_boolean(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        _ => false,
    }
}

impl WorkloadTools {
    fn benchmark(&self, scope: &Scope, id: &str) -> Result<Benchmark> {
        ident(id)?;
        self.journal
            .get("benchmark", id, scope)?
            .ok_or_else(|| anyhow!("benchmark not found in authenticated scope"))
    }
    fn validate_benchmark(&self, d: &BenchmarkDefinition) -> Result<()> {
        ensure!(
            !d.name.trim().is_empty()
                && d.name.len() <= 128
                && !d.name.chars().any(char::is_control),
            "invalid benchmark name"
        );
        ensure!(
            !d.workload_version.is_empty()
                && d.workload_version.len() <= 256
                && d.workload_version.bytes().all(|b| b.is_ascii_graphic()),
            "immutable workload version required"
        );
        ensure!(
            d.repetitions > 0
                && d.warmup_runs
                    .checked_add(d.repetitions)
                    .is_some_and(|n| n <= self.config.max_benchmark_runs)
                && (1..=self.config.max_benchmark_parallelism).contains(&d.parallelism),
            "benchmark run/parallelism bounds exceeded"
        );
        ensure!(
            !d.idempotency_key.is_empty() && d.idempotency_key.len() <= 128,
            "benchmark idempotency key required"
        );
        self.config.artifact(&format!(
            "{}/probe.json",
            d.output_prefix.trim_end_matches('/')
        ))?;
        let t = self.template(&d.workload_template)?;
        render(
            t,
            &d.parameters,
            &d.input_artifacts,
            &BTreeMap::new(),
            "missy-validation",
            &self.config,
            "validation",
        )?;
        Ok(())
    }
    async fn plan_benchmark(&self, ctx: &ToolContext, d: BenchmarkDefinition) -> Result<Benchmark> {
        self.validate_benchmark(&d)?;
        let id = format!(
            "bench-{}",
            &digest(&(&ctx.scope, &d.idempotency_key))?[..32]
        );
        let previous = self
            .journal
            .get::<Benchmark>("benchmark", &id, &ctx.scope)?;
        if let Some(b) = &previous {
            ensure!(
                digest(&b.definition)? == digest(&d)? && b.config_hash == self.fingerprint,
                "benchmark idempotency/config conflict"
            );
            if b.runs.len() == d.warmup_runs + d.repetitions {
                return Ok(b.clone());
            }
            ensure!(
                !b.started && !b.cancelled,
                "partial benchmark already started/cancelled"
            );
        }
        let total = d.warmup_runs + d.repetitions;
        let mut b = previous.unwrap_or(Benchmark {
            id,
            scope: ctx.scope.clone(),
            definition: d,
            config_hash: self.fingerprint.clone(),
            runs: vec![],
            started: false,
            cancelled: false,
            comparison: None,
        });
        self.journal.put("benchmark", &b.id, &b.scope, &b)?;
        for i in 0..total {
            if i < b.runs.len() {
                continue;
            }
            let kind = if i < b.definition.warmup_runs {
                "warmup"
            } else {
                "measured"
            };
            let number = if i < b.definition.warmup_runs {
                i + 1
            } else {
                i - b.definition.warmup_runs + 1
            };
            let mut outputs = BTreeMap::new();
            outputs.insert(
                "RESULT".into(),
                format!(
                    "{}/{}/{kind}-{number}.json",
                    b.definition.output_prefix.trim_end_matches('/'),
                    b.id
                ),
            );
            let data = WorkloadData {
                workload_template: b.definition.workload_template.clone(),
                parameters: b.definition.parameters.clone(),
                input_artifacts: b.definition.input_artifacts.clone(),
                output_artifacts: outputs,
                idempotency_key: format!("{}:{kind}:{number}", b.id),
            };
            let mut task = self.plan(ctx, data).await?;
            let expected = json!({"benchmark":b.id,"kind":kind,"number":number,"version":b.definition.workload_version});
            if task.spec["Meta"].get("missy_benchmark").is_none() {
                task.spec["Meta"]["missy_benchmark"] = json!(digest(&expected)?);
                task.spec["TaskGroups"][0]["Tasks"][0]["Env"]["MISSY_BENCHMARK_ID"] = json!(b.id);
                task.spec["TaskGroups"][0]["Tasks"][0]["Env"]["MISSY_RUN_KIND"] = json!(kind);
                task.spec["TaskGroups"][0]["Tasks"][0]["Env"]["MISSY_RUN_NUMBER"] =
                    json!(number.to_string());
                task.spec["TaskGroups"][0]["Tasks"][0]["Env"]["MISSY_WORKLOAD_VERSION"] =
                    json!(b.definition.workload_version);
                task.spec_hash = digest(&task.spec)?;
                self.discover(&ctx.cancellation, &task.spec, 1).await?;
                let plan = self
                    .json(
                        &ctx.cancellation,
                        "POST",
                        &format!("/v1/job/{}/plan?{}", task.id, self.q()),
                        Some(json!({"Job":task.spec,"Diff":true})),
                    )
                    .await?;
                ensure!(
                    plan["JobModifyIndex"].as_u64() == Some(0)
                        && plan["FailedTGAllocs"]
                            .as_object()
                            .is_none_or(|m| m.is_empty()),
                    "benchmark manifest plan conflict"
                );
                self.save(&task)?;
            }
            b.runs.push(task.id);
            self.journal.put("benchmark", &b.id, &b.scope, &b)?;
        }
        let first = self.task(&ctx.scope, &b.runs[0])?;
        self.discover(&ctx.cancellation, &first.spec, b.definition.parallelism)
            .await?;
        Ok(b)
    }
    async fn advance_benchmark(
        &self,
        ctx: &ToolContext,
        id: &str,
        start: bool,
        results: bool,
    ) -> Result<Value> {
        let mut b = self.benchmark(&ctx.scope, id)?;
        ensure!(
            b.config_hash == self.fingerprint,
            "benchmark configuration changed"
        );
        ensure!(
            b.runs.len() == b.definition.warmup_runs + b.definition.repetitions,
            "benchmark planning interrupted; inspect child plans before use"
        );
        if start {
            ensure!(!b.cancelled, "cancelled benchmark cannot start");
            b.started = true;
            self.journal.put("benchmark", &b.id, &b.scope, &b)?;
        }
        for id in &b.runs {
            let t = self.task(&ctx.scope, id)?;
            if !["planned", "planning", "cancelled"].contains(&t.state.as_str()) {
                self.observe(ctx, id, results).await?;
            }
        }
        let terminal = |s: &str| {
            [
                "completed",
                "completed_unverified",
                "failed",
                "cancelled",
                "timed_out",
                "result_validation_failed",
            ]
            .contains(&s)
        };
        let mut active = 0;
        for id in &b.runs {
            let t = self.task(&ctx.scope, id)?;
            if t.state != "planned" && !terminal(&t.state) {
                active += 1;
            }
        }
        if b.started && !b.cancelled && !results {
            let warm_done = b.runs[..b.definition.warmup_runs]
                .iter()
                .all(|id| self.task(&ctx.scope, id).is_ok_and(|t| terminal(&t.state)));
            for (i, id) in b.runs.iter().enumerate() {
                if active >= b.definition.parallelism {
                    break;
                }
                if i >= b.definition.warmup_runs && !warm_done {
                    continue;
                }
                let t = self.task(&ctx.scope, id)?;
                if t.state == "planned" {
                    self.submit(ctx, id).await?;
                    active += 1;
                }
            }
        }
        let mut rows = Vec::new();
        let mut conditions = Vec::new();
        let mut valid = true;
        for (i, id) in b.runs.iter().enumerate() {
            let t = self.task(&ctx.scope, id)?;
            let measured = i >= b.definition.warmup_runs;
            if measured {
                valid &= t.state == "completed";
                if let Some(r) = &t.result {
                    conditions.push(digest(&r["node_conditions"])?);
                } else {
                    valid = false
                }
            }
            rows.push(json!({"task_id":id,"kind":if measured{"measured"}else{"warmup"},"state":t.state,"result":if results{t.result}else{None}}));
        }
        valid &= !conditions.is_empty() && conditions.windows(2).all(|w| w[0] == w[1]);
        if results {
            b.comparison = Some(
                json!({"valid":valid,"reason":if valid{"all measured runs verified under matching observed node conditions"}else{"runs incomplete/failed/unverified or node conditions differ"},"workload_version":b.definition.workload_version}),
            );
            self.journal.put("benchmark", &b.id, &b.scope, &b)?;
        }
        Ok(
            json!({"benchmark_id":b.id,"started":b.started,"cancelled":b.cancelled,"runs":rows,"comparison":b.comparison,"parallelism":b.definition.parallelism}),
        )
    }
    async fn cancel_benchmark(&self, ctx: &ToolContext, id: &str) -> Result<Value> {
        let mut b = self.benchmark(&ctx.scope, id)?;
        b.cancelled = true;
        self.journal.put("benchmark", &b.id, &b.scope, &b)?;
        let mut cancelled = Vec::new();
        for id in &b.runs {
            let t = self.task(&ctx.scope, id)?;
            if ![
                "completed",
                "failed",
                "cancelled",
                "result_validation_failed",
            ]
            .contains(&t.state.as_str())
            {
                cancelled.push(self.cancel_task(ctx, id).await?.id);
            }
        }
        Ok(json!({"benchmark_id":id,"cancelled":true,"cancelled_tasks":cancelled}))
    }
    async fn schedule_call(&self, ctx: &ToolContext, args: ScheduleArgs) -> Result<Value> {
        match args {
            ScheduleArgs::Create {
                data,
                expression,
                timezone,
                max_runs,
                idempotency_key,
            } => {
                self.validate_data(&data)?;
                ensure!(
                    (1..=self.config.max_schedule_runs).contains(&max_runs),
                    "finite schedule run limit required"
                );
                ensure!(
                    !idempotency_key.is_empty() && idempotency_key.len() <= 128,
                    "schedule idempotency key required"
                );
                let spec = ScheduleSpec::parse(&expression, &timezone)?;
                let id = format!("sched-{}", &digest(&(&ctx.scope, &idempotency_key))?[..32]);
                if let Some(s) = self.journal.get::<Schedule>("schedule", &id, &ctx.scope)? {
                    ensure!(
                        digest(&s.data)? == digest(&data)?
                            && s.spec == spec
                            && s.max_runs == max_runs
                            && s.config_hash == self.fingerprint,
                        "schedule idempotency/config conflict"
                    );
                    return Ok(serde_json::to_value(s)?);
                }
                let s = Schedule {
                    id,
                    scope: ctx.scope.clone(),
                    data,
                    spec: spec.clone(),
                    next: spec.next_occurrence_after(Utc::now())?,
                    max_runs,
                    fired: 0,
                    paused: false,
                    removed: false,
                    config_hash: self.fingerprint.clone(),
                };
                self.journal.put("schedule", &s.id, &s.scope, &s)?;
                Ok(serde_json::to_value(s)?)
            }
            ScheduleArgs::List => Ok(serde_json::to_value(
                self.journal
                    .list::<Schedule>("schedule", Some(&ctx.scope))?,
            )?),
            ScheduleArgs::Pause { schedule_id } => self.change_schedule(ctx, &schedule_id, "pause"),
            ScheduleArgs::Resume { schedule_id } => {
                self.change_schedule(ctx, &schedule_id, "resume")
            }
            ScheduleArgs::Remove { schedule_id } => {
                self.change_schedule(ctx, &schedule_id, "remove")
            }
        }
    }
    fn change_schedule(&self, ctx: &ToolContext, id: &str, action: &str) -> Result<Value> {
        let mut s: Schedule = self
            .journal
            .get("schedule", id, &ctx.scope)?
            .ok_or_else(|| anyhow!("schedule not found in authenticated scope"))?;
        match action {
            "pause" => s.paused = true,
            "resume" => {
                ensure!(
                    !s.removed && s.fired < s.max_runs && s.config_hash == self.fingerprint,
                    "schedule cannot resume"
                );
                s.paused = false;
                s.next = s.spec.next_occurrence_after(Utc::now())?;
            }
            _ => {
                s.removed = true;
                s.paused = true;
                s.next = None;
            }
        }
        self.journal.put("schedule", &s.id, &s.scope, &s)?;
        Ok(serde_json::to_value(s)?)
    }
    /// Explicit supervisor tick. Creation's exact approval authorizes only its
    /// bounded template/data/config and finite count, never arbitrary commands.
    pub async fn run_due(
        &self,
        now: DateTime<Utc>,
        cancellation: CancellationToken,
        limit: usize,
    ) -> Result<Value> {
        ensure!(
            self.config.enabled && (1..=128).contains(&limit),
            "workload scheduler disabled or tick bound invalid"
        );
        let _serial = tokio::select! {_=cancellation.cancelled()=>bail!("workload tick cancelled"),s=self.serial.lock()=>s};
        let schedules = self.journal.list::<Schedule>("schedule", None)?;
        let mut out = Vec::new();
        for mut s in schedules {
            if out.len() >= limit {
                break;
            }
            if s.paused || s.removed || s.fired >= s.max_runs || s.config_hash != self.fingerprint {
                continue;
            }
            let Some(due) = s.next.filter(|d| *d <= now) else {
                continue;
            };
            let oid = format!("occ-{}", &digest(&(&s.id, due))?[..32]);
            if self
                .journal
                .get::<Occurrence>("occurrence", &oid, &s.scope)?
                .is_some()
            {
                s.fired += 1;
                s.next = s.spec.next_occurrence_after(now)?;
                self.journal.put("schedule", &s.id, &s.scope, &s)?;
                continue;
            }
            let mut occurrence = Occurrence {
                id: oid.clone(),
                scope: s.scope.clone(),
                schedule_id: s.id.clone(),
                scheduled_at: due,
                state: "claimed".into(),
                task_id: String::new(),
                error: None,
            };
            self.journal
                .put("occurrence", &oid, &s.scope, &occurrence)?;
            s.fired += 1;
            s.next = if s.fired < s.max_runs {
                s.spec.next_occurrence_after(now)?
            } else {
                None
            };
            self.journal.put("schedule", &s.id, &s.scope, &s)?;
            let ctx = ToolContext {
                scope: s.scope.clone(),
                turn_id: format!("workload:{oid}"),
                cancellation: cancellation.clone(),
            };
            let mut data = s.data.clone();
            data.idempotency_key = oid.clone();
            let operation = async {
                let t = self.plan(&ctx, data).await?;
                occurrence.task_id = t.id.clone();
                self.journal
                    .put("occurrence", &oid, &s.scope, &occurrence)?;
                let t = self.submit(&ctx, &t.id).await?;
                Ok::<_, anyhow::Error>(t)
            }
            .await;
            match operation {
                Ok(t) => {
                    occurrence.state = t.state;
                }
                Err(_) => {
                    occurrence.state = "failed_or_unknown_no_replay".into();
                    occurrence.error = Some(
                        "inspect scoped task journal and reconcile; occurrence will not replay"
                            .into(),
                    );
                }
            }
            self.journal
                .put("occurrence", &oid, &s.scope, &occurrence)?;
            out.push(occurrence);
        }
        for t in self.journal.list::<Task>("task", None)? {
            if !["submitted", "running", "timed_out"].contains(&t.state.as_str())
                || t.config_hash != self.fingerprint
            {
                continue;
            }
            let template = self.template(&t.data.workload_template)?;
            if t.submitted_at
                .is_some_and(|at| (now - at).num_seconds() > i64::from(template.max_run_seconds))
            {
                let ctx = ToolContext {
                    scope: t.scope.clone(),
                    turn_id: format!("timeout:{}", t.id),
                    cancellation: cancellation.clone(),
                };
                let _ = self.cancel_task(&ctx, &t.id).await;
            }
        }
        Ok(
            json!({"occurrences":out,"tick_at":now,"catchup":"one due occurrence per schedule; missed intervals coalesced without replay"}),
        )
    }
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum OffloadArgs {
    Plan {
        #[serde(flatten)]
        data: WorkloadData,
    },
    Submit {
        task_id: String,
    },
    Status {
        task_id: String,
    },
    Result {
        task_id: String,
    },
    Cancel {
        task_id: String,
    },
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum BenchmarkArgs {
    Plan { definition: BenchmarkDefinition },
    Start { benchmark_id: String },
    Status { benchmark_id: String },
    Results { benchmark_id: String },
    Cancel { benchmark_id: String },
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum ScheduleArgs {
    Create {
        data: WorkloadData,
        expression: String,
        timezone: String,
        max_runs: u32,
        idempotency_key: String,
    },
    List,
    Pause {
        schedule_id: String,
    },
    Resume {
        schedule_id: String,
    },
    Remove {
        schedule_id: String,
    },
}
fn closed(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
}
fn data_schema() -> Value {
    closed(
        json!({"workload_template":{"type":"string"},"parameters":{"type":"object","additionalProperties":{"type":"string"}},"input_artifacts":{"type":"object","additionalProperties":{"type":"string"}},"output_artifacts":{"type":"object","additionalProperties":{"type":"string"}},"idempotency_key":{"type":"string"}}),
        &["workload_template", "idempotency_key"],
    )
}
fn schema_union(variants: Vec<Value>) -> Value {
    json!({"type":"object","oneOf":variants})
}
fn action_schema(action: &str, key: Option<&str>) -> Value {
    let mut p = json!({"action":{"const":action,"type":"string"}});
    let mut r = vec!["action"];
    if let Some(k) = key {
        p[k] = json!({"type":"string"});
        r.push(k);
    }
    closed(p, &r)
}
#[async_trait]
impl ToolExecutor for WorkloadTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        if !self.config.enabled {
            return vec![];
        }
        let mut off = data_schema();
        off["properties"]["action"] = json!({"type":"string","const":"plan"});
        off["required"]
            .as_array_mut()
            .unwrap()
            .push(json!("action"));
        let offload = schema_union(
            std::iter::once(off)
                .chain(
                    ["submit", "status", "result", "cancel"]
                        .map(|a| action_schema(a, Some("task_id"))),
                )
                .collect(),
        );
        let mut bplan = action_schema("plan", None);
        bplan["properties"]["definition"] = closed(
            json!({"name":{"type":"string"},"workload_template":{"type":"string"},"workload_version":{"type":"string"},"parameters":{"type":"object","additionalProperties":{"type":"string"}},"input_artifacts":{"type":"object","additionalProperties":{"type":"string"}},"output_prefix":{"type":"string"},"warmup_runs":{"type":"integer","minimum":0},"repetitions":{"type":"integer","minimum":1},"parallelism":{"type":"integer","minimum":1},"idempotency_key":{"type":"string"}}),
            &[
                "name",
                "workload_template",
                "workload_version",
                "output_prefix",
                "repetitions",
                "parallelism",
                "idempotency_key",
            ],
        );
        bplan["required"]
            .as_array_mut()
            .unwrap()
            .push(json!("definition"));
        let benchmark = schema_union(
            std::iter::once(bplan)
                .chain(
                    ["start", "status", "results", "cancel"]
                        .map(|a| action_schema(a, Some("benchmark_id"))),
                )
                .collect(),
        );
        let mut create = action_schema("create", None);
        create["properties"]["data"] = data_schema();
        for (k, v) in [
            ("expression", json!({"type":"string"})),
            ("timezone", json!({"type":"string"})),
            (
                "max_runs",
                json!({"type":"integer","minimum":1,"maximum":self.config.max_schedule_runs}),
            ),
            ("idempotency_key", json!({"type":"string"})),
        ] {
            create["properties"][k] = v;
        }
        create["required"] = json!([
            "action",
            "data",
            "expression",
            "timezone",
            "max_runs",
            "idempotency_key"
        ]);
        let schedule = schema_union(
            std::iter::once(create)
                .chain(std::iter::once(action_schema("list", None)))
                .chain(["pause", "resume", "remove"].map(|a| action_schema(a, Some("schedule_id"))))
                .collect(),
        );
        [("nomad_offload_task","Durable scoped template plan/submit/status/result/cancel. Real bounded logs and approved output artifacts with digests; unknown mutations never replay.",offload),("nomad_benchmark","Durable finite reproducible warmup/measured run set. Exact approved templates, bounded parallelism, result verification and node-condition comparability.",benchmark),("nomad_reconcile","Read remote jobs against exact owned scoped manifests. Never adopts unknown jobs or replays uncertain effects.",closed(json!({}),&[])),("nomad_schedule","Finite durable approved template schedules. Fresh discovery and optimistic plan per occurrence; explicit supervisor tick, no hidden worker.",schedule)].into_iter().map(|(n,d,p)|ToolDefinition{name:n.into(),description:d.into(),parameters:p,side_effect:true}).collect()
    }
    fn configuration_fingerprint(&self) -> String {
        format!("workloads-v1:{}", self.fingerprint)
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        let check = || -> Result<()> {
            ensure!(
                self.config.enabled && !ctx.cancellation.is_cancelled(),
                "workloads disabled/cancelled"
            );
            ctx.scope.validate()?;
            ensure!(
                !ctx.turn_id.is_empty()
                    && ctx.turn_id.len() <= 256
                    && serde_json::to_vec(&call.arguments)?.len() <= 65536,
                "workload request bounds exceeded"
            );
            match call.name.as_str() {
                "nomad_offload_task" => {
                    let a: OffloadArgs = serde_json::from_value(call.arguments.clone())?;
                    if let OffloadArgs::Plan { data } = a {
                        self.validate_data(&data)?;
                    }
                }
                "nomad_benchmark" => {
                    let a: BenchmarkArgs = serde_json::from_value(call.arguments.clone())?;
                    if let BenchmarkArgs::Plan { definition } = a {
                        self.validate_benchmark(&definition)?;
                    }
                }
                "nomad_schedule" => {
                    let _: ScheduleArgs = serde_json::from_value(call.arguments.clone())?;
                }
                "nomad_reconcile" => ensure!(
                    call.arguments.as_object().is_some_and(|v| v.is_empty()),
                    "reconcile takes no parameters"
                ),
                _ => bail!("unknown workload tool"),
            };
            Ok(())
        };
        check().map_err(|e| {
            ToolDispatchRefused {
                reason: e.to_string(),
            }
            .into()
        })
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.preflight(ctx, call).await?;
        let _serial = tokio::select! {_=ctx.cancellation.cancelled()=>bail!("workload cancelled"),s=self.serial.lock()=>s};
        let v = match call.name.as_str() {
            "nomad_offload_task" => {
                match serde_json::from_value::<OffloadArgs>(call.arguments.clone())? {
                    OffloadArgs::Plan { data } => {
                        serde_json::to_value(self.plan(ctx, data).await?)?
                    }
                    OffloadArgs::Submit { task_id } => {
                        serde_json::to_value(self.submit(ctx, &task_id).await?)?
                    }
                    OffloadArgs::Status { task_id } => self.observe(ctx, &task_id, false).await?,
                    OffloadArgs::Result { task_id } => self.observe(ctx, &task_id, true).await?,
                    OffloadArgs::Cancel { task_id } => {
                        serde_json::to_value(self.cancel_task(ctx, &task_id).await?)?
                    }
                }
            }
            "nomad_benchmark" => {
                match serde_json::from_value::<BenchmarkArgs>(call.arguments.clone())? {
                    BenchmarkArgs::Plan { definition } => {
                        serde_json::to_value(self.plan_benchmark(ctx, definition).await?)?
                    }
                    BenchmarkArgs::Start { benchmark_id } => {
                        self.advance_benchmark(ctx, &benchmark_id, true, false)
                            .await?
                    }
                    BenchmarkArgs::Status { benchmark_id } => {
                        self.advance_benchmark(ctx, &benchmark_id, false, false)
                            .await?
                    }
                    BenchmarkArgs::Results { benchmark_id } => {
                        self.advance_benchmark(ctx, &benchmark_id, false, true)
                            .await?
                    }
                    BenchmarkArgs::Cancel { benchmark_id } => {
                        self.cancel_benchmark(ctx, &benchmark_id).await?
                    }
                }
            }
            "nomad_reconcile" => self.reconcile(ctx).await?,
            "nomad_schedule" => {
                self.schedule_call(ctx, serde_json::from_value(call.arguments.clone())?)
                    .await?
            }
            _ => bail!("unknown workload tool"),
        };
        Ok(json!({"trust":"untrusted_data","data":v}))
    }
}
