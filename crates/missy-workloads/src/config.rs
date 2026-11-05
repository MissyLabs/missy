use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkloadsConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub token_env: Option<String>,
    pub namespace: String,
    pub owner: String,
    pub templates: BTreeMap<String, WorkloadTemplate>,
    pub allowed_node_ids: BTreeSet<String>,
    pub approved_artifact_prefixes: BTreeSet<String>,
    pub max_benchmark_runs: usize,
    pub max_benchmark_parallelism: usize,
    pub max_records: usize,
    pub request_timeout_seconds: u64,
    pub max_log_bytes: usize,
    pub max_artifact_bytes: usize,
    pub max_schedule_runs: u32,
}
impl Default for WorkloadsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://127.0.0.1:4646".into(),
            token_env: None,
            namespace: "default".into(),
            owner: "missy-rust".into(),
            templates: BTreeMap::new(),
            allowed_node_ids: BTreeSet::new(),
            approved_artifact_prefixes: BTreeSet::new(),
            max_benchmark_runs: 16,
            max_benchmark_parallelism: 2,
            max_records: 4096,
            request_timeout_seconds: 30,
            max_log_bytes: 65536,
            max_artifact_bytes: 1024 * 1024,
            max_schedule_runs: 100,
        }
    }
}
/// Fully operator-authored finite Docker batch graph. Caller data goes only
/// into named environment slots, never a command, URL, image or template text.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadTemplate {
    pub job: Value,
    #[serde(default)]
    pub parameter_env: BTreeMap<String, String>,
    #[serde(default)]
    pub required_parameters: BTreeSet<String>,
    #[serde(default)]
    pub expected_output: String,
    #[serde(default)]
    pub result_fields: BTreeMap<String, String>,
    pub max_run_seconds: u32,
}
pub(crate) fn ident(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
        "invalid exact identifier"
    );
    Ok(())
}
pub(crate) fn digest(value: &impl Serialize) -> Result<String> {
    use sha2::{Digest, Sha256};
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
pub(crate) fn env_name(s: &str) -> Result<()> {
    ensure!(
        !s.is_empty()
            && s.len() <= 100
            && s.bytes()
                .enumerate()
                .all(|(i, b)| b == b'_' || b.is_ascii_uppercase() || (i > 0 && b.is_ascii_digit())),
        "invalid environment slot"
    );
    let folded = s.to_ascii_lowercase();
    ensure!(
        ![
            "secret",
            "token",
            "password",
            "credential",
            "api_key",
            "authorization"
        ]
        .iter()
        .any(|s| folded.contains(s)),
        "secret-shaped workload data forbidden"
    );
    Ok(())
}
impl WorkloadsConfig {
    pub fn validate(&self) -> Result<()> {
        let u = Url::parse(&self.endpoint)?;
        ensure!(
            matches!(u.scheme(), "http" | "https")
                && u.origin().ascii_serialization() == self.endpoint
                && u.username().is_empty()
                && u.password().is_none()
                && u.query().is_none()
                && u.fragment().is_none(),
            "Nomad endpoint must be canonical exact HTTP(S) origin"
        );
        ident(&self.namespace)?;
        ident(&self.owner)?;
        if let Some(e) = &self.token_env {
            ensure!(
                e.starts_with("MISSY_NOMAD_")
                    && e.len() <= 128
                    && e.bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "dedicated Nomad token reference required"
            );
        }
        ensure!(
            (1..=64).contains(&self.max_benchmark_runs)
                && (1..=16).contains(&self.max_benchmark_parallelism)
                && (1..=100000).contains(&self.max_records)
                && (1..=120).contains(&self.request_timeout_seconds)
                && (1..=262144).contains(&self.max_log_bytes)
                && (1..=4 * 1024 * 1024).contains(&self.max_artifact_bytes)
                && (1..=10000).contains(&self.max_schedule_runs),
            "invalid workload bounds"
        );
        for n in &self.allowed_node_ids {
            ident(n)?;
        }
        if self.enabled {
            ensure!(
                !self.templates.is_empty() && !self.allowed_node_ids.is_empty(),
                "enabled workloads require templates and exact allowed nodes"
            );
        }
        for p in &self.approved_artifact_prefixes {
            let u = Url::parse(p)?;
            ensure!(
                matches!(u.scheme(), "http" | "https")
                    && u.username().is_empty()
                    && u.password().is_none()
                    && u.query().is_none()
                    && u.fragment().is_none()
                    && p.ends_with('/'),
                "artifact prefixes require credential-free HTTP(S) directory URLs"
            );
        }
        for (name, t) in &self.templates {
            ident(name)?;
            ensure!(
                (1..=86400).contains(&t.max_run_seconds),
                "finite template wall time required"
            );
            ensure!(
                t.expected_output.len() <= 4096
                    && t.parameter_env.len() <= 32
                    && t.result_fields.len() <= 32,
                "template bounds exceeded"
            );
            let mut slots = BTreeSet::new();
            for (k, e) in &t.parameter_env {
                ident(k)?;
                env_name(e)?;
                ensure!(
                    !e.starts_with("MISSY_") && slots.insert(e),
                    "runtime-reserved or duplicate environment slot"
                );
            }
            ensure!(
                t.required_parameters
                    .iter()
                    .all(|k| t.parameter_env.contains_key(k)),
                "required parameter has no approved slot"
            );
            for (k, v) in &t.result_fields {
                ident(k)?;
                ensure!(
                    ["string", "number", "integer", "boolean", "object", "array"]
                        .contains(&v.as_str()),
                    "unsupported result type"
                );
            }
            validate_job(&t.job)?;
        }
        Ok(())
    }
    pub(crate) fn artifact(&self, s: &str) -> Result<()> {
        let u = Url::parse(s)?;
        ensure!(
            u.username().is_empty()
                && u.password().is_none()
                && u.fragment().is_none()
                && u.query().is_none()
                && u.as_str() == s,
            "artifact URL must be canonical and credential-free"
        );
        ensure!(
            !u.path().contains('%')
                && self
                    .approved_artifact_prefixes
                    .iter()
                    .any(|p| s.starts_with(p)),
            "artifact URL outside approved storage prefixes"
        );
        Ok(())
    }
}
fn validate_job(job: &Value) -> Result<()> {
    ensure!(
        serde_json::to_vec(job)?.len() <= 262144 && job.is_object(),
        "bounded job graph required"
    );
    let allowed = [
        "Name",
        "Type",
        "Priority",
        "Datacenters",
        "NodePool",
        "TaskGroups",
        "Constraints",
        "Meta",
    ];
    ensure!(
        job.as_object()
            .unwrap()
            .keys()
            .all(|k| allowed.contains(&k.as_str())),
        "unsupported template job field"
    );
    ensure!(job["Type"] == "batch", "only finite batch jobs supported");
    let groups = job["TaskGroups"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("task group required"))?;
    ensure!(groups.len() == 1, "one bounded task group required");
    let g = &groups[0];
    ensure!(
        g["Count"] == 1 && g["Name"].as_str().is_some_and(|s| ident(s).is_ok()),
        "one named group instance required"
    );
    let tasks = g["Tasks"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tasks required"))?;
    ensure!(tasks.len() == 1, "one workload task required");
    let t = &tasks[0];
    ensure!(
        t["Name"] == "workload" && t["Driver"] == "docker",
        "only workload Docker task supported"
    );
    let c = &t["Config"];
    ensure!(
        c.as_object().is_some_and(|m| m
            .keys()
            .all(|k| ["image", "command", "args"].contains(&k.as_str()))),
        "unsafe Docker config field"
    );
    let image = c["image"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("image required"))?;
    let (_, pin) = image
        .rsplit_once("@sha256:")
        .ok_or_else(|| anyhow::anyhow!("digest-pinned image required"))?;
    ensure!(
        pin.len() == 64 && pin.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid image digest"
    );
    ensure!(
        t["Resources"]["CPU"]
            .as_u64()
            .is_some_and(|n| n > 0 && n <= 100000)
            && t["Resources"]["MemoryMB"]
                .as_u64()
                .is_some_and(|n| n > 0 && n <= 1048576),
        "bounded CPU/memory required"
    );
    ensure!(
        t["Resources"]
            .as_object()
            .is_some_and(|m| m.keys().all(|k| ["CPU", "MemoryMB"].contains(&k.as_str()))),
        "resource devices/network/shared allocations forbidden"
    );
    if let Some(disk) = g.get("EphemeralDisk") {
        ensure!(
            disk.as_object().is_some_and(|m| m
                .keys()
                .all(|k| ["SizeMB", "Sticky", "Migrate"].contains(&k.as_str())))
                && disk["Sticky"] != true
                && disk["Migrate"] != true
                && disk["SizeMB"]
                    .as_u64()
                    .is_some_and(|n| n > 0 && n <= 102400),
            "only bounded nonpersistent ephemeral disk supported"
        );
    }
    if let Some(env) = t.get("Env") {
        let m = env
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("env object required"))?;
        for (k, v) in m {
            env_name(k)?;
            ensure!(
                v.as_str()
                    .is_some_and(|s| s.len() <= 4096 && !s.contains('\0')),
                "bounded environment string required"
            );
        }
    }
    if g.as_object().is_none_or(|m| {
        !m.keys().all(|k| {
            [
                "Name",
                "Count",
                "Tasks",
                "EphemeralDisk",
                "RestartPolicy",
                "ReschedulePolicy",
                "Constraints",
            ]
            .contains(&k.as_str())
        })
    }) {
        bail!("unsupported group field");
    }
    ensure!(
        t.as_object().is_some_and(|m| m.keys().all(|k| [
            "Name",
            "Driver",
            "Config",
            "Env",
            "Resources",
            "KillTimeout",
            "ShutdownDelay"
        ]
        .contains(&k.as_str()))),
        "unsupported task field"
    );
    Ok(())
}
pub(crate) fn render(
    t: &WorkloadTemplate,
    params: &BTreeMap<String, String>,
    inputs: &BTreeMap<String, String>,
    outputs: &BTreeMap<String, String>,
    id: &str,
    cfg: &WorkloadsConfig,
    scope_hash: &str,
) -> Result<Value> {
    ensure!(
        params.len() <= 32 && inputs.len() <= 16 && outputs.len() <= 16,
        "workload data map bounds exceeded"
    );
    ensure!(
        t.required_parameters.iter().all(|k| params.contains_key(k)),
        "required parameter missing"
    );
    let mut env = t.job["TaskGroups"][0]["Tasks"][0]["Env"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    for (k, v) in params {
        let e = t
            .parameter_env
            .get(k)
            .ok_or_else(|| anyhow::anyhow!("parameter not allowed"))?;
        ensure!(v.len() <= 4096 && !v.contains('\0'), "parameter too large");
        env.insert(e.clone(), json!(v));
    }
    for (prefix, map) in [("MISSY_INPUT_", inputs), ("MISSY_OUTPUT_", outputs)] {
        for (k, v) in map {
            env_name(k)?;
            cfg.artifact(v)?;
            env.insert(format!("{prefix}{k}"), json!(v));
        }
    }
    let mut job = t.job.clone();
    job["ID"] = json!(id);
    job["Name"] = json!(id);
    job["Namespace"] = json!(cfg.namespace);
    job["Meta"] = json!({"missy_owner":cfg.owner,"missy_scope":scope_hash,"missy_job":id});
    job["TaskGroups"][0]["Tasks"][0]["Env"] = Value::Object(env);
    job["TaskGroups"][0]["RestartPolicy"] =
        json!({"Attempts":0,"Mode":"fail","Interval":60000000000u64,"Delay":1000000000u64});
    job["TaskGroups"][0]["ReschedulePolicy"] = json!({"Attempts":0,"Unlimited":false});
    let mut constraints = job["Constraints"].as_array().cloned().unwrap_or_default();
    constraints.push(json!({"LTarget":"${node.unique.id}","RTarget":format!("^({})$",cfg.allowed_node_ids.iter().cloned().collect::<Vec<_>>().join("|")),"Operand":"regexp"}));
    job["Constraints"] = json!(constraints);
    Ok(job)
}
