use anyhow::{Result, bail, ensure};
use missy_core::ToolContext;
use missy_policy::PolicyHttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
#[derive(Default)]
pub struct FoundryState {
    listed: bool,
    repos: HashSet<String>,
    plans: BTreeMap<String, Value>,
    binding: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FoundryConfig {
    pub base_url: String,
    pub project_id: String,
    pub token_env: String,
    pub repository_ids: Vec<String>,
    pub run_ids: Vec<String>,
    pub snapshot_ids: Vec<String>,
    pub allow_mutations: bool,
    pub reviewed_workload_sha256: Vec<String>,
    pub idempotency_prefix: String,
}
fn id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s != "."
        && s != ".."
        && s.as_bytes()[0].is_ascii_alphanumeric()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn repo_id(s: &str) -> bool {
    let s = if let Some((prefix, tail)) = s.split_once(':') {
        if !id(prefix) {
            return false;
        }
        tail
    } else {
        s
    };
    let p: Vec<_> = s.split('/').collect();
    (1..=2).contains(&p.len()) && p.iter().all(|p| id(p))
}
impl FoundryConfig {
    pub fn validate(&self) -> Result<()> {
        super::validate_base(&self.base_url, true)?;
        let url = url::Url::parse(&self.base_url)?;
        ensure!(
            url.scheme() == "https"
                || matches!(
                    url.host_str(),
                    Some("localhost" | "127.0.0.1" | "[::1]" | "::1")
                ),
            "Foundry bearer requires HTTPS except explicit loopback fixtures"
        );
        ensure!(id(&self.project_id), "invalid project id");
        ensure!(
            !self.token_env.is_empty()
                && self.token_env.len() <= 128
                && self
                    .token_env
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "invalid credential env name"
        );
        ensure!(
            self.repository_ids.len() <= 128 && self.repository_ids.iter().all(|s| repo_id(s)),
            "invalid repository allowlist"
        );
        for ids in [&self.run_ids, &self.snapshot_ids] {
            ensure!(
                ids.len() <= 128 && ids.iter().all(|s| id(s)),
                "invalid resource allowlist"
            );
        }
        ensure!(
            self.reviewed_workload_sha256.len() <= 128
                && self.reviewed_workload_sha256.iter().all(|s| digest(s)),
            "invalid workload digest allowlist"
        );
        ensure!(
            self.idempotency_prefix.len() <= 64
                && self
                    .idempotency_prefix
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b)),
            "invalid idempotency namespace"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum FoundryRequest {
    List {},
    Plan {
        workload: Value,
    },
    Start {
        plan_id: String,
        idempotency_key: String,
        acknowledge_project_scope: bool,
    },
    Status {
        resource_type: ResourceType,
        resource_id: String,
    },
    Artifacts {
        run_id: String,
    },
    Compare {
        run_ids: Vec<String>,
    },
    Report {
        run_ids: Vec<String>,
    },
    Snapshot {
        repository_id: String,
        commit_sha: String,
        idempotency_key: String,
        acknowledge_project_scope: bool,
    },
    Cancel {
        run_id: String,
        acknowledge_project_scope: bool,
    },
}
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceType {
    Run,
    Snapshot,
}
impl FoundryRequest {
    fn mutation(&self) -> bool {
        matches!(
            self,
            Self::Snapshot { .. } | Self::Cancel { .. } | Self::Start { .. }
        )
    }
    pub fn validate(&self, c: &FoundryConfig, surface: &str) -> Result<()> {
        ensure!(
            !self.mutation() || (c.allow_mutations && surface != "repoeval_foundry_read"),
            "Foundry mutation denied"
        );
        ensure!(
            surface != "repoeval_foundry_mutate" || self.mutation(),
            "use read surface"
        );
        let run = |s: &String| -> Result<()> {
            ensure!(
                id(s) && c.run_ids.contains(s),
                "run not operator-allowlisted"
            );
            Ok(())
        };
        match self {
            Self::List {} => {}
            Self::Plan { workload } => validate_workload(c, workload)?,
            Self::Start {
                plan_id,
                idempotency_key,
                acknowledge_project_scope,
            } => {
                ensure!(
                    id(plan_id) && *acknowledge_project_scope,
                    "reviewed plan and acknowledgement required"
                );
                key(c, idempotency_key)?;
            }
            Self::Status {
                resource_type,
                resource_id,
            } => {
                let ids = match resource_type {
                    ResourceType::Run => &c.run_ids,
                    ResourceType::Snapshot => &c.snapshot_ids,
                };
                ensure!(
                    id(resource_id) && ids.contains(resource_id),
                    "resource not operator-allowlisted"
                );
            }
            Self::Artifacts { run_id } => run(run_id)?,
            Self::Cancel {
                run_id,
                acknowledge_project_scope,
            } => {
                run(run_id)?;
                ensure!(
                    *acknowledge_project_scope,
                    "project acknowledgement required"
                );
            }
            Self::Compare { run_ids } | Self::Report { run_ids } => {
                ensure!(
                    (1..=16).contains(&run_ids.len())
                        && (!matches!(self, Self::Compare { .. }) || run_ids.len() >= 2),
                    "invalid run count"
                );
                let mut set = std::collections::HashSet::new();
                for s in run_ids {
                    run(s)?;
                    ensure!(set.insert(s), "duplicate run");
                }
            }
            Self::Snapshot {
                repository_id,
                commit_sha,
                idempotency_key,
                acknowledge_project_scope,
            } => {
                ensure!(
                    *acknowledge_project_scope && c.repository_ids.contains(repository_id),
                    "repository/project scope denied"
                );
                ensure!(
                    matches!(commit_sha.len(), 40 | 64)
                        && commit_sha
                            .bytes()
                            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                    "immutable lowercase commit SHA required"
                );
                key(c, idempotency_key)?;
            }
        }
        Ok(())
    }
    pub fn wire(&self) -> (&'static str, String, Option<Value>, Option<String>) {
        match self {
            Self::List {} => ("GET", "/repositories".into(), None, None),
            Self::Plan { workload } => (
                "POST",
                "/benchmark/plan".into(),
                Some(json!({"workload":workload})),
                None,
            ),
            Self::Start {
                plan_id,
                idempotency_key,
                ..
            } => (
                "POST",
                "/benchmark/start".into(),
                Some(json!({"plan_id":plan_id})),
                Some(idempotency_key.clone()),
            ),
            Self::Status {
                resource_type,
                resource_id,
            } => (
                "GET",
                format!(
                    "/{}/{}",
                    match resource_type {
                        ResourceType::Run => "runs",
                        ResourceType::Snapshot => "snapshots",
                    },
                    resource_id
                ),
                None,
                None,
            ),
            Self::Artifacts { run_id } => ("GET", format!("/runs/{run_id}/artifacts"), None, None),
            Self::Compare { run_ids } => (
                "POST",
                "/compare".into(),
                Some(json!({"run_ids":run_ids})),
                None,
            ),
            Self::Report { run_ids } => (
                "POST",
                "/report".into(),
                Some(json!({"run_ids":run_ids})),
                None,
            ),
            Self::Snapshot {
                repository_id,
                commit_sha,
                idempotency_key,
                ..
            } => (
                "POST",
                "/snapshots".into(),
                Some(json!({"repository_id":repository_id,"commit_sha":commit_sha})),
                Some(idempotency_key.clone()),
            ),
            Self::Cancel { run_id, .. } => (
                "POST",
                format!("/runs/{run_id}/cancel"),
                Some(json!({})),
                None,
            ),
        }
    }
}
pub async fn execute(
    http: &PolicyHttpClient,
    c: &FoundryConfig,
    state: &mut FoundryState,
    ctx: &ToolContext,
    r: FoundryRequest,
) -> Result<Value> {
    let token = std::env::var(&c.token_env)
        .map_err(|_| anyhow::anyhow!("Foundry credential unavailable"))?;
    execute_authenticated(http, c, state, ctx, r, &token).await
}
pub(crate) async fn execute_authenticated(
    http: &PolicyHttpClient,
    c: &FoundryConfig,
    state: &mut FoundryState,
    ctx: &ToolContext,
    r: FoundryRequest,
    token: &str,
) -> Result<Value> {
    r.validate(
        c,
        if r.mutation() {
            "repoeval_foundry_mutate"
        } else {
            "repoeval_foundry_read"
        },
    )?;
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "Foundry cancelled before dispatch"
    );
    let binding = identity(&json!([
        ctx.scope,
        hex::encode(Sha256::digest(token.as_bytes()))
    ]));
    if binding != state.binding {
        *state = FoundryState {
            binding,
            ..Default::default()
        };
    }
    match &r {
        FoundryRequest::List {} => {
            state.listed = false;
            state.repos.clear();
            state.plans.clear();
        }
        FoundryRequest::Plan { workload } => {
            state.plans.clear();
            ensure!(
                state.listed
                    && state.repos.contains(
                        workload
                            .pointer("/repository/repository_id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    ),
                "repository must be present in fresh authenticated list"
            );
        }
        FoundryRequest::Snapshot { repository_id, .. } => ensure!(
            state.listed && state.repos.contains(repository_id),
            "repository must be present in authenticated list"
        ),
        FoundryRequest::Start { plan_id, .. } => ensure!(
            state.plans.get(plan_id).is_some_and(approved),
            "reviewed staging plan required"
        ),
        _ => {}
    }
    ensure!(
        !token.is_empty()
            && token.len() <= 4096
            && token
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._~+/-=".contains(&b)),
        "Foundry credential invalid"
    );
    let (method, path, body, key) = r.wire();
    let mut headers = vec![
        ("Authorization".into(), format!("Bearer {token}")),
        ("Accept".into(), "application/json".into()),
    ];
    if let Some(k) = key {
        headers.push(("Idempotency-Key".into(), k));
    }
    let base = c.base_url.trim_end_matches('/');
    let url = format!("{base}/projects/{}{path}", c.project_id);
    let body = body.map(|v| serde_json::to_vec(&v)).transpose()?;
    if body.is_some() {
        headers.push(("Content-Type".into(), "application/json".into()));
    }
    let response = tokio::select! { _ = ctx.cancellation.cancelled() => bail!("Foundry cancelled; remote mutation outcome unknown"), v = http.request(method,&url,headers,body) => v? };
    ensure!(
        response.status == if r.mutation() { 202 } else { 200 },
        "Foundry response status rejected; mutation outcome may be unknown"
    );
    ensure!(response.body.len() <= 1024 * 1024, "Foundry response cap");
    let envelope: Value = serde_json::from_slice(&response.body)
        .map_err(|_| anyhow::anyhow!("invalid Foundry JSON"))?;
    ensure!(
        !serde_json::to_string(&envelope)?.contains(token),
        "Foundry response echoed credential"
    );
    ensure!(
        envelope.get("ok") == Some(&Value::Bool(true)),
        "Foundry response missing success envelope"
    );
    let result = envelope
        .get("data")
        .ok_or_else(|| anyhow::anyhow!("Foundry data missing"))?;
    validate_response(c, &r, result)?;
    match &r {
        FoundryRequest::List {} => {
            state.repos = result
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| c.repository_ids.iter().any(|r| r == s))
                .map(str::to_owned)
                .collect();
            state.listed = true;
        }
        FoundryRequest::Plan { .. } => {
            state
                .plans
                .insert(result["id"].as_str().unwrap().into(), result.clone());
        }
        FoundryRequest::Start {
            plan_id,
            idempotency_key,
            ..
        } => validate_start(
            c,
            state.plans.get(plan_id).unwrap(),
            plan_id,
            idempotency_key,
            result,
        )?,
        _ => {}
    }
    Ok(
        json!({"route_project_id":c.project_id,"scope_source":"authenticated_project_route","response":safe(result,0),"execution_confirmed":false}),
    )
}
pub fn validate_response(c: &FoundryConfig, r: &FoundryRequest, v: &Value) -> Result<()> {
    if !matches!(r, FoundryRequest::List {} | FoundryRequest::Cancel { .. }) {
        ensure!(
            v.is_object()
                && v.get("project_id").and_then(Value::as_str) == Some(c.project_id.as_str()),
            "Foundry project identity mismatch"
        );
    }
    match r {
        FoundryRequest::Plan { workload } => ensure!(
            v.get("id").and_then(Value::as_str).is_some_and(id)
                && approved(v)
                && v.get("workload") == Some(workload),
            "plan not reviewed staging placement or workload mismatch"
        ),
        FoundryRequest::Start { plan_id, .. } => ensure!(
            v.get("plan_id").and_then(Value::as_str) == Some(plan_id)
                && v.get("id").and_then(Value::as_str).is_some_and(id),
            "start identity mismatch"
        ),
        FoundryRequest::List {} => {
            let rows = v
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("invalid repository list"))?;
            ensure!(rows.len() <= 1000, "repository count cap");
            let mut seen = std::collections::HashSet::new();
            for row in rows {
                let s = row
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("repository id required"))?;
                ensure!(
                    repo_id(s) && seen.insert(s),
                    "invalid/duplicate repository id"
                );
            }
        }
        FoundryRequest::Status { resource_id, .. } => ensure!(
            v.get("id").and_then(Value::as_str) == Some(resource_id),
            "resource identity mismatch"
        ),
        FoundryRequest::Artifacts { run_id } => {
            ensure!(
                exact_keys(v, &["project_id", "run_id", "artifacts"]),
                "invalid artifact response fields"
            );
            ensure!(
                v.get("run_id").and_then(Value::as_str) == Some(run_id),
                "artifact run mismatch"
            );
            let rows = v
                .get("artifacts")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("artifacts missing"))?;
            ensure!(rows.len() <= 100, "artifact cap");
            let mut seen = std::collections::HashSet::new();
            for row in rows {
                let aid = row.get("artifact_id").and_then(Value::as_str).unwrap_or("");
                let hash = row.get("sha256").and_then(Value::as_str).unwrap_or("");
                let kind = row.get("kind").and_then(Value::as_str).unwrap_or("");
                let media = row.get("media_type").and_then(Value::as_str).unwrap_or("");
                ensure!(
                    exact_keys(
                        row,
                        &[
                            "artifact_id",
                            "run_id",
                            "kind",
                            "sha256",
                            "size_bytes",
                            "media_type"
                        ]
                    ) && id(aid)
                        && aid.starts_with("artifact-")
                        && seen.insert(aid)
                        && row.get("run_id").and_then(Value::as_str) == Some(run_id)
                        && digest(hash)
                        && id(kind)
                        && kind.len() <= 64
                        && kind.bytes().all(|b| b.is_ascii_lowercase()
                            || b.is_ascii_digit()
                            || b"._-".contains(&b))
                        && !media.is_empty()
                        && media.len() <= 255
                        && !media.contains("://")
                        && !media.contains(char::is_control)
                        && row
                            .get("size_bytes")
                            .and_then(Value::as_u64)
                            .is_some_and(|n| n <= 64_000_000),
                    "artifact metadata invalid"
                );
            }
        }
        FoundryRequest::Compare { run_ids } => validate_compare(run_ids, v)?,
        FoundryRequest::Report { run_ids } => validate_report(run_ids, v)?,
        FoundryRequest::Snapshot {
            repository_id,
            commit_sha,
            ..
        } => ensure!(
            v.get("id").and_then(Value::as_str).is_some_and(id)
                && v.get("state").and_then(Value::as_str) == Some("requested")
                && v.get("repository_id").and_then(Value::as_str) == Some(repository_id)
                && v.get("commit_sha").and_then(Value::as_str) == Some(commit_sha),
            "snapshot identities mismatch"
        ),
        FoundryRequest::Cancel { run_id, .. } => ensure!(
            v.as_object().is_some_and(|o| o.len() == 3)
                && v.get("id").and_then(Value::as_str) == Some(run_id)
                && matches!(
                    v.get("state").and_then(Value::as_str),
                    Some("cancelled" | "cancel_pending" | "failed")
                )
                && v.get("cancel_requested") == Some(&Value::Bool(true)),
            "cancel identity mismatch"
        ),
    }
    Ok(())
}
fn safe(v: &Value, depth: u32) -> Value {
    if depth >= 8 {
        return json!("[omitted]");
    }
    match v {
        Value::Object(o) => Value::Object(
            o.iter()
                .take(100)
                .map(|(k, v)| {
                    let key = k.to_ascii_lowercase();
                    let redacted = [
                        "token",
                        "password",
                        "secret",
                        "authorization",
                        "cookie",
                        "credential",
                        "url",
                        "uri",
                        "href",
                        "location",
                        "stdout",
                        "stderr",
                        "environment",
                        "content",
                    ]
                    .iter()
                    .any(|s| key.contains(s));
                    (
                        k.clone(),
                        if redacted {
                            json!("[redacted]")
                        } else {
                            safe(v, depth + 1)
                        },
                    )
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().take(100).map(|v| safe(v, depth + 1)).collect()),
        Value::String(s) => {
            let l = s.to_ascii_lowercase();
            if l.contains("://")
                || l.contains("bearer ")
                || ["token=", "password=", "secret=", "api_key="]
                    .iter()
                    .any(|s| l.contains(s))
            {
                json!("[redacted]")
            } else {
                json!(s.chars().take(2048).collect::<String>())
            }
        }
        _ => v.clone(),
    }
}
pub fn schema(mutations: bool) -> Value {
    let actions = if mutations {
        json!([
            "list",
            "plan",
            "start",
            "status",
            "artifacts",
            "compare",
            "report",
            "snapshot",
            "cancel"
        ])
    } else {
        json!(["list", "plan", "status", "artifacts", "compare", "report"])
    };
    super::schema(
        json!({"action":{"enum":actions},"workload":{"type":"object"},"plan_id":{"type":"string"},"resource_type":{"enum":["run","snapshot"]},"resource_id":{"type":"string"},"run_id":{"type":"string"},"run_ids":{"type":"array","items":{"type":"string"},"maxItems":16},"repository_id":{"type":"string"},"commit_sha":{"type":"string"},"idempotency_key":{"type":"string"},"acknowledge_project_scope":{"type":"boolean"}}),
        &["action"],
    )
}
fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn exact_keys(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
}
fn reason(s: &str) -> bool {
    [
        "repository.repository_id",
        "repository.commit_sha",
        "repository.snapshot_id",
        "repository.subdirectory",
        "task.class",
        "task.prompt_sha256",
        "task.fixture_digests",
        "task.tool_schema_uris",
        "workload_id",
        "workload_version",
        "definition_sha256",
        "sandbox.image_digest",
        "sandbox.network_policy",
        "sandbox.cpu_mhz",
        "sandbox.memory_mb",
        "sandbox.disk_mb",
        "sandbox.timeout_seconds",
        "sandbox.architecture",
        "evaluator_version",
        "validators",
        "provider_independent_settings_sha256",
        "tool_schemas_sha256",
        "comparability_key_invalid",
        "comparability_key_mismatch",
        "comparability_rules_version_mismatch",
        "claimed_key_mismatch",
        "missing_or_invalid_comparability_key",
    ]
    .contains(&s)
}
fn reasons(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| a.len() <= 32 && a.iter().all(|v| v.as_str().is_some_and(reason)))
}
fn tagged_digest(s: &str, prefix: &str) -> bool {
    s.strip_prefix(prefix).is_some_and(|s| {
        s.len() == 24
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}
fn validate_compare(ids: &[String], v: &Value) -> Result<()> {
    ensure!(
        exact_keys(
            v,
            &[
                "project_id",
                "comparison_id",
                "run_ids",
                "comparable",
                "pairs"
            ]
        ) && v.get("run_ids") == Some(&json!(ids))
            && v.get("comparison_id")
                .and_then(Value::as_str)
                .is_some_and(|s| tagged_digest(s, "comparison-")),
        "invalid comparison identity"
    );
    let pairs = v
        .get("pairs")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("comparison pairs absent"))?;
    ensure!(
        pairs.len() == ids.len() * (ids.len() - 1) / 2,
        "comparison pair count mismatch"
    );
    let mut seen = HashSet::new();
    let mut all = true;
    for pair in pairs {
        ensure!(
            exact_keys(
                pair,
                &[
                    "left_run_id",
                    "right_run_id",
                    "comparable",
                    "key",
                    "reasons"
                ]
            ),
            "invalid pair fields"
        );
        let a = pair["left_run_id"].as_str().unwrap_or("");
        let b = pair["right_run_id"].as_str().unwrap_or("");
        let ai = ids
            .iter()
            .position(|s| s == a)
            .ok_or_else(|| anyhow::anyhow!("foreign comparison run"))?;
        let bi = ids
            .iter()
            .position(|s| s == b)
            .ok_or_else(|| anyhow::anyhow!("foreign comparison run"))?;
        ensure!(
            ai != bi && seen.insert((ai.min(bi), ai.max(bi))),
            "duplicate comparison pair"
        );
        let comparable = pair["comparable"]
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("comparable must be bool"))?;
        ensure!(reasons(&pair["reasons"]), "invalid comparison reasons");
        if comparable {
            ensure!(
                pair["reasons"].as_array().unwrap().is_empty()
                    && pair["key"].as_str().is_some_and(digest),
                "comparable evidence invalid"
            );
        } else {
            ensure!(
                pair["key"].is_null() && !pair["reasons"].as_array().unwrap().is_empty(),
                "incomparable evidence invalid"
            );
        }
        all &= comparable;
    }
    ensure!(
        v["comparable"].as_bool() == Some(all),
        "aggregate comparability mismatch"
    );
    Ok(())
}
fn validate_report(ids: &[String], v: &Value) -> Result<()> {
    ensure!(
        exact_keys(
            v,
            &[
                "project_id",
                "report_id",
                "status",
                "published",
                "run_ids",
                "schema_version",
                "groups",
                "incomparable"
            ]
        ) && v["run_ids"] == json!(ids)
            && v["report_id"]
                .as_str()
                .is_some_and(|s| tagged_digest(s, "report-"))
            && v["status"] == "draft"
            && v["published"] == false
            && v["schema_version"] == "1.0",
        "invalid draft report"
    );
    let groups = v["groups"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("report groups required"))?;
    let excluded = v["incomparable"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("report exclusions required"))?;
    ensure!(
        groups.len() + excluded.len() <= ids.len(),
        "report group cap"
    );
    let mut seen = HashSet::new();
    for group in groups {
        ensure!(
            exact_keys(group, &["comparability_key", "run_ids"])
                && group["comparability_key"].as_str().is_some_and(digest),
            "invalid report group"
        );
        let rows = group["run_ids"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("report run ids required"))?;
        ensure!(
            !rows.is_empty() && rows.len() <= ids.len(),
            "invalid report group size"
        );
        for row in rows {
            let s = row.as_str().unwrap_or("");
            ensure!(
                ids.iter().any(|id| id == s) && seen.insert(s),
                "foreign or duplicate report run"
            );
        }
    }
    for row in excluded {
        let s = row["run_id"].as_str().unwrap_or("");
        ensure!(
            exact_keys(row, &["run_id", "reason"])
                && ids.iter().any(|id| id == s)
                && seen.insert(s)
                && reasons(&row["reason"])
                && row["reason"].as_array().is_some_and(|a| !a.is_empty()),
            "invalid report exclusion"
        );
    }
    ensure!(seen.len() == ids.len(), "report coverage incomplete");
    Ok(())
}
pub fn identity(v: &Value) -> String {
    hex::encode(Sha256::digest(serde_json::to_vec(v).expect("JSON value")))
}
fn key(c: &FoundryConfig, s: &str) -> Result<()> {
    let prefix = if c.idempotency_prefix.is_empty() {
        "missy-"
    } else {
        &c.idempotency_prefix
    };
    ensure!(
        (8..=128).contains(&s.len())
            && s.starts_with(prefix)
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b)),
        "explicit namespaced idempotency key required"
    );
    Ok(())
}
fn approved(v: &Value) -> bool {
    v.get("state").and_then(Value::as_str) == Some("planned")
        && v.pointer("/placement/pool").and_then(Value::as_str) == Some("staging")
        && [
            "repository",
            "image",
            "providers",
            "budget",
            "quota",
            "egress",
            "audit",
        ]
        .iter()
        .all(|k| v.get("policy_checks").and_then(|v| v.get(*k)) == Some(&Value::Bool(true)))
}
fn validate_workload(c: &FoundryConfig, w: &Value) -> Result<()> {
    let o = w
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("workload object required"))?;
    ensure!(
        o.keys().all(|k| [
            "schema_version",
            "id",
            "version",
            "description",
            "repository",
            "task",
            "providers",
            "sandbox",
            "execution",
            "validation",
            "artifacts"
        ]
        .contains(&k.as_str())),
        "unknown workload field"
    );
    ensure!(
        serde_json::to_vec(w)?.len() <= 128 * 1024
            && c.reviewed_workload_sha256.contains(&identity(w)),
        "workload not operator-digest-reviewed"
    );
    let rid = w
        .pointer("/repository/repository_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        c.repository_ids.iter().any(|s| s == rid),
        "workload repository denied"
    );
    let sha = w
        .pointer("/repository/commit_sha")
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        matches!(sha.len(), 40 | 64)
            && sha
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "workload commit must be immutable"
    );
    ensure!(
        w.pointer("/repository/snapshot_id")
            .and_then(Value::as_str)
            .is_some_and(id),
        "snapshot id required"
    );
    let image = w
        .pointer("/sandbox/image_digest")
        .and_then(Value::as_str)
        .unwrap_or("");
    let (name, hash) = image
        .split_once("@sha256:")
        .ok_or_else(|| anyhow::anyhow!("pinned image digest required"))?;
    ensure!(
        !name.is_empty()
            && !name.contains(char::is_whitespace)
            && !name.contains('@')
            && digest(hash),
        "invalid pinned image"
    );
    for (k, max) in [
        ("repetitions", 32),
        ("parallelism", 8),
        ("timeout_seconds", 86400),
        ("max_attempts", 3),
    ] {
        ensure!(
            w.get("execution")
                .and_then(|v| v.get(k))
                .and_then(Value::as_u64)
                .is_some_and(|n| n >= 1 && n <= max),
            "execution bounds required"
        );
    }
    ensure!(
        w.get("providers")
            .and_then(Value::as_array)
            .is_some_and(|a| (1..=8).contains(&a.len())),
        "provider count bounds"
    );
    ensure!(
        no_secrets(w, 0),
        "workload contains forbidden credential fields"
    );
    Ok(())
}
fn no_secrets(v: &Value, depth: usize) -> bool {
    if depth > 10 {
        return false;
    }
    match v {
        Value::Object(o) => {
            o.len() <= 100
                && o.iter().all(|(k, v)| {
                    ![
                        "token",
                        "password",
                        "secret",
                        "api_key",
                        "authorization",
                        "cookie",
                        "credential",
                        "private_key",
                        "url",
                        "uri",
                        "href",
                        "location",
                        "stdout",
                        "stderr",
                        "raw_output",
                        "environment",
                        "env",
                        "content",
                    ]
                    .contains(&k.to_ascii_lowercase().as_str())
                        && no_secrets(v, depth + 1)
                })
        }
        Value::Array(a) => a.len() <= 100 && a.iter().all(|v| no_secrets(v, depth + 1)),
        Value::String(s) => s.len() <= 2048 && !s.to_ascii_lowercase().contains("bearer "),
        _ => true,
    }
}
pub fn validate_start(
    c: &FoundryConfig,
    plan: &Value,
    plan_id: &str,
    key: &str,
    v: &Value,
) -> Result<()> {
    ensure!(
        v.get("project_id").and_then(Value::as_str) == Some(c.project_id.as_str())
            && v.get("plan_id").and_then(Value::as_str) == Some(plan_id)
            && v.get("id").and_then(Value::as_str).is_some_and(id),
        "start project/plan/run mismatch"
    );
    if v.get("state").and_then(Value::as_str) == Some("submitted") {
        ensure!(
            v.get("job_id").and_then(Value::as_str).is_some_and(id),
            "submitted job identity required"
        );
        return Ok(());
    }
    ensure!(
        v.get("state").and_then(Value::as_str) == Some("reserved"),
        "unknown start state"
    );
    let expected = format!(
        "run-{}",
        &identity(&json!([c.project_id, plan_id, key]))[..24]
    );
    ensure!(
        v.as_object().is_some_and(|o| o.len() == 8)
            && v.get("id").and_then(Value::as_str) == Some(&expected)
            && v.get("job_id") == Some(&Value::Null)
            && v.get("cancel_requested") == Some(&Value::Bool(false)),
        "invalid inert reservation"
    );
    let repetitions = plan
        .pointer("/workload/execution/repetitions")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let providers = plan
        .pointer("/workload/providers")
        .and_then(Value::as_array)
        .map_or(0, Vec::len) as u64;
    ensure!(
        (1..=32).contains(&repetitions) && (1..=8).contains(&providers),
        "invalid cached plan execution"
    );
    let children = v
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("children missing"))?;
    ensure!(
        children.len() as u64 == providers * repetitions,
        "reservation child count mismatch"
    );
    let mut seen = HashSet::new();
    for child in children {
        let i = child
            .get("provider_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("provider index missing"))?;
        let r = child
            .get("repetition")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("repetition missing"))?;
        let cid = format!("run-{}", &identity(&json!([expected, i, r]))[..24]);
        ensure!(
            i < providers
                && r < repetitions
                && seen.insert((i, r))
                && child.as_object().is_some_and(|o| o.len() == 5)
                && child.get("run_id").and_then(Value::as_str) == Some(&cid)
                && child.get("job_id").and_then(Value::as_str) == Some(&format!("foundry-{cid}"))
                && child.get("state").and_then(Value::as_str) == Some("reserved"),
            "reservation child identity mismatch"
        );
    }
    let required = v
        .get("required_artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("required artifacts absent"))?;
    ensure!(required.len() <= 32, "artifact kind cap");
    let mut kinds = HashSet::new();
    for kind in required {
        let s = kind.as_str().unwrap_or("");
        ensure!(
            id(s)
                && s.len() <= 64
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
                && kinds.insert(s),
            "invalid artifact kinds"
        );
    }
    if plan
        .pointer("/workload/artifacts")
        .is_some_and(Value::is_object)
    {
        ensure!(
            plan.pointer("/workload/artifacts/required_kinds") == Some(&json!(required)),
            "artifact policy mismatch"
        );
    }
    Ok(())
}
