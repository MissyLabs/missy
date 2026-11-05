use axum::{
    Router,
    body::Bytes,
    extract::{OriginalUri, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_workloads::{WorkloadTemplate, WorkloadTools, WorkloadsConfig};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio_util::sync::CancellationToken;
#[derive(Default)]
struct Mock {
    jobs: Mutex<BTreeMap<String, Value>>,
    posts: AtomicUsize,
    stops: AtomicUsize,
    complete: AtomicUsize,
    unknown: AtomicUsize,
    tamper: AtomicUsize,
    no_capacity: AtomicUsize,
    wrong_result: AtomicUsize,
    plan_conflict: AtomicUsize,
    calls: Mutex<Vec<(String, String)>>,
}
async fn handler(
    State(s): State<Arc<Mock>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    body: Bytes,
) -> Response {
    let path = uri.path();
    s.calls
        .lock()
        .unwrap()
        .push((method.to_string(), path.into()));
    if path == "/v1/nodes" {
        return axum::Json(json!([{"ID":"node-1","Status":"ready","SchedulingEligibility":"eligible","Drain":false}])).into_response();
    }
    if path == "/v1/node/node-1" {
        if s.no_capacity.load(Ordering::SeqCst) > 0 {
            return axum::Json(json!({"ID":"node-1","Status":"ready","SchedulingEligibility":"eligible","Drain":false,"NodeResources":{"Cpu":{"CpuShares":1},"Memory":{"MemoryMB":1},"Disk":{"DiskMB":1}}})).into_response();
        }
        return axum::Json(json!({"ID":"node-1","Name":"test","Status":"ready","SchedulingEligibility":"eligible","Drain":false,"Attributes":{"cpu.arch":"x86_64"},"NodeResources":{"Cpu":{"CpuShares":10000},"Memory":{"MemoryMB":10000},"Disk":{"DiskMB":10000}}})).into_response();
    }
    if path == "/v1/node/node-1/allocations" {
        return axum::Json(json!([])).into_response();
    }
    if path.ends_with("/plan") {
        if s.plan_conflict.load(Ordering::SeqCst) > 0 {
            return axum::Json(json!({"JobModifyIndex":42,"FailedTGAllocs":{}})).into_response();
        }
        return axum::Json(json!({"JobModifyIndex":0,"FailedTGAllocs":{}})).into_response();
    }
    if path == "/v1/jobs" && method == Method::POST {
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["EnforceIndex"], true);
        assert_eq!(v["JobModifyIndex"], 0);
        let job = v["Job"].clone();
        let id = job["ID"].as_str().unwrap().to_string();
        s.posts.fetch_add(1, Ordering::SeqCst);
        s.jobs.lock().unwrap().insert(id, job);
        if s.unknown.load(Ordering::SeqCst) > 0 {
            return (StatusCode::INTERNAL_SERVER_ERROR, "lost acknowledgement").into_response();
        };
        return axum::Json(json!({"EvalID":"eval-1"})).into_response();
    }
    if let Some(aid) = path.strip_prefix("/v1/client/fs/logs/") {
        let stderr = uri.query().unwrap_or("").contains("type=stderr");
        if stderr {
            return "".into_response();
        };
        let jobs = s.jobs.lock().unwrap();
        let job = jobs
            .get(aid.strip_prefix("alloc-").unwrap_or(""))
            .cloned()
            .unwrap();
        drop(jobs);
        let env = &job["TaskGroups"][0]["Tasks"][0]["Env"];
        if s.wrong_result.load(Ordering::SeqCst) > 0 {
            return "missing marker\nMISSY_RESULT_JSON={\"score\":true,\"outputs\":{}}"
                .into_response();
        }
        let mut outputs = serde_json::Map::new();
        for (k, v) in env.as_object().unwrap() {
            if let Some(k) = k.strip_prefix("MISSY_OUTPUT_") {
                outputs.insert(k.into(), v.clone());
            }
        }
        return format!(
            "DONE\nMISSY_RESULT_JSON={}\n",
            json!({"score":42,"outputs":outputs})
        )
        .into_response();
    }
    if path.starts_with("/artifacts/") {
        return "real fixture artifact bytes".into_response();
    }
    if let Some(rest) = path.strip_prefix("/v1/job/") {
        if let Some(id) = rest.strip_suffix("/allocations") {
            let jobs = s.jobs.lock().unwrap();
            if !jobs.contains_key(id) {
                return StatusCode::NOT_FOUND.into_response();
            };
            return axum::Json(json!([{"ID":format!("alloc-{id}"),"JobID":id,"Namespace":"test","NodeID":"node-1","ClientStatus":if s.complete.load(Ordering::SeqCst)>0{"complete"}else{"running"},"TaskStates":{"workload":{"Failed":false}}}])).into_response();
        }
        let mut jobs = s.jobs.lock().unwrap();
        let Some(job) = jobs.get_mut(rest) else {
            return StatusCode::NOT_FOUND.into_response();
        };
        if method == Method::DELETE {
            s.stops.fetch_add(1, Ordering::SeqCst);
            job["Stop"] = json!(true);
            return axum::Json(json!({"EvalID":"stop-1"})).into_response();
        }
        let mut job = job.clone();
        if s.tamper.load(Ordering::SeqCst) > 0 {
            job["TaskGroups"][0]["Tasks"][0]["Config"]["privileged"] = json!(true);
        }
        return axum::Json(job).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}
fn template() -> WorkloadTemplate {
    WorkloadTemplate {
        job: json!({"Type":"batch","Datacenters":["dc1"],"TaskGroups":[{"Name":"work","Count":1,"Tasks":[{"Name":"workload","Driver":"docker","Config":{"image":format!("example/work@sha256:{}","a".repeat(64)),"command":"/work","args":[]},"Env":{},"Resources":{"CPU":500,"MemoryMB":128}}]}]}),
        parameter_env: BTreeMap::from([("input".into(), "INPUT_VALUE".into())]),
        required_parameters: BTreeSet::new(),
        expected_output: "DONE".into(),
        result_fields: BTreeMap::from([("score".into(), "integer".into())]),
        max_run_seconds: 3600,
    }
}
async fn fixture() -> (
    Arc<Mock>,
    WorkloadsConfig,
    PolicyConfig,
    tokio::task::JoinHandle<()>,
) {
    let s = Arc::new(Mock::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let state = s.clone();
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().fallback(any(handler)).with_state(state),
        )
        .await
        .unwrap()
    });
    let cfg = WorkloadsConfig {
        enabled: true,
        endpoint: endpoint.clone(),
        namespace: "test".into(),
        templates: BTreeMap::from([("test".into(), template())]),
        allowed_node_ids: BTreeSet::from(["node-1".into()]),
        approved_artifact_prefixes: BTreeSet::from([format!("{endpoint}/artifacts/")]),
        ..Default::default()
    };
    let p = PolicyConfig {
        allowed_origins: vec![endpoint],
        allow_private_network: true,
        max_file_bytes: 4 * 1024 * 1024,
        ..Default::default()
    };
    (s, cfg, p, task)
}
fn ctx() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "test".into(),
            user: "alice".into(),
            channel: "test".into(),
            session: "s".into(),
        },
        turn_id: "turn1".into(),
        cancellation: CancellationToken::new(),
    }
}
fn private_dir() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    d
}
fn call(name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: "call1".into(),
        name: name.into(),
        arguments: args,
    }
}
async fn exec(t: &WorkloadTools, c: &ToolContext, name: &str, v: Value) -> Value {
    t.execute(c, &call(name, v)).await.unwrap()["data"].clone()
}
fn plan_args(output: Option<String>) -> Value {
    let mut v = json!({"action":"plan","workload_template":"test","parameters":{"input":"literal; no shell"},"idempotency_key":"request-1"});
    if let Some(o) = output {
        v["output_artifacts"] = json!({"RESULT":o});
    }
    v
}
#[tokio::test]
async fn offload_real_http_plan_submit_logs_artifact_restart_and_scope() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).unwrap();
    let plan = exec(
        &t,
        &c,
        "nomad_offload_task",
        plan_args(Some(format!("{}/artifacts/result.json", cfg.endpoint))),
    )
    .await;
    let id = plan["id"].as_str().unwrap();
    assert_eq!(plan["state"], "planned");
    assert_eq!(s.posts.load(Ordering::SeqCst), 0);
    exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"submit","task_id":id}),
    )
    .await;
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    drop(t);
    let t = WorkloadTools::open(p, cfg, dir.path()).unwrap();
    exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"submit","task_id":id}),
    )
    .await;
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    s.complete.store(1, Ordering::SeqCst);
    let r = exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"result","task_id":id}),
    )
    .await;
    assert_eq!(r["state"], "completed");
    assert!(r["logs"][0]["stdout"].as_str().unwrap().contains("DONE"));
    assert_eq!(r["artifacts"][0]["bytes"], 27);
    let mut other = c.clone();
    other.scope.user = "bob".into();
    assert!(
        t.execute(
            &other,
            &call(
                "nomad_offload_task",
                json!({"action":"status","task_id":id})
            )
        )
        .await
        .is_err()
    );
    server.abort();
}
#[tokio::test]
async fn uncertain_submit_reconciles_without_repost_and_changed_remote_denies_cancel() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p, cfg, dir.path()).unwrap();
    let a = exec(&t, &c, "nomad_offload_task", plan_args(None)).await;
    let id = a["id"].as_str().unwrap();
    s.unknown.store(1, Ordering::SeqCst);
    assert!(
        t.execute(
            &c,
            &call(
                "nomad_offload_task",
                json!({"action":"submit","task_id":id})
            )
        )
        .await
        .is_err()
    );
    assert!(
        t.execute(
            &c,
            &call(
                "nomad_offload_task",
                json!({"action":"submit","task_id":id})
            )
        )
        .await
        .is_err()
    );
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    let r = exec(&t, &c, "nomad_reconcile", json!({})).await;
    assert_eq!(r["jobs"][0]["state"], "submitted_reconciled");
    s.tamper.store(1, Ordering::SeqCst);
    assert!(
        t.execute(
            &c,
            &call(
                "nomad_offload_task",
                json!({"action":"cancel","task_id":id})
            )
        )
        .await
        .is_err()
    );
    assert_eq!(s.stops.load(Ordering::SeqCst), 0);
    server.abort();
}
#[tokio::test]
async fn benchmark_expands_durable_runset_and_never_starts_on_status() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p, cfg.clone(), dir.path()).unwrap();
    let b=exec(&t,&c,"nomad_benchmark",json!({"action":"plan","definition":{"name":"fixture","workload_template":"test","workload_version":"sha256:abc","output_prefix":format!("{}/artifacts/bench",cfg.endpoint),"warmup_runs":1,"repetitions":2,"parallelism":2,"idempotency_key":"bench-1"}})).await;
    let id = b["id"].as_str().unwrap();
    assert_eq!(b["runs"].as_array().unwrap().len(), 3);
    exec(
        &t,
        &c,
        "nomad_benchmark",
        json!({"action":"status","benchmark_id":id}),
    )
    .await;
    assert_eq!(s.posts.load(Ordering::SeqCst), 0);
    exec(
        &t,
        &c,
        "nomad_benchmark",
        json!({"action":"start","benchmark_id":id}),
    )
    .await;
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    s.complete.store(1, Ordering::SeqCst);
    exec(
        &t,
        &c,
        "nomad_benchmark",
        json!({"action":"status","benchmark_id":id}),
    )
    .await;
    assert_eq!(s.posts.load(Ordering::SeqCst), 3);
    let result = exec(
        &t,
        &c,
        "nomad_benchmark",
        json!({"action":"results","benchmark_id":id}),
    )
    .await;
    assert_eq!(result["comparison"]["valid"], true);
    server.abort();
}
#[tokio::test]
async fn finite_schedule_tick_restart_has_no_duplicate_and_cancel_before_submit_no_remote_effect() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).unwrap();
    exec(&t,&c,"nomad_schedule",json!({"action":"create","data":{"workload_template":"test","idempotency_key":"unused"},"expression":"every 1 second","timezone":"UTC","max_runs":1,"idempotency_key":"sched-1"})).await;
    let now = chrono::Utc::now() + chrono::Duration::seconds(2);
    let r = t.run_due(now, CancellationToken::new(), 10).await.unwrap();
    assert_eq!(r["occurrences"].as_array().unwrap().len(), 1);
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    drop(t);
    let t = WorkloadTools::open(p, cfg, dir.path()).unwrap();
    t.run_due(
        now + chrono::Duration::seconds(10),
        CancellationToken::new(),
        10,
    )
    .await
    .unwrap();
    assert_eq!(s.posts.load(Ordering::SeqCst), 1);
    server.abort();
}
#[tokio::test]
async fn rejects_parameter_artifact_privilege_and_config_unknown_fields_before_network() {
    let (s, mut cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).unwrap();
    let mut a = plan_args(None);
    a["parameters"] = json!({"arbitrary":"no"});
    assert!(t.execute(&c, &call("nomad_offload_task", a)).await.is_err());
    assert!(
        t.execute(
            &c,
            &call(
                "nomad_offload_task",
                plan_args(Some("https://evil.invalid/result".into()))
            )
        )
        .await
        .is_err()
    );
    assert_eq!(s.calls.lock().unwrap().len(), 0);
    drop(t);
    cfg.templates.get_mut("test").unwrap().job["TaskGroups"][0]["Tasks"][0]["Config"]["privileged"] =
        json!(true);
    assert!(WorkloadTools::open(p, cfg, dir.path()).is_err());
    assert!(serde_json::from_value::<WorkloadsConfig>(json!({"enabeld":true})).is_err());
    server.abort();
}

#[tokio::test]
async fn fresh_capacity_and_optimistic_conflicts_block_mutation_and_read_only_plan_can_resume() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p, cfg, dir.path()).unwrap();
    s.no_capacity.store(1, Ordering::SeqCst);
    assert!(
        t.execute(&c, &call("nomad_offload_task", plan_args(None)))
            .await
            .is_err()
    );
    assert_eq!(s.posts.load(Ordering::SeqCst), 0);
    s.no_capacity.store(0, Ordering::SeqCst);
    let planned = exec(&t, &c, "nomad_offload_task", plan_args(None)).await;
    assert_eq!(planned["state"], "planned");
    let id = planned["id"].as_str().unwrap();
    s.plan_conflict.store(1, Ordering::SeqCst);
    assert!(
        t.execute(
            &c,
            &call(
                "nomad_offload_task",
                json!({"action":"submit","task_id":id})
            )
        )
        .await
        .is_err()
    );
    assert_eq!(s.posts.load(Ordering::SeqCst), 0);
    server.abort();
}

#[tokio::test]
async fn completed_allocation_is_not_verified_result_and_cancel_before_submit_has_no_delete() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p, cfg, dir.path()).unwrap();
    let planned = exec(&t, &c, "nomad_offload_task", plan_args(None)).await;
    let id = planned["id"].as_str().unwrap();
    let cancelled = exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"cancel","task_id":id}),
    )
    .await;
    assert_eq!(cancelled["state"], "cancelled");
    assert_eq!(s.stops.load(Ordering::SeqCst), 0);
    let mut args = plan_args(None);
    args["idempotency_key"] = json!("second-task");
    let planned = exec(&t, &c, "nomad_offload_task", args).await;
    let id = planned["id"].as_str().unwrap();
    exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"submit","task_id":id}),
    )
    .await;
    s.complete.store(1, Ordering::SeqCst);
    s.wrong_result.store(1, Ordering::SeqCst);
    let r = exec(
        &t,
        &c,
        "nomad_offload_task",
        json!({"action":"result","task_id":id}),
    )
    .await;
    assert_eq!(r["state"], "result_validation_failed");
    assert_eq!(r["complete"], false);
    server.abort();
}

#[tokio::test]
async fn schedule_foreign_scope_and_configuration_change_never_execute_unattended() {
    let (s, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let c = ctx();
    let t = WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).unwrap();
    let v=exec(&t,&c,"nomad_schedule",json!({"action":"create","data":{"workload_template":"test","idempotency_key":"unused"},"expression":"every 1 second","timezone":"UTC","max_runs":2,"idempotency_key":"sched-1"})).await;
    let mut other = c.clone();
    other.scope.user = "bob".into();
    assert!(
        t.execute(
            &other,
            &call(
                "nomad_schedule",
                json!({"action":"remove","schedule_id":v["id"]})
            )
        )
        .await
        .is_err()
    );
    drop(t);
    let mut changed = cfg;
    changed.templates.get_mut("test").unwrap().expected_output = "OTHER".into();
    let t = WorkloadTools::open(p, changed, dir.path()).unwrap();
    let v = t
        .run_due(
            chrono::Utc::now() + chrono::Duration::seconds(3),
            CancellationToken::new(),
            10,
        )
        .await
        .unwrap();
    assert!(v["occurrences"].as_array().unwrap().is_empty());
    assert_eq!(s.posts.load(Ordering::SeqCst), 0);
    server.abort();
}

#[tokio::test]
async fn private_state_single_coordinator_symlink_and_permission_guards() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let (_, cfg, p, server) = fixture().await;
    let dir = private_dir();
    let t = WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).unwrap();
    assert!(WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).is_err());
    drop(t);
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(WorkloadTools::open(p.clone(), cfg.clone(), dir.path()).is_err());
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let parent = private_dir();
    symlink(dir.path(), parent.path().join("linked")).unwrap();
    assert!(WorkloadTools::open(p, cfg, parent.path().join("linked")).is_err());
    server.abort();
}
