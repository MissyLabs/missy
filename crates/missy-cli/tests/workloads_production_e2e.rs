//! Actual native gateway child + provider/cluster loopback fixtures. No accounts.
use axum::{
    Router,
    body::Bytes,
    extract::{OriginalUri, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use missy_core::{Config, ProviderConfig, Scope};
use missy_governance::{ApprovalRegistry, ApprovalStatus, Decision};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
#[derive(Default)]
struct Fixture {
    provider_calls: AtomicUsize,
    submits: AtomicUsize,
    jobs: Mutex<BTreeMap<String, Value>>,
}
async fn route(
    State(s): State<Arc<Fixture>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    body: Bytes,
) -> Response {
    let p = uri.path();
    if p == "/v1/chat/completions" || p == "/chat/completions" {
        let request: Value = serde_json::from_slice(&body).unwrap();
        let n = s.provider_calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            let names: Vec<_> = request["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap())
                .collect();
            for name in [
                "nomad_schedule",
                "nomad_offload_task",
                "nomad_benchmark",
                "nomad_reconcile",
            ] {
                assert!(names.contains(&name), "unwired tool {name}");
            }
            let args = json!({"action":"create","data":{"workload_template":"test","idempotency_key":"fixture-data"},"expression":"every 1 second","timezone":"UTC","max_runs":1,"idempotency_key":"fixture-schedule"});
            return axum::Json(json!({"model":"fixture","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"schedule-call","type":"function","function":{"name":"nomad_schedule","arguments":args.to_string()}}]}}],"usage":{"prompt_tokens":50,"completion_tokens":20}})).into_response();
        }
        assert!(
            request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["role"] == "tool" && m["tool_call_id"] == "schedule-call"),
            "approved original call result missing on resume"
        );
        return axum::Json(json!({"model":"fixture","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"The finite schedule was saved. Its job runs only when due."}}],"usage":{"prompt_tokens":80,"completion_tokens":20}})).into_response();
    }
    if p == "/v1/nodes" {
        return axum::Json(json!([{"ID":"node-1","Status":"ready","SchedulingEligibility":"eligible","Drain":false}])).into_response();
    }
    if p == "/v1/node/node-1" {
        return axum::Json(json!({"ID":"node-1","Status":"ready","SchedulingEligibility":"eligible","Drain":false,"NodeResources":{"Cpu":{"CpuShares":10000},"Memory":{"MemoryMB":10000},"Disk":{"DiskMB":10000}}})).into_response();
    }
    if p == "/v1/node/node-1/allocations" {
        return axum::Json(json!([])).into_response();
    }
    if p.ends_with("/plan") {
        return axum::Json(json!({"JobModifyIndex":0,"FailedTGAllocs":{}})).into_response();
    }
    if p == "/v1/jobs" && method == Method::POST {
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["EnforceIndex"], true);
        assert_eq!(v["JobModifyIndex"], 0);
        let job = v["Job"].clone();
        s.jobs
            .lock()
            .unwrap()
            .insert(job["ID"].as_str().unwrap().into(), job);
        s.submits.fetch_add(1, Ordering::SeqCst);
        return axum::Json(json!({"EvalID":"eval-1"})).into_response();
    }
    if let Some(id) = p.strip_prefix("/v1/job/")
        && let Some(v) = s.jobs.lock().unwrap().get(id)
    {
        return axum::Json(v.clone()).into_response();
    }
    StatusCode::NOT_FOUND.into_response()
}
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn child(path: &Path) -> Process {
    Process(
        Command::new(env!("CARGO_BIN_EXE_missy-rs"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env(
                "MISSY_API_TOKEN",
                "DisposableGatewayFixtureToken0123456789abcdef",
            )
            .args(["--config", path.to_str().unwrap(), "gateway", "start"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}
fn private_write(path: &Path, v: &impl serde::Serialize) {
    fs::write(path, serde_yaml_ng::to_string(v).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn http(port: u16, method: &str, path: &str, body: Value) -> Value {
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let body = body.to_string();
    write!(socket,"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer DisposableGatewayFixtureToken0123456789abcdef\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
    let mut bytes = String::new();
    socket.read_to_string(&mut bytes).unwrap();
    assert!(
        bytes.starts_with("HTTP/1.1 200") || bytes.starts_with("HTTP/1.1 202"),
        "{bytes}"
    );
    serde_json::from_str(bytes.split_once("\r\n\r\n").unwrap().1).unwrap()
}
fn wait(mut ready: impl FnMut() -> bool, label: &str) {
    let start = Instant::now();
    while !ready() {
        assert!(
            start.elapsed() < Duration::from_secs(25),
            "timed out waiting for {label}"
        );
        std::thread::sleep(Duration::from_millis(40));
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn gateway_shared_workloads_model_approval_resume_due_tick_shutdown_restart() {
    let fixture = Arc::new(Fixture::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .fallback(any(route))
        .with_state(fixture.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let root = tempfile::tempdir().unwrap();
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let workload = root.path().join("workloads.yaml");
    private_write(
        &workload,
        &json!({"enabled":true,"endpoint":origin,"namespace":"test","allowed_node_ids":["node-1"],"templates":{"test":{"job":{"Type":"batch","TaskGroups":[{"Name":"work","Count":1,"Tasks":[{"Name":"workload","Driver":"docker","Config":{"image":format!("example/work@sha256:{}","a".repeat(64))},"Resources":{"CPU":500,"MemoryMB":128}}]}]},"max_run_seconds":3600}}}),
    );
    let approvals = root.path().join("approvals.yaml");
    private_write(
        &approvals,
        &json!({"sensitive_tool_names":["nomad_schedule","nomad_offload_task","nomad_benchmark","nomad_reconcile"],"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":120}),
    );
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        ..Default::default()
    };
    cfg.gateway.http_enabled = true;
    cfg.api.bind = format!("127.0.0.1:{port}");
    cfg.api.tenant = "test".into();
    cfg.api.user = "alice".into();
    cfg.policy.allowed_origins = vec![origin.clone()];
    cfg.policy.allow_private_network = true;
    cfg.extensions.workloads_config = Some(workload);
    cfg.extensions.approvals_config = Some(approvals);
    cfg.providers = vec![ProviderConfig {
        name: "fixture".into(),
        kind: "openai".into(),
        base_url: origin,
        model: "fixture".into(),
        input_microusd_per_million: Some(0),
        output_microusd_per_million: Some(0),
        ..Default::default()
    }];
    let path = root.path().join("config.yaml");
    private_write(&path, &cfg);
    let mut process = child(&path);
    let status_path = cfg.state_dir.join("run/gateway_status.json");
    wait(
        || {
            if let Ok(b) = fs::read(&status_path) {
                let s: Value = serde_json::from_slice(&b).unwrap();
                assert!(
                    s["phase"] != "failed",
                    "gateway failed startup, probable double coordinator open: {s}"
                );
                s["ready"] == true
            } else {
                false
            }
        },
        "gateway readiness",
    );
    let response = http(
        port,
        "POST",
        "/api/v1/chat",
        json!({"session":"fixture","prompt":"Create the finite approved test schedule.","request_id":"schedule-request"}),
    );
    let turn = response["turn_id"].as_str().unwrap().to_string();
    let scope = Scope {
        tenant: "test".into(),
        user: "alice".into(),
        channel: "api".into(),
        session: "fixture".into(),
    };
    let registry = ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3")).unwrap();
    wait(
        || {
            registry
                .list(&scope, 20)
                .unwrap()
                .iter()
                .any(|r| r.status == ApprovalStatus::Pending)
        },
        "exact schedule approval",
    );
    assert_eq!(fixture.submits.load(Ordering::SeqCst), 0);
    let pending = registry
        .list(&scope, 20)
        .unwrap()
        .into_iter()
        .find(|r| r.status == ApprovalStatus::Pending)
        .unwrap();
    registry
        .decide(
            &scope,
            &pending.id,
            &pending.binding_hash,
            Decision::Approve,
            "fixture-operator",
        )
        .unwrap();
    http(
        port,
        "POST",
        &format!("/api/v1/turns/{turn}/resume"),
        json!({"session":"fixture"}),
    );
    wait(
        || fixture.submits.load(Ordering::SeqCst) == 1,
        "supervisor workload tick",
    );
    assert!(fixture.provider_calls.load(Ordering::SeqCst) >= 2);
    unsafe {
        libc::kill(process.0.id() as i32, libc::SIGTERM);
    }
    assert!(process.0.wait().unwrap().success());
    let mut process = child(&path);
    wait(
        || {
            fs::read(&status_path)
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .is_some_and(|s| s["ready"] == true && s["pid"] == process.0.id())
        },
        "restart readiness",
    );
    tokio::time::sleep(Duration::from_secs(6)).await;
    assert_eq!(
        fixture.submits.load(Ordering::SeqCst),
        1,
        "completed occurrence replayed on restart"
    );
    unsafe {
        libc::kill(process.0.id() as i32, libc::SIGTERM);
    }
    assert!(process.0.wait().unwrap().success());
    server.abort();
}
