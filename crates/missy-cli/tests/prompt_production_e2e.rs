//! Real native daemon, real loopback HTTP, disposable governance and provider.
use axum::{Router, body::Bytes, extract::State, routing::post};
use missy_core::{Config, ProviderConfig, Scope};
use missy_governance::{ApprovalRegistry, ApprovalStatus, Decision, DocumentKind};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const KEY: &str = "DisposableGatewayFixtureToken0123456789abcdef";
const OLD: &str = "SCOPED_PERSONA_OLD_UNIQUE_MARKER";
const NEW: &str = "SCOPED_PERSONA_NEW_UNIQUE_MARKER";

#[derive(Default)]
struct Fixture {
    requests: Mutex<Vec<Value>>,
    artifact: Mutex<String>,
}
async fn provider(State(state): State<Arc<Fixture>>, body: Bytes) -> axum::Json<Value> {
    let request: Value = serde_json::from_slice(&body).unwrap();
    let mut requests = state.requests.lock().unwrap();
    let first = requests.is_empty();
    requests.push(request);
    let message = if first {
        let args = json!({"path":*state.artifact.lock().unwrap(),"content":"approved"});
        json!({"role":"assistant","content":"","tool_calls":[{"id":"write-1","type":"function","function":{"name":"file_write","arguments":args.to_string()}}]})
    } else {
        json!({"role":"assistant","content":"fixture complete"})
    };
    axum::Json(
        json!({"model":"fixture","choices":[{"finish_reason":if first {"tool_calls"} else {"stop"},"message":message}],"usage":{"prompt_tokens":50,"completion_tokens":10}}),
    )
}
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn private_write(path: &Path, data: &impl serde::Serialize) {
    fs::write(path, serde_yaml_ng::to_string(data).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn http(port: u16, path: &str, body: Value) -> Value {
    request(port, "POST", path, body)
}
fn request(port: u16, method: &str, path: &str, body: Value) -> Value {
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    let body = body.to_string();
    write!(socket, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.1 202"),
        "{response}"
    );
    serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap()
}
fn wait(mut ready: impl FnMut() -> bool, label: &str) {
    let started = Instant::now();
    while !ready() {
        assert!(
            started.elapsed() < Duration::from_secs(25),
            "timeout: {label}"
        );
        std::thread::sleep(Duration::from_millis(40));
    }
}
fn approve_persona(registry: &ApprovalRegistry, scope: &Scope, version: i64, content: &str) {
    let context = "0".repeat(64);
    let proposal = registry
        .propose_document(scope, DocumentKind::Persona, version, content, &context)
        .unwrap();
    registry
        .decide_document(
            scope,
            &proposal.id,
            &proposal.proposal_hash,
            &context,
            Decision::Approve,
            "fixture-operator",
        )
        .unwrap();
}
fn system(request: &Value) -> String {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "system")
        .map(|m| m["content"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn daemon_reads_latest_exact_scoped_persona_and_pins_paused_turn() {
    let fixture = Arc::new(Fixture::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/v1/chat/completions", post(provider))
        .route("/chat/completions", post(provider))
        .with_state(fixture.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    fs::create_dir(&workspace).unwrap();
    *fixture.artifact.lock().unwrap() = workspace
        .join("approved.txt")
        .to_string_lossy()
        .into_owned();
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let approvals = root.path().join("approvals.yaml");
    private_write(
        &approvals,
        &json!({"sensitive_tool_names":["file_write"],"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":300}),
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
    cfg.policy.write_roots = vec![workspace];
    cfg.policy.max_file_bytes = 65536;
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
    let mut child = Process(
        Command::new(env!("CARGO_BIN_EXE_missy-rs"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("MISSY_API_TOKEN", KEY)
            .args(["--config", path.to_str().unwrap(), "gateway", "start"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let status = cfg.state_dir.join("run/gateway_status.json");
    wait(
        || {
            fs::read(&status)
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .is_some_and(|s| {
                    assert_ne!(s["phase"], "failed", "{s}");
                    s["ready"] == true
                })
        },
        "gateway readiness",
    );
    let scope = Scope {
        tenant: "test".into(),
        user: "alice".into(),
        channel: "api".into(),
        session: "approved".into(),
    };
    let registry = ApprovalRegistry::open(cfg.state_dir.join("governance.sqlite3")).unwrap();
    // Deliberately approve only after the daemon and runtime exist.
    approve_persona(&registry, &scope, 0, OLD);
    let result = http(
        port,
        "/api/v1/chat",
        json!({"session":"approved","prompt":"write the fixture file","request_id":"prompt-1"}),
    );
    let turn = result["turn_id"].as_str().unwrap().to_owned();
    wait(
        || {
            registry
                .list(&scope, 20)
                .unwrap()
                .iter()
                .any(|r| r.status == ApprovalStatus::Pending)
        },
        "tool approval",
    );
    assert!(system(&fixture.requests.lock().unwrap()[0]).contains(OLD));
    approve_persona(&registry, &scope, 1, NEW);
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
    // Registry publication precedes the durable interrupted checkpoint. Wait for
    // the runtime's actual pause before requesting resume, never retry effects.
    wait(
        || {
            request(
                port,
                "GET",
                &format!("/api/v1/turns/{turn}?session=approved"),
                json!({}),
            )["turn"]["status"]
                == "interrupted"
        },
        "durable approval pause",
    );
    http(
        port,
        &format!("/api/v1/turns/{turn}/resume"),
        json!({"session":"approved"}),
    );
    wait(
        || fixture.requests.lock().unwrap().len() >= 2,
        "resumed provider call",
    );
    let pinned = system(&fixture.requests.lock().unwrap()[1]);
    assert!(pinned.contains(OLD), "{pinned}");
    assert!(!pinned.contains(NEW), "{pinned}");
    http(
        port,
        "/api/v1/chat",
        json!({"session":"other","prompt":"other session","request_id":"prompt-2"}),
    );
    wait(
        || fixture.requests.lock().unwrap().len() >= 3,
        "other scope provider call",
    );
    let other = system(&fixture.requests.lock().unwrap()[2]);
    assert!(!other.contains(OLD) && !other.contains(NEW), "{other}");
    http(
        port,
        "/api/v1/chat",
        json!({"session":"approved","prompt":"fresh turn","request_id":"prompt-3"}),
    );
    wait(
        || fixture.requests.lock().unwrap().len() >= 4,
        "latest persona provider call",
    );
    let latest = system(&fixture.requests.lock().unwrap()[3]);
    assert!(latest.contains(NEW), "{latest}");
    assert!(!latest.contains(OLD), "{latest}");
    unsafe {
        libc::kill(child.0.id() as i32, libc::SIGTERM);
    }
    assert!(child.0.wait().unwrap().success());
    server.abort();
}
