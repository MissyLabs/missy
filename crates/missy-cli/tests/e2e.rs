//! Executable-level acceptance: real HTTP protocol -> native tool loop -> SQLite -> restart.
use missy_core::{Config, ProviderConfig};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Command, Output},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

struct Mock {
    origin: String,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    requests: Arc<Mutex<Vec<Value>>>,
}
impl Mock {
    fn start(replies: Vec<Value>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let requests = Arc::new(Mutex::new(vec![]));
        let recorded = requests.clone();
        let handle = thread::spawn(move || {
            let mut replies = replies.into_iter();
            while !worker_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(5)))
                            .unwrap();
                        let request = read_request(&mut stream);
                        recorded.lock().unwrap().push(request);
                        let body = serde_json::to_vec(
                            &replies.next().expect("unexpected provider replay"),
                        )
                        .unwrap();
                        write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                        stream.write_all(&body).unwrap();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("mock accept failed: {e}"),
                }
            }
        });
        Self {
            origin,
            stop,
            handle: Some(handle),
            requests,
        }
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.take().unwrap().join().unwrap();
    }
}
fn read_request(stream: &mut TcpStream) -> Value {
    let mut bytes = vec![];
    let mut byte = [0u8; 1];
    while !bytes.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 65536);
    }
    let headers = String::from_utf8(bytes).unwrap();
    assert!(headers.starts_with("POST /v1/chat/completions "));
    let size: usize = headers
        .lines()
        .find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .map(|(_, v)| v.trim().parse().unwrap())
        })
        .unwrap();
    assert!(size < 2_000_000);
    let mut body = vec![0; size];
    stream.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}
fn reply(content: &str, tools: Vec<Value>) -> Value {
    json!({"choices":[{"message":{"role":"assistant","content":content,"tool_calls":tools},"finish_reason":if tools.is_empty(){"stop"}else{"tool_calls"}}],"usage":{"prompt_tokens":100,"completion_tokens":20}})
}
fn call(id: &str, name: &str, args: Value) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":serde_json::to_string(&args).unwrap()}})
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_missy-rs"))
        .arg("--config")
        .arg(root.join("config.yaml"))
        .args(args)
        .env_clear()
        .env("HOME", root)
        .env("PATH", "/usr/bin:/bin")
        .env("MISSY_TEST_KEY", "fixture-only-not-a-real-secret")
        .current_dir(root)
        .output()
        .unwrap()
}
fn ok(output: Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|_| panic!("not JSON: {}", String::from_utf8_lossy(&output.stdout)))
}

#[test]
fn actual_binary_benchmark_is_no_tools_and_durable_no_replay() {
    let root = tempfile::tempdir().unwrap();
    let mock = Mock::start(vec![reply("measured", vec![])]);
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", mock.origin),
            model: "fixture".into(),
            max_output_tokens: 32,
            input_microusd_per_million: Some(1),
            output_microusd_per_million: Some(1),
            ..ProviderConfig::default()
        }],
        ..Config::default()
    };
    cfg.policy.allowed_origins = vec![mock.origin.clone()];
    cfg.policy.allow_private_network = true;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let spec = root.path().join("benchmark.yaml");
    std::fs::write(&spec,"ledger_path: /unused/overridden\nprompt: Literal read only probe\niterations: 1\nmax_calls: 1\nmax_total_cost_microusd: 10\ntimeout_seconds: 2\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&spec, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let first = ok(invoke(
        root.path(),
        &[
            "benchmark",
            spec.to_str().unwrap(),
            "--request-id",
            "benchmark-one",
        ],
    ));
    assert_eq!(first["status"], "completed");
    let second = ok(invoke(
        root.path(),
        &[
            "benchmark",
            spec.to_str().unwrap(),
            "--request-id",
            "benchmark-one",
        ],
    ));
    assert_eq!(second["deduplicated"], true);
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
    assert!(
        mock.requests.lock().unwrap()[0]["tools"].is_null()
            || mock.requests.lock().unwrap()[0]["tools"]
                .as_array()
                .is_some_and(|v| v.is_empty())
    );
}

#[test]
fn actual_binary_hatching_and_experiments_never_auto_promote() {
    let root = tempfile::tempdir().unwrap();
    let cfg = Config {
        state_dir: root.path().join("state"),
        ..Config::default()
    };
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let answers = root.path().join("answers.yaml");
    std::fs::write(
        &answers,
        "name: Missy\ntone: Direct\npurpose: Useful assistance\nprovider_label: local\n",
    )
    .unwrap();
    let variants = root.path().join("variants.yaml");
    std::fs::write(
        &variants,
        "- name: concise\n  prompt: Be concise\n- name: detailed\n  prompt: Be detailed\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for file in [&answers, &variants] {
            std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
    }
    let hatched = ok(invoke(root.path(), &["hatch", answers.to_str().unwrap()]));
    assert_eq!(hatched["config_proposal"]["contains_credentials"], false);
    assert!(!hatched["persona_proposal"].is_null());
    let created = ok(invoke(
        root.path(),
        &["experiments", "create", variants.to_str().unwrap()],
    ));
    let id = created[0]["id"].as_str().unwrap();
    let assignment = ok(invoke(root.path(), &["experiments", "assign", id]));
    assert_eq!(
        assignment,
        ok(invoke(root.path(), &["experiments", "assign", id]))
    );
    ok(invoke(
        root.path(),
        &["experiments", "record", id, "observation", "--success"],
    ));
    ok(invoke(
        root.path(),
        &["experiments", "record", id, "observation", "--success"],
    ));
    let compared = ok(invoke(root.path(), &["experiments", "compare", id]));
    let total: u64 = compared
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["observations"].as_u64().unwrap())
        .sum();
    assert_eq!(total, 1);
    assert!(
        compared
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["trusted_for_promotion"] == false)
    );
    assert!(
        !invoke(root.path(), &["experiments", "show", id, "--user", "other"])
            .status
            .success()
    );
}

#[test]
fn actual_binary_explicit_parallel_agents_reserve_budgets_and_restart_without_inference() {
    let root = tempfile::tempdir().unwrap();
    let mock = Mock::start(vec![
        reply("review one", vec![]),
        reply("review two", vec![]),
    ]);
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", mock.origin),
            model: "fixture".into(),
            input_microusd_per_million: Some(0),
            output_microusd_per_million: Some(0),
            ..ProviderConfig::default()
        }],
        ..Config::default()
    };
    cfg.policy.allowed_origins = vec![mock.origin.clone()];
    cfg.policy.allow_private_network = true;
    cfg.runtime.max_cost_microusd = 100_000;
    cfg.runtime.max_steps = 4;
    cfg.runtime.max_tokens = 10_000;
    cfg.runtime.max_turn_seconds = 30;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let plan = root.path().join("agents.yaml");
    std::fs::write(&plan,"limits:\n  max_workers: 2\nplan:\n  agents:\n    - id: architecture\n      depth: 1\n      provider: fixture\n      prompt: Review architecture\n    - id: risks\n      depth: 1\n      provider: fixture\n      prompt: Review risks\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&plan, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let first = ok(invoke(
        root.path(),
        &[
            "agents",
            plan.to_str().unwrap(),
            "--request-id",
            "agent-acceptance",
        ],
    ));
    assert_eq!(first["status"], "completed");
    assert_eq!(first["children"].as_object().unwrap().len(), 2);
    assert!(first["reserved"]["cost_microusd"].as_u64().unwrap() > 0);
    let again = ok(invoke(
        root.path(),
        &[
            "agents",
            plan.to_str().unwrap(),
            "--request-id",
            "agent-acceptance",
        ],
    ));
    assert_eq!(again["root_id"], first["root_id"]);
    assert_eq!(again["deduplicated"], true);
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[test]
fn actual_binary_personal_recall_survives_session_and_process_but_not_user_or_tenant() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        ..Config::default()
    };
    cfg.extensions.personal_memory_enabled = true;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let record = ok(invoke(
        root.path(),
        &[
            "personal-memory",
            "put",
            "repair",
            "personal-canary engine repair",
            "--session",
            "first",
        ],
    ));
    assert_eq!(record["trust"], "untrusted_data");
    let recalled = ok(invoke(
        root.path(),
        &[
            "personal-memory",
            "get",
            "repair",
            "--session",
            "second",
            "--channel",
            "another",
        ],
    ));
    assert_eq!(recalled["content"], "personal-canary engine repair");
    assert_eq!(recalled["source_scope"]["session"], "first");
    let other = ok(invoke(
        root.path(),
        &["personal-memory", "get", "repair", "--user", "other"],
    ));
    assert!(other.is_null());
    let tenant = ok(invoke(
        root.path(),
        &["personal-memory", "search", "engine", "--tenant", "other"],
    ));
    assert!(tenant.as_array().unwrap().is_empty());
    let found = ok(invoke(
        root.path(),
        &["personal-memory", "search", "engine", "--session", "third"],
    ));
    assert_eq!(found.as_array().unwrap().len(), 1);
    ok(invoke(
        root.path(),
        &["personal-memory", "delete", "repair", "--session", "third"],
    ));
    assert!(ok(invoke(root.path(), &["personal-memory", "get", "repair"])).is_null());
    cfg.extensions.personal_memory_enabled = false;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    assert!(
        !invoke(root.path(), &["personal-memory", "get", "repair"])
            .status
            .success()
    );
}

#[test]
fn actual_binary_schedule_preview_needs_no_provider_and_opens_no_state() {
    let root = tempfile::tempdir().unwrap();
    let cfg = Config {
        state_dir: root.path().join("must-not-exist"),
        ..Config::default()
    };
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let preview = ok(invoke(
        root.path(),
        &[
            "schedule",
            "next",
            "daily at 09:00",
            "--timezone",
            "America/New_York",
        ],
    ));
    assert!(preview["next"].is_string());
    assert!(!cfg.state_dir.exists());
}

#[test]
fn actual_binary_human_schedule_and_literal_workflow_restart() {
    let root = tempfile::tempdir().unwrap();
    let mock = Mock::start(vec![
        reply("scheduled", vec![]),
        reply("first", vec![]),
        reply("second", vec![]),
    ]);
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", mock.origin),
            model: "fixture".into(),
            input_microusd_per_million: Some(0),
            output_microusd_per_million: Some(0),
            ..ProviderConfig::default()
        }],
        ..Config::default()
    };
    cfg.policy.allowed_origins = vec![mock.origin.clone()];
    cfg.policy.allow_private_network = true;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let serialized = serde_yaml_ng::to_string(&cfg).unwrap();
    let _: Config = serde_yaml_ng::from_str(&serialized).unwrap();
    let next = ok(invoke(
        root.path(),
        &[
            "schedule",
            "next",
            "daily at 09:00",
            "--timezone",
            "America/New_York",
        ],
    ));
    assert!(next["next"].is_string());
    let job = ok(invoke(
        root.path(),
        &["schedule", "add-text", "run once", "in 1 second"],
    ));
    assert!(job["schedule_spec"].is_object());
    std::thread::sleep(Duration::from_millis(2100));
    let ran = ok(invoke(root.path(), &["schedule", "run-due"]));
    assert_eq!(ran.as_array().unwrap().len(), 1);
    assert_eq!(ran[0]["status"], "completed");
    let listed = ok(invoke(root.path(), &["schedule", "list"]));
    assert_eq!(listed[0]["enabled"], false);
    assert!(
        ok(invoke(root.path(), &["schedule", "run-due"]))
            .as_array()
            .unwrap()
            .is_empty()
    );
    let workflow = root.path().join("workflow.yaml");
    std::fs::write(&workflow,"id: acceptance\nnodes:\n  - id: first\n    dependencies: []\n    prompt: First step\n  - id: second\n    dependencies: [first]\n    prompt: Second step\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&workflow, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let result = ok(invoke(
        root.path(),
        &["workflow", workflow.to_str().unwrap()],
    ));
    assert_eq!(result["turns"].as_object().unwrap().len(), 2);
    let duplicate = ok(invoke(
        root.path(),
        &["workflow", workflow.to_str().unwrap()],
    ));
    assert_eq!(
        duplicate["turns"]["first"]["turn_id"],
        result["turns"]["first"]["turn_id"]
    );
    assert_eq!(mock.requests.lock().unwrap().len(), 3);
}

#[test]
fn actual_binary_multitool_restart_dedup_memory_history_and_user_isolation() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let artifact = workspace.join("result.txt");
    let artifact_text = artifact.to_string_lossy().into_owned();
    let mock = Mock::start(vec![
        reply(
            "",
            vec![
                call(
                    "write-1",
                    "file_write",
                    json!({"path":artifact_text,"content":"durable artifact"}),
                ),
                call(
                    "remember-1",
                    "memory_put",
                    json!({"id":"private-note","content":"scope-canary-73f8 is the repair code"}),
                ),
            ],
        ),
        reply("Saved the artifact and private memory.", vec![]),
        reply(
            "",
            vec![call(
                "recall-1",
                "memory_search",
                json!({"query":"scope-canary-73f8","limit":5}),
            )],
        ),
        reply("The remembered repair code is scope-canary-73f8.", vec![]),
        reply("This is a separate user session.", vec![]),
    ]);
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", mock.origin),
            model: "fixture-model".into(),
            api_key_env: Some("MISSY_TEST_KEY".into()),
            input_microusd_per_million: Some(0),
            output_microusd_per_million: Some(0),
            ..ProviderConfig::default()
        }],
        ..Config::default()
    };
    cfg.policy.allowed_origins = vec![mock.origin.clone()];
    cfg.policy.allow_private_network = true;
    cfg.policy.write_roots = vec![workspace.clone()];
    cfg.policy.read_roots = vec![workspace];
    cfg.policy.max_file_bytes = 65536;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let first = ok(invoke(
        root.path(),
        &[
            "ask",
            "Write the artifact and remember my repair code",
            "--request-id",
            "first",
        ],
    ));
    assert_eq!(first["status"], "completed");
    assert_eq!(
        std::fs::read_to_string(&artifact).unwrap(),
        "durable artifact"
    );
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
    let duplicate = ok(invoke(
        root.path(),
        &[
            "ask",
            "Write the artifact and remember my repair code",
            "--request-id",
            "first",
        ],
    ));
    assert_eq!(duplicate["turn_id"], first["turn_id"]);
    assert_eq!(duplicate["deduplicated"], true);
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
    let conflict = invoke(
        root.path(),
        &["ask", "DIFFERENT payload", "--request-id", "first"],
    );
    assert!(!conflict.status.success());
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
    let second = ok(invoke(
        root.path(),
        &[
            "ask",
            "Find my repair code in memory",
            "--request-id",
            "second",
        ],
    ));
    assert_eq!(second["status"], "completed");
    let requests = mock.requests.lock().unwrap();
    assert!(requests[2].to_string().contains("untrusted_history"));
    assert!(requests[3].to_string().contains("scope-canary-73f8"));
    drop(requests);
    ok(invoke(
        root.path(),
        &[
            "ask",
            "Who am I",
            "--user",
            "different-user",
            "--request-id",
            "third",
        ],
    ));
    assert!(
        !mock.requests.lock().unwrap()[4]
            .to_string()
            .contains("scope-canary-73f8")
    );
    let history = ok(invoke(root.path(), &["history"]));
    assert!(history.to_string().contains("remembered repair code"));
    let other = ok(invoke(
        root.path(),
        &["history", "--user", "different-user"],
    ));
    assert!(!other.to_string().contains("scope-canary-73f8"));
    let denied = invoke(
        root.path(),
        &[
            "status",
            first["turn_id"].as_str().unwrap(),
            "--user",
            "different-user",
        ],
    );
    assert!(!denied.status.success());
    assert!(!root.path().join(".missy").exists());
}

#[test]
fn actual_binary_waits_for_exact_operator_approval_then_resumes_without_regenerating_effect() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let artifact = workspace.join("approved.txt");
    let mock = Mock::start(vec![
        reply(
            "",
            vec![call(
                "guarded-write",
                "file_write",
                json!({"path":artifact,"content":"operator approved once"}),
            )],
        ),
        reply("Completed the approved write.", vec![]),
    ]);
    let guard = root.path().join("approvals.yaml");
    std::fs::write(&guard,serde_json::to_vec(&json!({"sensitive_tool_names":["file_write"],"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":300})).unwrap()).unwrap();
    std::fs::set_permissions(&guard, std::fs::Permissions::from_mode(0o600)).unwrap();
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", mock.origin),
            model: "fixture".into(),
            api_key_env: Some("MISSY_TEST_KEY".into()),
            input_microusd_per_million: Some(0),
            output_microusd_per_million: Some(0),
            ..Default::default()
        }],
        ..Default::default()
    };
    cfg.policy.allowed_origins = vec![mock.origin.clone()];
    cfg.policy.allow_private_network = true;
    cfg.policy.write_roots = vec![workspace];
    cfg.policy.max_file_bytes = 65536;
    cfg.extensions.approvals_config = Some(guard);
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    let pending = ok(invoke(
        root.path(),
        &[
            "ask",
            "Write approved.txt only after operator approval",
            "--request-id",
            "approval-request",
        ],
    ));
    assert_eq!(pending["status"], "interrupted");
    assert_eq!(pending["resumable"], true);
    assert!(!artifact.exists());
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
    let approvals = ok(invoke(root.path(), &["approvals", "list"]));
    let record = &approvals.as_array().unwrap()[0];
    assert_eq!(record["status"], "pending");
    ok(invoke(
        root.path(),
        &[
            "approvals",
            "approve",
            record["id"].as_str().unwrap(),
            "--binding-hash",
            record["binding_hash"].as_str().unwrap(),
            "--operator",
            "fixture-operator",
        ],
    ));
    let result = ok(invoke(
        root.path(),
        &["resume", pending["turn_id"].as_str().unwrap()],
    ));
    assert_eq!(result["status"], "completed");
    assert_eq!(
        std::fs::read_to_string(&artifact).unwrap(),
        "operator approved once"
    );
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
    let receipts = ok(invoke(root.path(), &["approvals", "list"]));
    assert_eq!(receipts[0]["status"], "consumed");
    assert!(
        !invoke(
            root.path(),
            &["resume", pending["turn_id"].as_str().unwrap()]
        )
        .status
        .success()
    );
    assert_eq!(mock.requests.lock().unwrap().len(), 2);
}

#[test]
fn actual_binary_retrieval_preserves_citations_and_scope() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let docs = root.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    let document = docs.join("reference.txt");
    let text = "Original é𐐀 UTF-8 citation. The scoped repair code is CITATION_CANARY.";
    std::fs::write(&document, text).unwrap();
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        ..Default::default()
    };
    cfg.extensions.retrieval_enabled = true;
    cfg.policy.read_roots = vec![docs];
    cfg.policy.max_file_bytes = 65536;
    std::fs::write(
        root.path().join("config.yaml"),
        serde_yaml_ng::to_string(&cfg).unwrap(),
    )
    .unwrap();
    // The native store creates this parent0700; ensure no environment umask ambiguity.
    std::fs::create_dir(&cfg.state_dir).unwrap();
    std::fs::set_permissions(&cfg.state_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    ok(invoke(
        root.path(),
        &[
            "retrieval",
            "ingest",
            "reference",
            document.to_str().unwrap(),
        ],
    ));
    let hits = ok(invoke(
        root.path(),
        &["retrieval", "query", "CITATION_CANARY"],
    ));
    let hit = &hits[0];
    let start = hit["start_byte"].as_u64().unwrap() as usize;
    let end = hit["end_byte"].as_u64().unwrap() as usize;
    assert_eq!(hit["text"].as_str().unwrap(), &text[start..end]);
    assert!(!hit["document_hash"].as_str().unwrap().is_empty());
    assert_eq!(
        ok(invoke(
            root.path(),
            &[
                "retrieval",
                "query",
                "CITATION_CANARY",
                "--user",
                "stranger"
            ]
        )),
        json!([])
    );
    ok(invoke(root.path(), &["retrieval", "delete", "reference"]));
    assert_eq!(
        ok(invoke(
            root.path(),
            &["retrieval", "query", "CITATION_CANARY"]
        )),
        json!([])
    );
}
