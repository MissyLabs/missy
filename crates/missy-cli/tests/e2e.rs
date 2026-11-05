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
