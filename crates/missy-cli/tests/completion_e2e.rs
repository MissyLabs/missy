//! Completion acceptance through the native executable, isolated Git and loopback HTTP only.
#![cfg(target_os = "linux")]
use missy_core::{Config, ProviderConfig};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

fn private_write(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_missy-rs"))
        .arg("--config")
        .arg(root.join("config.yaml"))
        .args(args)
        .env_clear()
        .env("HOME", root)
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("MISSY_COMPLETION_KEY", "fixture-secret-do-not-export")
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("native CLI exceeded fixture deadline");
        }
        thread::sleep(Duration::from_millis(2));
    }
}
fn ok(output: Output) -> Value {
    assert!(
        output.status.success(),
        "native CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("CLI JSON response")
}
fn rejected(output: Output, expected: &str) {
    assert!(
        !output.status.success(),
        "unsafe command unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected),
        "wrong rejection category: {stderr}"
    );
}
fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("/usr/bin/git")
        .env_clear()
        .env("HOME", repo.parent().unwrap())
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_AUTHOR_NAME", "Fixture")
        .env("GIT_AUTHOR_EMAIL", "fixture@localhost")
        .env("GIT_COMMITTER_NAME", "Fixture")
        .env("GIT_COMMITTER_EMAIL", "fixture@localhost")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgSign=false",
        ])
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(output.status.success(), "synthetic Git fixture failed");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
fn evolution_fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let repo = root.path().join("repo");
    fs::create_dir(&repo).unwrap();
    git(&repo, &["init", "-q", "-b", "fixture"]);
    fs::write(repo.join("a.txt"), "old\n").unwrap();
    fs::write(repo.join("b.txt"), "untouched\n").unwrap();
    git(&repo, &["add", "--", "a.txt", "b.txt"]);
    git(&repo, &["commit", "-q", "-m", "base"]);
    private_write(
        &root.path().join("change.diff"),
        "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n",
    );
    let policy = missy_evolution::EvolutionPolicy {
        read_roots: vec![repo.clone()],
        write_roots: vec![repo],
        operator_user: "operator".into(),
        allow_git_commits: true,
    };
    private_write(
        &root.path().join("evolution.yaml"),
        &serde_yaml_ng::to_string(&policy).unwrap(),
    );
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        ..Config::default()
    };
    cfg.policy.read_roots = vec![root.path().to_owned()];
    cfg.policy.max_file_bytes = 65_536;
    cfg.extensions.evolution_config = Some(root.path().join("evolution.yaml"));
    private_write(
        &root.path().join("config.yaml"),
        &serde_yaml_ng::to_string(&cfg).unwrap(),
    );
    root
}
fn propose(root: &Path) -> Value {
    ok(invoke(
        root,
        &[
            "evolve",
            "propose",
            root.join("repo").to_str().unwrap(),
            root.join("change.diff").to_str().unwrap(),
            "synthetic edit",
        ],
    ))
}
fn approve(root: &Path, p: &Value) {
    let reviewed = ok(invoke(
        root,
        &[
            "evolve",
            "review",
            p["plan"]["id"].as_str().unwrap(),
            "--plan-hash",
            p["plan_hash"].as_str().unwrap(),
            "--operator",
            "operator",
            "--approve",
        ],
    ));
    assert_eq!(reviewed["status"], "approved");
}

#[test]
fn binary_evolution_pins_review_commit_and_rollback_preserving_other_files() {
    let temp = evolution_fixture();
    let root = temp.path();
    let repo = root.join("repo");
    let p = propose(root);
    let id = p["plan"]["id"].as_str().unwrap();
    let hash = p["plan_hash"].as_str().unwrap();
    let head = p["plan"]["base_head"].as_str().unwrap();
    assert_eq!(hash.len(), 64);
    assert_eq!(head, git(&repo, &["rev-parse", "HEAD"]));
    assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "old\n");
    rejected(
        invoke(root, &["evolve", "show", id, "--session", "another"]),
        "scope",
    );
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "review",
                id,
                "--plan-hash",
                "wrong",
                "--operator",
                "operator",
                "--approve",
            ],
        ),
        "hash",
    );
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "review",
                id,
                "--plan-hash",
                hash,
                "--operator",
                "model",
                "--approve",
            ],
        ),
        "operator",
    );
    approve(root, &p);
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "apply",
                id,
                "--plan-hash",
                hash,
                "--head",
                head,
                "--operator",
                "operator",
            ],
        ),
        "confirmation",
    );
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "apply",
                id,
                "--plan-hash",
                "wrong",
                "--head",
                head,
                "--operator",
                "operator",
                "--confirm",
            ],
        ),
        "hash",
    );
    let applied = ok(invoke(
        root,
        &[
            "evolve",
            "apply",
            id,
            "--plan-hash",
            hash,
            "--head",
            head,
            "--operator",
            "operator",
            "--confirm",
        ],
    ));
    assert_eq!(applied["status"], "applied");
    let commit = applied["commit"].as_str().unwrap();
    assert_ne!(commit, head);
    assert_eq!(commit, git(&repo, &["rev-parse", "HEAD"]));
    assert_eq!(head, git(&repo, &["rev-parse", "HEAD^"]));
    assert_eq!(git(&repo, &["show", "HEAD:a.txt"]), "new");
    assert_eq!(
        fs::read_to_string(repo.join("b.txt")).unwrap(),
        "untouched\n"
    );
    assert!(git(&repo, &["status", "--porcelain"]).is_empty());
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "rollback",
                id,
                "--plan-hash",
                hash,
                "--head",
                head,
                "--operator",
                "operator",
                "--confirm",
            ],
        ),
        "HEAD",
    );
    let rolled = ok(invoke(
        root,
        &[
            "evolve",
            "rollback",
            id,
            "--plan-hash",
            hash,
            "--head",
            commit,
            "--operator",
            "operator",
            "--confirm",
        ],
    ));
    assert_eq!(rolled["status"], "rolled_back");
    assert_eq!(
        rolled["rollback_commit"],
        git(&repo, &["rev-parse", "HEAD"])
    );
    assert_eq!(commit, git(&repo, &["rev-parse", "HEAD^"]));
    assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "old\n");
    assert_eq!(
        fs::read_to_string(repo.join("b.txt")).unwrap(),
        "untouched\n"
    );
    assert!(git(&repo, &["status", "--porcelain"]).is_empty());
    let reloaded = ok(invoke(root, &["evolve", "show", id]));
    assert_eq!(reloaded, rolled);
}

#[test]
fn binary_evolution_rejects_changed_head_and_cross_scope_without_writes() {
    let temp = evolution_fixture();
    let root = temp.path();
    let repo = root.join("repo");
    let p = propose(root);
    approve(root, &p);
    let id = p["plan"]["id"].as_str().unwrap();
    let hash = p["plan_hash"].as_str().unwrap();
    let head = p["plan"]["base_head"].as_str().unwrap();
    rejected(
        invoke(
            root,
            &[
                "evolve",
                "apply",
                id,
                "--plan-hash",
                hash,
                "--head",
                head,
                "--operator",
                "operator",
                "--confirm",
                "--session",
                "other",
            ],
        ),
        "scope",
    );
    git(
        &repo,
        &["commit", "--allow-empty", "-q", "-m", "intervening commit"],
    );
    let changed = git(&repo, &["rev-parse", "HEAD"]);
    for supplied in [head, changed.as_str()] {
        rejected(
            invoke(
                root,
                &[
                    "evolve",
                    "apply",
                    id,
                    "--plan-hash",
                    hash,
                    "--head",
                    supplied,
                    "--operator",
                    "operator",
                    "--confirm",
                ],
            ),
            "HEAD",
        );
    }
    assert_eq!(git(&repo, &["rev-parse", "HEAD"]), changed);
    assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "old\n");
    assert_eq!(
        fs::read_to_string(repo.join("b.txt")).unwrap(),
        "untouched\n"
    );
    assert!(git(&repo, &["status", "--porcelain"]).is_empty());
    assert_eq!(
        ok(invoke(root, &["evolve", "show", id]))["status"],
        "approved"
    );
}

struct HttpFixture {
    origin: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    completed: Arc<Mutex<usize>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl HttpFixture {
    fn new(provider: bool, status: u16) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(Mutex::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (record, done, stopped) = (requests.clone(), completed.clone(), stop.clone());
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        socket
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        socket
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let mut header = Vec::new();
                        let mut byte = [0];
                        while !header.ends_with(b"\r\n\r\n") {
                            socket.read_exact(&mut byte).unwrap();
                            header.push(byte[0]);
                            assert!(header.len() < 65_536);
                        }
                        let header = String::from_utf8(header).unwrap();
                        let length: usize = header
                            .lines()
                            .find_map(|line| {
                                line.split_once(':')
                                    .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                                    .map(|(_, value)| value.trim().parse().unwrap())
                            })
                            .unwrap();
                        assert!(length < 1_048_576);
                        let mut bytes = vec![0; length];
                        socket.read_exact(&mut bytes).unwrap();
                        record
                            .lock()
                            .unwrap()
                            .push((header, serde_json::from_slice(&bytes).unwrap()));
                        if !provider {
                            thread::sleep(Duration::from_millis(100));
                        }
                        let body = if provider {
                            json!({"choices":[{"message":{"role":"assistant","content":"private-assistant-marker","tool_calls":[]},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}})
                        } else {
                            json!({})
                        };
                        let body = serde_json::to_vec(&body).unwrap();
                        *done.lock().unwrap() += 1;
                        // Simulate an uncertain POST: collector received the body but no acknowledgement.
                        if status == 0 {
                            continue;
                        }
                        // A timed-out collector connection is also recorded; never panic on broken pipe.
                        let _ = write!(
                            socket,
                            "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = socket.write_all(&body);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(_) => panic!("loopback fixture accept failed"),
                }
            }
        });
        Self {
            origin,
            requests,
            completed,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for HttpFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.worker.take().unwrap().join().unwrap();
    }
}
fn telemetry_case(collector_status: u16) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let provider = HttpFixture::new(true, 200);
    let collector = HttpFixture::new(false, collector_status);
    let mut cfg = Config {
        state_dir: root.join("state"),
        providers: vec![ProviderConfig {
            name: "private-provider-marker".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", provider.origin),
            model: "private-model-marker".into(),
            api_key_env: Some("MISSY_COMPLETION_KEY".into()),
            input_microusd_per_million: Some(0),
            output_microusd_per_million: Some(0),
            ..ProviderConfig::default()
        }],
        ..Config::default()
    };
    cfg.policy.allowed_origins = vec![provider.origin.clone(), collector.origin.clone()];
    cfg.policy.allow_private_network = true;
    cfg.runtime.max_turn_seconds = 5;
    cfg.runtime.max_cost_microusd = 100_000;
    cfg.extensions.telemetry_config = Some(root.join("telemetry.yaml"));
    let telemetry = missy_operations::TelemetryConfig {
        endpoint: Some(format!("{}/v1/traces", collector.origin)),
        // Final spans cannot depend on the periodic tick before process exit.
        flush_interval_ms: 60_000,
        request_timeout_ms: 1000,
        shutdown_timeout_ms: 2000,
        ..missy_operations::TelemetryConfig::default()
    };
    private_write(
        &root.join("telemetry.yaml"),
        &serde_yaml_ng::to_string(&telemetry).unwrap(),
    );
    private_write(
        &root.join("config.yaml"),
        &serde_yaml_ng::to_string(&cfg).unwrap(),
    );
    let started = Instant::now();
    let answer = ok(invoke(
        root,
        &[
            "ask",
            "private-user-marker private-tool-name-marker",
            "--request-id",
            "private-request-marker",
        ],
    ));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "shutdown exceeded bounded fixture deadline"
    );
    assert!(answer.to_string().contains("private-assistant-marker"));
    let requests = collector.requests.lock().unwrap().clone();
    assert!(
        !requests.is_empty(),
        "CLI exited without exporting final telemetry"
    );
    assert_eq!(
        *collector.completed.lock().unwrap(),
        requests.len(),
        "CLI did not await collector response"
    );
    assert!(started.elapsed() >= Duration::from_millis(100));
    let provider_requests = provider.requests.lock().unwrap();
    assert_eq!(
        provider_requests.len(),
        1,
        "provider operation was replayed"
    );
    assert!(
        provider_requests[0]
            .0
            .starts_with("POST /v1/chat/completions ")
    );
    assert!(
        provider_requests[0].1["messages"]
            .to_string()
            .contains("private-user-marker")
    );
    let tool_names: Vec<&str> = provider_requests[0].1["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    assert!(
        !tool_names.is_empty(),
        "privacy test needs actual advertised tool names"
    );
    let mut kinds = Vec::new();
    let mut span_ids = std::collections::HashSet::new();
    for (headers, body) in &requests {
        assert!(headers.starts_with("POST /v1/traces "));
        assert!(!headers.to_ascii_lowercase().contains("authorization:"));
        let serialized = body.to_string();
        for tool_name in &tool_names {
            assert!(
                !serialized.contains(tool_name),
                "telemetry included an advertised tool name"
            );
        }
        for forbidden in [
            "private-user-marker",
            "private-assistant-marker",
            "private-tool-name-marker",
            "private-provider-marker",
            "private-model-marker",
            "private-request-marker",
            "fixture-secret-do-not-export",
            "MISSY_COMPLETION_KEY",
        ] {
            assert!(
                !serialized.contains(forbidden),
                "telemetry leaked fixture marker"
            );
        }
        for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array()
            .unwrap()
        {
            assert!(
                span_ids.insert(span["spanId"].as_str().unwrap().to_owned()),
                "collector received retried span"
            );
            assert_eq!(span["name"], "missy.operation");
            assert_eq!(span["status"]["code"], 1);
            let attrs = span["attributes"].as_array().unwrap();
            assert_eq!(attrs.len(), 2, "unreviewed telemetry attributes");
            kinds.push(
                attrs
                    .iter()
                    .find(|a| a["key"] == "missy.operation.kind")
                    .unwrap()["value"]["stringValue"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            );
        }
    }
    kinds.sort();
    assert_eq!(
        kinds,
        ["runtime_provider", "runtime_settlement", "runtime_turn"],
        "final turn/settlement spans missing or repeated"
    );
}
#[test]
fn binary_ask_awaits_final_telemetry_flush_without_payloads() {
    telemetry_case(200);
}
#[test]
fn binary_ask_does_not_retry_rejected_telemetry_or_provider_operations() {
    telemetry_case(503);
}
#[test]
fn binary_ask_does_not_retry_uncertain_telemetry_post() {
    telemetry_case(0);
}
