//! Real child processes, disposable config/state, no live credentials.
use missy_core::{Config, ProviderConfig, Scope};
use serde_json::Value;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
fn command(path: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_missy-rs"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .arg("--config")
        .arg(path)
        .args(args);
    command
}
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn config(root: &Path) -> (Config, std::path::PathBuf) {
    let mut cfg = Config {
        state_dir: root.join("state"),
        ..Default::default()
    };
    cfg.providers.push(ProviderConfig {
        model: "never-contacted".into(),
        input_microusd_per_million: Some(0),
        output_microusd_per_million: Some(0),
        ..Default::default()
    });
    cfg.policy
        .allowed_origins
        .push("http://127.0.0.1:11434".into());
    cfg.policy.allow_private_network = true;
    cfg.gateway.scheduler.enabled = true;
    cfg.gateway.scheduler.poll_seconds = 1;
    cfg.gateway.scheduler.scopes.push(Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "cli".into(),
        session: "test".into(),
    });
    let path = root.join("config.yaml");
    write(&path, &cfg);
    (cfg, path)
}
fn write(path: &Path, cfg: &Config) {
    fs::write(path, serde_yaml_ng::to_string(cfg).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn wait_ready(path: &Path) {
    let start = Instant::now();
    loop {
        if command(path, &["gateway", "status"])
            .output()
            .unwrap()
            .status
            .success()
        {
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "gateway failed readiness"
        );
        thread::sleep(Duration::from_millis(30));
    }
}
#[test]
fn supervisor_lock_reload_sigterm_and_restart() {
    let temp = tempfile::tempdir().unwrap();
    let (mut cfg, path) = config(temp.path());
    let mut child = Process(
        command(&path, &["gateway", "start"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    wait_ready(&path);
    let duplicate = command(&path, &["gateway", "start"]).output().unwrap();
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("another native gateway"));
    assert!(
        command(&path, &["gateway", "check-reload"])
            .output()
            .unwrap()
            .status
            .success()
    );
    cfg.system_prompt.push_str(" Changed after admission.");
    write(&path, &cfg);
    let reload = command(&path, &["gateway", "check-reload"])
        .output()
        .unwrap();
    assert!(!reload.status.success());
    let reload: Value = serde_json::from_slice(&reload.stdout).unwrap();
    assert_eq!(reload["restart_required"], true);
    unsafe {
        libc::kill(child.0.id() as i32, libc::SIGTERM);
    }
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(10));
        thread::sleep(Duration::from_millis(25));
    }
    assert!(
        !command(&path, &["gateway", "status"])
            .output()
            .unwrap()
            .status
            .success()
    );
    let status: Value =
        serde_json::from_slice(&fs::read(cfg.state_dir.join("run/gateway_status.json")).unwrap())
            .unwrap();
    assert_eq!(status["phase"], "stopped");
    let mut restarted = Process(
        command(&path, &["gateway", "start"])
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_ready(&path);
    unsafe {
        libc::kill(restarted.0.id() as i32, libc::SIGTERM);
    }
    assert!(restarted.0.wait().unwrap().success());
}
#[test]
fn failed_http_startup_persists_failure_and_no_readiness() {
    let temp = tempfile::tempdir().unwrap();
    let (mut cfg, path) = config(temp.path());
    cfg.gateway.http_enabled = true;
    write(&path, &cfg);
    let result = command(&path, &["gateway", "start"]).output().unwrap();
    assert!(!result.status.success());
    let status: Value =
        serde_json::from_slice(&fs::read(cfg.state_dir.join("run/gateway_status.json")).unwrap())
            .unwrap();
    assert_eq!(status["ready"], false);
    assert_eq!(status["phase"], "failed");
    assert_eq!(status["subsystems"]["http"], "startup_failed");
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
fn http(port: u16, route: &str, key: &str) -> String {
    use std::io::{Read, Write};
    let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(socket,"GET {route} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {key}\r\nConnection: close\r\n\r\n").unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    response
}
#[test]
fn shared_http_operations_voice_worker_real_listeners_and_sigterm() {
    let temp = tempfile::tempdir().unwrap();
    let (mut cfg, path) = config(temp.path());
    let api_port = free_port();
    let operations_port = free_port();
    let voice_port = free_port();
    cfg.gateway.http_enabled = true;
    cfg.api.bind = format!("127.0.0.1:{api_port}");
    let operations = temp.path().join("operations.yaml");
    fs::write(&operations, format!("bind: 127.0.0.1:{operations_port}\n")).unwrap();
    fs::set_permissions(&operations, fs::Permissions::from_mode(0o600)).unwrap();
    cfg.extensions.operations_config = Some(operations);
    let voice = temp.path().join("voice.yaml");
    fs::write(&voice,format!("server:\n  bind: 127.0.0.1:{voice_port}\nstt:\n  executable: /usr/bin/true\n  args: ['{{input}}','{{output}}']\n  timeout_seconds: 1\ntts:\n  executable: /usr/bin/true\n  args: ['{{output}}']\n  timeout_seconds: 1\n")).unwrap();
    fs::set_permissions(&voice, fs::Permissions::from_mode(0o600)).unwrap();
    cfg.extensions.voice_config = Some(voice);
    write(&path, &cfg);
    let key = "DisposableTestKey0123456789abcdefABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let mut cmd = command(&path, &["gateway", "start"]);
    cmd.env("MISSY_API_TOKEN", key)
        .env("MISSY_OPERATIONS_READ_KEY", key)
        .env(
            "MISSY_WEBHOOK_HMAC_KEY",
            "DifferentDisposableWebhookKey0123456789abcdef",
        );
    let mut child = Process(
        cmd.stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    wait_ready(&path);
    assert!(http(api_port, "/api/v1/health", key).starts_with("HTTP/1.1 200"));
    assert!(http(operations_port, "/diagnostics", key).starts_with("HTTP/1.1 200"));
    assert!(std::net::TcpStream::connect(("127.0.0.1", voice_port)).is_ok());
    let status: Value =
        serde_json::from_slice(&fs::read(cfg.state_dir.join("run/gateway_status.json")).unwrap())
            .unwrap();
    for service in ["http", "operations", "voice", "scheduler"] {
        assert_eq!(status["subsystems"][service], "ready");
    }
    unsafe {
        libc::kill(child.0.id() as i32, libc::SIGTERM);
    }
    assert!(child.0.wait().unwrap().success());
    assert!(std::net::TcpStream::connect(("127.0.0.1", api_port)).is_err());
}

#[test]
fn private_config_backup_redacted_plan_restore_and_tamper_refusal() {
    let temp = tempfile::tempdir().unwrap();
    let (mut cfg, path) = config(temp.path());
    let original = fs::read(&path).unwrap();
    let snapshot = temp.path().join("snapshot.json");
    let result = command(&path, &["config", "backup", snapshot.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::metadata(&snapshot).unwrap().permissions().mode() & 0o777,
        0o600
    );
    cfg.system_prompt = "DisposableSecretMarkerNeverPrint".into();
    write(&path, &cfg);
    let candidate = temp.path().join("candidate.yaml");
    fs::write(&candidate, &original).unwrap();
    fs::set_permissions(&candidate, fs::Permissions::from_mode(0o600)).unwrap();
    let plan = command(&path, &["config", "plan", candidate.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(plan.status.success());
    assert!(!String::from_utf8_lossy(&plan.stdout).contains("DisposableSecretMarkerNeverPrint"));
    let plan: Value = serde_json::from_slice(&plan.stdout).unwrap();
    assert_eq!(plan["restart_required"], true);
    assert_eq!(plan["running_services_changed"], false);
    assert!(
        plan["changed_fields"]
            .as_array()
            .unwrap()
            .contains(&Value::String("system_prompt".into()))
    );
    assert!(
        !command(&path, &["config", "rollback", snapshot.to_str().unwrap()])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(
        command(
            &path,
            &[
                "config",
                "rollback",
                snapshot.to_str().unwrap(),
                "--confirm"
            ]
        )
        .output()
        .unwrap()
        .status
        .success()
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::write(&path, "this: [invalid YAML").unwrap();
    assert!(
        command(
            &path,
            &[
                "config",
                "rollback",
                snapshot.to_str().unwrap(),
                "--confirm"
            ]
        )
        .output()
        .unwrap()
        .status
        .success()
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    let mut bad: Value = serde_json::from_slice(&fs::read(&snapshot).unwrap()).unwrap();
    bad["sha256"] = "bad".into();
    fs::write(&snapshot, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(
        !command(
            &path,
            &[
                "config",
                "rollback",
                snapshot.to_str().unwrap(),
                "--confirm"
            ]
        )
        .output()
        .unwrap()
        .status
        .success()
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(
        !cfg.state_dir.exists(),
        "config commands must not open runtime state"
    );
}

#[test]
fn packaging_stage_activate_rollback_only_changes_disposable_binary_links() {
    let temp = tempfile::tempdir().unwrap();
    let prefix = temp.path().join("package");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packaging/rust/release.sh");
    use sha2::{Digest, Sha256};
    let binary = env!("CARGO_BIN_EXE_missy-rs");
    let digest = hex::encode(Sha256::digest(fs::read(binary).unwrap()));
    let run = |args: &[&str]| {
        Command::new("/bin/sh")
            .arg(&script)
            .args(args)
            .output()
            .unwrap()
    };
    assert!(run(&["-n"]).status.code() != Some(0)); // Not a release action.
    let syntax = Command::new("/bin/sh")
        .arg("-n")
        .arg(&script)
        .status()
        .unwrap();
    assert!(syntax.success());
    for version in ["one", "two"] {
        let out = run(&["stage", prefix.to_str().unwrap(), version, binary, &digest]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            run(&["activate", prefix.to_str().unwrap(), version])
                .status
                .success()
        );
    }
    assert_eq!(
        fs::read_link(prefix.join("current")).unwrap(),
        Path::new("releases/two")
    );
    assert!(
        run(&["rollback", prefix.to_str().unwrap()])
            .status
            .success()
    );
    assert_eq!(
        fs::read_link(prefix.join("current")).unwrap(),
        Path::new("releases/one")
    );
    assert!(run(&["inspect", prefix.to_str().unwrap()]).status.success());
    assert!(
        !run(&[
            "stage",
            prefix.to_str().unwrap(),
            "bad",
            binary,
            &"0".repeat(64)
        ])
        .status
        .success()
    );
    // Slot directories intentionally immutable. Permit disposable cleanup only.
    for version in ["one", "two"] {
        fs::set_permissions(
            prefix.join("releases").join(version),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
    }
}

#[test]
fn plugin_source_cli_proposes_reviews_without_implicitly_building() {
    let temp = tempfile::tempdir().unwrap();
    let (_, path) = config(temp.path());
    let source = temp.path().join("source.yaml");
    fs::write(&source,"id: fixture\nsource: 'fn main() {}'\ndefinition:\n  name: plugin__fixture\n  description: fixture\n  parameters: {type: object, properties: {}, required: [], additionalProperties: false}\n  side_effect: true\nargs: []\ncapabilities: {}\n").unwrap();
    fs::set_permissions(&source, fs::Permissions::from_mode(0o600)).unwrap();
    let proposal = command(
        &path,
        &["plugins", "source-propose", source.to_str().unwrap()],
    )
    .output()
    .unwrap();
    assert!(
        proposal.status.success(),
        "{}",
        String::from_utf8_lossy(&proposal.stderr)
    );
    let proposal: Value = serde_json::from_slice(&proposal.stdout).unwrap();
    let hash = proposal["hash"].as_str().unwrap();
    assert_eq!(proposal["reviewed"], false);
    assert!(
        command(
            &path,
            &[
                "plugins",
                "source-review",
                "fixture",
                "--hash",
                hash,
                "--approve"
            ]
        )
        .output()
        .unwrap()
        .status
        .success()
    );
    let show = command(&path, &["plugins", "source-show", "fixture"])
        .output()
        .unwrap();
    assert!(show.status.success());
    let show: Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(show["reviewed"], true);
    assert!(
        !command(
            &path,
            &[
                "plugins",
                "source-review",
                "fixture",
                "--hash",
                &"0".repeat(64),
                "--approve"
            ]
        )
        .output()
        .unwrap()
        .status
        .success()
    );
}

#[test]
fn offline_discord_outbox_requires_exact_binding_and_confirm_never_replays() {
    let temp = tempfile::tempdir().unwrap();
    let (cfg, path) = config(temp.path());
    let dir = cfg.state_dir.join("discord");
    fs::create_dir_all(&dir).unwrap();
    fs::set_permissions(&cfg.state_dir, fs::Permissions::from_mode(0o700)).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(
        command(&path, &["discord-outbox", "inspect"])
            .output()
            .unwrap()
            .status
            .success()
    );
    let db = rusqlite::Connection::open(dir.join("discord.sqlite")).unwrap();
    db.execute(
        "INSERT INTO intake(id,payload,hash,denied,interaction) VALUES('r','{}','h',0,0)",
        [],
    )
    .unwrap();
    db.execute("INSERT INTO outbox(nonce,request,chunk,text,state) VALUES('n','r',0,'must not print payload','uncertain')",[]).unwrap();
    drop(db);
    let result = command(&path, &["discord-outbox", "inspect"])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("must not print payload"));
    let rows: Value = serde_json::from_slice(&result.stdout).unwrap();
    let binding = rows[0]["binding_sha256"].as_str().unwrap();
    let base = [
        "discord-outbox",
        "resolve",
        "n",
        "--binding",
        binding,
        "--operator",
        "test",
        "--evidence",
        "manual abandonment",
    ];
    assert!(!command(&path, &base).output().unwrap().status.success());
    let mut confirmed = base.to_vec();
    confirmed.push("--confirm");
    assert!(
        command(&path, &confirmed)
            .output()
            .unwrap()
            .status
            .success()
    );
    let db = rusqlite::Connection::open(dir.join("discord.sqlite")).unwrap();
    let row: (String, i64) = db
        .query_row(
            "SELECT state,attempts FROM outbox WHERE nonce='n'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(row, ("blocked".into(), 0));
}

#[test]
fn configured_watchdog_persists_trusted_health_under_supervision() {
    let temp = tempfile::tempdir().unwrap();
    let (mut cfg, path) = config(temp.path());
    let watchdog = temp.path().join("watchdog.yaml");
    fs::write(&watchdog,"enabled: true\nscope: {tenant: t, user: u, channel: cli, session: health}\npoll_seconds: 1\nfailure_threshold: 2\nprobes:\n  - {kind: database, name: runtime_db}\n").unwrap();
    fs::set_permissions(&watchdog, fs::Permissions::from_mode(0o600)).unwrap();
    cfg.extensions.watchdog_config = Some(watchdog);
    write(&path, &cfg);
    let mut child = Process(
        command(&path, &["gateway", "start"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    wait_ready(&path);
    let start = Instant::now();
    loop {
        let db = rusqlite::Connection::open(cfg.state_dir.join("scheduler.sqlite3")).unwrap();
        let healthy:bool=db.query_row("SELECT EXISTS(SELECT 1 FROM automation_health WHERE name='runtime_db' AND healthy=1)",[],|r|r.get(0)).unwrap();
        if healthy {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(20));
    }
    unsafe {
        libc::kill(child.0.id() as i32, libc::SIGTERM);
    }
    assert!(child.0.wait().unwrap().success());
    let status: Value =
        serde_json::from_slice(&fs::read(cfg.state_dir.join("run/gateway_status.json")).unwrap())
            .unwrap();
    assert_eq!(status["subsystems"]["watchdog"], "ready");
    assert_eq!(status["phase"], "stopped");
}
