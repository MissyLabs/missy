use missy_core::{Config, Scope};
use missy_store::Store;
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
    process::{Command, Output},
};

fn private(path: &Path, data: impl AsRef<[u8]>) {
    fs::write(path, data).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn run(root: &Path, config: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_missy-rs"))
        .env_clear()
        .env("HOME", root)
        .env("PATH", "/usr/bin:/bin")
        .current_dir(root)
        .arg("--config")
        .arg(config)
        .args([
            "admin",
            "--tenant",
            "t",
            "--user",
            "u",
            "--channel",
            "c",
            "--session",
            "ignored",
        ])
        .args(args)
        .output()
        .unwrap()
}
fn ok(out: Output) -> Value {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn exact_scope_inventory_and_metadata_scan() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.yaml");
    let cfg = Config {
        state_dir: root.path().join("state"),
        system_prompt: "NEVER_ECHO_SECRET".into(),
        ..Default::default()
    };
    private(&config, serde_yaml_ng::to_string(&cfg).unwrap());
    let store = Store::open(cfg.state_dir.join("runtime.sqlite3")).unwrap();
    let scope = Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "match-one".into(),
    };
    store.admit(&scope, "r", "private prompt").unwrap();
    let mut other = scope.clone();
    other.user = "other".into();
    other.session = "match-leak".into();
    store.admit(&other, "r", "secret").unwrap();
    let output = run(root.path(), &config, &["sessions", "--prefix", "match"]);
    let result = ok(output);
    assert_eq!(result["sessions"], json!(["match-one"]));
    let scan = run(root.path(), &config, &["security-scan"]);
    assert!(scan.status.success());
    assert!(!String::from_utf8_lossy(&scan.stdout).contains("NEVER_ECHO_SECRET"));
    assert_eq!(
        serde_json::from_slice::<Value>(&scan.stdout).unwrap()["config"]["private"],
        true
    );
}

#[test]
fn mcp_plan_apply_binding_and_symlink_refusal() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.yaml");
    let mut cfg = Config {
        state_dir: root.path().join("state"),
        system_prompt: "NEVER_ECHO_SECRET".into(),
        ..Default::default()
    };
    cfg.policy.mcp_enabled = true;
    cfg.policy.allowed_commands = vec![vec!["/usr/bin/example".into()]];
    private(&config, serde_yaml_ng::to_string(&cfg).unwrap());
    let manifest = root.path().join("manifest.json");
    private(&manifest, br#"{"tools":[]}"#);
    let instruction = root.path().join("instruction.json");
    private(&instruction,serde_json::to_vec(&json!({"operation":"add","entry":{"name":"sample","command":["/usr/bin/example"],"env":{},"timeout_seconds":3},"manifest":manifest})).unwrap());
    let i = instruction.to_str().unwrap();
    let plan = ok(run(root.path(), &config, &["mcp-plan", i]));
    let c = plan["config_sha256"].as_str().unwrap();
    let p = plan["instruction_sha256"].as_str().unwrap();
    let snapshot = root.path().join("snapshot.json");
    let s = snapshot.to_str().unwrap();
    private(&manifest, br#"{"tools":[{"name":"changed"}]}"#);
    assert!(
        !run(
            root.path(),
            &config,
            &[
                "mcp-apply",
                i,
                "--config-sha256",
                c,
                "--instruction-sha256",
                p,
                "--snapshot",
                s,
                "--confirm"
            ]
        )
        .status
        .success()
    );
    private(&manifest, br#"{"tools":[]}"#);
    let applied = run(
        root.path(),
        &config,
        &[
            "mcp-apply",
            i,
            "--config-sha256",
            c,
            "--instruction-sha256",
            p,
            "--snapshot",
            s,
            "--confirm",
        ],
    );
    assert!(!String::from_utf8_lossy(&applied.stdout).contains("NEVER_ECHO_SECRET"));
    assert_eq!(ok(applied)["applied"], true);
    let next: Config = serde_yaml_ng::from_slice(&fs::read(&config).unwrap()).unwrap();
    assert_eq!(
        next.mcp[0].manifest_sha256,
        missy_tools::manifest_digest(&json!({"tools":[]}))
    );
    let linked = root.path().join("link");
    symlink(&instruction, &linked).unwrap();
    assert!(
        !run(
            root.path(),
            &config,
            &["mcp-plan", linked.to_str().unwrap()]
        )
        .status
        .success()
    );
}
