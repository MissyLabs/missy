//! Real executable workflows. Disposable HOME/config/state; no provider, service,
//! network, legacy directory, SQL maintenance or real credentials are used.
use missy_core::{Config, Message, Scope, ToolCall, Usage};
use missy_store::{Store, ToolStatus, TurnStatus, hash_text};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    process::{Command, Output},
    time::Duration,
};

struct Lab {
    root: tempfile::TempDir,
    config: PathBuf,
    state: PathBuf,
    scope: Scope,
}
impl Lab {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        let config = root.path().join("config.yaml");
        let cfg = Config {
            state_dir: state.clone(),
            ..Default::default()
        };
        fs::write(&config, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();
        fs::set_permissions(&config, fs::Permissions::from_mode(0o600)).unwrap();
        fs::create_dir(root.path().join("home")).unwrap();
        Self {
            root,
            config,
            state,
            scope: Scope {
                tenant: "tenant".into(),
                user: "principal".into(),
                channel: "cli".into(),
                session: "session".into(),
            },
        }
    }
    fn store(&self) -> Store {
        Store::open(self.state.join("runtime.sqlite3")).unwrap()
    }
    fn run_scope(&self, scope: &Scope, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_missy-rs"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", self.root.path().join("home"))
            .env("XDG_CONFIG_HOME", self.root.path().join("home/config"))
            .env("XDG_STATE_HOME", self.root.path().join("home/state"))
            .current_dir(self.root.path())
            .arg("--config")
            .arg(&self.config)
            .args([
                "operator",
                "--tenant",
                &scope.tenant,
                "--user",
                &scope.user,
                "--channel",
                &scope.channel,
                "--session",
                &scope.session,
            ])
            .args(args)
            .output()
            .unwrap()
    }
    fn run(&self, args: &[&str]) -> Output {
        self.run_scope(&self.scope, args)
    }
    fn ok(&self, args: &[&str]) -> Value {
        parse(self.run(args))
    }
    fn fail(&self, args: &[&str], expected: &str) {
        let output = self.run(args);
        assert!(
            !output.status.success(),
            "unexpected success: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "expected {expected:?}, got {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fn file(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.root.path().join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
    fn completed(&self, scope: &Scope, request: &str) -> String {
        let store = self.store();
        let turn = store.admit(scope, request, "original prompt").unwrap().turn;
        let lease = store
            .acquire_lease(scope, &turn.id, "fixture", Duration::from_secs(60))
            .unwrap();
        store
            .append_message(scope, &lease, &Message::text("user", "preserved history"))
            .unwrap();
        store
            .finish(scope, &lease, TurnStatus::Completed, Some("complete"), None)
            .unwrap();
        turn.id
    }
}
fn parse(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn as_refs(args: &[String]) -> Vec<&str> {
    args.iter().map(String::as_str).collect()
}

#[test]
fn display_name_and_logical_clear_are_exact_four_component_scope() {
    let lab = Lab::new();
    let id = lab.completed(&lab.scope, "original");
    let store = lab.store();
    let original_turn = store.get_turn(&lab.scope, &id).unwrap();
    let original_audit = store.audit_entries(&lab.scope).unwrap().len();
    assert_eq!(
        lab.ok(&["session", "display-name", "Operator's presentation"])["display_name"],
        "Operator's presentation"
    );
    for component in 0..4 {
        let mut other = lab.scope.clone();
        match component {
            0 => other.tenant.push('x'),
            1 => other.user.push('x'),
            2 => other.channel.push('x'),
            _ => other.session.push('x'),
        }
        assert!(
            parse(lab.run_scope(&other, &["session", "display-name"]))["display_name"].is_null()
        );
        assert!(store.get_turn(&other, &id).is_err());
    }
    lab.fail(&["session", "clear"], "--confirm");
    assert_eq!(store.history(&lab.scope, 100).unwrap().len(), 1);
    let cleared = lab.ok(&["session", "clear", "--confirm"]);
    assert_eq!(cleared["logical_clear"], true);
    assert!(store.history(&lab.scope, 100).unwrap().is_empty());
    assert_eq!(
        store.get_turn(&lab.scope, &id).unwrap().prompt,
        original_turn.prompt
    );
    assert!(store.audit_entries(&lab.scope).unwrap().len() > original_audit);
    assert!(
        !store
            .admit(&lab.scope, "original", "original prompt")
            .unwrap()
            .created
    );
    assert!(
        store
            .admit(&lab.scope, "original", "different prompt")
            .is_err()
    );
    lab.fail(
        &["session", "display-name", "bad\nname"],
        "invalid session display name",
    );
    lab.fail(
        &["session", "show", "--limit", "0"],
        "limit must be 1..1000",
    );
}

#[test]
fn cleanup_dry_run_terminal_purge_and_nonterminal_live_rejection() {
    let lab = Lab::new();
    let id = lab.completed(&lab.scope, "old");
    let store = lab.store();
    let cutoff = "9223372036854775807";
    lab.fail(
        &["session", "cleanup", "--before-unix-ms", cutoff],
        "--dry-run or --confirm",
    );
    let audit_before = store.audit_entries(&lab.scope).unwrap().len();
    let preview = lab.ok(&[
        "session",
        "cleanup",
        "--before-unix-ms",
        cutoff,
        "--dry-run",
    ]);
    assert!(preview["content_records"].as_u64().unwrap() > 0);
    assert_eq!(
        store.get_turn(&lab.scope, &id).unwrap().prompt,
        "original prompt"
    );
    assert_eq!(store.audit_entries(&lab.scope).unwrap().len(), audit_before);
    let mut other = lab.scope.clone();
    other.session = "other".into();
    let other_id = lab.completed(&other, "other");
    let cleaned = lab.ok(&[
        "session",
        "cleanup",
        "--before-unix-ms",
        cutoff,
        "--confirm",
    ]);
    assert_eq!(cleaned["dry_run"], false);
    assert_eq!(store.get_turn(&lab.scope, &id).unwrap().prompt, "");
    assert_eq!(
        store.get_turn(&other, &other_id).unwrap().prompt,
        "original prompt"
    );
    lab.ok(&["session", "clear", "--confirm"]);
    assert!(lab.ok(&["session", "display-name"])["display_name"].is_null());
    assert!(
        !store
            .admit(&lab.scope, "old", "original prompt")
            .unwrap()
            .created
    );
    assert!(store.admit(&lab.scope, "old", "changed").is_err());
    assert!(store.verify_audit(&lab.scope).unwrap() > audit_before);

    let pending = store
        .admit(&lab.scope, "pending", "never dispatched")
        .unwrap()
        .turn;
    for action in [
        vec!["session", "clear", "--confirm"],
        vec![
            "session",
            "cleanup",
            "--before-unix-ms",
            cutoff,
            "--dry-run",
        ],
        vec![
            "session",
            "cleanup",
            "--before-unix-ms",
            cutoff,
            "--confirm",
        ],
    ] {
        lab.fail(&action, "nonterminal");
    }
    let lease = store
        .acquire_lease(&lab.scope, &pending.id, "fixture", Duration::from_secs(60))
        .unwrap();
    lab.fail(&["session", "clear", "--confirm"], "leased");
    lab.fail(
        &[
            "session",
            "cleanup",
            "--before-unix-ms",
            cutoff,
            "--confirm",
        ],
        "active",
    );
    store
        .finish(
            &lab.scope,
            &lease,
            TurnStatus::Failed,
            None,
            Some("fixture"),
        )
        .unwrap();
    lab.ok(&[
        "session",
        "cleanup",
        "--before-unix-ms",
        cutoff,
        "--confirm",
    ]);
}

#[test]
fn terminal_turn_with_uncertain_tool_blocks_clear_and_cleanup() {
    let lab = Lab::new();
    let store = lab.store();
    let turn = store
        .admit(&lab.scope, "terminal-unknown", "operator must investigate")
        .unwrap()
        .turn;
    let lease = store
        .acquire_lease(&lab.scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    let call = ToolCall {
        id: "uncertain".into(),
        name: "external-mutation".into(),
        arguments: json!({"value":1}),
    };
    store.tool_begin(&lab.scope, &lease, &call).unwrap();
    store.tool_running(&lab.scope, &lease, &call.id).unwrap();
    store
        .finish(
            &lab.scope,
            &lease,
            TurnStatus::Failed,
            None,
            Some("unknown effect"),
        )
        .unwrap();
    assert_eq!(
        store.get_turn(&lab.scope, &turn.id).unwrap().status,
        TurnStatus::Failed
    );
    lab.fail(&["session", "clear", "--confirm"], "uncertain");
    lab.fail(
        &[
            "session",
            "cleanup",
            "--before-unix-ms",
            "9223372036854775807",
            "--dry-run",
        ],
        "uncertain",
    );
    lab.fail(
        &[
            "session",
            "cleanup",
            "--before-unix-ms",
            "9223372036854775807",
            "--confirm",
        ],
        "uncertain",
    );
    assert_eq!(store.tool_journal(&lab.scope, &turn.id).unwrap().len(), 1);
}

#[test]
fn signed_export_is_pinned_scoped_offline_nonoverwriting_and_chain_checked() {
    let lab = Lab::new();
    lab.completed(&lab.scope, "audit");
    lab.ok(&["session", "display-name", "Audit session"]);
    let verified = lab.ok(&["audit", "verify"]);
    assert_eq!(verified["verified"], true);
    let recent = lab.ok(&[
        "audit",
        "recent",
        "--kind",
        "session_renamed",
        "--limit",
        "1",
    ]);
    assert_eq!(recent["entries"].as_array().unwrap().len(), 1);
    lab.fail(
        &["audit", "recent", "--limit", "1001"],
        "limit must be 1..1000",
    );
    let path = lab.root.path().join("export.json");
    let output = lab.ok(&["audit", "export", path.to_str().unwrap()]);
    let pin = output["public_key"].as_str().unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    lab.fail(
        &["audit", "export", path.to_str().unwrap()],
        "must not already exist",
    );
    assert_eq!(
        lab.ok(&[
            "audit",
            "verify-export",
            path.to_str().unwrap(),
            "--public-key",
            pin
        ])["verified"],
        true
    );
    lab.fail(
        &[
            "audit",
            "verify-export",
            path.to_str().unwrap(),
            "--public-key",
            "not-the-pinned-key",
        ],
        "identity or digest mismatch",
    );
    let mut other = lab.scope.clone();
    other.user = "other".into();
    assert!(
        !lab.run_scope(
            &other,
            &[
                "audit",
                "verify-export",
                path.to_str().unwrap(),
                "--public-key",
                pin
            ]
        )
        .status
        .success()
    );

    let mut document: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    document["payload"]["entries"][0]["kind"] = json!("tampered");
    let tampered = lab.file("tampered.json", serde_json::to_vec(&document).unwrap());
    lab.fail(
        &[
            "audit",
            "verify-export",
            tampered.to_str().unwrap(),
            "--public-key",
            pin,
        ],
        "identity or digest mismatch",
    );
    // A correctly signed but internally inconsistent chain is still rejected.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Payload {
        format: String,
        scope: Scope,
        entries: Vec<missy_store::AuditEntry>,
    }
    let payload: Payload = serde_json::from_value(document["payload"].clone()).unwrap();
    let signature = missy_security::Vault::open(lab.state.join("vault"))
        .unwrap()
        .sign_export(&serde_json::to_vec(&payload).unwrap())
        .unwrap();
    document["signature"] = serde_json::to_value(signature).unwrap();
    let broken_chain = lab.file("broken-chain.json", serde_json::to_vec(&document).unwrap());
    lab.fail(
        &[
            "audit",
            "verify-export",
            broken_chain.to_str().unwrap(),
            "--public-key",
            pin,
        ],
        "audit hash mismatch",
    );

    // Verification needs neither the local database nor the signing identity.
    fs::rename(&lab.state, lab.root.path().join("retired-state")).unwrap();
    assert_eq!(
        lab.ok(&[
            "audit",
            "verify-export",
            path.to_str().unwrap(),
            "--public-key",
            pin
        ])["verified"],
        true
    );
    assert!(!lab.state.exists());
    assert!(!lab.root.path().join("home/.missy").exists());
}

#[test]
fn exact_tool_and_billing_reconciliation_never_dispatch_or_allow_replay() {
    let lab = Lab::new();
    let store = lab.store();
    let turn = store
        .admit(&lab.scope, "unknown", "external request")
        .unwrap()
        .turn;
    let lease = store
        .acquire_lease(&lab.scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    store
        .reserve_turn_budget(
            &lab.scope,
            &lease,
            Usage {
                input_tokens: 10,
                output_tokens: 20,
                cost_microusd: 100,
                cost_known: true,
            },
            10000,
            10000,
        )
        .unwrap();
    let call = ToolCall {
        id: "effect".into(),
        name: "shell_exec".into(),
        arguments: json!({"argv":["never-run"]}),
    };
    store.tool_begin(&lab.scope, &lease, &call).unwrap();
    store.tool_running(&lab.scope, &lease, &call.id).unwrap();
    let args_hash = hash_text(&serde_json::to_string(&call.arguments).unwrap());
    let result = lab.file("result.json", br#"{"externally_verified":true}"#);
    let evidence = lab.file(
        "evidence.txt",
        "Independent operator checked external receipt #42",
    );
    let usage = lab.file(
        "usage.json",
        br#"{"input_tokens":10,"output_tokens":20,"cost_microusd":75,"cost_known":true}"#,
    );
    let tool_args = |hash: &str| {
        vec![
            "effects".to_owned(),
            "reconcile-tool".into(),
            turn.id.clone(),
            call.id.clone(),
            "--arguments-sha256".into(),
            hash.into(),
            "--result".into(),
            result.to_str().unwrap().into(),
            "--evidence".into(),
            evidence.to_str().unwrap().into(),
            "--succeeded".into(),
            "--confirm".into(),
        ]
    };
    let attempt = tool_args(&args_hash);
    lab.fail(&as_refs(&attempt), "leased");
    store
        .finish(
            &lab.scope,
            &lease,
            TurnStatus::Interrupted,
            None,
            Some("lost receipt"),
        )
        .unwrap();
    let inspected = lab.ok(&["effects", "inspect", &turn.id]);
    assert_eq!(inspected["tools"][0]["arguments_sha256"], args_hash);
    assert_eq!(inspected["tools"][0]["record"]["status"], "outcome_unknown");
    assert_eq!(inspected["prompt_sha256"], hash_text("external request"));
    lab.fail(&["session", "clear", "--confirm"], "nonterminal");
    let billing_args = [
        "effects",
        "reconcile-billing",
        &turn.id,
        "--prompt-sha256",
        &hash_text("external request"),
        "--usage",
        usage.to_str().unwrap(),
        "--evidence",
        evidence.to_str().unwrap(),
        "--confirm",
    ];
    lab.fail(&billing_args, "reconcile all tool effects");
    let wrong = tool_args(&"0".repeat(64));
    lab.fail(&as_refs(&wrong), "argument hash mismatch");
    let mut other = lab.scope.clone();
    other.tenant = "other".into();
    assert!(!lab.run_scope(&other, &as_refs(&attempt)).status.success());
    let mut unconfirmed = attempt.clone();
    unconfirmed.pop();
    lab.fail(&as_refs(&unconfirmed), "--confirm");
    assert_eq!(lab.ok(&as_refs(&attempt))["automatic_replay"], false);
    assert!(matches!(
        store.tool_journal(&lab.scope, &turn.id).unwrap()[0].status,
        ToolStatus::Succeeded
    ));
    lab.fail(&as_refs(&attempt), "final disposition");
    let abandoned = lab.ok(&billing_args);
    assert_eq!(abandoned["turn_status"], "failed");
    assert_eq!(
        store.get_turn(&lab.scope, &turn.id).unwrap().status,
        TurnStatus::Failed
    );
    let totals = store.budget_totals().unwrap();
    assert_eq!(totals.used_cost_microusd, 75);
    assert_eq!(totals.reserved_cost_microusd, 75);
    let budget = lab.ok(&["budget"]);
    assert_eq!(budget["scope_kind"], "database_global");
    assert_eq!(budget["scope_filter_applied"], false);
    assert_eq!(budget["totals"]["used_cost_microusd"], 75);
    assert!(
        !store
            .admit(&lab.scope, "unknown", "external request")
            .unwrap()
            .created
    );
    assert!(
        store
            .acquire_lease(&lab.scope, &turn.id, "replay", Duration::from_secs(60))
            .is_err()
    );
    lab.fail(&billing_args, "disposition or prompt hash mismatch");
    let security = lab.ok(&["audit", "security"]);
    assert!(
        security["entries"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "operator_tool_reconciliation")
    );
    lab.ok(&["audit", "verify"]);
}

#[test]
fn bounded_operator_inputs_refuse_symlinks_and_contradictory_dispositions() {
    let lab = Lab::new();
    let file = lab.file("small.json", b"{}");
    let link = lab.root.path().join("symlink.json");
    symlink(&file, &link).unwrap();
    lab.fail(
        &[
            "audit",
            "verify-export",
            link.to_str().unwrap(),
            "--public-key",
            "irrelevant",
        ],
        "cannot open operator input",
    );
    let large = lab.file("large.json", vec![b' '; 8 * 1024 * 1024 + 1]);
    lab.fail(
        &[
            "audit",
            "verify-export",
            large.to_str().unwrap(),
            "--public-key",
            "irrelevant",
        ],
        "bounded regular file",
    );
    lab.fail(
        &[
            "effects",
            "reconcile-tool",
            "turn",
            "call",
            "--arguments-sha256",
            &"0".repeat(64),
            "--result",
            file.to_str().unwrap(),
            "--evidence",
            file.to_str().unwrap(),
            "--succeeded",
            "--failed",
            "--confirm",
        ],
        "cannot be used with",
    );
    assert!(!lab.root.path().join("home/.missy").exists());
}
