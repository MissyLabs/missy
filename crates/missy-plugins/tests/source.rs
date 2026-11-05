use missy_core::{Scope, ToolDefinition};
use missy_plugins::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};
use tokio_util::sync::CancellationToken;

fn scope() -> Scope {
    Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "s".into(),
    }
}
fn source(id: &str, text: &str) -> RustSource {
    RustSource {
        id: id.into(),
        source: text.into(),
        definition: ToolDefinition {
            name: format!("plugin__{id}"),
            description: "Reviewed fixture printer".into(),
            parameters: json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
            side_effect: true,
        },
        args: vec![],
        capabilities: PluginCapabilities::default(),
    }
}
fn config() -> RustBuildConfig {
    let result = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let sysroot = fs::canonicalize(String::from_utf8(result.stdout).unwrap().trim()).unwrap();
    let rustc = sysroot.join("bin/rustc");
    let linker =
        fs::canonicalize(sysroot.join("lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld"))
            .unwrap();
    RustBuildConfig {
        rustc_sha256: hex::encode(Sha256::digest(fs::read(&rustc).unwrap())),
        rustc,
        linker_sha256: hex::encode(Sha256::digest(fs::read(&linker).unwrap())),
        linker,
        read_roots: vec![
            sysroot.clone(),
            PathBuf::from("/usr/lib"),
            PathBuf::from("/usr/bin"),
        ],
        native_library_roots: vec![
            "/usr/lib/x86_64-linux-gnu".into(),
            "/usr/lib/gcc/x86_64-linux-gnu/13".into(),
        ],
        linker_args: vec![
            "/usr/lib/x86_64-linux-gnu/Scrt1.o".into(),
            "/usr/lib/x86_64-linux-gnu/crti.o".into(),
            "/usr/lib/x86_64-linux-gnu/crtn.o".into(),
            "--dynamic-linker=/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2".into(),
        ],
        sysroot,
        timeout_ms: 30000,
        max_output_bytes: 65536,
    }
}

#[tokio::test]
async fn actual_reviewed_native_rust_compile_is_separate_from_manifest_review() {
    let t = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::open(t.path().join("registry")).unwrap();
    let proposal = registry
        .propose_rust_source(
            &scope(),
            source("echo", r#"fn main(){println!("SAFE_COMPILED_ECHO");}"#),
        )
        .unwrap();
    assert!(!proposal.reviewed);
    let cfg = config();
    assert!(
        registry
            .build_rust_source(
                &scope(),
                "echo",
                &proposal.hash,
                &cfg,
                true,
                CancellationToken::new()
            )
            .await
            .is_err()
    );
    assert!(
        registry
            .review_rust_source(&scope(), "echo", &"0".repeat(64), true)
            .is_err()
    );
    registry
        .review_rust_source(&scope(), "echo", &proposal.hash, true)
        .unwrap();
    let built = registry
        .build_rust_source(
            &scope(),
            "echo",
            &proposal.hash,
            &cfg,
            true,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(
        fs::read(&built.manifest.command.executable)
            .unwrap()
            .starts_with(b"\x7fELF")
    );
    assert_ne!(built.hash, proposal.hash);
    let active = PluginConfig {
        enabled: true,
        scope: scope(),
        allowed_ids: vec!["echo".into()],
        runtime_read_roots: vec!["/usr/lib/x86_64-linux-gnu".into()],
        timeout_ms: 1000,
        max_output_bytes: 1024,
    };
    assert!(
        registry
            .enable(&scope(), "echo", &built.hash, &active)
            .is_err(),
        "source review must not approve binary"
    );
    registry
        .review(&scope(), "echo", &built.hash, true)
        .unwrap();
    registry
        .enable(&scope(), "echo", &built.hash, &active)
        .unwrap();
    // Test deliberately does not invoke the artifact. Building is not execution.
}

#[tokio::test]
async fn compiler_pin_cancel_denial_and_source_drift_fail_closed() {
    let t = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::open(t.path().join("registry")).unwrap();
    let p = registry
        .propose_rust_source(&scope(), source("safe", "fn main(){}"))
        .unwrap();
    registry
        .review_rust_source(&scope(), "safe", &p.hash, true)
        .unwrap();
    let mut cfg = config();
    cfg.rustc_sha256 = "0".repeat(64);
    assert!(
        registry
            .build_rust_source(
                &scope(),
                "safe",
                &p.hash,
                &cfg,
                true,
                CancellationToken::new()
            )
            .await
            .is_err()
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(
        registry
            .build_rust_source(&scope(), "safe", &p.hash, &config(), true, cancelled)
            .await
            .is_err()
    );
    assert!(
        registry
            .build_rust_source(
                &scope(),
                "safe",
                &p.hash,
                &config(),
                false,
                CancellationToken::new()
            )
            .await
            .is_err()
    );
    let path = fs::read_dir(registry.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut json: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    json["source"]["source"] = json!("fn main(){panic!(\"drift\")}");
    fs::write(path, serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(registry.rust_source(&scope(), "safe").is_err());
}

#[tokio::test]
async fn compiler_cannot_include_external_private_source_or_publish_diagnostics() {
    let t = tempfile::tempdir().unwrap();
    let secret = t.path().join("not-readable-by-compiler");
    fs::write(&secret, "PRIVATE_BUILD_CANARY").unwrap();
    let registry = ToolRegistry::open(t.path().join("registry")).unwrap();
    let text = format!(
        "fn main(){{println!(\"{{}}\",include_str!({:?}));}}",
        secret
    );
    let p = registry
        .propose_rust_source(&scope(), source("denied", &text))
        .unwrap();
    registry
        .review_rust_source(&scope(), "denied", &p.hash, true)
        .unwrap();
    let error = registry
        .build_rust_source(
            &scope(),
            "denied",
            &p.hash,
            &config(),
            true,
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(!format!("{error:#}").contains("PRIVATE_BUILD_CANARY"));
    assert!(registry.get(&scope(), "denied").is_err());
}
