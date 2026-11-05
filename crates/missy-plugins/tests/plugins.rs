use anyhow::Result;
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_plugins::*;
use missy_policy::Policy;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};
struct Approval;
struct Denied;
#[async_trait]
impl PluginApproval for Denied {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        anyhow::bail!("durable approval missing")
    }
    fn configuration_fingerprint(&self) -> String {
        "test-denied".into()
    }
}
#[async_trait]
impl PluginApproval for Approval {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "test-human-approved".into()
    }
}
fn scope() -> Scope {
    Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "s".into(),
    }
}
fn cfg() -> PluginConfig {
    PluginConfig {
        enabled: true,
        scope: scope(),
        allowed_ids: vec!["printf".into()],
        runtime_read_roots: vec![PathBuf::from("/usr/lib/x86_64-linux-gnu")],
        timeout_ms: 1000,
        max_output_bytes: 1024,
    }
}
fn manifest() -> PluginManifest {
    let path = PathBuf::from("/usr/bin/printf");
    PluginManifest {
        id: "printf".into(),
        definition: ToolDefinition {
            name: "plugin__printf".into(),
            description: "Literal text printer".into(),
            parameters: json!({"type":"object","additionalProperties":false,"properties":{"text":{"type":"string","maxLength":1000}},"required":["text"]}),
            side_effect: true,
        },
        command: CommandTemplate {
            executable: path.clone(),
            sha256: hex::encode(Sha256::digest(std::fs::read(path).unwrap())),
            args: vec!["%s".into(), "{text}".into()],
        },
        capabilities: PluginCapabilities::default(),
    }
}
fn ctx() -> ToolContext {
    ToolContext {
        scope: scope(),
        turn_id: "turn".into(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    }
}
fn call(text: &str) -> ToolCall {
    ToolCall {
        id: "original-call-id".into(),
        name: "plugin__printf".into(),
        arguments: json!({"text":text}),
    }
}
fn policy(text: &str) -> Arc<Policy> {
    Arc::new(
        Policy::new(PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![vec!["/usr/bin/printf".into(), "%s".into(), text.into()]],
            ..Default::default()
        })
        .unwrap(),
    )
}
fn activate(reg: &ToolRegistry, c: &PluginConfig) -> PluginProposal {
    let p = reg.propose(&scope(), manifest()).unwrap();
    reg.review(&scope(), &p.id, &p.hash, true).unwrap();
    reg.enable(&scope(), &p.id, &p.hash, c).unwrap();
    p
}

#[test]
fn registry_approval_scope_and_closed_schema() {
    let d = tempfile::tempdir().unwrap();
    let reg = ToolRegistry::open(d.path().join("private")).unwrap();
    let c = cfg();
    let p = reg.propose(&scope(), manifest()).unwrap();
    assert!(reg.enable(&scope(), &p.id, &p.hash, &c).is_err());
    assert!(reg.review(&scope(), &p.id, "wrong", true).is_err());
    let mut other = scope();
    other.user = "other".into();
    assert!(reg.review(&other, &p.id, &p.hash, true).is_err());
    reg.review(&scope(), &p.id, &p.hash, true).unwrap();
    reg.enable(&scope(), &p.id, &p.hash, &c).unwrap();
    let mut m = manifest();
    m.id = "remote".into();
    m.definition.name = "plugin__remote".into();
    m.definition.parameters["$ref"] = json!("https://invalid/schema");
    assert!(reg.propose(&scope(), m).is_err());
    let mut m = manifest();
    m.definition.parameters["properties"]["text"]["$ref"] = json!("file:///etc/passwd");
    assert!(m.validate().is_err());
    let mut m = manifest();
    m.definition.parameters["additionalProperties"] = json!(true);
    assert!(m.validate().is_err());
    let mut m = manifest();
    m.command.executable = "/usr/bin/bash".into();
    assert!(m.validate().is_err());
}
#[tokio::test]
async fn literal_elf_execution_and_scope_policy_guard() {
    let d = tempfile::tempdir().unwrap();
    let reg = ToolRegistry::open(d.path().join("private")).unwrap();
    activate(&reg, &cfg());
    let literal = "$(touch /tmp/never-plugin-injection); 'literal'";
    let denied = PluginTools::new(
        ToolRegistry::open(d.path().join("private")).unwrap(),
        cfg(),
        policy("denied"),
        Arc::new(Denied),
    )
    .unwrap();
    assert!(denied.preflight(&ctx(), &call("denied")).await.is_err());
    assert!(denied.execute(&ctx(), &call("denied")).await.is_err());
    let tools = PluginTools::new(reg, cfg(), policy(literal), Arc::new(Approval)).unwrap();
    assert!(tools.definitions().iter().all(|d| d.side_effect));
    let result = tools.execute(&ctx(), &call(literal)).await.unwrap();
    assert_eq!(result["stdout"], literal);
    assert_eq!(result["success"], true);
    assert!(
        tools
            .execute(&ctx(), &call("not-policy-approved"))
            .await
            .is_err()
    );
    let mut context = ctx();
    context.scope.user = "other".into();
    assert!(tools.execute(&context, &call(literal)).await.is_err());
    let mut invalid = call(literal);
    invalid.arguments["extra"] = json!("bad");
    assert!(tools.execute(&ctx(), &invalid).await.is_err());
}
#[tokio::test]
async fn no_environment_and_timeout_and_hash_drift() {
    let d = tempfile::tempdir().unwrap();
    let binary = d.path().join("fixture");
    // Compile a disposable native fixture only in tests. Runtime never compiles.
    let src = d.path().join("fixture.c");
    std::fs::write(&src,b"#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n#include <sys/socket.h>\nint main(int c,char**v){ if(c>1 && v[1][0]=='s'){sleep(5);return 0;} printf(\"%s|%s|%d|%d\",getenv(\"PATH\")?\"bad\":\"empty\",getenv(\"HOME\")?\"bad\":\"empty\",fopen(\"/etc/passwd\",\"r\")?1:0,socket(AF_INET,SOCK_STREAM,0)>=0?1:0);return 0;}\n").unwrap();
    assert!(
        std::process::Command::new("cc")
            .args([src.to_str().unwrap(), "-o", binary.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
    let mut m = manifest();
    m.command.executable = binary.clone();
    m.command.sha256 = hex::encode(Sha256::digest(std::fs::read(&binary).unwrap()));
    m.command.args = vec!["{text}".into()];
    let reg = ToolRegistry::open(d.path().join("private")).unwrap();
    let p = reg.propose(&scope(), m).unwrap();
    reg.review(&scope(), &p.id, &p.hash, true).unwrap();
    reg.enable(&scope(), &p.id, &p.hash, &cfg()).unwrap();
    let mut c = cfg();
    c.timeout_ms = 100;
    let policy = Arc::new(
        Policy::new(PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![
                vec![binary.to_str().unwrap().into(), "env".into()],
                vec![binary.to_str().unwrap().into(), "sleep".into()],
            ],
            ..Default::default()
        })
        .unwrap(),
    );
    let tools = PluginTools::new(reg, c, policy, Arc::new(Approval)).unwrap();
    assert_eq!(
        tools.execute(&ctx(), &call("env")).await.unwrap()["stdout"],
        "empty|empty|0|0"
    );
    assert!(
        tools
            .execute(&ctx(), &call("sleep"))
            .await
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    // A dropped execution future must synchronously kill and reap its process.
    let tools = Arc::new(tools);
    let task = tokio::spawn({
        let tools = tools.clone();
        async move { tools.execute(&ctx(), &call("sleep")).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    let prefix = binary.as_os_str().as_encoded_bytes();
    let pids: Vec<u32> = std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(|e| {
            let e = e.ok()?;
            let pid = e.file_name().to_str()?.parse().ok()?;
            let bytes = std::fs::read(e.path().join("cmdline")).ok()?;
            bytes.starts_with(prefix).then_some(pid)
        })
        .collect();
    assert!(!pids.is_empty(), "sleep fixture must be alive before abort");
    task.abort();
    let _ = task.await;
    for pid in pids {
        assert!(
            !PathBuf::from(format!("/proc/{pid}")).exists(),
            "child must be reaped on future drop"
        );
    }
    std::fs::write(&binary, b"#!/bin/sh\necho unsafe").unwrap();
    assert!(tools.execute(&ctx(), &call("env")).await.is_err());
}
#[test]
fn snapshots_and_fingerprints_pin_config() {
    let d = tempfile::tempdir().unwrap();
    let reg = ToolRegistry::open(d.path().join("private")).unwrap();
    activate(&reg, &cfg());
    let tools = PluginTools::new(reg, cfg(), policy("x"), Arc::new(Approval)).unwrap();
    let other = PluginTools::new(
        ToolRegistry::open(d.path().join("private")).unwrap(),
        cfg(),
        policy("different-vector"),
        Arc::new(Approval),
    )
    .unwrap();
    assert_ne!(
        tools.configuration_fingerprint(),
        other.configuration_fingerprint()
    );
    let reg = ToolRegistry::open(d.path().join("private")).unwrap();
    let p = reg.get(&scope(), "printf").unwrap();
    reg.review(&scope(), "printf", &p.hash, false).unwrap();
    assert_eq!(tools.definitions().len(), 2);
    assert!(PluginTools::new(reg, cfg(), policy("x"), Arc::new(Approval)).is_err());
}
