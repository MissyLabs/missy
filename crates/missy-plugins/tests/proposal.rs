use anyhow::Result;
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_plugins::*;
use missy_policy::Policy;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;
struct Gate(bool);
#[async_trait]
impl PluginApproval for Gate {
    async fn authorize(&self, _: &ToolContext, call: &ToolCall) -> Result<()> {
        assert_eq!(call.id, "original-proposal-id");
        anyhow::ensure!(self.0, "approval denied");
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        format!("test-gate-{}", self.0)
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
fn context() -> ToolContext {
    ToolContext {
        scope: scope(),
        turn_id: "turn".into(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    }
}
fn manifest() -> PluginManifest {
    PluginManifest {
        id: "printer".into(),
        definition: ToolDefinition {
            name: "plugin__printer".into(),
            description: "Literal printer".into(),
            parameters: json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}),
            side_effect: true,
        },
        command: CommandTemplate {
            executable: "/usr/bin/printf".into(),
            sha256: hex::encode(Sha256::digest(std::fs::read("/usr/bin/printf").unwrap())),
            args: vec!["%s".into(), "{text}".into()],
        },
        capabilities: PluginCapabilities::default(),
    }
}
#[tokio::test]
async fn model_proposes_only_pending_gated_scope_bound_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("private");
    let config = PluginConfig {
        enabled: true,
        scope: scope(),
        allowed_ids: vec![],
        runtime_read_roots: vec!["/usr/lib/x86_64-linux-gnu".into()],
        timeout_ms: 1000,
        max_output_bytes: 1024,
    };
    let policy = Arc::new(
        Policy::new(PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![vec!["/usr/bin/printf".into(), "%s".into(), "x".into()]],
            ..Default::default()
        })
        .unwrap(),
    );
    let tools = PluginTools::new(
        ToolRegistry::open(&path).unwrap(),
        config.clone(),
        policy.clone(),
        Arc::new(Gate(true)),
    )
    .unwrap();
    assert_eq!(tools.definitions().len(), 1);
    assert_eq!(tools.definitions()[0].name, "self_create_tool");
    assert!(tools.definitions()[0].side_effect);
    let fingerprint = tools.configuration_fingerprint();
    let call = ToolCall {
        id: "original-proposal-id".into(),
        name: "self_create_tool".into(),
        arguments: json!({"manifest":manifest()}),
    };
    let denied = PluginTools::new(
        ToolRegistry::open(&path).unwrap(),
        config.clone(),
        policy.clone(),
        Arc::new(Gate(false)),
    )
    .unwrap();
    assert!(denied.preflight(&context(), &call).await.is_err());
    assert!(denied.execute(&context(), &call).await.is_err());
    assert!(
        ToolRegistry::open(&path)
            .unwrap()
            .get(&scope(), "printer")
            .is_err()
    );
    let mut other = context();
    other.scope.user = "other".into();
    assert!(tools.execute(&other, &call).await.is_err());
    let mut invalid = call.clone();
    invalid.arguments["manifest"]["definition"]["parameters"]["$ref"] =
        json!("https://invalid/schema");
    assert!(tools.execute(&context(), &invalid).await.is_err());
    let mut invalid = call.clone();
    invalid.arguments["approve"] = json!(true);
    assert!(tools.execute(&context(), &invalid).await.is_err());
    let mut invalid = call.clone();
    invalid.arguments["manifest"]["definition"]["approved"] = json!(true);
    assert!(tools.execute(&context(), &invalid).await.is_err());
    let mut invalid = call.clone();
    invalid.arguments["manifest"]["command"]["executable"] = json!("/usr/bin/bash");
    assert!(tools.execute(&context(), &invalid).await.is_err());
    let mut invalid = call.clone();
    invalid.arguments["manifest"]["capabilities"]["read_roots"] = json!(["/etc"]);
    assert!(tools.execute(&context(), &invalid).await.is_err());
    let cancelled = context();
    cancelled.cancellation.cancel();
    assert!(tools.execute(&cancelled, &call).await.is_err());
    tools.preflight(&context(), &call).await.unwrap();
    let result = tools.execute(&context(), &call).await.unwrap();
    assert_eq!(result["status"], "pending_operator_review");
    assert_eq!(result["approved"], false);
    assert_eq!(result["enabled"], false);
    assert_eq!(tools.configuration_fingerprint(), fingerprint);
    assert_eq!(tools.definitions().len(), 1);
    let reg = ToolRegistry::open(&path).unwrap();
    let pending = reg.get(&scope(), "printer").unwrap();
    assert_eq!(result["proposal"]["hash"], pending.hash);
    let mut active = config.clone();
    active.allowed_ids = vec!["printer".into()];
    assert!(
        reg.enable(&scope(), "printer", &pending.hash, &active)
            .is_err()
    );
    assert!(
        PluginTools::new(
            ToolRegistry::open(&path).unwrap(),
            active.clone(),
            policy.clone(),
            Arc::new(Gate(true))
        )
        .is_err()
    );
    reg.review(&scope(), "printer", &pending.hash, true)
        .unwrap();
    reg.enable(&scope(), "printer", &pending.hash, &active)
        .unwrap();
    assert_eq!(tools.definitions().len(), 1, "no hotload");
    let source_call = ToolCall {
        id: "original-proposal-id".into(),
        name: "self_create_tool".into(),
        arguments: json!({"rust_source":{"id":"generated","source":"fn main(){println!(\"not executed\");}",
            "definition":{"name":"plugin__generated","description":"generated fixture",
                "parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false},"side_effect":true},
            "args":[],"capabilities":{"read_roots":[]}}}),
    };
    let pending_source = tools.execute(&context(), &source_call).await.unwrap();
    assert_eq!(pending_source["approved"], false);
    assert_eq!(pending_source["proposal"]["reviewed"], false);
    assert!(
        ToolRegistry::open(&path)
            .unwrap()
            .get(&scope(), "generated")
            .is_err()
    );
    assert_eq!(tools.definitions().len(), 1, "Rust proposal never hotloads");
    let next = PluginTools::new(reg, active, policy.clone(), Arc::new(Gate(true))).unwrap();
    assert_eq!(next.definitions().len(), 2);
    let exec = ToolCall {
        id: call.id.clone(),
        name: "plugin__printer".into(),
        arguments: json!({"text":"x"}),
    };
    assert_eq!(
        next.execute(&context(), &exec).await.unwrap()["stdout"],
        "x"
    );
    let mut off = config;
    off.enabled = false;
    let off = PluginTools::new(
        ToolRegistry::open(&path).unwrap(),
        off,
        policy,
        Arc::new(Gate(true)),
    )
    .unwrap();
    assert!(off.definitions().is_empty());
    assert!(off.execute(&context(), &call).await.is_err());
}
