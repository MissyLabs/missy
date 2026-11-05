//! Owned tool/security API acceptance against the real durable runtime.
use async_trait::async_trait;
use missy_core::{
    Completion, Message, PolicyConfig, Provider, RuntimeConfig, Scope, ToolCall, ToolDefinition,
};
use missy_runtime::Runtime;
use missy_store::Store;
use missy_tools::BuiltinTools;
use serde_json::json;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct NoProvider;
#[async_trait]
impl Provider for NoProvider {
    fn name(&self) -> &str {
        "disposable-no-provider"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> anyhow::Result<Completion> {
        anyhow::bail!("exact-tool integration must never contact a provider")
    }
}
fn scope(user: &str) -> Scope {
    Scope {
        tenant: "disposable".into(),
        user: user.into(),
        channel: "test".into(),
        session: "test".into(),
    }
}
#[tokio::test]
async fn native_document_runtime_receipt_projection_scope_and_reopen_are_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.html");
    let synthetic_credential = format!("ghp_{}", "a".repeat(36));
    let original = format!(
        "<h1>Readable content</h1><script>raw-only-canary</script><p>{synthetic_credential}</p>"
    );
    std::fs::write(&source, &original).unwrap();
    let config = RuntimeConfig::default();
    let store = Arc::new(Store::open(dir.path().join("runtime.sqlite")).unwrap());
    let tools = Arc::new(
        BuiltinTools::new(
            PolicyConfig {
                read_roots: vec![dir.path().into()],
                max_file_bytes: 1024 * 1024,
                ..Default::default()
            },
            config.clone(),
            store.clone(),
        )
        .unwrap(),
    );
    let rt = Runtime::new(
        store.clone(),
        Arc::new(NoProvider),
        tools.clone(),
        "trusted immutable template".into(),
        config.clone(),
    )
    .unwrap();
    let calls = vec![ToolCall {
        id: "document".into(),
        name: "read_document".into(),
        arguments: json!({"path":source,"format":"html"}),
    }];
    let result = rt
        .run_tools(scope("a"), calls.clone(), "document-request".into())
        .await
        .unwrap();
    assert_eq!(result.status, "completed");
    assert!(!result.content.contains(&synthetic_credential));
    assert!(!result.content.contains("raw-only-canary"));
    assert!(!result.content.contains("raw_evidence"));
    assert!(result.content.contains("Readable content"));
    let evidence = store
        .tool_evidence(&scope("a"), &result.turn_id, "document")
        .unwrap()
        .unwrap();
    assert_eq!(evidence["data"]["raw_evidence"]["source_utf8"], original);
    assert!(
        store
            .tool_evidence(&scope("b"), &result.turn_id, "document")
            .is_err()
    );
    let reopened = Runtime::new(
        Arc::new(Store::open(dir.path().join("runtime.sqlite")).unwrap()),
        Arc::new(NoProvider),
        tools,
        "trusted immutable template".into(),
        config,
    )
    .unwrap();
    let replay = reopened
        .run_tools(scope("a"), calls, "document-request".into())
        .await
        .unwrap();
    assert!(replay.deduplicated);
    assert_eq!(replay.content, result.content);
    assert_eq!(
        reopened
            .store()
            .tool_evidence(&scope("a"), &result.turn_id, "document")
            .unwrap()
            .unwrap(),
        evidence
    );
}

#[tokio::test]
async fn actual_native_catalog_runtime_schema_rejects_wrong_types_before_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(dir.path().join("schema.sqlite")).unwrap());
    let config = RuntimeConfig::default();
    let tools = Arc::new(
        BuiltinTools::new(PolicyConfig::default(), config.clone(), store.clone()).unwrap(),
    );
    let rt = Runtime::new(
        store,
        Arc::new(NoProvider),
        tools,
        "trusted schema template".into(),
        config,
    )
    .unwrap();
    for (id, args) in [
        (
            "unknown",
            json!({"query":"x","ungranted_url":"https://invalid.example"}),
        ),
        ("type", json!({"query":"x","limit":"5"})),
        ("range", json!({"query":"x","limit":11})),
    ] {
        let result = rt
            .run_tools(
                scope("a"),
                vec![ToolCall {
                    id: id.into(),
                    name: "web_search".into(),
                    arguments: args,
                }],
                format!("schema-{id}"),
            )
            .await;
        assert!(result.is_err() || result.as_ref().is_ok_and(|r| r.status != "completed"));
    }
}
