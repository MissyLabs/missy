use super::*;
use anyhow::Result;
use async_trait::async_trait;
use missy_core::*;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub(crate) struct MockProvider;
#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        Ok(Completion {
            message: Message::text("assistant", "private model response"),
            usage: Usage {
                input_tokens: 11,
                output_tokens: 7,
                cost_microusd: 5,
                cost_known: true,
            },
            finish_reason: "stop".into(),
        })
    }
}
struct NoTools;
#[async_trait]
impl ToolExecutor for NoTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        anyhow::bail!("no tools")
    }
}
pub(crate) fn runtime(dir: &std::path::Path) -> SharedRuntime {
    Arc::new(
        missy_runtime::Runtime::new(
            Arc::new(missy_store::Store::open(dir.join("store.sqlite3")).unwrap()),
            Arc::new(MockProvider),
            Arc::new(NoTools),
            "test system".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    )
}
pub(crate) fn scope() -> Scope {
    Scope {
        tenant: "tenant".into(),
        user: "trusted".into(),
        channel: "local".into(),
        session: "s".into(),
    }
}

#[tokio::test]
async fn diagnostic_usage_and_redaction_are_real_and_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path());
    runtime
        .run(scope(), "PROMPT_SECRET".into(), "r1".into())
        .await
        .unwrap();
    let mut other = scope();
    other.user = "other".into();
    runtime
        .run(other, "OTHER_SECRET".into(), "r2".into())
        .await
        .unwrap();
    let report = snapshot(runtime.store(), &scope()).unwrap();
    assert_eq!(report.sampled_turns, 1);
    assert_eq!(report.recorded_usage.input_tokens, 11);
    assert_eq!(report.recorded_usage.cost_microusd, 5);
    assert!(report.recorded_usage.cost_known);
    let serialized = serde_json::to_string(&report).unwrap();
    for secret in [
        "PROMPT_SECRET",
        "OTHER_SECRET",
        "private model response",
        "trusted",
        "tenant",
        "test system",
    ] {
        assert!(!serialized.contains(secret));
        assert!(!report.prometheus().contains(secret));
    }
    assert_eq!(
        report.subsystem_observations["provider"],
        LifecycleState::Unknown
    );
}
#[test]
fn lifecycle_does_not_claim_restart_without_starting() {
    let registry = LifecycleRegistry::default();
    assert!(registry.observations().unwrap().is_empty());
    registry
        .record(Subsystem::Gateway, LifecycleState::Stopped)
        .unwrap();
    assert!(
        registry
            .record(Subsystem::Gateway, LifecycleState::Ready)
            .is_err()
    );
    registry
        .record(Subsystem::Gateway, LifecycleState::Starting)
        .unwrap();
    registry
        .record(Subsystem::Gateway, LifecycleState::Ready)
        .unwrap();
}
#[test]
fn collector_payload_is_otlp_and_bounded() {
    let client = missy_policy::PolicyHttpClient::new(PolicyConfig::default()).unwrap();
    let exporter = OtlpExporter::new(
        client,
        "https://collector.invalid/v1/traces".into(),
        1,
        4096,
    )
    .unwrap();
    let span = OperationalSpan::new(OperationalKind::WebhookAccepted, 10, 20, true).unwrap();
    let body: Value =
        serde_json::from_slice(&exporter.payload(std::slice::from_ref(&span)).unwrap()).unwrap();
    let out = &body["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
    assert_eq!(out["traceId"].as_str().unwrap().len(), 32);
    assert_eq!(out["spanId"].as_str().unwrap().len(), 16);
    assert_eq!(out["startTimeUnixNano"], "10");
    assert_eq!(out["attributes"].as_array().unwrap().len(), 2);
    assert!(exporter.payload(&[]).is_err());
    assert!(exporter.payload(&[span.clone(), span.clone()]).is_err());
    let mut invalid = span;
    invalid.trace_id = "0".repeat(32);
    assert!(exporter.payload(&[invalid]).is_err());
}
#[tokio::test]
async fn collector_cannot_bypass_network_policy() {
    let http = missy_policy::PolicyHttpClient::new(PolicyConfig::default()).unwrap();
    let exporter = OtlpExporter::new(http, "http://127.0.0.1:1/v1/traces".into(), 1, 4096).unwrap();
    assert!(
        exporter
            .export(&[OperationalSpan::new(OperationalKind::ExportSucceeded, 1, 2, true).unwrap()])
            .await
            .is_err()
    );
}
#[tokio::test]
async fn workflow_is_literal_bounded_and_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path());
    let workflow = Workflow {
        id: "workflow".into(),
        nodes: vec![
            WorkflowNode {
                id: "a".into(),
                dependencies: vec![],
                prompt: "literal ${a.result}".into(),
            },
            WorkflowNode {
                id: "b".into(),
                dependencies: vec!["a".into()],
                prompt: "next".into(),
            },
        ],
    };
    let result = workflow
        .execute(runtime.clone(), scope(), WorkflowBounds::default())
        .await
        .unwrap();
    assert_eq!(result.turns.len(), 2);
    assert_eq!(result.recorded_usage.cost_microusd, 10);
    let repeated = workflow
        .execute(runtime.clone(), scope(), WorkflowBounds::default())
        .await
        .unwrap();
    assert_eq!(result.turns["a"].turn_id, repeated.turns["a"].turn_id);
    assert_eq!(runtime.store().list_turns(&scope(), 100).unwrap().len(), 2);
    assert!(
        Workflow::parse(
            &serde_json::to_vec(&json!({"id":"x","scope":scope(),"nodes":[]})).unwrap()
        )
        .is_err()
    );
    let mut cyclic = workflow;
    cyclic.nodes[0].dependencies = vec!["b".into()];
    assert!(cyclic.ordered_nodes(&WorkflowBounds::default()).is_err());
}
#[tokio::test]
async fn workflow_stops_before_new_node_when_cumulative_budget_used() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path());
    let workflow = Workflow {
        id: "budget".into(),
        nodes: (0..3)
            .map(|i| WorkflowNode {
                id: format!("n{i}"),
                dependencies: vec![],
                prompt: "prompt".into(),
            })
            .collect(),
    };
    let bounds = WorkflowBounds {
        max_recorded_cost_microusd: 1,
        ..Default::default()
    };
    assert!(
        workflow
            .execute(runtime.clone(), scope(), bounds)
            .await
            .is_err()
    );
    assert_eq!(runtime.store().list_turns(&scope(), 100).unwrap().len(), 1);
}
