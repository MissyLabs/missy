use crate::{Runtime, RuntimeEvent, RuntimeEventKind, RuntimeObserver};
use anyhow::Result;
use async_trait::async_trait;
use missy_core::*;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;

struct ProviderFixture(AtomicUsize);
#[async_trait]
impl Provider for ProviderFixture {
    fn name(&self) -> &str {
        "fixture"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        let first = self.0.fetch_add(1, Ordering::Relaxed) == 0;
        Ok(Completion {
            message: if first {
                Message {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "call".into(),
                        name: "effect".into(),
                        arguments: json!({}),
                    }],
                    tool_call_id: None,
                }
            } else {
                Message::text("assistant", "done")
            },
            usage: Usage {
                cost_known: true,
                ..Default::default()
            },
            finish_reason: if first { "tool_calls" } else { "stop" }.into(),
        })
    }
}
struct FailingTool;
#[async_trait]
impl ToolExecutor for FailingTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "effect".into(),
            description: "fixture".into(),
            parameters: json!({"type":"object"}),
            side_effect: true,
        }]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        anyhow::bail!("uncertain external effect")
    }
}
#[derive(Default)]
struct Capture(Mutex<Vec<RuntimeEvent>>);
impl RuntimeObserver for Capture {
    fn event(&self, event: RuntimeEvent) {
        self.0.lock().unwrap().push(event);
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

#[tokio::test]
async fn unknown_effect_reports_failed_tool_and_successful_durable_receipts() {
    let dir = tempfile::tempdir().unwrap();
    let observer = Arc::new(Capture::default());
    let runtime = Runtime::new_observed(
        Arc::new(missy_store::Store::open(dir.path().join("db")).unwrap()),
        Arc::new(ProviderFixture(AtomicUsize::new(0))),
        Arc::new(FailingTool),
        "system".into(),
        RuntimeConfig::default(),
        observer.clone(),
    )
    .unwrap();
    assert_eq!(
        runtime
            .run(scope(), "request".into(), "id".into())
            .await
            .unwrap()
            .status,
        "interrupted"
    );
    let events = observer.0.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == RuntimeEventKind::Tool && !e.succeeded)
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == RuntimeEventKind::Settlement && e.succeeded)
            .count(),
        2
    );
    let root = events
        .iter()
        .find(|e| e.kind == RuntimeEventKind::Turn)
        .unwrap();
    assert!(!root.succeeded);
    assert_eq!(root.parent_span_id, None);
    for event in events.iter() {
        assert_eq!(event.trace_id, root.trace_id);
        assert!(event.end_time_unix_nano >= event.start_time_unix_nano);
    }
}

struct Panics;
impl RuntimeObserver for Panics {
    fn event(&self, _: RuntimeEvent) {
        panic!("broken instrumentation");
    }
}
#[tokio::test]
async fn observer_panic_cannot_interrupt_execution_or_receipt_settlement() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = Runtime::new_observed(
        Arc::new(missy_store::Store::open(dir.path().join("db")).unwrap()),
        Arc::new(ProviderFixture(AtomicUsize::new(1))),
        Arc::new(FailingTool),
        "system".into(),
        RuntimeConfig::default(),
        Arc::new(Panics),
    )
    .unwrap();
    let result = runtime
        .run(scope(), "request".into(), "id".into())
        .await
        .unwrap();
    assert_eq!(result.status, "completed");
    assert_eq!(
        runtime.status(&scope(), &result.turn_id).unwrap().status,
        missy_store::TurnStatus::Completed
    );
}
