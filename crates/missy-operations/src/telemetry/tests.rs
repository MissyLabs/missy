use super::*;
use async_trait::async_trait;
use axum::{Router, body::Bytes, extract::State, http::StatusCode, routing::post};
use missy_core::*;
use serde_json::{Value, json};
use std::sync::{Mutex, atomic::AtomicUsize};
use tokio_util::sync::CancellationToken;

fn event() -> RuntimeEvent {
    RuntimeEvent {
        trace_id: [1; 16],
        span_id: [2; 8],
        parent_span_id: None,
        kind: RuntimeEventKind::Turn,
        start_time_unix_nano: 1,
        end_time_unix_nano: 2,
        succeeded: true,
    }
}
async fn collector(status: StatusCode) -> (String, Arc<Mutex<Vec<Value>>>, JoinHandle<()>) {
    let received = Arc::new(Mutex::new(vec![]));
    let captured = received.clone();
    let app = Router::new()
        .route(
            "/v1/traces",
            post(
                move |State(values): State<Arc<Mutex<Vec<Value>>>>, bytes: Bytes| async move {
                    values
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&bytes).unwrap());
                    (status, "{}")
                },
            ),
        )
        .with_state(captured);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let worker = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (origin, received, worker)
}
fn enabled(origin: &str) -> (PolicyConfig, TelemetryConfig) {
    (
        PolicyConfig {
            allowed_origins: vec![origin.into()],
            allow_private_network: true,
            ..Default::default()
        },
        TelemetryConfig {
            endpoint: Some(format!("{origin}/v1/traces")),
            ..Default::default()
        },
    )
}
struct Script(AtomicUsize);
#[async_trait]
impl Provider for Script {
    fn name(&self) -> &str {
        "fixture"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        let index = self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Completion {
            message: if index == 0 {
                Message {
                    role: "assistant".into(),
                    content: "RESPONSE_SECRET".into(),
                    tool_calls: vec![ToolCall {
                        id: "CALL_SECRET".into(),
                        name: "private_tool".into(),
                        arguments: json!({"secret":"ARG_SECRET"}),
                    }],
                    tool_call_id: None,
                }
            } else {
                Message::text("assistant", "RESPONSE_SECRET")
            },
            usage: Usage {
                cost_known: true,
                ..Default::default()
            },
            finish_reason: if index == 0 { "tool_calls" } else { "stop" }.into(),
        })
    }
}
struct Tool;
#[async_trait]
impl ToolExecutor for Tool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "private_tool".into(),
            description: "TOOL_SECRET".into(),
            parameters: json!({"type":"object"}),
            side_effect: true,
        }]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        Ok(json!({"secret":"RESULT_SECRET"}))
    }
}

#[tokio::test]
async fn actual_runtime_provider_tool_settlement_reach_http_with_one_private_trace() {
    let (origin, received, server) = collector(StatusCode::OK).await;
    let (policy, config) = enabled(&origin);
    let telemetry = TelemetryRuntime::build_client(policy, config).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let runtime = missy_runtime::Runtime::new_observed(
        Arc::new(missy_store::Store::open(dir.path().join("db")).unwrap()),
        Arc::new(Script(AtomicUsize::new(0))),
        Arc::new(Tool),
        "SYSTEM_SECRET".into(),
        RuntimeConfig::default(),
        telemetry.observer(),
    )
    .unwrap();
    let result = runtime
        .run(
            Scope {
                tenant: "TENANT_SECRET".into(),
                user: "USER_SECRET".into(),
                channel: "CHANNEL_SECRET".into(),
                session: "SESSION_SECRET".into(),
            },
            "PROMPT_SECRET".into(),
            "REQUEST_SECRET".into(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, "completed");
    let metrics = telemetry.shutdown().await.unwrap();
    assert_eq!(metrics.exported_spans, 6); // 2 provider, tool, tool receipt, turn receipt, turn.
    assert_eq!(metrics.pending_spans, 0);
    assert_eq!(metrics.dropped_spans, 0);
    let bodies = received.lock().unwrap();
    let spans: Vec<&Value> = bodies
        .iter()
        .flat_map(|v| {
            v["resourceSpans"][0]["scopeSpans"][0]["spans"]
                .as_array()
                .unwrap()
        })
        .collect();
    assert_eq!(spans.len(), 6);
    let root = spans
        .iter()
        .find(|s| s["attributes"][0]["value"]["stringValue"] == "runtime_turn")
        .unwrap();
    assert!(root.get("parentSpanId").is_none());
    let mut ids = std::collections::HashSet::new();
    for s in &spans {
        assert_eq!(s["traceId"], root["traceId"]);
        assert!(ids.insert(s["spanId"].as_str().unwrap()));
        if s["spanId"] != root["spanId"] {
            assert_eq!(s["parentSpanId"], root["spanId"]);
        }
        assert_eq!(s["status"]["code"], 1);
    }
    let bytes = serde_json::to_string(&*bodies).unwrap();
    assert!(!bytes.contains("SECRET"));
    assert!(!bytes.contains("private_tool"));
    server.abort();
}

#[tokio::test]
async fn callback_never_waits_and_queue_overflow_is_counted() {
    let (origin, _, server) = collector(StatusCode::OK).await;
    let (policy, mut config) = enabled(&origin);
    config.queue_capacity = 2;
    let telemetry = TelemetryRuntime::build_client(policy, config).unwrap();
    let observer = telemetry.observer();
    // Current-thread executor cannot drain until this synchronous loop yields.
    for _ in 0..100 {
        observer.event(event());
    }
    let metrics = telemetry.metrics();
    assert_eq!(metrics.accepted_spans, 2);
    assert_eq!(metrics.pending_spans, 2);
    assert_eq!(metrics.dropped_spans, 98);
    let metrics = telemetry.shutdown().await.unwrap();
    assert_eq!(metrics.exported_spans, 2);
    assert_eq!(metrics.pending_spans, 0);
    server.abort();
}

#[tokio::test]
async fn collector_failure_is_counted_without_retry_or_runtime_failure() {
    let (origin, received, server) = collector(StatusCode::INTERNAL_SERVER_ERROR).await;
    let (policy, config) = enabled(&origin);
    let telemetry = TelemetryRuntime::build_client(policy, config).unwrap();
    telemetry.observer().event(event());
    let metrics = telemetry.shutdown().await.unwrap();
    assert_eq!(metrics.failed_spans, 1);
    assert_eq!(metrics.failed_batches, 1);
    assert_eq!(metrics.exported_spans, 0);
    assert_eq!(metrics.pending_spans, 0);
    assert_eq!(received.lock().unwrap().len(), 1);
    server.abort();
}

#[tokio::test]
async fn dropped_owner_does_not_shutdown_live_observer_and_last_drop_flushes() {
    let (origin, received, server) = collector(StatusCode::OK).await;
    let (policy, config) = enabled(&origin);
    let telemetry = TelemetryRuntime::build_client(policy, config).unwrap();
    let observer = telemetry.observer();
    let counters = telemetry.observer.counters.clone();
    drop(telemetry);
    observer.event(event());
    drop(observer);
    tokio::time::timeout(Duration::from_secs(2), async {
        while counters.exported.load(Ordering::Relaxed) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(received.lock().unwrap().len(), 1);
    assert_eq!(counters.pending.load(Ordering::Relaxed), 0);
    server.abort();
}

#[tokio::test]
async fn policy_denial_isolated_and_disabled_is_noop() {
    let config = TelemetryConfig {
        endpoint: Some("http://127.0.0.1:1/v1/traces".into()),
        ..Default::default()
    };
    let telemetry = TelemetryRuntime::build_client(PolicyConfig::default(), config).unwrap();
    telemetry.observer().event(event());
    assert_eq!(telemetry.shutdown().await.unwrap().failed_spans, 1);
    let disabled =
        TelemetryRuntime::build_client(PolicyConfig::default(), TelemetryConfig::default())
            .unwrap();
    disabled.observer().event(event());
    assert_eq!(disabled.shutdown().await.unwrap().accepted_spans, 0);
}

#[test]
fn configuration_is_explicit_and_strict() {
    assert!(TelemetryConfig::default().endpoint.is_none());
    assert!(serde_json::from_value::<TelemetryConfig>(json!({"prompts":true})).is_err());
    let config = TelemetryConfig {
        queue_capacity: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn conservative_encoded_bound_fits_the_largest_permitted_batch() {
    let config = TelemetryConfig {
        max_batch: 1024,
        max_bytes: 512 + 1024 * 768,
        ..Default::default()
    };
    config.validate().unwrap();
    let exporter = OtlpExporter::new(
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        "https://collector.invalid/v1/traces".into(),
        config.max_batch,
        config.max_bytes,
    )
    .unwrap();
    let mut event = event();
    event.parent_span_id = Some([3; 8]);
    event.start_time_unix_nano = u64::MAX - 1;
    event.end_time_unix_nano = u64::MAX;
    event.kind = RuntimeEventKind::Settlement;
    assert!(exporter.payload(&vec![span(event); 1024]).unwrap().len() <= config.max_bytes);
}

#[tokio::test]
async fn stalled_collector_shutdown_has_a_hard_deadline_and_counts_abandoned_spans() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/v1/traces",
        post(|| async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            "{}"
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let (policy, mut config) = enabled(&origin);
    config.shutdown_timeout_ms = 30;
    let telemetry = TelemetryRuntime::build_client(policy, config).unwrap();
    let monitor = telemetry.monitor();
    telemetry.observer().event(event());
    let start = tokio::time::Instant::now();
    assert!(telemetry.shutdown().await.is_err());
    assert!(start.elapsed() < Duration::from_secs(1));
    let metrics = monitor.metrics();
    assert_eq!(metrics.pending_spans, 0);
    assert_eq!(metrics.dropped_spans, 1);
    server.abort();
}
