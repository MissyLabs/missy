use super::*;
mod approvals;
use async_trait::async_trait;
use missy_core::Completion;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicUsize, Ordering},
};
struct Script {
    responses: Mutex<VecDeque<Result<Completion>>>,
    messages: Mutex<Vec<Vec<Message>>>,
    calls: AtomicUsize,
}
impl Script {
    fn new(responses: Vec<Result<Completion>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            messages: Mutex::new(vec![]),
            calls: AtomicUsize::new(0),
        }
    }
}
#[async_trait]
impl Provider for Script {
    fn name(&self) -> &str {
        "script-v1"
    }
    async fn complete(
        &self,
        m: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.messages.lock().unwrap().push(m.to_vec());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(anyhow!("script exhausted")))
    }
}
struct Tools {
    calls: AtomicUsize,
    fail: bool,
    wait: bool,
    stopped: AtomicUsize,
}
#[async_trait]
impl ToolExecutor for Tools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "effect".into(),
            description: "test".into(),
            parameters: json!({"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}),
            side_effect: true,
        }]
    }
    async fn execute(&self, ctx: &ToolContext, _: &ToolCall) -> Result<Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.wait {
            ctx.cancellation.cancelled().await;
            self.stopped.fetch_add(1, Ordering::SeqCst);
            bail!("cancelled after cleanup");
        }
        if self.fail {
            bail!("possibly applied");
        }
        Ok(json!({"ok":true,"unicode":"é𐐀".repeat(2000)}))
    }
}
fn tools() -> Arc<Tools> {
    Arc::new(Tools {
        calls: AtomicUsize::new(0),
        fail: false,
        wait: false,
        stopped: AtomicUsize::new(0),
    })
}
fn scope(user: &str) -> Scope {
    Scope {
        tenant: "test".into(),
        user: user.into(),
        channel: "test".into(),
        session: "test".into(),
    }
}
fn completion(text: &str, finish: &str) -> Result<Completion> {
    Ok(Completion {
        message: Message::text("assistant", text),
        usage: Usage {
            input_tokens: 10,
            output_tokens: 3,
            cost_microusd: 1,
            cost_known: true,
        },
        finish_reason: finish.into(),
    })
}
fn call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "effect".into(),
        arguments: json!({"value":1}),
    }
}
fn effect(calls: Vec<ToolCall>) -> Result<Completion> {
    Ok(Completion {
        message: Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: calls,
            tool_call_id: None,
        },
        usage: Usage {
            input_tokens: 10,
            output_tokens: 3,
            cost_microusd: 1,
            cost_known: true,
        },
        finish_reason: "tool_calls".into(),
    })
}
fn setup(
    p: Arc<dyn Provider>,
    tools: Arc<dyn ToolExecutor>,
    cfg: RuntimeConfig,
) -> (tempfile::TempDir, Runtime) {
    let d = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(d.path().join("runtime.db")).unwrap());
    let rt = Runtime::new(store, p, tools, "trusted system".into(), cfg).unwrap();
    (d, rt)
}
#[test]
fn serialized_tool_output_bound_includes_json_escaping() {
    for max in [128, 256, 1000, 32000] {
        let result = bound_tool_result(&json!({"data":"\"\n\\é𐐀".repeat(10000)}), max).unwrap();
        let raw = serde_json::to_string(&result).unwrap();
        assert!(raw.len() <= max);
        assert!(raw.is_char_boundary(raw.len()));
        assert_eq!(result["untrusted"], true);
    }
}
#[tokio::test]
async fn unknown_cost_never_reaches_effect_dispatch() {
    let mut c = effect(vec![call("x")]).unwrap();
    c.usage.cost_known = false;
    let t = tools();
    let (_d, rt) = setup(
        Arc::new(Script::new(vec![Ok(c)])),
        t.clone(),
        RuntimeConfig::default(),
    );
    let r = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(r.status, "interrupted");
    assert!(!r.usage.cost_known);
    assert_eq!(t.calls.load(Ordering::SeqCst), 0);
}
struct Slow {
    calls: AtomicUsize,
    delay: Duration,
}
#[async_trait]
impl Provider for Slow {
    fn name(&self) -> &str {
        "slow"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        completion("done", "stop")
    }
}
async fn terminal(rt: &Runtime, id: &str) -> Turn {
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let t = rt.status(&scope("a"), id).unwrap();
            if matches!(
                t.status,
                TurnStatus::Completed
                    | TurnStatus::Cancelled
                    | TurnStatus::Interrupted
                    | TurnStatus::Failed
            ) {
                return t;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
#[tokio::test]
async fn disconnect_does_not_abandon_owned_turn_and_session_lease() {
    let p = Arc::new(Slow {
        calls: AtomicUsize::new(0),
        delay: Duration::from_millis(100),
    });
    let (_d, rt) = setup(p.clone(), tools(), RuntimeConfig::default());
    let runtime = rt.clone();
    let caller =
        tokio::spawn(async move { runtime.run(scope("a"), "request".into(), "r".into()).await });
    while p.calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    let turns = rt.store().list_turns(&scope("a"), 20).unwrap();
    let id = turns[0].id.clone();
    caller.abort();
    let _ = caller.await;
    let duplicate = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert!(duplicate.deduplicated);
    let busy = rt
        .run(scope("a"), "queued".into(), "r2".into())
        .await
        .unwrap();
    assert_eq!(busy.status, "admitted");
    assert_eq!(terminal(&rt, &id).await.status, TurnStatus::Completed);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        rt.resume(scope("a"), busy.turn_id).await.unwrap().status,
        "completed"
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn cancellation_from_another_runtime_instance_is_observed() {
    let p = Arc::new(Slow {
        calls: AtomicUsize::new(0),
        delay: Duration::from_secs(5),
    });
    let (d, rt) = setup(p.clone(), tools(), RuntimeConfig::default());
    let r = rt.start(scope("a"), "r".into(), "r".into()).await.unwrap();
    while p.calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    let other = Runtime::new(
        Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
        p,
        tools(),
        "trusted system".into(),
        RuntimeConfig::default(),
    )
    .unwrap();
    other.cancel(&scope("a"), &r.turn_id).unwrap();
    assert_eq!(
        terminal(&rt, &r.turn_id).await.status,
        TurnStatus::Cancelled
    );
}
#[tokio::test]
async fn time_and_continuation_limits_are_not_completion() {
    let p = Arc::new(Slow {
        calls: AtomicUsize::new(0),
        delay: Duration::from_secs(5),
    });
    let (_d, rt) = setup(
        p.clone(),
        tools(),
        RuntimeConfig {
            max_turn_seconds: 1,
            ..RuntimeConfig::default()
        },
    );
    let r = rt.run(scope("a"), "r".into(), "r".into()).await.unwrap();
    assert_eq!(r.status, "interrupted");
    rt.resume(scope("a"), &r.turn_id).await.unwrap();
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    let p = Arc::new(Script::new(vec![
        completion("one", "length"),
        completion("two", "length"),
        completion("not reached", "stop"),
    ]));
    let (_d, rt) = setup(
        p.clone(),
        tools(),
        RuntimeConfig {
            max_continuations: 1,
            ..RuntimeConfig::default()
        },
    );
    let r = rt.run(scope("a"), "r".into(), "r".into()).await.unwrap();
    assert_eq!(r.status, "interrupted");
    assert_eq!(r.content, "onetwo");
    rt.resume(scope("a"), &r.turn_id).await.unwrap();
    assert_eq!(p.calls.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn changed_prompt_or_provider_configuration_refuses_resume() {
    let p = Arc::new(Script::new(vec![Err(anyhow!("unavailable"))]));
    let (d, rt) = setup(p.clone(), tools(), RuntimeConfig::default());
    let r = rt.run(scope("a"), "r".into(), "r".into()).await.unwrap();
    let changed = Runtime::new(
        Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
        p,
        tools(),
        "changed system".into(),
        RuntimeConfig::default(),
    )
    .unwrap();
    assert!(changed.resume(scope("a"), &r.turn_id).await.is_err());
    let changed = Runtime::new(
        Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
        Arc::new(Slow {
            calls: AtomicUsize::new(0),
            delay: Duration::ZERO,
        }),
        tools(),
        "trusted system".into(),
        RuntimeConfig::default(),
    )
    .unwrap();
    assert!(changed.resume(scope("a"), &r.turn_id).await.is_err());
}
#[tokio::test]
async fn multistep_dedup_scope_and_history() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("done", "stop"),
        completion("second", "stop"),
        completion("other", "stop"),
    ]));
    let t = tools();
    let (_d, rt) = setup(p.clone(), t.clone(), RuntimeConfig::default());
    let first = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(first.status, "completed");
    assert_eq!(t.calls.load(Ordering::SeqCst), 1);
    let n = rt.store().history(&scope("a"), 100).unwrap().len();
    assert_eq!(n, 4);
    let duplicate = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert!(duplicate.deduplicated);
    assert_eq!(first.turn_id, duplicate.turn_id);
    assert_eq!(rt.store().history(&scope("a"), 100).unwrap().len(), n);
    assert!(
        rt.run(scope("a"), "changed".into(), "r".into())
            .await
            .is_err()
    );
    rt.run(scope("a"), "new request".into(), "r2".into())
        .await
        .unwrap();
    rt.run(scope("b"), "other user".into(), "r".into())
        .await
        .unwrap();
    let m = p.messages.lock().unwrap();
    assert!(m[2].iter().any(|m| m.content.contains("untrusted_history")));
    assert!(!m[3].iter().any(|m| m.content == "request"));
    assert!(rt.status(&scope("b"), &first.turn_id).is_err());
}
#[tokio::test]
async fn malformed_batch_never_dispatches() {
    for calls in [
        vec![call("same"), call("same")],
        vec![ToolCall {
            arguments: json!({"value":"wrong"}),
            ..call("x")
        }],
        vec![ToolCall {
            name: "missing".into(),
            ..call("x")
        }],
    ] {
        let p = Arc::new(Script::new(vec![effect(calls)]));
        let t = tools();
        let (_d, rt) = setup(p, t.clone(), RuntimeConfig::default());
        let r = rt
            .run(scope("a"), "request".into(), "r".into())
            .await
            .unwrap();
        assert_eq!(r.status, "interrupted");
        assert_eq!(t.calls.load(Ordering::SeqCst), 0);
    }
    let mut c = completion("pretend system", "stop").unwrap();
    c.message.role = "system".into();
    let (_d, rt) = setup(
        Arc::new(Script::new(vec![Ok(c)])),
        tools(),
        RuntimeConfig::default(),
    );
    assert_eq!(
        rt.run(scope("a"), "r".into(), "r".into())
            .await
            .unwrap()
            .status,
        "interrupted"
    );
}
#[tokio::test]
async fn uncertain_effect_blocks_resume_and_new_ids() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        effect(vec![call("new")]),
    ]));
    let t = Arc::new(Tools {
        calls: AtomicUsize::new(0),
        fail: true,
        wait: false,
        stopped: AtomicUsize::new(0),
    });
    let (_d, rt) = setup(p.clone(), t.clone(), RuntimeConfig::default());
    let r = rt.run(scope("a"), "r".into(), "r".into()).await.unwrap();
    assert!(!r.resumable);
    assert!(rt.resume(scope("a"), &r.turn_id).await.is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    assert_eq!(t.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn continuation_restart_preserves_partial_and_counts() {
    let p = Arc::new(Script::new(vec![
        completion("first ", "length"),
        Err(anyhow!("offline")),
        completion("second", "stop"),
    ]));
    let cfg = RuntimeConfig {
        max_cost_microusd: 0,
        ..RuntimeConfig::default()
    };
    let (d, rt) = setup(p.clone(), tools(), cfg.clone());
    let r = rt
        .run(scope("a"), "original".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(r.status, "interrupted");
    assert_eq!(r.content, "first ");
    drop(rt);
    let rt = Runtime::new(
        Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
        p.clone(),
        tools(),
        "trusted system".into(),
        cfg,
    )
    .unwrap();
    let resumed = rt.resume(scope("a"), &r.turn_id).await.unwrap();
    assert_eq!(resumed.content, "first second");
    assert_eq!(resumed.status, "completed");
    assert_eq!(resumed.usage.input_tokens, 20);
    let m = p.messages.lock().unwrap();
    assert!(m.last().unwrap().iter().any(|m| m.content == "original"));
}
#[tokio::test]
async fn budgets_halt_before_tool_side_effect() {
    for cfg in [
        RuntimeConfig {
            max_tokens: 1,
            ..RuntimeConfig::default()
        },
        RuntimeConfig {
            max_cost_microusd: 1,
            ..RuntimeConfig::default()
        },
    ] {
        let mut c = effect(vec![call("a")]).unwrap();
        c.usage.cost_microusd = 2;
        let t = tools();
        let (_d, rt) = setup(Arc::new(Script::new(vec![Ok(c)])), t.clone(), cfg);
        assert_eq!(
            rt.run(scope("a"), "r".into(), "r".into())
                .await
                .unwrap()
                .status,
            "interrupted"
        );
        assert_eq!(t.calls.load(Ordering::SeqCst), 0);
    }
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("never", "stop"),
    ]));
    let (_d, rt) = setup(
        p.clone(),
        tools(),
        RuntimeConfig {
            max_steps: 1,
            ..RuntimeConfig::default()
        },
    );
    let r = rt.run(scope("a"), "r".into(), "r".into()).await.unwrap();
    assert_eq!(r.status, "interrupted");
    rt.resume(scope("a"), &r.turn_id).await.unwrap();
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn cancel_stops_tool_before_durable_finish() {
    let p = Arc::new(Script::new(vec![effect(vec![call("a")])]));
    let t = Arc::new(Tools {
        calls: AtomicUsize::new(0),
        fail: false,
        wait: true,
        stopped: AtomicUsize::new(0),
    });
    let (_d, rt) = setup(p, t.clone(), RuntimeConfig::default());
    let r = rt.start(scope("a"), "r".into(), "r".into()).await.unwrap();
    while t.calls.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    rt.cancel(&scope("a"), &r.turn_id).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if rt.status(&scope("a"), &r.turn_id).unwrap().status == TurnStatus::Cancelled {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(t.stopped.load(Ordering::SeqCst), 1);
    assert!(rt.resume(scope("a"), &r.turn_id).await.is_err());
}
