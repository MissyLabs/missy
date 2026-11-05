use super::*;
#[path = "retention_tests.rs"]
mod retention_tests;
use missy_core::{Completion, Message, Provider, Usage};
use missy_store::{ToolStatus, TurnStatus};
use std::{
    os::unix::fs::PermissionsExt,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
fn scope() -> Scope {
    Scope {
        tenant: "tenant".into(),
        user: "user".into(),
        channel: "channel".into(),
        session: "session".into(),
    }
}
fn context(scope: Scope) -> ToolContext {
    ToolContext {
        scope,
        turn_id: "parent".into(),
        cancellation: CancellationToken::new(),
    }
}
fn call(name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: "call".into(),
        name: name.into(),
        arguments: args,
    }
}
fn setup() -> (tempfile::TempDir, Arc<Store>, HistoryTools, ToolContext) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Arc::new(Store::open(dir.path().join("history.sqlite")).unwrap());
    let tools = HistoryTools::new(
        store.clone(),
        HistoryConfig {
            enabled: true,
            retention_path: Some(dir.path().join("evidence.sqlite3")),
            ..Default::default()
        },
    )
    .unwrap();
    (dir, store, tools, context(scope()))
}
fn turn(store: &Store, scope: &Scope, request: &str, prompt: &str, response: &str) -> String {
    let t = store.admit(scope, request, prompt).unwrap().turn;
    let lease = store
        .acquire_lease(scope, &t.id, "fixture", Duration::from_secs(60))
        .unwrap();
    store
        .finish(scope, &lease, TurnStatus::Completed, Some(response), None)
        .unwrap();
    t.id
}
async fn search(tools: &HistoryTools, ctx: &ToolContext, query: &str) -> Value {
    tools
        .execute(ctx, &call("history_search", json!({"query":query})))
        .await
        .unwrap()
}
#[tokio::test]
async fn disabled_default_and_strict_config_arguments() {
    let (_d, s, _, ctx) = setup();
    let off = HistoryTools::new(s, HistoryConfig::default()).unwrap();
    assert!(off.definitions().is_empty());
    assert!(
        off.execute(&ctx, &call("history_search", json!({"query":"x"})))
            .await
            .is_err()
    );
    assert!(serde_json::from_value::<HistoryConfig>(json!({"scope":scope()})).is_err());
    let (_, _, on, ctx) = setup();
    assert!(
        on.execute(
            &ctx,
            &call("history_search", json!({"query":"x","scope":scope()}))
        )
        .await
        .is_err()
    );
    assert!(
        !on.definitions()
            .iter()
            .any(|d| d.name == "memory_search" || d.name == "context_shunt")
    );
    assert!(on.definitions().iter().all(|d| !d.side_effect));
}
#[tokio::test]
async fn literal_history_stable_receipts_describe_and_all_scope_dimensions() {
    let (_d, s, t, ctx) = setup();
    let id = turn(&s, &ctx.scope, "a", "database outage needle", "fixed");
    let first = search(&t, &ctx, "NEEDLE").await;
    assert_eq!(first["results"][0]["turn_id"], id);
    assert_eq!(
        first["results"][0]["item_id"],
        search(&t, &ctx, "needle").await["results"][0]["item_id"]
    );
    let item = first["results"][0]["item_id"].as_str().unwrap();
    let desc = call("memory_describe", json!({"item_ids":[item]}));
    assert_eq!(t.execute(&ctx, &desc).await.unwrap()["source_count"], 1);
    for dimension in 0..4 {
        let mut other = ctx.scope.clone();
        match dimension {
            0 => other.tenant.push('2'),
            1 => other.user.push('2'),
            2 => other.channel.push('2'),
            _ => other.session.push('2'),
        };
        let other = context(other);
        assert_eq!(search(&t, &other, "needle").await["results"], json!([]));
        assert!(t.execute(&other, &desc).await.is_err());
    }
    for literal in ["' OR 1=1 --", ".*", "needle OR outage"] {
        assert_eq!(search(&t, &ctx, literal).await["results"], json!([]));
    }
    assert!(
        t.execute(
            &ctx,
            &call("memory_describe", json!({"item_ids":["hist_forged"]}))
        )
        .await
        .is_err()
    );
}
#[tokio::test]
async fn expansion_exact_utf8_checksum_turn_and_tool_binding() {
    let (_d, s, t, ctx) = setup();
    let id = turn(&s, &ctx.scope, "a", "aé水needle end", "");
    let found = search(&t, &ctx, "needle").await;
    let row = &found["results"][0];
    let args = json!({"item_id":row["item_id"],"content_sha256":row["content_sha256"],"turn_id":id,"start_byte":1,"end_byte":6});
    assert_eq!(
        t.execute(&ctx, &call("memory_expand", args.clone()))
            .await
            .unwrap()["text"],
        "é水"
    );
    for (key, value) in [
        ("start_byte", json!(2)),
        ("end_byte", json!(4)),
        ("content_sha256", json!("forged")),
        ("turn_id", json!("forged")),
        ("tool_id", json!("forged")),
    ] {
        let mut bad = args.clone();
        bad[key] = value;
        assert!(
            t.execute(&ctx, &call("memory_expand", bad)).await.is_err(),
            "{key}"
        );
    }
    let mut other = ctx.clone();
    other.scope.session.push('2');
    assert!(
        t.execute(&other, &call("memory_expand", args))
            .await
            .is_err()
    );
}
#[tokio::test]
async fn tool_outputs_retained_without_reexecution_and_missing_is_not_null() {
    let (_d, s, t, ctx) = setup();
    let turn = s.admit(&ctx.scope, "tools", "prompt").unwrap().turn;
    let lease = s
        .acquire_lease(&ctx.scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    let tool = call("expensive_real_tool", json!({}));
    s.tool_begin(&ctx.scope, &lease, &tool).unwrap();
    s.tool_running(&ctx.scope, &lease, &tool.id).unwrap();
    s.tool_settle(
        &ctx.scope,
        &lease,
        &tool.id,
        ToolStatus::Succeeded,
        Some(&json!({"result":"needle 水"})),
        None,
    )
    .unwrap();
    let missing = ToolCall {
        id: "missing".into(),
        ..call("missing_tool", json!({}))
    };
    s.tool_begin(&ctx.scope, &lease, &missing).unwrap();
    s.tool_running(&ctx.scope, &lease, &missing.id).unwrap();
    s.tool_settle(
        &ctx.scope,
        &lease,
        &missing.id,
        ToolStatus::Failed,
        None,
        Some("failure"),
    )
    .unwrap();
    let result = search(&t, &ctx, "needle").await;
    let row = &result["results"][0];
    assert_eq!(row["tool_id"], "call");
    assert_eq!(search(&t, &ctx, "null").await["results"], json!([]));
    let expanded=t.execute(&ctx,&call("memory_expand",json!({"item_id":row["item_id"],"content_sha256":row["content_sha256"],"turn_id":turn.id,"tool_id":"call","start_byte":0,"end_byte":row["content_bytes"]}))).await.unwrap();
    assert_eq!(expanded["text"], json!({"result":"needle 水"}).to_string());
    assert_eq!(s.tools(&ctx.scope, &turn.id).unwrap().len(), 2);
}
#[tokio::test]
async fn honest_bounded_window() {
    let (d, s, _, ctx) = setup();
    turn(&s, &ctx.scope, "a", "old needle", "");
    std::thread::sleep(Duration::from_millis(3));
    turn(&s, &ctx.scope, "b", "new needle", "");
    let t = HistoryTools::new(
        s,
        HistoryConfig {
            enabled: true,
            max_window_turns: 1,
            retention_path: Some(d.path().join("evidence.sqlite3")),
            max_retained_items: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let r = search(&t, &ctx, "needle").await;
    assert_eq!(r["window"]["older_turns_omitted"], true);
    assert_eq!(r["window"]["full_corpus"], false);
    assert_eq!(r["results"].as_array().unwrap().len(), 1);
    assert_eq!(r["results"][0]["text"], "new needle");
    assert_eq!(
        search(&t, &ctx, "new").await["results"][0]["item_id"],
        r["results"][0]["item_id"]
    );
}
struct MockProvider {
    calls: AtomicUsize,
    tool_call: bool,
    unknown_cost: bool,
}
#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "fixture"
    }
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(tools.is_empty());
        assert_eq!(messages.len(), 2);
        let input: Value = serde_json::from_str(&messages[1].content)?;
        let mut message=Message::text("assistant",json!({"summary":"Evidence says needle.","citations":[{"item_id":input["sources"][0]["citation"]["item_id"],"start_byte":0,"end_byte":6}]}).to_string());
        if self.tool_call {
            message.tool_calls.push(call("forbidden", json!({})));
        }
        Ok(Completion {
            message,
            usage: Usage {
                input_tokens: 30,
                output_tokens: 10,
                cost_microusd: if self.unknown_cost { 0 } else { 1 },
                cost_known: !self.unknown_cost,
            },
            finish_reason: "stop".into(),
        })
    }
}
fn policy(dir: &tempfile::TempDir) -> SummaryPolicy {
    SummaryPolicy {
        ledger_path: dir.path().join("summary.jsonl"),
        max_calls: 1,
        max_cost_microusd: 100000,
        input_microusd_per_million: 1000,
        output_microusd_per_million: 1000,
        ..Default::default()
    }
}
#[tokio::test]
async fn summary_one_no_tools_completion_billed_and_durable_budget() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let provider = Arc::new(MockProvider {
        calls: AtomicUsize::new(0),
        tool_call: false,
        unknown_cost: true,
    });
    let t = t
        .with_summary_backend(SummaryBackend::new(provider.clone(), policy(&d)).unwrap())
        .unwrap();
    assert!(
        t.definitions()
            .iter()
            .find(|d| d.name == "context_shunt")
            .unwrap()
            .side_effect
    );
    let c = call(
        "context_shunt",
        json!({"query":"what happened?","item_ids":[row["item_id"]]}),
    );
    let r = t.execute(&ctx, &c).await.unwrap();
    assert_eq!(r["provider_calls"], 1);
    assert_eq!(r["usage"]["cost_known"], false);
    assert!(r["calculated_cost_microusd"].as_u64().unwrap() > 0);
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let rebuilt = HistoryTools::new(
        s,
        HistoryConfig {
            enabled: true,
            retention_path: Some(d.path().join("evidence.sqlite3")),
            ..Default::default()
        },
    )
    .unwrap()
    .with_summary_backend(SummaryBackend::new(provider.clone(), policy(&d)).unwrap())
    .unwrap();
    let row2 = search(&rebuilt, &ctx, "needle").await["results"][0].clone();
    let mut c2 = call(
        "context_shunt",
        json!({"query":"again?","item_ids":[row2["item_id"]]}),
    );
    c2.id = "newcall".into();
    assert!(rebuilt.execute(&ctx, &c2).await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn summary_denies_cross_scope_and_tools_and_disabled_backend() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let c = call(
        "context_shunt",
        json!({"query":"what?","item_ids":[row["item_id"]]}),
    );
    assert!(t.execute(&ctx, &c).await.is_err());
    let provider = Arc::new(MockProvider {
        calls: AtomicUsize::new(0),
        tool_call: true,
        unknown_cost: false,
    });
    let t = t
        .with_summary_backend(SummaryBackend::new(provider.clone(), policy(&d)).unwrap())
        .unwrap();
    let mut other = ctx.clone();
    other.scope.channel = "other".into();
    assert!(t.execute(&other, &c).await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

struct InvalidCitationProvider {
    calls: AtomicUsize,
    wait: bool,
}
#[async_trait]
impl Provider for InvalidCitationProvider {
    fn name(&self) -> &str {
        "invalid-citation-fixture"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.wait {
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
        Ok(Completion {message:Message::text("assistant",json!({"summary":"invented","citations":[{"item_id":"forged","start_byte":0,"end_byte":1}]}).to_string()),usage:Usage {input_tokens:1,output_tokens:1,..Default::default()},finish_reason:"stop".into()})
    }
}
#[tokio::test]
async fn summary_forged_citation_denied_and_no_budget_dispatch() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let p = Arc::new(InvalidCitationProvider {
        calls: AtomicUsize::new(0),
        wait: false,
    });
    let c = call(
        "context_shunt",
        json!({"query":"what?","item_ids":[row["item_id"]]}),
    );
    let mut pol = policy(&d);
    pol.max_cost_microusd = 1;
    let t = t
        .with_summary_backend(SummaryBackend::new(p.clone(), pol).unwrap())
        .unwrap();
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let c = call(
        "context_shunt",
        json!({"query":"what?","item_ids":[row["item_id"]]}),
    );
    let t = t
        .with_summary_backend(SummaryBackend::new(p.clone(), policy(&d)).unwrap())
        .unwrap();
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn summary_timeout_holds_budget_and_never_retries() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let p = Arc::new(InvalidCitationProvider {
        calls: AtomicUsize::new(0),
        wait: true,
    });
    let mut pol = policy(&d);
    pol.timeout_seconds = 1;
    let t = t
        .with_summary_backend(SummaryBackend::new(p.clone(), pol).unwrap())
        .unwrap();
    let c = call(
        "context_shunt",
        json!({"query":"what?","item_ids":[row["item_id"]]}),
    );
    assert!(
        t.execute(&ctx, &c)
            .await
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert!(t.execute(&ctx, &c).await.is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn summary_cancellation_returns_promptly_with_held_reservation() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let p = Arc::new(InvalidCitationProvider {
        calls: AtomicUsize::new(0),
        wait: true,
    });
    let t = t
        .with_summary_backend(SummaryBackend::new(p.clone(), policy(&d)).unwrap())
        .unwrap();
    let c = call(
        "context_shunt",
        json!({"query":"what?","item_ids":[row["item_id"]]}),
    );
    let cancelled = ctx.cancellation.clone();
    let cancel = async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancelled.cancel();
    };
    let (result, _) = tokio::join!(t.execute(&ctx, &c), cancel);
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancelled_or_invalid_scope_never_reads_and_duplicate_sources_denied() {
    let (_d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    assert!(
        t.execute(
            &ctx,
            &call(
                "memory_describe",
                json!({"item_ids":[row["item_id"],row["item_id"]]})
            )
        )
        .await
        .is_err()
    );
    let mut bad = ctx.clone();
    bad.scope.session.clear();
    assert!(
        t.execute(&bad, &call("history_search", json!({"query":"needle"})))
            .await
            .is_err()
    );
    ctx.cancellation.cancel();
    assert!(
        t.execute(&ctx, &call("history_search", json!({"query":"needle"})))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn incomplete_ledger_fails_closed_before_dispatch() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "a", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let pol = policy(&d);
    std::fs::write(&pol.ledger_path, b"{\"event\":\"reserved\"").unwrap();
    std::fs::set_permissions(&pol.ledger_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let p = Arc::new(InvalidCitationProvider {
        calls: AtomicUsize::new(0),
        wait: false,
    });
    let t = t
        .with_summary_backend(SummaryBackend::new(p.clone(), pol).unwrap())
        .unwrap();
    assert!(
        t.execute(
            &ctx,
            &call(
                "context_shunt",
                json!({"query":"what?","item_ids":[row["item_id"]]})
            )
        )
        .await
        .is_err()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
}
