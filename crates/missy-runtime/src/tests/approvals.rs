use super::*;

#[derive(Clone, Copy)]
enum Decision {
    Pending,
    Approved,
    Denied,
    RevokedAtDispatch,
}
struct Gate {
    decisions: Mutex<HashMap<String, Decision>>,
    effects: AtomicUsize,
    preflights: AtomicUsize,
    policy: String,
}
impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            decisions: Mutex::new(HashMap::new()),
            effects: AtomicUsize::new(0),
            preflights: AtomicUsize::new(0),
            policy: "guard-v1".into(),
        })
    }
    fn decide(&self, id: &str, decision: Decision) {
        self.decisions.lock().unwrap().insert(id.into(), decision);
    }
}
#[async_trait]
impl ToolExecutor for Gate {
    fn definitions(&self) -> Vec<ToolDefinition> {
        tools().definitions()
    }
    fn configuration_fingerprint(&self) -> String {
        self.policy.clone()
    }
    async fn preflight(&self, _: &ToolContext, call: &ToolCall) -> Result<()> {
        self.preflights.fetch_add(1, Ordering::SeqCst);
        match self
            .decisions
            .lock()
            .unwrap()
            .get(&call.id)
            .copied()
            .unwrap_or(Decision::Pending)
        {
            Decision::Pending => Err(ApprovalRequired {
                request_id: format!("approval-{}", call.id),
            }
            .into()),
            Decision::Approved | Decision::RevokedAtDispatch => Ok(()),
            Decision::Denied => bail!("approval denied or expired"),
        }
    }
    async fn execute(&self, _: &ToolContext, call: &ToolCall) -> Result<Value> {
        let decision = self.decisions.lock().unwrap().remove(&call.id);
        if matches!(decision, Some(Decision::RevokedAtDispatch)) {
            return Err(ToolDispatchRefused {
                reason: "approval revoked before dispatch".into(),
            }
            .into());
        }
        ensure!(
            matches!(decision, Some(Decision::Approved)),
            "approval not available"
        );
        self.effects.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"effect": call.arguments}))
    }
}

fn replace_state(rt: &Runtime, id: &str, state: &State) {
    let lease = rt
        .store()
        .acquire_lease(&scope("a"), id, "test-editor", Duration::from_secs(30))
        .unwrap();
    rt.save(&scope("a"), &lease, state).unwrap();
    rt.store()
        .finish(
            &scope("a"),
            &lease,
            TurnStatus::Interrupted,
            None,
            Some("test checkpoint"),
        )
        .unwrap();
}

#[tokio::test]
async fn approval_pause_restart_resume_retains_exact_batch_without_provider_repeat() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("done", "stop"),
    ]));
    let gate = Gate::new();
    let (d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let paused = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(paused.status, "interrupted");
    assert!(paused.resumable);
    assert_eq!(paused.usage.input_tokens, 10);
    assert!(
        rt.store()
            .tools(&scope("a"), &paused.turn_id)
            .unwrap()
            .is_empty()
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    let turn = rt.status(&scope("a"), &paused.turn_id).unwrap();
    let before = rt.load(&scope("a"), &turn).unwrap();
    assert_eq!(
        before.waiting_tools.as_ref().unwrap().calls,
        vec![call("a")]
    );
    assert_eq!(
        before
            .waiting_tools
            .as_ref()
            .unwrap()
            .approval_request_id
            .as_deref(),
        Some("approval-a")
    );
    let again = rt.resume(scope("a"), &paused.turn_id).await.unwrap();
    assert!(again.resumable);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    gate.decide("a", Decision::Approved);
    drop(rt);
    let rt = Runtime::new(
        Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
        p.clone(),
        gate.clone(),
        "trusted system".into(),
        RuntimeConfig::default(),
    )
    .unwrap();
    let done = rt.resume(scope("a"), &paused.turn_id).await.unwrap();
    assert_eq!(done.status, "completed");
    assert_eq!(done.content, "done");
    assert_eq!(done.usage.input_tokens, 20);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 1);
    assert_eq!(p.calls.load(Ordering::SeqCst), 2);
    let after = rt
        .load(
            &scope("a"),
            &rt.status(&scope("a"), &paused.turn_id).unwrap(),
        )
        .unwrap();
    assert_eq!(after.steps, 2);
    assert_eq!(after.deadline_ms, before.deadline_ms);
    assert!(after.waiting_tools.is_none());
    assert!(rt.resume(scope("a"), &paused.turn_id).await.is_err());
}

#[tokio::test]
async fn whole_batch_preflight_prevents_approved_first_effect() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a"), call("b")]),
        completion("done", "stop"),
    ]));
    let gate = Gate::new();
    gate.decide("a", Decision::Approved);
    let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let paused = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert!(paused.resumable);
    assert_eq!(gate.preflights.load(Ordering::SeqCst), 2);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert!(
        rt.store()
            .tools(&scope("a"), &paused.turn_id)
            .unwrap()
            .is_empty()
    );
    gate.decide("b", Decision::Approved);
    assert_eq!(
        rt.resume(scope("a"), &paused.turn_id).await.unwrap().status,
        "completed"
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 2);
    assert_eq!(p.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn approval_denial_or_expiry_is_known_nonexecution_not_unknown() {
    for pending_first in [false, true] {
        let p = Arc::new(Script::new(vec![effect(vec![call("a"), call("b")])]));
        let gate = Gate::new();
        gate.decide("a", Decision::Approved);
        if !pending_first {
            gate.decide("b", Decision::Denied);
        }
        let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
        let mut result = rt
            .run(scope("a"), "request".into(), "r".into())
            .await
            .unwrap();
        if pending_first {
            assert!(result.resumable);
            gate.decide("b", Decision::Denied);
            result = rt.resume(scope("a"), &result.turn_id).await.unwrap();
        }
        assert_eq!(result.status, "interrupted");
        assert!(!result.resumable);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert!(
            rt.store()
                .tools(&scope("a"), &result.turn_id)
                .unwrap()
                .is_empty()
        );
        assert!(
            rt.store()
                .unresolved_tools(&scope("a"), &result.turn_id)
                .unwrap()
                .is_empty()
        );
        assert!(rt.resume(scope("a"), &result.turn_id).await.is_err());
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn waiting_batch_changed_arguments_ids_and_policy_fail_closed() {
    for change in ["arguments", "ids", "missing-journal", "policy", "seen"] {
        let p = Arc::new(Script::new(vec![effect(vec![call("a")])]));
        let gate = Gate::new();
        let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
        let paused = rt
            .run(scope("a"), "request".into(), "r".into())
            .await
            .unwrap();
        if change == "policy" {
            let changed = Arc::new(Gate {
                policy: "guard-v2".into(),
                decisions: Mutex::new(HashMap::new()),
                effects: AtomicUsize::new(0),
                preflights: AtomicUsize::new(0),
            });
            let other = Runtime::new(
                rt.store().clone(),
                p.clone(),
                changed,
                "trusted system".into(),
                RuntimeConfig::default(),
            )
            .unwrap();
            assert!(other.resume(scope("a"), &paused.turn_id).await.is_err());
        } else {
            let mut state = rt
                .load(
                    &scope("a"),
                    &rt.status(&scope("a"), &paused.turn_id).unwrap(),
                )
                .unwrap();
            match change {
                "arguments" => {
                    state.transcript.last_mut().unwrap().tool_calls[0].arguments =
                        json!({"value":2})
                }
                "ids" => state.waiting_tools.as_mut().unwrap().calls[0].id = "different".into(),
                "seen" => {
                    state.seen_ids.clear();
                }
                _ => state.waiting_tools = None,
            }
            replace_state(&rt, &paused.turn_id, &state);
            assert!(
                rt.resume(scope("a"), &paused.turn_id).await.is_err(),
                "{change}"
            );
        }
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn waiting_batch_cannot_bypass_existing_journal() {
    let gate = Gate::new();
    let p = Arc::new(Script::new(vec![effect(vec![call("a")])]));
    let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let paused = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    let lease = rt
        .store()
        .acquire_lease(
            &scope("a"),
            &paused.turn_id,
            "journal-test",
            Duration::from_secs(30),
        )
        .unwrap();
    rt.store()
        .tool_begin(&scope("a"), &lease, &call("a"))
        .unwrap();
    rt.store().tool_running(&scope("a"), &lease, "a").unwrap();
    rt.store()
        .tool_settle(
            &scope("a"),
            &lease,
            "a",
            ToolStatus::Succeeded,
            Some(&json!({"ok":true})),
            None,
        )
        .unwrap();
    rt.store()
        .finish(&scope("a"), &lease, TurnStatus::Interrupted, None, None)
        .unwrap();
    assert!(rt.resume(scope("a"), &paused.turn_id).await.is_err());
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn approval_wait_retains_original_step_and_wallclock_budgets() {
    for expired in [false, true] {
        let p = Arc::new(Script::new(vec![
            effect(vec![call("a")]),
            completion("unreachable", "stop"),
        ]));
        let gate = Gate::new();
        let (_d, rt) = setup(
            p.clone(),
            gate.clone(),
            RuntimeConfig {
                max_steps: 1,
                max_turn_seconds: if expired { 1 } else { 300 },
                ..RuntimeConfig::default()
            },
        );
        let paused = rt
            .run(scope("a"), "request".into(), "r".into())
            .await
            .unwrap();
        assert!(paused.resumable);
        gate.decide("a", Decision::Approved);
        if expired {
            tokio::time::sleep(Duration::from_millis(1100)).await;
        }
        let resumed = rt.resume(scope("a"), &paused.turn_id).await.unwrap();
        assert_eq!(resumed.status, "interrupted");
        assert!(!resumed.resumable);
        assert_eq!(gate.effects.load(Ordering::SeqCst), usize::from(!expired));
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn cancellation_while_waiting_has_no_effect_and_no_provider_repeat() {
    let p = Arc::new(Script::new(vec![effect(vec![call("a")])]));
    let gate = Gate::new();
    let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let paused = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    gate.decide("a", Decision::Approved);
    rt.cancel(&scope("a"), &paused.turn_id).unwrap();
    if let Ok(result) = rt.resume(scope("a"), &paused.turn_id).await {
        assert_eq!(result.status, "cancelled");
    }
    assert!(
        rt.store()
            .tools(&scope("a"), &paused.turn_id)
            .unwrap()
            .is_empty()
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn legacy_state_defaults_waiting_tools_to_none() {
    let (_d, rt) = setup(
        Arc::new(Script::new(vec![])),
        tools(),
        RuntimeConfig::default(),
    );
    let admitted = rt.store().admit(&scope("a"), "legacy", "request").unwrap();
    let state = rt.fresh_state(&scope("a"), &admitted.turn).unwrap();
    let mut value = serde_json::to_value(state).unwrap();
    value.as_object_mut().unwrap().remove("waiting_tools");
    assert!(
        serde_json::from_value::<State>(value)
            .unwrap()
            .waiting_tools
            .is_none()
    );
}

#[tokio::test]
async fn crash_before_exact_waiting_checkpoint_never_regenerates() {
    let p = Arc::new(Script::new(vec![effect(vec![call("new-identity")])]));
    let gate = Gate::new();
    let (_d, rt) = setup(
        p.clone(),
        gate.clone(),
        RuntimeConfig {
            max_cost_microusd: 0,
            ..RuntimeConfig::default()
        },
    );
    let admitted = rt.store().admit(&scope("a"), "crash", "request").unwrap();
    let mut state = rt.fresh_state(&scope("a"), &admitted.turn).unwrap();
    state.steps = 1;
    state.provider_pending = true;
    replace_state(&rt, &admitted.turn.id, &state);
    let result = rt.resume(scope("a"), &admitted.turn.id).await.unwrap();
    assert_eq!(result.status, "interrupted");
    assert!(!result.resumable);
    assert!(!result.usage.cost_known);
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert!(
        rt.store()
            .tools(&scope("a"), &admitted.turn.id)
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn typed_execute_refusal_is_failed_not_unknown_and_never_replayed() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("unreachable", "stop"),
    ]));
    let gate = Gate::new();
    gate.decide("a", Decision::RevokedAtDispatch);
    let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let result = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(result.status, "interrupted");
    assert!(!result.resumable);
    let journal = rt.store().tools(&scope("a"), &result.turn_id).unwrap();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0].status, ToolStatus::Failed);
    assert_eq!(journal[0].result.as_ref().unwrap()["executed"], false);
    assert!(
        rt.store()
            .unresolved_tools(&scope("a"), &result.turn_id)
            .unwrap()
            .is_empty()
    );
    assert!(rt.resume(scope("a"), &result.turn_id).await.is_err());
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn typed_execute_refusal_keeps_later_prepared_intents_conservatively_unknown() {
    let p = Arc::new(Script::new(vec![effect(vec![call("a"), call("b")])]));
    let gate = Gate::new();
    gate.decide("a", Decision::RevokedAtDispatch);
    gate.decide("b", Decision::Approved);
    let (_d, rt) = setup(p.clone(), gate.clone(), RuntimeConfig::default());
    let result = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    let journal = rt.store().tools(&scope("a"), &result.turn_id).unwrap();
    assert_eq!(journal[0].status, ToolStatus::Failed);
    assert_eq!(journal[1].status, ToolStatus::OutcomeUnknown);
    assert!(!result.resumable);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

struct DriftingTools {
    gate: Arc<Gate>,
    revision: AtomicUsize,
    drift_in_preflight: bool,
}
#[async_trait]
impl ToolExecutor for DriftingTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.gate.definitions()
    }
    fn configuration_fingerprint(&self) -> String {
        format!("mutable-policy-{}", self.revision.load(Ordering::SeqCst))
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.gate.preflight(ctx, call).await?;
        if self.drift_in_preflight {
            self.revision.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.gate.execute(ctx, call).await
    }
}

#[tokio::test]
async fn in_place_policy_drift_blocks_paused_resume_before_effect() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("must not complete", "stop"),
    ]));
    let gate = Gate::new();
    let executor = Arc::new(DriftingTools {
        gate: gate.clone(),
        revision: AtomicUsize::new(0),
        drift_in_preflight: false,
    });
    let (_d, rt) = setup(p.clone(), executor.clone(), RuntimeConfig::default());
    let paused = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert!(paused.resumable);
    gate.decide("a", Decision::Approved);
    executor.revision.fetch_add(1, Ordering::SeqCst);
    assert!(rt.resume(scope("a"), &paused.turn_id).await.is_err());
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn in_place_policy_drift_during_preflight_blocks_batch_prepare() {
    let p = Arc::new(Script::new(vec![
        effect(vec![call("a")]),
        completion("must not complete", "stop"),
    ]));
    let gate = Gate::new();
    gate.decide("a", Decision::Approved);
    let executor = Arc::new(DriftingTools {
        gate: gate.clone(),
        revision: AtomicUsize::new(0),
        drift_in_preflight: true,
    });
    let (_d, rt) = setup(p.clone(), executor, RuntimeConfig::default());
    let result = rt
        .run(scope("a"), "request".into(), "r".into())
        .await
        .unwrap();
    assert_eq!(result.status, "interrupted");
    assert!(!result.resumable);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    assert!(
        rt.store()
            .tools(&scope("a"), &result.turn_id)
            .unwrap()
            .is_empty()
    );
}
