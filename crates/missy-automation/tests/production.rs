use anyhow::Result;
use async_trait::async_trait;
use chrono::{Duration, Utc};
use missy_automation::{
    AutomationTools, MaintenanceConfig, ObservationSource, Scheduler, TriggerConfig, TriggerKind,
    TriggerObservation, TypedPlan, observe, validate_source,
};
use missy_core::{
    Completion, Message, Provider, RuntimeConfig, Scope, ToolCall, ToolContext, ToolDefinition,
    ToolExecutor, Usage,
};
use missy_runtime::Runtime;
use missy_store::Store;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;
struct ProviderFixture(AtomicUsize);
#[tokio::test]
async fn maintenance_creates_real_untrusted_digests_and_learnings_once() {
    let (_d, s, p, _) = setup();
    let now = Utc::now();
    let plan: TypedPlan = serde_json::from_value(
        json!({"nodes":[{"id":"a","tool":"echo","arguments":{"value":"observable"}}]}),
    )
    .unwrap();
    s.execute_plan(&scope(), "memory-source", plan)
        .await
        .unwrap();
    let mut cfg = MaintenanceConfig {
        min_unprocessed_turns: 1,
        batch_size: 2,
        ..Default::default()
    };
    assert_eq!(
        s.run_maintenance(&scope(), &cfg, now + Duration::seconds(600))
            .unwrap()
            .processed_turns,
        0
    );
    cfg.enabled = true;
    cfg.operator_consent = true;
    assert_eq!(
        s.run_maintenance(&scope(), &cfg, now)
            .unwrap()
            .skipped
            .as_deref(),
        Some("not_idle")
    );
    let r = s
        .run_maintenance(&scope(), &cfg, now + Duration::seconds(600))
        .unwrap();
    assert_eq!(r.processed_turns, 1, "{r:?}");
    assert_eq!(r.summaries_created, 1, "{r:?}");
    assert_eq!(r.learnings_created, 1, "{r:?}");
    assert_eq!(p.0.load(Ordering::SeqCst), 0);
    let memories = s_store(&s)
        .memory_search(&scope(), "observed_journal_v1", 10)
        .unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].trust, "untrusted");
    assert!(memories[0].content.contains("confirmed_success"));
    assert_eq!(
        s.run_maintenance(&scope(), &cfg, now + Duration::seconds(700))
            .unwrap()
            .processed_turns,
        0
    );
}
fn s_store(s: &Scheduler) -> Arc<Store> {
    s.runtime_store()
}
#[tokio::test]
async fn trusted_collectors_bound_paths_file_changes_disk_load_and_idle() {
    let (d, s, _, _) = setup();
    let file = d.path().join("watched");
    std::fs::write(&file, "one").unwrap();
    let mut c = TriggerConfig {
        name: "watch".into(),
        enabled: true,
        operator_consent: true,
        kind: TriggerKind::FileChange,
        prompt: "check".into(),
        cooldown_seconds: 30,
        active_utc: None,
    };
    let source = ObservationSource::File {
        path: file.clone(),
        allowed_root: d.path().into(),
        max_bytes: 100,
    };
    validate_source(&source, &c).unwrap();
    let a = observe(&source, &c, &s_store(&s), &scope(), Utc::now()).unwrap();
    std::fs::write(&file, "two").unwrap();
    assert_ne!(
        a.file_revision,
        observe(&source, &c, &s_store(&s), &scope(), Utc::now())
            .unwrap()
            .file_revision
    );
    let short = ObservationSource::File {
        path: file.clone(),
        allowed_root: d.path().into(),
        max_bytes: 1,
    };
    assert!(observe(&short, &c, &s_store(&s), &scope(), Utc::now()).is_err());
    let link = d.path().join("link");
    std::os::unix::fs::symlink(&file, &link).unwrap();
    assert!(
        validate_source(
            &ObservationSource::File {
                path: link,
                allowed_root: d.path().into(),
                max_bytes: 100
            },
            &c
        )
        .is_err()
    );
    c.kind = TriggerKind::DiskAbove { percent: 50.0 };
    let disk = observe(
        &ObservationSource::Disk {
            path: d.path().into(),
            allowed_root: d.path().into(),
        },
        &c,
        &s_store(&s),
        &scope(),
        Utc::now(),
    )
    .unwrap();
    assert!(disk.disk_percent.unwrap().is_finite());
    c.kind = TriggerKind::LoadAbove {
        normalized_load: 1.0,
    };
    assert!(
        observe(
            &ObservationSource::Load,
            &c,
            &s_store(&s),
            &scope(),
            Utc::now()
        )
        .unwrap()
        .normalized_load
        .unwrap()
        .is_finite()
    );
    c.kind = TriggerKind::Idle { seconds: 30 };
    assert!(
        observe(
            &ObservationSource::Idle,
            &c,
            &s_store(&s),
            &scope(),
            Utc::now()
        )
        .unwrap()
        .idle_seconds
        .is_none()
    );
}
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
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Completion {
            message: Message::text("assistant", "checked"),
            usage: Usage::default(),
            finish_reason: "stop".into(),
        })
    }
}
#[derive(Default)]
struct Echo {
    calls: Mutex<Vec<ToolCall>>,
}
struct ApprovalGate {
    pending: AtomicBool,
    effects: AtomicUsize,
}
#[async_trait]
impl ToolExecutor for ApprovalGate {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "effect".into(),
            description: "fixture".into(),
            side_effect: true,
            parameters: json!({"type":"object"}),
        }]
    }
    async fn preflight(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        if self.pending.load(Ordering::SeqCst) {
            return Err(missy_core::ApprovalRequired {
                request_id: "approval".into(),
            }
            .into());
        }
        Ok(())
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        self.effects.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"approved_effect":true}))
    }
}
#[async_trait]
impl ToolExecutor for Echo {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "echo".into(),
            description: "fixture".into(),
            side_effect: true,
            parameters: json!({"type":"object"}),
        }]
    }
    async fn execute(&self, _: &ToolContext, c: &ToolCall) -> Result<Value> {
        self.calls.lock().unwrap().push(c.clone());
        Ok(c.arguments.clone())
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
fn setup() -> (
    tempfile::TempDir,
    Scheduler,
    Arc<ProviderFixture>,
    Arc<Echo>,
) {
    let d = tempfile::tempdir().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let p = Arc::new(ProviderFixture(AtomicUsize::new(0)));
    let e = Arc::new(Echo::default());
    let rt = Arc::new(
        Runtime::new(
            Arc::new(Store::open(d.path().join("runtime.db")).unwrap()),
            p.clone(),
            e.clone(),
            "fixture".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let s = Scheduler::open(d.path().join("scheduler.db"), rt).unwrap();
    (d, s, p, e)
}
#[tokio::test]
async fn exact_dag_dispatch_refs_conditions_dedup_and_scope() {
    let (_d, s, p, e) = setup();
    let plan:TypedPlan=serde_json::from_value(json!({"nodes":[
        {"id":"b","tool":"echo","arguments":{"number":{"$ref":{"node":"a","pointer":"/value"}}},"preconditions":[{"node":"a","pointer":"/value","test":{"kind":"is_number"}}],"postconditions":[{"pointer":"/number","test":{"kind":"equals","value":42}}]},
        {"id":"a","tool":"echo","arguments":{"value":42}}
    ]})).unwrap();
    let r = s
        .execute_plan(&scope(), "fixed", plan.clone())
        .await
        .unwrap();
    assert_eq!(r.status, "completed");
    assert_eq!(r.nodes["b"].output.as_ref().unwrap()["number"], 42);
    let again = s
        .execute_plan(&scope(), "fixed", plan.clone())
        .await
        .unwrap();
    assert_eq!(r.nodes["a"].operation_id, again.nodes["a"].operation_id);
    assert_eq!(e.calls.lock().unwrap().len(), 2);
    assert_eq!(p.0.load(Ordering::SeqCst), 0);
    let mut changed = plan.clone();
    changed.nodes[1].arguments = json!({"value":99});
    assert!(s.execute_plan(&scope(), "fixed", changed).await.is_err());
    let mut other = scope();
    other.user = "other".into();
    assert!(s.plan_receipt(&other, "fixed").unwrap().is_none());
    assert_eq!(
        s.execute_plan(&other, "fixed", plan).await.unwrap().status,
        "completed"
    );
    assert_eq!(e.calls.lock().unwrap().len(), 4);
}
#[tokio::test]
async fn postcondition_failure_blocks_all_later_effects() {
    let (_d, s, _, e) = setup();
    let plan:TypedPlan=serde_json::from_value(json!({"nodes":[{"id":"a","tool":"echo","arguments":{"x":1},"postconditions":[{"pointer":"/x","test":{"kind":"equals","value":2}}]},{"id":"b","tool":"echo","arguments":{},"depends_on":["a"]}]})).unwrap();
    let r = s.execute_plan(&scope(), "fail", plan).await.unwrap();
    assert_eq!(r.status, "failed");
    assert_eq!(r.nodes["b"].status, "blocked");
    assert_eq!(e.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn interrupted_approval_requires_explicit_resume_and_reconcile() {
    let d = tempfile::tempdir().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let gate = Arc::new(ApprovalGate {
        pending: AtomicBool::new(true),
        effects: AtomicUsize::new(0),
    });
    let rt = Arc::new(
        Runtime::new(
            Arc::new(Store::open(d.path().join("r.db")).unwrap()),
            Arc::new(ProviderFixture(AtomicUsize::new(0))),
            gate.clone(),
            "test".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let s = Scheduler::open(d.path().join("s.db"), rt.clone()).unwrap();
    let plan:TypedPlan=serde_json::from_value(json!({"nodes":[{"id":"a","tool":"effect","arguments":{}},{"id":"b","tool":"effect","arguments":{},"depends_on":["a"]}]})).unwrap();
    let r = s
        .execute_plan(&scope(), "approved", plan.clone())
        .await
        .unwrap();
    assert_eq!(r.status, "interrupted");
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert!(
        s.reconcile_plan(&scope(), "approved", plan.clone())
            .await
            .is_err()
    );
    gate.pending.store(false, Ordering::SeqCst);
    let still = s
        .execute_plan(&scope(), "approved", plan.clone())
        .await
        .unwrap();
    assert_eq!(still.status, "interrupted");
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    rt.resume(scope(), r.nodes["a"].turn_id.as_ref().unwrap())
        .await
        .unwrap();
    assert_eq!(gate.effects.load(Ordering::SeqCst), 1);
    let final_r = s.reconcile_plan(&scope(), "approved", plan).await.unwrap();
    assert_eq!(final_r.status, "completed");
    assert_eq!(gate.effects.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn triggers_require_consent_durable_cooldown_and_config_binding() {
    let (_d, s, p, _) = setup();
    let mut c = TriggerConfig {
        name: "heartbeat".into(),
        enabled: true,
        operator_consent: false,
        kind: TriggerKind::Heartbeat,
        prompt: "check".into(),
        cooldown_seconds: 60,
        active_utc: None,
    };
    let now = Utc::now();
    let o = TriggerObservation::default();
    assert!(
        s.fire_trigger(&scope(), &c, &o, now)
            .await
            .unwrap()
            .is_none()
    );
    c.operator_consent = true;
    assert_eq!(
        s.fire_trigger(&scope(), &c, &o, now)
            .await
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(10))
            .await
            .unwrap()
            .is_none()
    );
    c.prompt = "changed".into();
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(70))
            .await
            .is_err()
    );
    assert_eq!(p.0.load(Ordering::SeqCst), 1);
    s.reset_trigger(&scope(), &c.name).unwrap();
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(70))
            .await
            .unwrap()
            .is_some()
    );
}
struct TriggerProvider(AtomicUsize);
#[async_trait]
impl Provider for TriggerProvider {
    fn name(&self) -> &str {
        "trigger-fixture"
    }
    async fn complete(
        &self,
        messages: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut message = Message::text("assistant", "checked");
        if messages.last().is_some_and(|m| m.role == "user") {
            message.tool_calls.push(ToolCall {
                id: "sensitive".into(),
                name: "effect".into(),
                arguments: json!({}),
            });
        }
        Ok(Completion {
            message,
            usage: Usage::default(),
            finish_reason: "stop".into(),
        })
    }
}

fn trigger_config() -> TriggerConfig {
    TriggerConfig {
        name: "approval-trigger".into(),
        enabled: true,
        operator_consent: true,
        kind: TriggerKind::Heartbeat,
        prompt: "perform sensitive effect".into(),
        cooldown_seconds: 30,
        active_utc: None,
    }
}

fn approval_trigger_setup() -> (
    tempfile::TempDir,
    Arc<Runtime>,
    Scheduler,
    Arc<TriggerProvider>,
    Arc<ApprovalGate>,
) {
    let d = tempfile::tempdir().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let p = Arc::new(TriggerProvider(AtomicUsize::new(0)));
    let gate = Arc::new(ApprovalGate {
        pending: AtomicBool::new(true),
        effects: AtomicUsize::new(0),
    });
    let rt = Arc::new(
        Runtime::new(
            Arc::new(Store::open(d.path().join("r.db")).unwrap()),
            p.clone(),
            gate.clone(),
            "test".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let s = Scheduler::open(d.path().join("s.db"), rt.clone()).unwrap();
    (d, rt, s, p, gate)
}

#[tokio::test]
async fn trigger_approval_poll_is_nonfatal_no_replay_and_human_completion_refreshes() {
    let (d, rt, s, p, gate) = approval_trigger_setup();
    let c = trigger_config();
    let o = TriggerObservation::default();
    let now = Utc::now();
    let first = s
        .fire_trigger(&scope(), &c, &o, now)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.status, "interrupted");
    assert_eq!(p.0.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    drop(s);
    let s = Scheduler::open(d.path().join("s.db"), rt.clone()).unwrap();
    for offset in [1, 31, 90] {
        let poll = s
            .fire_trigger(&scope(), &c, &o, now + Duration::seconds(offset))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(poll.status, "paused");
        assert_eq!(poll.turn_id, first.turn_id);
        assert!(poll.deduplicated && !poll.resumable);
    }
    assert_eq!(p.0.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    let receipts = s.trigger_receipts(&scope(), &c.name).unwrap();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].status, "interrupted");
    assert_eq!(receipts[0].turn_id.as_ref().unwrap(), &first.turn_id);
    let mut other = scope();
    other.user = "other".into();
    assert!(s.trigger_receipts(&other, &c.name).unwrap().is_empty());
    assert!(s.reset_trigger(&scope(), &c.name).is_err());
    gate.pending.store(false, Ordering::SeqCst);
    assert_eq!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(91))
            .await
            .unwrap()
            .unwrap()
            .status,
        "paused"
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(
        rt.resume(receipts[0].turn_scope.clone(), &first.turn_id)
            .await
            .unwrap()
            .status,
        "completed"
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 1);
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(10))
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(p.0.load(Ordering::SeqCst), 2);
    let next = s
        .fire_trigger(&scope(), &c, &o, now + Duration::seconds(100))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(next.status, "completed");
    assert_ne!(next.turn_id, first.turn_id);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn trigger_failed_cancelled_occurrences_pause_until_explicit_safe_reset() {
    for status in [
        missy_store::TurnStatus::Failed,
        missy_store::TurnStatus::Cancelled,
    ] {
        let (_d, rt, s, p, gate) = approval_trigger_setup();
        let c = trigger_config();
        let o = TriggerObservation::default();
        let now = Utc::now();
        let first = s
            .fire_trigger(&scope(), &c, &o, now)
            .await
            .unwrap()
            .unwrap();
        let child = missy_automation::trigger_scope(&scope(), &c.name).unwrap();
        let lease = rt
            .store()
            .acquire_lease(
                &child,
                &first.turn_id,
                "operator",
                std::time::Duration::from_secs(60),
            )
            .unwrap();
        rt.store()
            .finish(
                &child,
                &lease,
                status,
                None,
                Some("operator terminal disposition"),
            )
            .unwrap();
        for offset in [31, 61, 91] {
            assert_eq!(
                s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(offset))
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                "paused"
            );
        }
        assert_eq!(p.0.load(Ordering::SeqCst), 1);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        s.reset_trigger(&scope(), &c.name).unwrap();
        let next = s
            .fire_trigger(&scope(), &c, &o, now + Duration::seconds(100))
            .await
            .unwrap()
            .unwrap();
        assert_ne!(next.turn_id, first.turn_id);
        assert_eq!(p.0.load(Ordering::SeqCst), 2);
    }
}

#[tokio::test]
async fn trigger_reset_does_not_discard_unknown_effects_on_terminal_turn() {
    let (_d, rt, s, p, gate) = approval_trigger_setup();
    let c = trigger_config();
    let o = TriggerObservation::default();
    let now = Utc::now();
    let first = s
        .fire_trigger(&scope(), &c, &o, now)
        .await
        .unwrap()
        .unwrap();
    let child = missy_automation::trigger_scope(&scope(), &c.name).unwrap();
    let lease = rt
        .store()
        .acquire_lease(
            &child,
            &first.turn_id,
            "operator",
            std::time::Duration::from_secs(60),
        )
        .unwrap();
    let call = ToolCall {
        id: "uncertain".into(),
        name: "effect".into(),
        arguments: json!({}),
    };
    rt.store().tool_begin(&child, &lease, &call).unwrap();
    rt.store().tool_running(&child, &lease, &call.id).unwrap();
    rt.store()
        .finish(
            &child,
            &lease,
            missy_store::TurnStatus::Cancelled,
            None,
            Some("uncertain effect"),
        )
        .unwrap();
    assert!(s.reset_trigger(&scope(), &c.name).is_err());
    assert_eq!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(100))
            .await
            .unwrap()
            .unwrap()
            .status,
        "paused"
    );
    assert_eq!(p.0.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn trigger_uncertain_reservation_and_admitted_turn_are_nonfatal_no_replay() {
    let (d, rt, s, p, gate) = approval_trigger_setup();
    let c = trigger_config();
    let o = TriggerObservation::default();
    let now = Utc::now();
    let db = rusqlite::Connection::open(d.path().join("s.db")).unwrap();
    let key = serde_json::to_string(&scope()).unwrap();
    db.execute("INSERT INTO automation_trigger_runs(scope,name,operation,status) VALUES(?1,?2,'crash-reservation','running')", rusqlite::params![key,c.name]).unwrap();
    let unknown = s
        .fire_trigger(&scope(), &c, &o, now)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unknown.status, "paused");
    assert_eq!(
        s.trigger_receipts(&scope(), &c.name).unwrap()[0].status,
        "outcome_unknown"
    );
    assert!(s.reset_trigger(&scope(), &c.name).is_err());
    let child = missy_automation::trigger_scope(&scope(), &c.name).unwrap();
    let admitted = rt
        .store()
        .admit(&child, "trigger:crash-reservation", "already admitted")
        .unwrap();
    for offset in [1, 31, 91] {
        let poll = s
            .fire_trigger(&scope(), &c, &o, now + Duration::seconds(offset))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(poll.status, "inflight");
        assert_eq!(poll.turn_id, admitted.turn.id);
    }
    assert!(s.reset_trigger(&scope(), &c.name).is_err());
    let _lease = rt
        .store()
        .acquire_lease(
            &child,
            &admitted.turn.id,
            "fixture",
            std::time::Duration::from_secs(60),
        )
        .unwrap();
    assert_eq!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(101))
            .await
            .unwrap()
            .unwrap()
            .status,
        "inflight"
    );
    assert_eq!(p.0.load(Ordering::SeqCst), 0);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn file_changes_baseline_and_scoped_model_schedule_crud() {
    let (_d, health, _, _) = setup();
    let time = Utc::now();
    assert!(
        health
            .record_health(&scope(), "provider", true, time, 3)
            .unwrap()
            .audited
    );
    assert!(
        !health
            .record_health(&scope(), "provider", true, time + Duration::seconds(60), 3)
            .unwrap()
            .audited
    );
    for i in 1..=3 {
        let r = health
            .record_health(
                &scope(),
                "provider",
                false,
                time + Duration::seconds(60 + i),
                3,
            )
            .unwrap();
        assert_eq!(r.consecutive_failures, i as u64);
        assert_eq!(r.threshold_exceeded, i == 3);
    }
    let recovered = health
        .record_health(&scope(), "provider", true, time + Duration::seconds(70), 3)
        .unwrap();
    assert!(recovered.recovered && recovered.audited);
    assert_eq!(recovered.consecutive_failures, 0);
    let (_d, s, p, e) = setup();
    let now = Utc::now();
    let c = TriggerConfig {
        name: "file".into(),
        enabled: true,
        operator_consent: true,
        kind: TriggerKind::FileChange,
        prompt: "changed".into(),
        cooldown_seconds: 30,
        active_utc: None,
    };
    let mut o = TriggerObservation {
        file_revision: Some("hash1".into()),
        ..Default::default()
    };
    assert!(
        s.fire_trigger(&scope(), &c, &o, now)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(31))
            .await
            .unwrap()
            .is_none()
    );
    o.file_revision = Some("hash2".into());
    assert!(
        s.fire_trigger(&scope(), &c, &o, now + Duration::seconds(31))
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(p.0.load(Ordering::SeqCst), 1);
    let tools = AutomationTools::new(e, s).unwrap();
    let ctx = ToolContext {
        scope: scope(),
        turn_id: "turn".into(),
        cancellation: CancellationToken::new(),
    };
    assert!(
        tools
            .definitions()
            .iter()
            .any(|d| d.name == "schedule_task")
    );
    let spec = json!({"timezone":"UTC","schedule":{"kind":"daily","hour":23,"minute":59}});
    let created = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "create".into(),
                name: "schedule_task".into(),
                arguments: json!({"prompt":"fixture scheduled check","spec":spec}),
            },
        )
        .await
        .unwrap();
    let id = created["id"].as_str().unwrap();
    tools
        .execute(
            &ctx,
            &ToolCall {
                id: "pause".into(),
                name: "update_schedule".into(),
                arguments: json!({"id":id,"paused":true}),
            },
        )
        .await
        .unwrap();
    let mut foreign = ctx.clone();
    foreign.scope.user = "foreign".into();
    assert!(
        tools
            .execute(
                &foreign,
                &ToolCall {
                    id: "remove-other".into(),
                    name: "delete_schedule".into(),
                    arguments: json!({"id":id})
                }
            )
            .await
            .is_err()
    );
    let listed = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "list-paused".into(),
                name: "list_schedules".into(),
                arguments: json!({}),
            },
        )
        .await
        .unwrap();
    assert_eq!(listed[0]["enabled"], false);
    tools
        .execute(
            &ctx,
            &ToolCall {
                id: "remove".into(),
                name: "delete_schedule".into(),
                arguments: json!({"id":id}),
            },
        )
        .await
        .unwrap();
    let call = ToolCall {
        id: "list".into(),
        name: "list_schedules".into(),
        arguments: json!({}),
    };
    assert_eq!(tools.execute(&ctx, &call).await.unwrap(), json!([]));
    let spoof = ToolCall {
        arguments: json!({"scope":{"user":"other"}}),
        ..call
    };
    assert!(tools.execute(&ctx, &spoof).await.is_err());
}
