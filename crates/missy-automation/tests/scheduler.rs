use anyhow::Result;
use async_trait::async_trait;
use chrono::{Duration, Timelike, Utc};
use missy_automation::{NewJob, OccurrenceStatus, Scheduler};
use missy_core::{
    Completion, Message, Provider, RuntimeConfig, Scope, ToolCall, ToolContext, ToolDefinition,
    ToolExecutor, Usage,
};
use missy_runtime::Runtime;
use missy_schedule_text::ScheduleSpec;
use missy_store::Store;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;

struct MockProvider {
    calls: AtomicUsize,
    fail: AtomicBool,
    gate: Option<Arc<tokio::sync::Semaphore>>,
}
#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "automation-test"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(g) = &self.gate {
            g.acquire().await?.forget();
        }
        if self.fail.load(Ordering::SeqCst) {
            anyhow::bail!("injected provider failure");
        }
        Ok(Completion {
            message: Message::text("assistant", "done"),
            usage: Usage::default(),
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
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<serde_json::Value> {
        anyhow::bail!("no tools")
    }
}
fn scope() -> Scope {
    Scope {
        tenant: "tenant".into(),
        user: "user".into(),
        channel: "channel".into(),
        session: "session".into(),
    }
}
fn private_dir(path: &std::path::Path) {
    std::fs::create_dir_all(path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
}
fn runtime(path: &std::path::Path, p: Arc<MockProvider>) -> Arc<Runtime> {
    Arc::new(
        Runtime::new(
            Arc::new(Store::open(path).unwrap()),
            p,
            Arc::new(NoTools),
            "test system".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    )
}
fn provider(fail: bool, gate: Option<Arc<tokio::sync::Semaphore>>) -> Arc<MockProvider> {
    Arc::new(MockProvider {
        calls: AtomicUsize::new(0),
        fail: AtomicBool::new(fail),
        gate,
    })
}
fn job() -> NewJob {
    NewJob {
        prompt: "do the job".into(),
        interval_seconds: 3600,
        first_due: Utc::now() - Duration::seconds(1),
    }
}

#[tokio::test]
async fn two_instances_only_one_active_claim_and_no_startup_sweep() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let p = provider(false, Some(gate.clone()));
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let a = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
    a.add(&scope(), job()).unwrap();
    let s = scope();
    let aa = a.clone();
    let running = tokio::spawn(async move { aa.run_due(&s, 4, CancellationToken::new()).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while p.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let b = Scheduler::open(root.join("scheduler.db"), rt).unwrap();
    assert!(
        b.run_due(&scope(), 4, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        b.occurrences(&scope(), 10).unwrap()[0].status,
        OccurrenceStatus::Running
    );
    gate.add_permits(1);
    assert_eq!(
        running.await.unwrap().unwrap()[0].status,
        OccurrenceStatus::Completed
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn restart_preserves_history_scope_and_failures() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let p = provider(true, None);
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let a = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
    let j = a.add(&scope(), job()).unwrap();
    let out = a
        .run_due(&scope(), 2, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_ne!(out[0].status, OccurrenceStatus::Completed);
    assert!(out[0].error.is_some());
    drop(a);
    let b = Scheduler::open(root.join("scheduler.db"), rt).unwrap();
    assert_eq!(
        b.occurrences(&scope(), 10).unwrap()[0].request_id,
        out[0].request_id
    );
    assert!(
        b.run_due(&scope(), 2, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    let mut other = scope();
    other.user = "other".into();
    assert!(b.list(&other).unwrap().is_empty());
    assert!(b.pause(&other, &j.id, true).is_err());
    assert!(b.remove(&other, &j.id).is_err());
    b.pause(&scope(), &j.id, true).unwrap();
    assert!(!b.list(&scope()).unwrap()[0].enabled);
    b.remove(&scope(), &j.id).unwrap();
    assert!(b.list(&scope()).unwrap().is_empty());
    assert_eq!(b.occurrences(&scope(), 10).unwrap().len(), 1);
}

#[tokio::test]
async fn cancellation_before_admission_runs_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let p = provider(false, None);
    let a = Scheduler::open(
        root.join("scheduler.db"),
        runtime(&root.join("runtime.db"), p.clone()),
    )
    .unwrap();
    a.add(&scope(), job()).unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(a.run_due(&scope(), 2, cancel).await.unwrap().is_empty());
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_drains_admitted_work_without_starting_next_job() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let p = provider(false, Some(gate.clone()));
    let scheduler = Scheduler::open(
        root.join("scheduler.db"),
        runtime(&root.join("runtime.db"), p.clone()),
    )
    .unwrap();
    scheduler.add(&scope(), job()).unwrap();
    scheduler.add(&scope(), job()).unwrap();
    let cancel = CancellationToken::new();
    let worker_cancel = cancel.clone();
    let s = scheduler.clone();
    let worker = tokio::spawn(async move { s.run_due(&scope(), 4, worker_cancel).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while p.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    cancel.cancel();
    assert!(!worker.is_finished());
    gate.add_permits(1);
    let finished = worker.await.unwrap().unwrap();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].status, OccurrenceStatus::Completed);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn rejects_runtime_database_nonprivate_parent_symlinks_and_hardlinks() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let rt = runtime(&root.join("runtime.db"), provider(false, None));
    assert!(Scheduler::open(root.join("runtime.db"), rt.clone()).is_err());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let public = root.join("public");
        std::fs::create_dir(&public).unwrap();
        std::fs::set_permissions(&public, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(Scheduler::open(public.join("scheduler.db"), rt.clone()).is_err());
        let a = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
        std::os::unix::fs::symlink(root.join("scheduler.db"), root.join("linked.db")).unwrap();
        assert!(Scheduler::open(root.join("linked.db"), rt.clone()).is_err());
        std::fs::hard_link(root.join("scheduler.db"), root.join("hard.db")).unwrap();
        assert!(Scheduler::open(root.join("hard.db"), rt).is_err());
        drop(a);
    }
}

#[tokio::test]
async fn crash_after_runtime_completion_reuses_request_and_does_not_repeat_effect() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let p = provider(false, None);
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let a = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
    a.add(&scope(), job()).unwrap();
    let completed = a
        .run_due(&scope(), 1, CancellationToken::new())
        .await
        .unwrap()
        .remove(0);
    assert_eq!(completed.status, OccurrenceStatus::Completed);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    // Emulate a crash between Runtime's durable completion and scheduler ack.
    // SQLite opens the existing file directly; no raw descriptor is opened/closed.
    let db = rusqlite::Connection::open(root.join("scheduler.db")).unwrap();
    db.execute(
        "UPDATE automation_jobs SET next_due=?1 WHERE id=?2",
        rusqlite::params![completed.due.timestamp(), completed.job_id],
    )
    .unwrap();
    db.execute("UPDATE automation_occurrences SET status='running',lease_until=0,owner='dead-worker' WHERE request_id=?1",[&completed.request_id]).unwrap();
    drop(db);
    drop(a);
    let b = Scheduler::open(root.join("scheduler.db"), rt).unwrap();
    let recovered = b
        .run_due(&scope(), 2, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].request_id, completed.request_id);
    assert_eq!(recovered[0].status, OccurrenceStatus::Completed);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    let db = rusqlite::Connection::open(root.join("scheduler.db")).unwrap();
    let fence: i64 = db
        .query_row(
            "SELECT fence FROM automation_occurrences WHERE request_id=?1",
            [completed.request_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fence, 2);
}

#[tokio::test]
async fn expired_lease_takeover_fences_stale_worker_without_duplicate_active_effect() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let p = provider(false, Some(gate.clone()));
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let a = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
    a.add(&scope(), job()).unwrap();
    let aa = a.clone();
    let worker =
        tokio::spawn(async move { aa.run_due(&scope(), 1, CancellationToken::new()).await });
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while p.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let old_request = a.occurrences(&scope(), 1).unwrap()[0].request_id.clone();
    let db = rusqlite::Connection::open(root.join("scheduler.db")).unwrap();
    db.execute(
        "UPDATE automation_occurrences SET lease_until=0 WHERE request_id=?1",
        [&old_request],
    )
    .unwrap();
    drop(db);
    let b = Scheduler::open(root.join("scheduler.db"), rt).unwrap();
    let recovered = b
        .run_due(&scope(), 1, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(recovered[0].request_id, old_request);
    assert_eq!(recovered[0].status, OccurrenceStatus::Unknown);
    gate.add_permits(1);
    assert!(worker.await.unwrap().is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        b.occurrences(&scope(), 1).unwrap()[0].status,
        OccurrenceStatus::Unknown
    );
}

#[tokio::test]
async fn expired_occurrence_with_unknown_effect_is_not_replayed_or_unpaused() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let store = Arc::new(Store::open(root.join("runtime.db")).unwrap());
    let p = provider(false, None);
    let rt = Arc::new(
        Runtime::new(
            store.clone(),
            p.clone(),
            Arc::new(NoTools),
            "test system".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let scheduler = Scheduler::open(root.join("scheduler.db"), rt.clone()).unwrap();
    let j = scheduler.add(&scope(), job()).unwrap();
    let request_id = "automation:unknown-effect";
    let admitted = store.admit(&scope(), request_id, &j.prompt).unwrap();
    let lease = store
        .acquire_lease(
            &scope(),
            &admitted.turn.id,
            "dead-runtime",
            std::time::Duration::from_secs(30),
        )
        .unwrap();
    let call = ToolCall {
        id: "effect".into(),
        name: "external_write".into(),
        arguments: serde_json::json!({}),
    };
    store.tool_begin(&scope(), &lease, &call).unwrap();
    store.tool_running(&scope(), &lease, &call.id).unwrap();
    store.release_lease(&scope(), &lease).unwrap();
    let db = rusqlite::Connection::open(root.join("scheduler.db")).unwrap();
    db.execute("INSERT INTO automation_occurrences(request_id,job_id,due,status,owner,fence,lease_until) VALUES(?1,?2,?3,'running','dead-scheduler',1,0)",rusqlite::params![request_id,j.id,j.next_due.timestamp()]).unwrap();
    drop(db);
    let results = scheduler
        .run_due(&scope(), 1, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(results[0].status, OccurrenceStatus::Unknown);
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    assert!(!scheduler.list(&scope()).unwrap()[0].enabled);
    assert!(scheduler.pause(&scope(), &j.id, false).is_err());
    assert!(rt.resume(scope(), &admitted.turn.id).await.is_err());
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn fractional_one_shot_rounds_up_completes_once_and_stays_exhausted() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let p = provider(false, None);
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let db_path = root.join("scheduler.db");
    let scheduler = Scheduler::open(&db_path, rt.clone()).unwrap();
    let anchor = chrono::DateTime::parse_from_rfc3339("2026-01-01T12:00:00.123456789Z")
        .unwrap()
        .with_timezone(&Utc);
    let spec = ScheduleSpec::parse_at("in 1 second", "UTC", anchor).unwrap();
    let j = scheduler
        .add_spec(&scope(), "once".into(), spec, anchor)
        .unwrap();
    assert_eq!(
        j.next_due,
        anchor.with_nanosecond(0).unwrap() + Duration::seconds(2)
    );
    assert_eq!(scheduler.list(&scope()).unwrap()[0].next_due, j.next_due);
    let mut other = scope();
    other.channel = "other".into();
    assert!(scheduler.list(&other).unwrap().is_empty());
    assert!(
        scheduler
            .run_due(&other, 1, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    assert!(scheduler.pause(&other, &j.id, false).is_err());
    assert!(scheduler.remove(&other, &j.id).is_err());
    let completed = scheduler
        .run_due(&scope(), 1, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].due, j.next_due);
    assert_eq!(completed[0].status, OccurrenceStatus::Completed);
    assert!(!scheduler.list(&scope()).unwrap()[0].enabled);
    assert!(scheduler.pause(&scope(), &j.id, false).is_err());
    assert!(scheduler.occurrences(&other, 10).unwrap().is_empty());
    drop(scheduler);
    let reopened = Scheduler::open(&db_path, rt).unwrap();
    assert!(!reopened.list(&scope()).unwrap()[0].enabled);
    assert!(reopened.pause(&scope(), &j.id, false).is_err());
    assert!(
        reopened
            .run_due(&scope(), 1, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fractional_interval_uses_ceiling_on_first_and_subsequent_dates() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let p = provider(false, Some(gate.clone()));
    let scheduler = Scheduler::open(
        root.join("scheduler.db"),
        runtime(&root.join("runtime.db"), p.clone()),
    )
    .unwrap();
    let anchor = chrono::DateTime::parse_from_rfc3339("2026-01-01T12:00:00.125Z")
        .unwrap()
        .with_timezone(&Utc);
    let spec = ScheduleSpec::parse_at("every 10 seconds", "UTC", anchor).unwrap();
    let j = scheduler
        .add_spec(&scope(), "interval".into(), spec.clone(), anchor)
        .unwrap();
    let running = scheduler.clone();
    let worker =
        tokio::spawn(async move { running.run_due(&scope(), 1, CancellationToken::new()).await });
    wait_for_call(&p).await;
    scheduler.pause(&scope(), &j.id, true).unwrap();
    gate.add_permits(1);
    let completed = worker.await.unwrap().unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].status, OccurrenceStatus::Completed);
    let next = scheduler.list(&scope()).unwrap().remove(0).next_due;
    assert_eq!(
        j.next_due,
        anchor.with_nanosecond(0).unwrap() + Duration::seconds(11)
    );
    assert_eq!(next, j.next_due + Duration::seconds(10));
    assert!(next >= spec.next_after(j.next_due).unwrap());
    assert_eq!(next.timestamp_subsec_nanos(), 0);
    // The legacy API deliberately retains its old floor-to-second behavior.
    let legacy = scheduler
        .add(
            &scope(),
            NewJob {
                prompt: "legacy".into(),
                interval_seconds: 10,
                first_due: anchor,
            },
        )
        .unwrap();
    assert_eq!(legacy.next_due, anchor.with_nanosecond(0).unwrap());
}

#[tokio::test]
async fn future_spec_is_not_admitted_early() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let p = provider(false, None);
    let scheduler = Scheduler::open(
        root.join("scheduler.db"),
        runtime(&root.join("runtime.db"), p.clone()),
    )
    .unwrap();
    let anchor = Utc::now().with_nanosecond(123_000_000).unwrap();
    let spec = ScheduleSpec::parse_at("in 1 hour", "UTC", anchor).unwrap();
    let exact = spec.next_after(anchor).unwrap();
    let j = scheduler
        .add_spec(&scope(), "future".into(), spec, anchor)
        .unwrap();
    assert!(j.next_due > exact);
    assert!(j.next_due - exact < Duration::seconds(1));
    assert_eq!(scheduler.list(&scope()).unwrap()[0].next_due, j.next_due);
    assert!(
        scheduler
            .run_due(&scope(), 1, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
}

async fn wait_for_call(p: &MockProvider) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while p.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn daily_and_cron_reschedule_wall_clock_across_dst_not_fixed_86400_seconds() {
    // 09:00 New York changes from 14:00Z to 13:00Z across spring DST.
    let after = chrono::DateTime::parse_from_rfc3339("2026-03-07T13:59:59Z")
        .unwrap()
        .with_timezone(&Utc);
    let first = chrono::DateTime::parse_from_rfc3339("2026-03-07T14:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let next = chrono::DateTime::parse_from_rfc3339("2026-03-08T13:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    for expression in ["daily at 09:00", "0 9 * * *"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("private");
        private_dir(&root);
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let p = provider(false, Some(gate.clone()));
        let rt = runtime(&root.join("runtime.db"), p.clone());
        let path = root.join("scheduler.db");
        let scheduler = Scheduler::open(&path, rt.clone()).unwrap();
        let spec = ScheduleSpec::parse_at(expression, "America/New_York", after).unwrap();
        let j = scheduler
            .add_spec(&scope(), "wall clock".into(), spec.clone(), after)
            .unwrap();
        assert_eq!(j.next_due, first);
        let running = scheduler.clone();
        let worker =
            tokio::spawn(
                async move { running.run_due(&scope(), 1, CancellationToken::new()).await },
            );
        wait_for_call(&p).await;
        // Pause admission, not completion, to inspect exactly the next occurrence.
        scheduler.pause(&scope(), &j.id, true).unwrap();
        gate.add_permits(1);
        let completed = worker.await.unwrap().unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].status, OccurrenceStatus::Completed);
        assert_eq!(scheduler.list(&scope()).unwrap()[0].next_due, next);
        assert_eq!(next - first, Duration::hours(23));
        drop(scheduler);
        let reopened = Scheduler::open(&path, rt).unwrap();
        let persisted = reopened.list(&scope()).unwrap().remove(0);
        assert_eq!(persisted.next_due, next);
        assert_eq!(persisted.schedule_spec, Some(spec));
        assert!(!persisted.enabled);
    }
}

fn create_v1(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let db = rusqlite::Connection::open(path).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA application_id=1297307969; PRAGMA user_version=1;
        CREATE TABLE automation_jobs(id TEXT PRIMARY KEY, scope TEXT NOT NULL, prompt TEXT NOT NULL,
            interval_seconds INTEGER NOT NULL CHECK(interval_seconds > 0), next_due INTEGER NOT NULL,
            enabled INTEGER NOT NULL, removed INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE automation_occurrences(request_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES automation_jobs(id),
            due INTEGER NOT NULL, status TEXT NOT NULL, owner TEXT NOT NULL, fence INTEGER NOT NULL,
            lease_until INTEGER NOT NULL, turn_id TEXT, error TEXT, UNIQUE(job_id,due));").unwrap();
    let due = Utc::now().timestamp() - 1;
    db.execute(
        "INSERT INTO automation_jobs VALUES('legacy',?1,'migrated legacy',3600,?2,1,0)",
        rusqlite::params![serde_json::to_string(&scope()).unwrap(), due],
    )
    .unwrap();
    db.execute("INSERT INTO automation_occurrences VALUES('legacy-completed','legacy',?1,'completed','old',7,0,'old-turn',NULL)", [due-3600]).unwrap();
}

#[tokio::test]
async fn v1_migration_preserves_jobs_history_and_legacy_execution() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let path = root.join("scheduler.db");
    create_v1(&path);
    let p = provider(false, None);
    let rt = runtime(&root.join("runtime.db"), p.clone());
    let scheduler = Scheduler::open(&path, rt.clone()).unwrap();
    let j = scheduler.list(&scope()).unwrap().remove(0);
    assert_eq!(j.id, "legacy");
    assert_eq!(j.prompt, "migrated legacy");
    assert_eq!(j.interval_seconds, 3600);
    assert!(j.schedule_spec.is_none());
    assert_eq!(
        scheduler.occurrences(&scope(), 10).unwrap()[0].request_id,
        "legacy-completed"
    );
    let out = scheduler
        .run_due(&scope(), 1, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].status, OccurrenceStatus::Completed);
    assert_eq!(
        scheduler.list(&scope()).unwrap()[0].next_due,
        j.next_due + Duration::hours(1)
    );
    drop(scheduler);
    let reopened = Scheduler::open(&path, rt).unwrap();
    assert_eq!(reopened.occurrences(&scope(), 10).unwrap().len(), 2);
    assert!(
        reopened
            .run_due(&scope(), 1, CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
    let db = rusqlite::Connection::open(&path).unwrap();
    assert_eq!(
        db.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn migration_open_child() {
    let Ok(root) = std::env::var("MISSY_SCHEDULER_MIGRATION_TEST_ROOT") else {
        return;
    };
    let root = std::path::Path::new(&root);
    let rt = runtime(
        &root.join(format!("runtime-{}.db", std::process::id())),
        provider(false, None),
    );
    let scheduler = Scheduler::open(root.join("scheduler.db"), rt).unwrap();
    assert_eq!(scheduler.list(&scope()).unwrap()[0].id, "legacy");
}

#[test]
fn separate_processes_migrate_v1_atomically() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("private");
    private_dir(&root);
    let path = root.join("scheduler.db");
    create_v1(&path);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    let children: Vec<_> = (0..4)
        .map(|_| {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "migration_open_child", "--nocapture"])
                .env("MISSY_SCHEDULER_MIGRATION_TEST_ROOT", &root)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    std::thread::sleep(std::time::Duration::from_millis(250));
    db.execute_batch("COMMIT").unwrap();
    for child in children {
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "child failed: {} {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        db.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        db.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('automation_jobs') WHERE name='schedule_spec'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}
