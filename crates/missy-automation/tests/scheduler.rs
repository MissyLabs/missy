use anyhow::Result;
use async_trait::async_trait;
use chrono::{Duration, Utc};
use missy_automation::{NewJob, OccurrenceStatus, Scheduler};
use missy_core::{
    Completion, Message, Provider, RuntimeConfig, Scope, ToolCall, ToolContext, ToolDefinition,
    ToolExecutor, Usage,
};
use missy_runtime::Runtime;
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
