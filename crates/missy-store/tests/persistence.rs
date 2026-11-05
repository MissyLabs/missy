use missy_core::{Message, Scope, ToolCall, Usage};
use missy_store::{Checkpoint, Store, ToolBegin, ToolStatus, TurnStatus, hash_text};
use serde_json::json;
use std::{
    sync::{Arc, Barrier},
    time::Duration,
};
fn scope() -> Scope {
    Scope {
        tenant: "tenant".into(),
        user: "alice".into(),
        channel: "api".into(),
        session: "session".into(),
    }
}
fn cp() -> Checkpoint {
    Checkpoint {
        transcript: vec![
            Message::text("system", "trusted system"),
            Message::text("user", "hello"),
            Message::text("assistant", "世界 café"),
        ],
        steps: 1,
        usage: Usage {
            input_tokens: 7,
            output_tokens: 3,
            cost_microusd: 2,
            cost_known: true,
        },
        provider_context: json!({"elapsed_ms":17}),
        context_hash: hash_text("provider/config/v1"),
        system_prompt_hash: hash_text("trusted system"),
    }
}
fn call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "write_file".into(),
        arguments: json!({"text":"αβγ","nested":{"x":1}}),
    }
}
#[test]
fn reopen_wal_checkpoint_journal_and_audit() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("store.db");
    let s = scope();
    let id;
    {
        let db = Store::open(&p).unwrap();
        id = db.admit(&s, "request", "hello").unwrap().turn.id;
        let l = db
            .acquire_lease(&s, &id, "worker", Duration::from_secs(3))
            .unwrap();
        db.save_checkpoint(&s, &l, &cp()).unwrap();
        assert_eq!(
            db.tool_begin(&s, &l, &call("tool1")).unwrap(),
            ToolBegin::Prepared
        );
        db.tool_running(&s, &l, "tool1").unwrap();
        db.tool_settle(
            &s,
            &l,
            "tool1",
            ToolStatus::Succeeded,
            Some(&json!({"ok":"✓"})),
            None,
        )
        .unwrap();
        db.release_lease(&s, &l).unwrap();
        assert!(db.verify_audit(&s).unwrap() > 0);
    }
    let db = Store::open(&p).unwrap();
    assert_eq!(
        db.checkpoint(&s, &id, &cp().context_hash)
            .unwrap()
            .unwrap()
            .transcript,
        cp().transcript
    );
    let l = db
        .acquire_lease(&s, &id, "worker2", Duration::from_secs(3))
        .unwrap();
    assert_eq!(
        db.tool_begin(&s, &l, &call("tool1")).unwrap(),
        ToolBegin::Cached(json!({"ok":"✓"}))
    );
    assert!(!db.admit(&s, "request", "hello").unwrap().created);
    db.finish(&s, &l, TurnStatus::Completed, Some("世界 café"), None)
        .unwrap();
    assert_eq!(db.history(&s, 10).unwrap(), cp().transcript[1..]);
    assert!(
        db.finish(&s, &l, TurnStatus::Completed, None, None)
            .is_err()
    );
    assert_eq!(db.history(&s, 10).unwrap().len(), 2);
    assert!(db.verify_audit(&s).unwrap() > 0);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
        for suffix in ["-wal", "-shm"] {
            let side = std::path::PathBuf::from(format!("{}{suffix}", p.display()));
            assert_eq!(
                std::fs::metadata(side).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}
#[test]
fn concurrent_admission_two_connections_is_atomic() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("store.db");
    let a = Arc::new(Store::open(&p).unwrap());
    let b = Arc::new(Store::open(&p).unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = vec![];
    for db in [a, b] {
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            db.admit(&scope(), "same-id", "same-payload").unwrap()
        }));
    }
    let a = threads.remove(0).join().unwrap();
    let b = threads.remove(0).join().unwrap();
    assert_eq!(a.turn.id, b.turn.id);
    assert_ne!(a.created, b.created);
}
#[test]
fn request_binding_scope_denial_and_session_lease() {
    let d = tempfile::tempdir().unwrap();
    let db = Store::open(d.path().join("state.db")).unwrap();
    let s = scope();
    let a = db.admit(&s, "r", "hello").unwrap();
    assert!(db.admit(&s, "r", "changed").is_err());
    for field in 0..4 {
        let mut other = s.clone();
        match field {
            0 => other.tenant = "other".into(),
            1 => other.user = "bob".into(),
            2 => other.channel = "discord".into(),
            _ => other.session = "other".into(),
        };
        assert!(db.get_turn(&other, &a.turn.id).is_err());
        assert!(db.events(&other, &a.turn.id).is_err());
        assert!(db.tool_journal(&other, &a.turn.id).is_err());
        assert!(
            db.checkpoint(&other, &a.turn.id, &cp().context_hash)
                .is_err()
        );
        assert!(db.request_cancel(&other, &a.turn.id).is_err());
        assert!(
            db.acquire_lease(&other, &a.turn.id, "worker", Duration::from_secs(1))
                .is_err()
        );
        assert!(db.admit(&other, "r", "hello").unwrap().created);
    }
    let l = db
        .acquire_lease(&s, &a.turn.id, "worker", Duration::from_secs(3))
        .unwrap();
    let second = db.admit(&s, "r2", "next").unwrap();
    assert!(
        db.acquire_lease(&s, &second.turn.id, "worker2", Duration::from_secs(3))
            .is_err()
    );
    let mut wrong = s.clone();
    wrong.user = "bob".into();
    assert!(db.heartbeat(&wrong, &l, Duration::from_secs(3)).is_err());
    assert!(
        db.finish(&wrong, &l, TurnStatus::Completed, None, None)
            .is_err()
    );
}
#[test]
fn stale_fence_and_expired_heartbeat_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("store.db");
    let a = Store::open(&p).unwrap();
    let b = Store::open(&p).unwrap();
    let s = scope();
    let id = a.admit(&s, "r", "hello").unwrap().turn.id;
    let old = a
        .acquire_lease(&s, &id, "owner", Duration::from_millis(20))
        .unwrap();
    std::thread::sleep(Duration::from_millis(40));
    assert!(a.heartbeat(&s, &old, Duration::from_secs(10)).is_err());
    let new = b
        .acquire_lease(&s, &id, "new", Duration::from_secs(10))
        .unwrap();
    assert!(new.fence > old.fence);
    assert!(a.release_lease(&s, &old).is_err());
    assert!(a.save_checkpoint(&s, &old, &cp()).is_err());
    assert!(a.tool_begin(&s, &old, &call("x")).is_err());
    assert!(
        a.append_message(&s, &old, &Message::text("assistant", "stale"))
            .is_err()
    );
    assert!(
        a.finish(&s, &old, TurnStatus::Completed, Some("stale"), None)
            .is_err()
    );
    b.heartbeat(&s, &new, Duration::from_secs(10)).unwrap();
}
#[test]
fn uncertain_tools_never_replay_and_live_worker_not_recovered() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("store.db");
    let s = scope();
    let id;
    {
        let db = Store::open(&p).unwrap();
        id = db.admit(&s, "r", "hello").unwrap().turn.id;
        let l = db
            .acquire_lease(&s, &id, "crasher", Duration::from_millis(20))
            .unwrap();
        db.tool_begin(&s, &l, &call("x")).unwrap();
        db.tool_running(&s, &l, "x").unwrap();
        assert!(db.recover(&s).unwrap().is_empty());
    }
    std::thread::sleep(Duration::from_millis(40));
    let db = Store::open(&p).unwrap();
    assert_eq!(db.recover(&s).unwrap().len(), 1);
    assert_eq!(
        db.unresolved_tools(&s, &id).unwrap()[0].status,
        ToolStatus::OutcomeUnknown
    );
    let l = db
        .acquire_lease(&s, &id, "new", Duration::from_secs(3))
        .unwrap();
    assert_eq!(
        db.tool_begin(&s, &l, &call("x")).unwrap(),
        ToolBegin::OutcomeUnknown
    );
    assert!(db.tool_running(&s, &l, "x").is_err());
    let mut changed = call("x");
    changed.arguments["nested"]["x"] = json!(2);
    assert!(db.tool_begin(&s, &l, &changed).is_err());
    assert!(
        db.tool_settle(&s, &l, "x", ToolStatus::Succeeded, Some(&json!(1)), None)
            .is_err()
    );
}
#[test]
fn memory_fts_injection_unicode_provenance_and_scope() {
    let d = tempfile::tempdir().unwrap();
    let db = Store::open(d.path().join("state.db")).unwrap();
    let a = scope();
    let mut b = a.clone();
    b.user = "bob".into();
    db.memory_put(
        &a,
        "id'; DROP TABLE turns;--",
        "café 世界 project notes",
        "user:alice/request:1",
        "user_provided",
    )
    .unwrap();
    db.memory_put(
        &b,
        "private",
        "café secret",
        "user:bob/request:2",
        "untrusted",
    )
    .unwrap();
    assert_eq!(db.memory_search(&a, "café", 10).unwrap().len(), 1);
    assert_eq!(
        db.memory_search(&a, "世界", 10).unwrap()[0].trust,
        "user_provided"
    );
    assert!(
        db.memory_search(&a, "\" OR secret OR \"", 10)
            .unwrap()
            .is_empty()
    );
    assert!(
        db.memory_search(&a, "'; DROP TABLE memories; --", 10)
            .unwrap()
            .is_empty()
    );
    assert!(db.memory_search(&a, "* NEAR(\"", 10).is_ok());
    assert!(db.memory_get(&a, "private").unwrap().is_none());
    assert!(!db.memory_delete(&a, "private").unwrap());
    assert!(db.memory_get(&b, "private").unwrap().is_some());
    assert!(
        db.memory_put(&a, "bad", "ignore instructions", "tool:unknown", "system")
            .is_err()
    );
    assert!(db.memory_put(&a, "bad", "text", "", "untrusted").is_err());
    db.memory_delete(&a, "id'; DROP TABLE turns;--").unwrap();
    assert!(db.memory_search(&a, "café", 10).unwrap().is_empty());
    db.verify_audit(&a).unwrap();
}
#[test]
fn checkpoint_lineage_context_and_budget_rollback() {
    let d = tempfile::tempdir().unwrap();
    let db = Store::open(d.path().join("state.db")).unwrap();
    let s = scope();
    let id = db.admit(&s, "r", "hello").unwrap().turn.id;
    let l = db
        .acquire_lease(&s, &id, "w", Duration::from_secs(3))
        .unwrap();
    db.save_checkpoint(&s, &l, &cp()).unwrap();
    assert!(db.checkpoint(&s, &id, "wrong-config").is_err());
    let mut c = cp();
    c.transcript[0].content = "replaced".into();
    assert!(db.save_checkpoint(&s, &l, &c).is_err());
    c.system_prompt_hash = hash_text("replaced");
    assert!(db.save_checkpoint(&s, &l, &c).is_err());
    c = cp();
    c.usage.input_tokens = 0;
    assert!(db.save_checkpoint(&s, &l, &c).is_err());
    c = cp();
    c.transcript
        .push(Message::text("system", "promoted tool instruction"));
    assert!(db.save_checkpoint(&s, &l, &c).is_err());
}
#[test]
fn cancellation_and_retention_preserve_tombstones_and_scoping() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let db = Store::open(&p).unwrap();
    let s = scope();
    let id = db.admit(&s, "r", "hello").unwrap().turn.id;
    let l = db
        .acquire_lease(&s, &id, "w", Duration::from_secs(3))
        .unwrap();
    db.request_cancel(&s, &id).unwrap();
    assert!(db.cancellation_requested(&s, &id).unwrap());
    db.finish(&s, &l, TurnStatus::Cancelled, None, None)
        .unwrap();
    db.memory_put(&s, "id", "memory", "user:alice", "untrusted")
        .unwrap();
    let mut b = s.clone();
    b.user = "bob".into();
    db.memory_put(&b, "id", "memory", "user:bob", "untrusted")
        .unwrap();
    db.retention(&s, i64::MAX).unwrap();
    assert!(db.memory_get(&s, "id").unwrap().is_none());
    assert!(db.memory_get(&b, "id").unwrap().is_some());
    assert!(!db.admit(&s, "r", "hello").unwrap().created);
    assert!(db.admit(&s, "r", "changed").is_err());
    db.verify_audit(&s).unwrap();
    drop(db);
    let db = Store::open(&p).unwrap();
    assert!(db.cancellation_requested(&s, &id).unwrap());
}
#[test]
fn migration_v1_to_v2_preserves_memory_and_rejects_future_schema() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    {
        let c = rusqlite::Connection::open(&p).unwrap();
        c.execute_batch(include_str!("../src/schema-v1.sql"))
            .unwrap();
        c.pragma_update(None, "user_version", 1).unwrap();
        c.execute("INSERT INTO memories(scope,id,content,provenance,trust,created_at,updated_at) VALUES(?1,'id','café migrated','fixture','untrusted',1,1)",[serde_json::to_string(&scope()).unwrap()]).unwrap();
    }
    let db = Store::open(&p).unwrap();
    assert_eq!(db.memory_search(&scope(), "migrated", 10).unwrap().len(), 1);
    drop(db);
    let c = rusqlite::Connection::open(&p).unwrap();
    c.pragma_update(None, "user_version", 999).unwrap();
    drop(c);
    assert!(Store::open(&p).is_err());
}
#[test]
#[cfg(unix)]
fn reject_symlink_hardlink_and_public_parent() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let other = d.path().join("other.db");
    std::fs::write(&other, []).unwrap();
    symlink(&other, &p).unwrap();
    assert!(Store::open(&p).is_err());
    std::fs::remove_file(&p).unwrap();
    std::fs::hard_link(&other, &p).unwrap();
    assert!(Store::open(&p).is_err());
    std::fs::remove_file(&p).unwrap();
    std::fs::set_permissions(d.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(Store::open(&p).is_err());
}
#[test]
fn checkpoint_corruption_and_audit_mutation_fail_closed() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let db = Store::open(&p).unwrap();
    let s = scope();
    let id = db.admit(&s, "r", "hello").unwrap().turn.id;
    let l = db
        .acquire_lease(&s, &id, "worker", Duration::from_secs(3))
        .unwrap();
    db.save_checkpoint(&s, &l, &cp()).unwrap();
    let raw = rusqlite::Connection::open(&p).unwrap();
    let mut changed = cp();
    changed.provider_context = json!({"elapsed_ms":0});
    raw.execute(
        "UPDATE checkpoints SET data=?1 WHERE turn_id=?2",
        rusqlite::params![serde_json::to_string(&changed).unwrap(), id],
    )
    .unwrap();
    assert!(db.checkpoint(&s, &id, &cp().context_hash).is_err());
    assert!(db.save_checkpoint(&s, &l, &cp()).is_err());
    assert!(
        db.finish(&s, &l, TurnStatus::Completed, None, None)
            .is_err()
    );
    assert!(raw.execute("DELETE FROM audit", []).is_err());
    assert!(raw.execute("UPDATE audit SET kind='forged'", []).is_err());
    raw.execute_batch(
        "DROP TRIGGER audit_no_update; UPDATE audit SET kind='forged' WHERE sequence=1;",
    )
    .unwrap();
    assert!(db.verify_audit(&s).is_err());
}
#[test]
fn process_fixture() {
    let Ok(mode) = std::env::var("MISSY_STORE_TEST_MODE") else {
        return;
    };
    let p = std::env::var_os("MISSY_STORE_TEST_PATH").unwrap();
    let db = Store::open(std::path::PathBuf::from(p)).unwrap();
    let a = db.admit(&scope(), "process-request", "hello").unwrap();
    if mode == "crash" {
        let l = db
            .acquire_lease(
                &scope(),
                &a.turn.id,
                "crashed-child",
                Duration::from_millis(100),
            )
            .unwrap();
        db.save_checkpoint(&scope(), &l, &cp()).unwrap();
        db.tool_begin(&scope(), &l, &call("effect")).unwrap();
        db.tool_running(&scope(), &l, "effect").unwrap();
        #[cfg(unix)]
        unsafe {
            libc::_exit(0)
        };
        #[cfg(not(unix))]
        std::process::exit(0);
    }
    println!("STORE_ADMISSION:{}:{}", a.turn.id, a.created);
}
#[test]
fn abrupt_child_exit_preserves_intents_wal_and_continuation() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "process_fixture", "--nocapture"])
        .env("MISSY_STORE_TEST_MODE", "crash")
        .env("MISSY_STORE_TEST_PATH", &p)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    std::thread::sleep(Duration::from_millis(150));
    let db = Store::open(&p).unwrap();
    let s = scope();
    let recovered = db.recover(&s).unwrap();
    assert_eq!(recovered.len(), 1);
    let id = &recovered[0].id;
    assert_eq!(
        db.unresolved_tools(&s, id).unwrap()[0].status,
        ToolStatus::OutcomeUnknown
    );
    let saved = db.checkpoint(&s, id, &cp().context_hash).unwrap().unwrap();
    assert_eq!(saved.provider_context, json!({"elapsed_ms":17}));
    assert_eq!(saved.usage.input_tokens, 7);
    assert!(db.verify_audit(&s).unwrap() > 0);
}
#[test]
fn cross_process_admission_returns_one_created() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    drop(Store::open(&p).unwrap());
    let mut children = vec![];
    for _ in 0..4 {
        children.push(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "process_fixture", "--nocapture"])
                .env("MISSY_STORE_TEST_MODE", "admit")
                .env("MISSY_STORE_TEST_PATH", &p)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let mut created = 0;
    let mut ids = vec![];
    for child in children {
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let line = text
            .lines()
            .find(|l| l.starts_with("STORE_ADMISSION:"))
            .unwrap();
        let fields: Vec<_> = line.split(':').collect();
        ids.push(fields[1].to_owned());
        if fields[2] == "true" {
            created += 1
        }
    }
    assert_eq!(created, 1);
    assert!(ids.iter().all(|id| id == &ids[0]));
}
#[test]
fn migrations_preserve_v2_checkpoint_and_foreign_db_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let s = scope();
    let id = "legacy-turn";
    {
        let c = rusqlite::Connection::open(&p).unwrap();
        c.execute_batch(include_str!("../src/schema-v1.sql"))
            .unwrap();
        c.execute_batch("CREATE VIRTUAL TABLE memory_fts USING fts5(content,tokenize='unicode61'); PRAGMA user_version=2;").unwrap();
        let key = serde_json::to_string(&s).unwrap();
        c.execute("INSERT INTO turns(id,scope,request_id,payload_hash,prompt,status,created_at,updated_at) VALUES(?1,?2,'r',?3,'hello','interrupted',1,1)",rusqlite::params![id,key,hash_text("hello")]).unwrap();
        c.execute("INSERT INTO checkpoints(scope,turn_id,context_hash,system_prompt_hash,data) VALUES(?1,?2,?3,?4,?5)",rusqlite::params![key,id,cp().context_hash,cp().system_prompt_hash,serde_json::to_string(&cp()).unwrap()]).unwrap();
    }
    let db = Store::open(&p).unwrap();
    assert_eq!(
        db.checkpoint(&s, id, &cp().context_hash)
            .unwrap()
            .unwrap()
            .transcript,
        cp().transcript
    );
    drop(db);
    let foreign = d.path().join("foreign.db");
    let c = rusqlite::Connection::open(&foreign).unwrap();
    c.execute_batch("PRAGMA application_id=1234; CREATE TABLE private_data(id);")
        .unwrap();
    drop(c);
    assert!(Store::open(&foreign).is_err());
    let unrecognized = d.path().join("unknown.db");
    let c = rusqlite::Connection::open(&unrecognized).unwrap();
    c.execute_batch("CREATE TABLE unknown(id);").unwrap();
    drop(c);
    assert!(Store::open(&unrecognized).is_err());
}
#[test]
fn prepared_unknown_and_new_id_are_blocked_after_recovery() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("state.db");
    let db = Store::open(&p).unwrap();
    let s = scope();
    let id = db.admit(&s, "r", "hello").unwrap().turn.id;
    let l = db
        .acquire_lease(&s, &id, "w", Duration::from_secs(3))
        .unwrap();
    assert_eq!(
        db.tool_begin(&s, &l, &call("old")).unwrap(),
        ToolBegin::Prepared
    );
    assert_eq!(
        db.tool_begin(&s, &l, &call("old")).unwrap(),
        ToolBegin::OutcomeUnknown
    );
    assert_eq!(db.unresolved_tools(&s, &id).unwrap().len(), 1);
    assert!(
        db.finish(&s, &l, TurnStatus::Completed, Some("false claim"), None)
            .is_err()
    );
    db.release_lease(&s, &l).unwrap();
    let new = db
        .acquire_lease(&s, &id, "new", Duration::from_secs(3))
        .unwrap();
    assert!(db.tool_begin(&s, &new, &call("fresh-id")).is_err());
    assert!(db.tool_running(&s, &new, "old").is_err());
    assert_eq!(
        db.unresolved_tools(&s, &id).unwrap()[0].status,
        ToolStatus::OutcomeUnknown
    );
}
