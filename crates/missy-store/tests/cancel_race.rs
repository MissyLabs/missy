use missy_core::{Scope, ToolCall};
use missy_store::{Store, ToolStatus, TurnStatus};
use std::time::Duration;

#[test]
fn cancellation_serializes_before_late_success_and_cannot_be_overwritten() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("cancel.sqlite3");
    let worker = Store::open(&path).unwrap();
    let canceller = Store::open(&path).unwrap();
    let scope = Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "s".into(),
    };
    let admission = worker.admit(&scope, "race", "request").unwrap();
    let lease = worker
        .acquire_lease(
            &scope,
            &admission.turn.id,
            "worker",
            Duration::from_secs(30),
        )
        .unwrap();
    canceller
        .request_cancel(&scope, &admission.turn.id)
        .unwrap();
    let settled = worker
        .finish(
            &scope,
            &lease,
            TurnStatus::Completed,
            Some("late response"),
            None,
        )
        .unwrap();
    assert_eq!(settled.status, TurnStatus::Cancelled);
    assert_eq!(
        canceller
            .get_turn(&scope, &admission.turn.id)
            .unwrap()
            .status,
        TurnStatus::Cancelled
    );
    assert!(
        worker
            .finish(
                &scope,
                &lease,
                TurnStatus::Completed,
                Some("even later"),
                None
            )
            .is_err()
    );
    assert!(worker.history(&scope, 100).unwrap().is_empty());
}

#[test]
fn cancelled_turn_blocks_new_dispatch_but_can_record_an_already_running_effect() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path().join("cancel.sqlite3")).unwrap();
    let scope = Scope {
        tenant: "t".into(),
        user: "u".into(),
        channel: "c".into(),
        session: "s".into(),
    };
    let admission = store.admit(&scope, "race", "request").unwrap();
    let lease = store
        .acquire_lease(
            &scope,
            &admission.turn.id,
            "worker",
            Duration::from_secs(30),
        )
        .unwrap();
    let running = ToolCall {
        id: "running".into(),
        name: "effect".into(),
        arguments: serde_json::json!({}),
    };
    let prepared = ToolCall {
        id: "prepared".into(),
        ..running.clone()
    };
    store.tool_begin(&scope, &lease, &running).unwrap();
    store.tool_begin(&scope, &lease, &prepared).unwrap();
    store.tool_running(&scope, &lease, &running.id).unwrap();
    store.request_cancel(&scope, &admission.turn.id).unwrap();
    assert!(store.tool_running(&scope, &lease, &prepared.id).is_err());
    let new_call = ToolCall {
        id: "new".into(),
        ..running.clone()
    };
    assert!(store.tool_begin(&scope, &lease, &new_call).is_err());
    store
        .tool_settle(
            &scope,
            &lease,
            &running.id,
            ToolStatus::Succeeded,
            Some(&serde_json::json!({"applied":true})),
            None,
        )
        .unwrap();
    assert_eq!(
        store.tools(&scope, &admission.turn.id).unwrap()[0].status,
        ToolStatus::Succeeded
    );
}
