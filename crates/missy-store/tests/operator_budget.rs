use missy_core::{Scope, Usage};
use missy_store::{Store, TurnStatus, hash_text};
use std::time::Duration;
fn scope(user: &str) -> Scope {
    Scope {
        tenant: "t".into(),
        user: user.into(),
        channel: "c".into(),
        session: "s".into(),
    }
}
fn usage(cost: u64, tokens: u64, known: bool) -> Usage {
    Usage {
        cost_microusd: cost,
        input_tokens: tokens,
        output_tokens: 0,
        cost_known: known,
    }
}
#[test]
fn cumulative_reservations_restart_unknown_and_downgrade() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let store = Store::open(&path).unwrap();
    let a = store.admit(&scope("a"), "a", "request").unwrap();
    let la = store
        .acquire_lease(&scope("a"), &a.turn.id, "a", Duration::from_secs(10))
        .unwrap();
    store
        .reserve_turn_budget(&scope("a"), &la, usage(80, 100, true), 100, 1000)
        .unwrap();
    let b = store.admit(&scope("b"), "b", "request").unwrap();
    let lb = store
        .acquire_lease(&scope("b"), &b.turn.id, "b", Duration::from_secs(10))
        .unwrap();
    assert!(
        store
            .reserve_turn_budget(&scope("b"), &lb, usage(80, 100, true), 100, 1000)
            .is_err()
    );
    assert!(
        store
            .reserve_turn_budget(&scope("b"), &lb, usage(80, 100, true), 0, 0)
            .is_err()
    );
    store
        .update_turn_budget(&scope("a"), &la, usage(20, 10, true), true)
        .unwrap();
    store
        .reserve_turn_budget(&scope("b"), &lb, usage(80, 100, true), 100, 1000)
        .unwrap();
    store
        .update_turn_budget(&scope("b"), &lb, usage(0, 0, false), false)
        .unwrap();
    store
        .finish(&scope("a"), &la, TurnStatus::Completed, Some("done"), None)
        .unwrap();
    store
        .finish(
            &scope("b"),
            &lb,
            TurnStatus::Interrupted,
            None,
            Some("unknown"),
        )
        .unwrap();
    drop(store);
    let store = Store::open(&path).unwrap();
    assert_eq!(store.budget_totals().unwrap().reserved_cost_microusd, 100);
    assert_eq!(store.budget_totals().unwrap().uncertain_turns, 1);
    assert!(
        store
            .reconcile_billing_and_abandon(
                &scope("b"),
                &b.turn.id,
                "wrong",
                usage(10, 3, true),
                "receipt"
            )
            .is_err()
    );
    store
        .reconcile_billing_and_abandon(
            &scope("b"),
            &b.turn.id,
            &hash_text("request"),
            usage(10, 3, true),
            "receipt",
        )
        .unwrap();
    assert_eq!(store.budget_totals().unwrap().reserved_cost_microusd, 30);
    assert_eq!(store.budget_totals().unwrap().uncertain_turns, 0);
}
#[test]
fn session_cleanup_dry_run_clear_and_tombstone_dedup() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("state.db")).unwrap();
    let s = scope("a");
    store.set_session_display_name(&s, "new title").unwrap();
    assert_eq!(
        store.session_display_name(&s).unwrap().as_deref(),
        Some("new title")
    );
    let a = store.admit(&s, "r", "hello").unwrap();
    assert!(store.session_clear(&s).is_err());
    let lease = store
        .acquire_lease(&s, &a.turn.id, "owner", Duration::from_secs(10))
        .unwrap();
    store
        .append_message(&s, &lease, &missy_core::Message::text("user", "hello"))
        .unwrap();
    assert!(store.session_cleanup(&s, i64::MAX, true).is_err());
    store
        .finish(&s, &lease, TurnStatus::Completed, Some("done"), None)
        .unwrap();
    assert!(store.session_clear(&s).unwrap() > 0);
    assert!(store.history(&s, 100).unwrap().is_empty());
    assert_eq!(store.session_cleanup(&s, i64::MAX, true).unwrap(), 1);
    assert_eq!(store.get_turn(&s, &a.turn.id).unwrap().prompt, "hello");
    assert_eq!(store.session_cleanup(&s, i64::MAX, false).unwrap(), 1);
    assert!(!store.admit(&s, "r", "hello").unwrap().created);
    assert!(store.admit(&s, "r", "different").is_err());
    assert!(store.verify_audit(&s).unwrap() > 0);
}

#[test]
fn display_name_handles_new_empty_and_legacy_nullable_rows() {
    for legacy in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.db");
        let store = Store::open(&path).unwrap();
        if legacy {
            rusqlite::Connection::open(&path).unwrap().execute_batch(
                "DROP TABLE session_metadata; CREATE TABLE session_metadata(scope TEXT PRIMARY KEY,display_name TEXT,history_cutoff INTEGER NOT NULL DEFAULT 0);"
            ).unwrap();
        }
        let s = scope("unnamed");
        assert_eq!(store.session_display_name(&s).unwrap(), None);
        store.session_clear(&s).unwrap();
        assert_eq!(store.session_display_name(&s).unwrap(), None);
        store.set_session_display_name(&s, "named").unwrap();
        assert_eq!(
            store.session_display_name(&s).unwrap().as_deref(),
            Some("named")
        );
    }
}

#[test]
fn expired_parent_reservation_is_not_released_or_bypassed_by_child() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let store = Store::open(&path).unwrap();
    let parent = scope("parent");
    let child = scope("child");
    let a = store.admit(&parent, "parent", "request").unwrap();
    let lease = store
        .acquire_lease(&parent, &a.turn.id, "parent", Duration::from_millis(15))
        .unwrap();
    store
        .reserve_turn_budget(&parent, &lease, usage(80, 100, true), 100, 1000)
        .unwrap();
    store
        .update_turn_budget(&parent, &lease, usage(0, 0, false), false)
        .unwrap();
    std::thread::sleep(Duration::from_millis(30));
    drop(store);
    let store = Store::open(&path).unwrap();
    store.recover(&parent).unwrap();
    let b = store.admit(&child, "child", "request").unwrap();
    let lease_b = store
        .acquire_lease(&child, &b.turn.id, "child", Duration::from_secs(10))
        .unwrap();
    assert!(
        store
            .reserve_turn_budget(&child, &lease_b, usage(1, 1, true), 100, 1000)
            .is_err()
    );
    assert!(
        store
            .reserve_turn_budget(&child, &lease_b, usage(0, 0, true), 0, 0)
            .is_err()
    );
    assert!(
        store
            .reserve_turn_budget(&child, &lease_b, usage(1, 1, true), 200, 2000)
            .is_err()
    );
    let totals = store.budget_totals().unwrap();
    assert_eq!(totals.reserved_cost_microusd, 80);
    assert_eq!(totals.reserved_tokens, 100);
    assert_eq!(totals.uncertain_turns, 1);
    store
        .reconcile_billing_and_abandon(
            &parent,
            &a.turn.id,
            &hash_text("request"),
            usage(5, 5, true),
            "fixture provider receipt",
        )
        .unwrap();
    store
        .reserve_turn_budget(&child, &lease_b, usage(80, 100, true), 100, 1000)
        .unwrap();
}
