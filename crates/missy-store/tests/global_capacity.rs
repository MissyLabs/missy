use missy_core::Scope;
use missy_store::Store;
use std::{
    sync::{Arc, Barrier},
    time::Duration,
};
fn scope(n: usize) -> Scope {
    Scope {
        tenant: "t".into(),
        user: format!("u{n}"),
        channel: "c".into(),
        session: "s".into(),
    }
}
#[test]
fn independent_connections_race_for_one_database_global_slot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let first = Store::open(&path).unwrap();
    let second = Store::open(&path).unwrap();
    let a = first.admit(&scope(0), "r", "one").unwrap();
    let b = second.admit(&scope(1), "r", "two").unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = [(first, a, 0), (second, b, 1)]
        .into_iter()
        .map(|(store, admission, n)| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                store
                    .acquire_lease_bounded(
                        &scope(n),
                        &admission.turn.id,
                        "worker",
                        Duration::from_secs(3),
                        1,
                    )
                    .is_ok()
            })
        })
        .collect();
    assert_eq!(
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|ok| *ok)
            .count(),
        1
    );
}
#[test]
fn restart_reclaims_expired_slot_but_stale_owner_cannot_renew() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let first = Store::open(&path).unwrap();
    let a = first.admit(&scope(0), "r", "one").unwrap();
    let lease = first
        .acquire_lease_bounded(&scope(0), &a.turn.id, "old", Duration::from_millis(1), 1)
        .unwrap();
    std::thread::sleep(Duration::from_millis(10));
    let second = Store::open(&path).unwrap();
    let b = second.admit(&scope(1), "r", "two").unwrap();
    let next = second
        .acquire_lease_bounded(&scope(1), &b.turn.id, "new", Duration::from_secs(3), 1)
        .unwrap();
    assert!(
        first
            .heartbeat(&scope(0), &lease, Duration::from_secs(3))
            .is_err()
    );
    assert!(
        first
            .acquire_lease_bounded(&scope(0), &a.turn.id, "retry", Duration::from_secs(3), 1)
            .is_err()
    );
    second.release_lease(&scope(1), &next).unwrap();
    assert!(
        first
            .acquire_lease_bounded(&scope(0), &a.turn.id, "retry", Duration::from_secs(3), 1)
            .is_ok()
    );
}
