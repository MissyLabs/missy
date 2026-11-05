use super::*;
#[tokio::test]
async fn checkpoint_and_raw_receipts_expand_and_summarize_after_reopen_without_search() {
    let (_d, s, t, ctx) = setup();
    let cfg = t.config.clone();
    let turn = s.admit(&ctx.scope, "pending", "ordinary").unwrap().turn;
    let lease = s
        .acquire_lease(&ctx.scope, &turn.id, "fixture", Duration::from_secs(60))
        .unwrap();
    let cp = missy_store::Checkpoint {
        transcript: vec![
            Message::text("system", "excluded-policy"),
            Message::text("assistant", "needle checkpoint 水"),
        ],
        steps: 1,
        usage: Usage::default(),
        provider_context: json!({}),
        context_hash: "fixture-context".into(),
        system_prompt_hash: hash_text("excluded-policy"),
    };
    s.save_checkpoint(&ctx.scope, &lease, &cp).unwrap();
    s.append_message(
        &ctx.scope,
        &lease,
        &Message::tool("raw-call", "needle raw é"),
    )
    .unwrap();
    let cp_row = search(&t, &ctx, "checkpoint").await["results"][0].clone();
    let raw_row = search(&t, &ctx, "raw").await["results"][0].clone();
    assert_eq!(cp_row["turn_id"], turn.id);
    assert_eq!(raw_row["turn_id"], Value::Null);
    assert_eq!(raw_row["tool_id"], "raw-call");
    assert!(
        search(&t, &ctx, "excluded-policy").await["results"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    drop(t);
    // Receipt selection is precisely what a durable pending approval stores.
    let rebuilt = HistoryTools::new(s, cfg).unwrap();
    for row in [&cp_row, &raw_row] {
        let value=rebuilt.execute(&ctx,&call("memory_expand",json!({"item_id":row["item_id"],"content_sha256":row["content_sha256"],"start_byte":0,"end_byte":row["content_bytes"]}))).await.unwrap();
        assert_eq!(value["text"], row["text"]);
    }
    let p = Arc::new(MockProvider {
        calls: AtomicUsize::new(0),
        tool_call: false,
        unknown_cost: false,
    });
    let rebuilt = rebuilt
        .with_summary_backend(SummaryBackend::new(p.clone(), policy(&_d)).unwrap())
        .unwrap();
    let summary = rebuilt
        .execute(
            &ctx,
            &call(
                "context_shunt",
                json!({"query":"old selection","item_ids":[cp_row["item_id"]]}),
            ),
        )
        .await
        .unwrap();
    assert_eq!(summary["provider_calls"], 1);
    assert_eq!(p.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn durable_receipts_reopen_preserve_snapshot_after_source_change_and_scope_denial() {
    let (d, s, t, ctx) = setup();
    let cfg = t.config.clone();
    turn(&s, &ctx.scope, "durable", "needle original 水", "response");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let old_id = row["item_id"].clone();
    drop(t);
    let reopened = HistoryTools::new(s.clone(), cfg.clone()).unwrap();
    assert_eq!(
        search(&reopened, &ctx, "needle").await["results"][0]["item_id"],
        old_id
    );
    assert_eq!(row["retention"], "durable_immutable_snapshot");
    let source = rusqlite::Connection::open(d.path().join("history.sqlite")).unwrap();
    source
        .execute(
            "UPDATE turns SET prompt='changed source' WHERE id=?1",
            [row["turn_id"].as_str().unwrap()],
        )
        .unwrap();
    drop(reopened);
    let reopened = HistoryTools::new(s, cfg).unwrap();
    let expand = call(
        "memory_expand",
        json!({"item_id":old_id,"content_sha256":row["content_sha256"],"start_byte":0,"end_byte":row["content_bytes"]}),
    );
    assert_eq!(
        reopened.execute(&ctx, &expand).await.unwrap()["text"],
        "needle original 水"
    );
    for dimension in 0..4 {
        let mut other = ctx.clone();
        match dimension {
            0 => other.scope.tenant.push('x'),
            1 => other.scope.user.push('x'),
            2 => other.scope.channel.push('x'),
            _ => other.scope.session.push('x'),
        }
        assert!(reopened.execute(&other, &expand).await.is_err());
    }
}
#[tokio::test]
async fn retention_global_capacity_preserves_existing_receipts_across_reopen() {
    let (_d, s, t, ctx) = setup();
    let mut cfg = t.config.clone();
    cfg.max_retained_items = 1;
    drop(t);
    let t = HistoryTools::new(s.clone(), cfg.clone()).unwrap();
    turn(&s, &ctx.scope, "one", "first needle", "");
    let row = search(&t, &ctx, "first").await["results"][0].clone();
    turn(&s, &ctx.scope, "two", "second needle", "");
    assert!(
        t.execute(&ctx, &call("history_search", json!({"query":"second"})))
            .await
            .unwrap_err()
            .to_string()
            .contains("capacity exhausted")
    );
    drop(t);
    let t = HistoryTools::new(s, cfg).unwrap();
    assert_eq!(
        t.execute(
            &ctx,
            &call("memory_describe", json!({"item_ids":[row["item_id"]]}))
        )
        .await
        .unwrap()["source_count"],
        1
    );
}
#[test]
fn retention_requires_path_disabled_does_not_create_and_refuses_unsafe_files() {
    use std::os::unix::fs::symlink;
    let (d, s, t, _ctx) = setup();
    let mut cfg = t.config.clone();
    let missing = d.path().join("disabled.sqlite");
    cfg.enabled = false;
    cfg.retention_path = Some(missing.clone());
    HistoryTools::new(s.clone(), cfg.clone()).unwrap();
    assert!(!missing.exists());
    cfg.enabled = true;
    cfg.retention_path = None;
    assert!(HistoryTools::new(s.clone(), cfg.clone()).is_err());
    let real = t.config.retention_path.clone().unwrap();
    cfg.retention_path = Some(d.path().join("linked.sqlite"));
    symlink(&real, cfg.retention_path.as_ref().unwrap()).unwrap();
    assert!(HistoryTools::new(s.clone(), cfg.clone()).is_err());
    std::fs::remove_file(cfg.retention_path.as_ref().unwrap()).unwrap();
    std::fs::hard_link(&real, cfg.retention_path.as_ref().unwrap()).unwrap();
    assert!(HistoryTools::new(s.clone(), cfg.clone()).is_err());
    std::fs::remove_file(cfg.retention_path.as_ref().unwrap()).unwrap();
    let alias = d.path().join("alias");
    symlink(d.path(), &alias).unwrap();
    cfg.retention_path = Some(alias.join("new.sqlite"));
    assert!(HistoryTools::new(s.clone(), cfg.clone()).is_err());
    cfg.retention_path = Some(d.path().join("corrupt.sqlite"));
    std::fs::write(cfg.retention_path.as_ref().unwrap(), b"not sqlite").unwrap();
    std::fs::set_permissions(
        cfg.retention_path.as_ref().unwrap(),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(HistoryTools::new(s, cfg).is_err());
}
#[tokio::test]
async fn retained_payload_corruption_and_inode_replacement_refused() {
    let (d, s, t, ctx) = setup();
    turn(&s, &ctx.scope, "one", "needle", "");
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    let cfg = t.config.clone();
    let path = cfg.retention_path.clone().unwrap();
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute("UPDATE evidence SET payload='{}'", [])
        .unwrap();
    assert!(
        t.execute(
            &ctx,
            &call("memory_describe", json!({"item_ids":[row["item_id"]]}))
        )
        .await
        .is_err()
    );
    drop(conn);
    drop(t);
    assert!(HistoryTools::new(s.clone(), cfg.clone()).is_err());
    std::fs::remove_file(&path).unwrap();
    let t = HistoryTools::new(s.clone(), cfg.clone()).unwrap();
    let row = search(&t, &ctx, "needle").await["results"][0].clone();
    std::fs::rename(&path, d.path().join("old-evidence.sqlite")).unwrap();
    let _replacement = HistoryTools::new(s, cfg).unwrap();
    assert!(
        t.execute(
            &ctx,
            &call("memory_describe", json!({"item_ids":[row["item_id"]]}))
        )
        .await
        .is_err()
    );
}
