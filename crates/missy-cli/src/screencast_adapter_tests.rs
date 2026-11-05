use super::*;
fn scope() -> Scope {
    Scope {
        tenant: "test".into(),
        user: "operator".into(),
        channel: "screen".into(),
        session: "parent".into(),
    }
}
fn fixture() -> (tempfile::TempDir, Config) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("screencast.json");
    let cfg = AdapterConfig {
        channel: ScreencastConfig {
            enabled: true,
            ..Default::default()
        },
        vision: None,
        analysis_prompt: default_prompt(),
        artifact_root: None,
        max_artifact_files: 1000,
    };
    fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    let mut app = Config {
        state_dir: dir.path().join("state"),
        ..Default::default()
    };
    app.extensions.screencast_config = Some(path);
    (dir, app)
}
#[tokio::test]
async fn durable_operator_pairing_scope_owner_modes_and_cancel() {
    let (_dir, cfg) = fixture();
    assert!(
        execute(
            &cfg,
            ScreencastCommand::Approve {
                confirm: false,
                ttl_seconds: 30,
                analyze: false,
                deliver_analysis: false,
                deliver_frame_artifacts: false
            },
            scope()
        )
        .await
        .is_err()
    );
    execute(
        &cfg,
        ScreencastCommand::Approve {
            confirm: true,
            ttl_seconds: 30,
            analyze: false,
            deliver_analysis: false,
            deliver_frame_artifacts: false,
        },
        scope(),
    )
    .await
    .unwrap();
    let prepared = prepare(&cfg).unwrap().unwrap();
    let receipt: String = prepared
        .authority
        .connection()
        .unwrap()
        .query_row("SELECT receipt FROM grants", [], |r| r.get(0))
        .unwrap();
    prepared.poll().await.unwrap();
    let path = prepared.authority.pairings.join(format!("{receipt}.json"));
    assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let id = value["session_id"].as_str().unwrap();
    assert_eq!(
        prepared.channel.status(&scope(), id).unwrap().phase,
        missy_screencast::SessionPhase::Paired
    );
    let mut wrong = scope();
    wrong.user = "other".into();
    assert!(
        execute(
            &cfg,
            ScreencastCommand::Cancel {
                receipt: receipt.clone()
            },
            wrong
        )
        .await
        .is_err()
    );
    execute(
        &cfg,
        ScreencastCommand::Cancel {
            receipt: receipt.clone(),
        },
        scope(),
    )
    .await
    .unwrap();
    prepared.poll().await.unwrap();
    assert_eq!(
        prepared.channel.status(&scope(), id).unwrap().phase,
        missy_screencast::SessionPhase::Closed
    );
    assert!(
        !prepared
            .authority
            .pairings
            .join(format!("{receipt}.json"))
            .exists()
    );
}
#[tokio::test]
async fn one_time_durable_approval_config_digest_and_parent_frame_reservation() {
    let (_dir, cfg) = fixture();
    let prepared = prepare(&cfg).unwrap().unwrap();
    let receipt = Uuid::new_v4().to_string();
    let grant = SessionApproval {
        scope: scope(),
        approval_receipt: receipt.clone(),
        ttl_seconds: 30,
        analyze: true,
        deliver_analysis: false,
        deliver_frame_artifacts: false,
    };
    prepared.authority.connection().unwrap().execute("INSERT INTO grants(receipt,scope,approval,fingerprint,expires,phase) VALUES(?1,?2,?3,?4,?5,'pending')",
        params![receipt,scope_json(&scope()).unwrap(),serde_json::to_string(&grant).unwrap(),prepared.authority.fingerprint,now()+30]).unwrap();
    assert!(
        prepared
            .authority
            .verify(&grant, "wrong-config")
            .await
            .is_err()
    );
    prepared
        .authority
        .verify(&grant, &prepared.authority.channel_fingerprint)
        .await
        .unwrap();
    assert!(
        prepared
            .authority
            .verify(&grant, &prepared.authority.channel_fingerprint)
            .await
            .is_err()
    );
    prepared
        .authority
        .connection()
        .unwrap()
        .execute(
            "UPDATE grants SET session_id='fixture' WHERE receipt=?1",
            [&receipt],
        )
        .unwrap();
    let frame = FrameRequest {
        context: missy_core::ToolContext {
            scope: scope(),
            turn_id: "screencast:fixture:1".into(),
            cancellation: CancellationToken::new(),
        },
        approval_receipt: receipt.clone(),
        session_id: "fixture".into(),
        frame_number: 1,
        sha256: hex::encode(Sha256::digest([])),
        width: 1,
        height: 1,
        png: vec![],
    };
    let reservation = prepared.authority.reserve(&frame, "prompt").await.unwrap();
    assert!(prepared.authority.reserve(&frame, "prompt").await.is_err());
    prepared
        .authority
        .finish(&reservation, false)
        .await
        .unwrap();
    assert!(prepared.authority.reserve(&frame, "prompt").await.is_err());
    let mut other = frame;
    other.context.turn_id = "screencast:fixture:2".into();
    other.context.scope.user = "alien".into();
    assert!(prepared.authority.reserve(&other, "prompt").await.is_err());
}
#[test]
fn symlink_and_permissive_authority_roots_denied() {
    let (dir, cfg) = fixture();
    fs::create_dir(&cfg.state_dir).unwrap();
    let leaf = cfg.state_dir.join("screencast");
    fs::create_dir(&leaf).unwrap();
    fs::set_permissions(&leaf, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(prepare(&cfg).is_err());
    fs::remove_dir(&leaf).unwrap();
    std::os::unix::fs::symlink(dir.path(), &leaf).unwrap();
    assert!(prepare(&cfg).is_err());
}
