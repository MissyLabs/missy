use missy_core::Scope;
use missy_migrate::{MigrationInput, StateDisposition, apply, plan, rollback, verify};
use rusqlite::Connection;
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::{Path, PathBuf},
};

fn private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn directory(path: &Path) {
    fs::create_dir(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn scope() -> Scope {
    Scope {
        tenant: "operator-tenant".into(),
        user: "alice".into(),
        channel: "legacy-discord".into(),
        session: "mapped-session".into(),
    }
}
fn export(t: &Path) -> (PathBuf, MigrationInput) {
    let root = t.join("offline-export");
    directory(&root);
    let map = t.join("mapping.yaml");
    private(
        &map,
        &serde_json::to_vec(&serde_json::json!({"legacy-session":scope()})).unwrap(),
    );
    let input = MigrationInput {
        backup_root: Some(root.clone()),
        mapping_path: Some(map),
        ..Default::default()
    };
    (root, input)
}
fn graph(path: &Path) {
    let db = Connection::open(path).unwrap();
    db.execute_batch(r#"
      PRAGMA foreign_keys=ON;
      CREATE TABLE entities(id TEXT PRIMARY KEY,name TEXT NOT NULL,entity_type TEXT NOT NULL,properties TEXT NOT NULL DEFAULT '{}',first_seen TEXT NOT NULL,last_seen TEXT NOT NULL,mention_count INTEGER NOT NULL DEFAULT 1,UNIQUE(name,entity_type));
      CREATE TABLE relationships(id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES entities(id),target_id TEXT NOT NULL REFERENCES entities(id),relation_type TEXT NOT NULL,weight REAL NOT NULL DEFAULT 0.5,context TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,last_seen TEXT NOT NULL,UNIQUE(source_id,target_id,relation_type));
      INSERT INTO entities VALUES('a','ALICE_GRAPH','person','{}','2026','2026',2);
      INSERT INTO entities VALUES('b','PROJECT_GRAPH','project','{}','2026','2026',1);
      INSERT INTO relationships VALUES('r','a','b','owns',0.5,'GLOBAL_GRAPH_CANARY','2026','2026');
    "#).unwrap();
    drop(db);
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn checkpoints(path: &Path) {
    let db = Connection::open(path).unwrap();
    db.execute_batch(r#"
      CREATE TABLE checkpoints(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,task_id TEXT NOT NULL,prompt TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'RUNNING',loop_messages TEXT NOT NULL DEFAULT '[]',tool_names_used TEXT NOT NULL DEFAULT '[]',iteration INTEGER NOT NULL DEFAULT 0,owner_id TEXT,lease_expires_at REAL,created_at REAL NOT NULL,updated_at REAL NOT NULL);
      INSERT INTO checkpoints VALUES('cp','legacy-session','task','MUTATION_MAY_HAVE_EXECUTED','RUNNING','[{"role":"tool","content":"ambiguous external effect"}]','["shell"]',4,'old-worker',1700000000,1,2);
    "#).unwrap();
    drop(db);
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

#[test]
fn broad_actual_legacy_state_is_durable_but_never_activated_or_replayed() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    graph(&root.join("graph_memory.db"));
    checkpoints(&root.join("checkpoints.db"));
    // Actual ScheduledJob.to_dict field names and top-level array envelope.
    private(&root.join("jobs.json"), br#"[{"id":"job-1","name":"backup","description":"legacy","schedule":"0 9 * * *","task":"BACKUP_TASK_CANARY","provider":"anthropic","enabled":true,"created_at":"2026-11-01T00:00:00+00:00","last_run":"2026-11-04T09:00:00+00:00","next_run":"2026-11-05T09:00:00+00:00","run_count":17,"last_result":"ok","max_attempts":3,"backoff_seconds":[30,60,300],"retry_on":["network","provider_error"],"consecutive_failures":0,"last_error":"","delete_after_run":false,"active_hours":"09:00-17:00","timezone":"America/New_York","capability_mode":"standard","execution_target":"nomad","nomad_request":{"job":{"ID":"batch"}},"nomad_overlap_policy":"skip","last_nomad_job_id":"batch","last_nomad_evaluation_id":"pending-evaluation","nomad_queued_runs":1,"nomad_runs":[{"status":"running"}],"updated_at":"2026-11-04T00:00:00+00:00","last_session_id":"legacy-session","last_cost_usd":0.01,"total_cost_usd":5.5,"last_duration_seconds":4.2}]"#);
    private(&root.join("persona.yaml"), b"version: 3\nname: Missy\ntone: [direct]\npersonality_traits: [curious]\nbehavioral_tendencies: [careful]\nresponse_style_rules: [concise]\nboundaries: [NEVER_LOSE_BOUNDARY]\nidentity_description: operator persona\n");
    private(
        &root.join("trust.json"),
        br#"{"user:alice":1000,"tool:shell":350}"#,
    );
    private(
        &root.join("personal_memories.json"),
        br#"{"alice":{"favorite":"PERSONAL_CANARY"}}"#,
    );
    directory(&root.join("skills"));
    private(
        &root.join("skills/plugin.py"),
        b"raise RuntimeError('MUST_NOT_EXECUTE')\n",
    );
    private(
        &root.join("global_budget.json"),
        br#"{"spent_usd":42,"reserved_usd":7}"#,
    );
    private(
        &root.join("future-operational-file"),
        b"DO_NOT_SILENTLY_DROP",
    );
    let before: Vec<_> = [
        "jobs.json",
        "graph_memory.db",
        "checkpoints.db",
        "persona.yaml",
    ]
    .iter()
    .map(|name| (root.join(name), fs::read(root.join(name)).unwrap()))
    .collect();
    let p = plan(input).unwrap();
    assert!(p.is_applicable(), "{p:?}");
    assert_eq!(p.counts.imported_memories, 0);
    assert_eq!(p.counts.quarantined_records, 8); // 3 graph + 1 checkpoint + 4 structured files
    assert!(p.resolutions.iter().any(|r| r.code == "checkpoint_effect_receipts_missing" && r.required_before_activation));
    assert!(
        p.inventory
            .iter()
            .any(|i| i.category == "unknown_state" && i.archive.is_some())
    );
    let report = serde_json::to_string(&p).unwrap();
    assert!(!report.contains("CANARY") && !report.contains("MUTATION_MAY_HAVE_EXECUTED"));
    let dest = t.path().join("native-import");
    apply(&p, &dest, true).unwrap();
    let manifest = verify(&dest).unwrap();
    assert_eq!(manifest.format_version, 2);
    let q = fs::read_to_string(dest.join("quarantine.json")).unwrap();
    assert!(
        q.contains("NEVER_LOSE_BOUNDARY")
            && q.contains("nomad_queued_runs")
            && q.contains("PERSONAL_CANARY")
    );
    assert!(q.contains("effect_receipts_verified\": false") && q.contains("enabled\": false"));
    for (path, original) in before {
        assert_eq!(fs::read(path).unwrap(), original);
    }
    let db = Connection::open(dest.join("state/runtime.sqlite3")).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn encrypted_vault_keys_oauth_config_and_identity_are_references_only() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    directory(&root.join("secrets"));
    for name in [
        "vault.enc",
        "vault.key",
        "openai-oauth-work.json",
        "web_console.key",
    ] {
        private(
            &root.join("secrets").join(name),
            b"SECRET_BYTES_MUST_NOT_COPY",
        );
    }
    private(&root.join("identity.pem"), b"IDENTITY_MUST_NOT_COPY");
    private(&root.join("config.yaml"), b"api_key: RAW_SECRET_CONFIG\n");
    let p = plan(input).unwrap();
    assert!(p.is_applicable());
    assert_eq!(
        p.inventory
            .iter()
            .filter(|i| i.disposition == StateDisposition::SecretReferenceOnly)
            .count(),
        6
    );
    let dest = t.path().join("native");
    apply(&p, &dest, true).unwrap();
    assert_eq!(fs::read_dir(dest.join("archive")).unwrap().count(), 0);
    for name in [
        "plan.json",
        "manifest.json",
        "quarantine.json",
        "operator-resolution.json",
        "config.yaml",
    ] {
        let text = fs::read_to_string(dest.join(name)).unwrap();
        assert!(
            !text.contains("SECRET_BYTES")
                && !text.contains("IDENTITY_MUST")
                && !text.contains("RAW_SECRET_CONFIG")
        );
    }
    assert!(verify(&dest).is_ok());
}

#[test]
fn inventory_changes_and_wal_journals_prevent_publication() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    private(&root.join("jobs.json"), b"[]");
    let p = plan(input.clone()).unwrap();
    private(&root.join("new-work.json"), b"{}");
    let dest = t.path().join("native");
    assert!(apply(&p, &dest, true).is_err());
    assert!(!dest.exists());
    graph(&root.join("graph_memory.db"));
    private(&root.join("graph_memory.db-wal"), b"");
    assert!(plan(input).is_err());
}

#[test]
fn verification_and_non_destructive_offline_rollback_refuse_corruption() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    private(&root.join("jobs.json"), b"[]");
    let p = plan(input).unwrap();
    let dest = t.path().join("native");
    apply(&p, &dest, true).unwrap();
    let retired = t.path().join("retired-import");
    assert!(rollback(&dest, &retired, false).is_err());
    let original = fs::read(dest.join("config.yaml")).unwrap();
    private(&dest.join("config.yaml"), b"CORRUPTION");
    assert!(verify(&dest).is_err());
    assert!(rollback(&dest, &retired, true).is_err());
    private(&dest.join("config.yaml"), &original);
    rollback(&dest, &retired, true).unwrap();
    assert!(!dest.exists() && retired.exists() && root.exists());
    verify(retired).unwrap();
}

#[test]
fn symlinks_unsafe_directories_missing_scopes_and_oversize_are_refused() {
    let t = tempfile::tempdir().unwrap();
    let (root, mut input) = export(t.path());
    private(&root.join("jobs.json"), b"[]");
    symlink(root.join("jobs.json"), root.join("link")).unwrap();
    assert!(plan(input.clone()).is_err());
    fs::remove_file(root.join("link")).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(plan(input.clone()).is_err());
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    input.limits.max_total_content_bytes = 1;
    assert!(plan(input.clone()).is_err());
    input.limits = Default::default();
    checkpoints(&root.join("checkpoints.db"));
    input.mapping_path = None;
    let p = plan(input).unwrap();
    assert!(!p.is_applicable());
}

#[test]
fn corrupt_unknown_database_and_dangling_graph_fail_consistency_validation() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    graph(&root.join("graph_memory.db"));
    let db = Connection::open(root.join("graph_memory.db")).unwrap();
    db.execute("PRAGMA foreign_keys=OFF", []).unwrap();
    db.execute("UPDATE relationships SET target_id='missing'", [])
        .unwrap();
    drop(db);
    assert!(plan(input.clone()).is_err());
    fs::remove_file(root.join("graph_memory.db")).unwrap();
    private(&root.join("request_tracker.db"), b"SQLite format 3\0broken");
    assert!(plan(input).is_err());
}

#[test]
fn malformed_jobs_preserved_with_useful_resolution_not_dropped() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    private(&root.join("jobs.json"), b"{bad JSON");
    let p = plan(input).unwrap();
    assert!(
        p.resolutions
            .iter()
            .any(|r| r.code == "malformed_legacy_state")
    );
    assert_eq!(
        p.inventory
            .iter()
            .find(|i| i.relative_path == "jobs.json")
            .unwrap()
            .review_records,
        0
    );
    let dest = t.path().join("native");
    let manifest = apply(&p, &dest, true).unwrap();
    let item = manifest
        .inventory
        .iter()
        .find(|i| i.relative_path == "jobs.json")
        .unwrap();
    assert_eq!(
        fs::read(dest.join(item.archive.as_ref().unwrap())).unwrap(),
        b"{bad JSON"
    );
    verify(dest).unwrap();
}

#[test]
fn verify_rejects_unmanifested_files_and_rollback_never_overwrites() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    private(&root.join("jobs.json"), b"[]");
    let dest = t.path().join("native");
    apply(&plan(input).unwrap(), &dest, true).unwrap();
    private(&dest.join("unexpected-effects.json"), b"{}");
    assert!(verify(&dest).is_err());
    fs::remove_file(dest.join("unexpected-effects.json")).unwrap();
    let occupied = t.path().join("occupied");
    directory(&occupied);
    private(&occupied.join("preserve"), b"ORIGINAL");
    assert!(rollback(&dest, &occupied, true).is_err());
    assert!(dest.exists());
    assert_eq!(fs::read(occupied.join("preserve")).unwrap(), b"ORIGINAL");
}

#[test]
fn database_schema_failure_never_exposes_embedded_schema_content() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    let path = root.join("request_tracker.db");
    let db = Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE ordinary(id TEXT); PRAGMA writable_schema=ON;
        UPDATE sqlite_master SET sql='CREATE TABLE SECRET_SCHEMA_CANARY syntax invalid' WHERE name='ordinary';
        PRAGMA writable_schema=OFF;").unwrap();
    drop(db);
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    let error = plan(input).unwrap_err();
    assert!(!format!("{error:#}").contains("SECRET_SCHEMA_CANARY"));
}

#[test]
fn explicit_state_mappings_create_real_paused_jobs_pending_persona_and_scoped_graph() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    let mut other = scope();
    other.user = "bob".into();
    private(
        input.mapping_path.as_ref().unwrap(),
        &serde_json::to_vec(&serde_json::json!({
            "legacy-session":scope(), "state:job:j1":scope(), "state:persona":scope(),
            "state:graph:graph_memory.db":scope(), "state:trust:tool:shell":scope(),
            "state:trust:user:bob":other
        }))
        .unwrap(),
    );
    graph(&root.join("graph_memory.db"));
    private(&root.join("jobs.json"),br#"[{"id":"j1","task":"retained task","schedule":"every 2 hours","timezone":"America/New_York","next_run":"2026-11-05T18:00:00Z","enabled":true,"active_hours":"09:00-17:00","execution_target":"nomad","nomad_queued_runs":3}]"#);
    private(
        &root.join("persona.yaml"),
        b"name: Missy\nboundaries: [NEVER_LOSE_BOUNDARY]\nversion: 7\n",
    );
    private(
        &root.join("trust.json"),
        br#"{"tool:shell":250,"user:bob":1000}"#,
    );
    let p = plan(input).unwrap();
    assert!(p.is_applicable(), "{p:?}");
    let dest = t.path().join("native");
    apply(&p, &dest, true).unwrap();
    verify(&dest).unwrap();
    let db = Connection::open(dest.join("state/automation.sqlite3")).unwrap();
    let (enabled, due, spec): (i64, i64, String) = db
        .query_row(
            "SELECT enabled,next_due,schedule_spec FROM automation_jobs",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(enabled, 0);
    let spec: missy_schedule_text::ScheduleSpec = serde_json::from_str(&spec).unwrap();
    let due = chrono::DateTime::from_timestamp(due, 0).unwrap();
    assert_eq!(
        spec.next_after(due).unwrap() - due,
        chrono::Duration::hours(2)
    );
    let governance =
        missy_governance::ApprovalRegistry::open(dest.join("state/governance.sqlite3")).unwrap();
    assert!(
        governance
            .document(&scope(), missy_governance::DocumentKind::Persona)
            .unwrap()
            .is_none()
    );
    let proposals = governance.document_proposals(&scope(), 10).unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(
        proposals[0].status,
        missy_governance::ProposalStatus::Pending
    );
    assert!(proposals[0].content.contains("NEVER_LOSE_BOUNDARY"));
    assert_eq!(governance.trust(&scope()).unwrap().score, -0.5);
    assert_eq!(governance.trust(&other).unwrap().score, 0.0);
    let graph = missy_retrieval::Retrieval::open(dest.join("state/retrieval.sqlite3")).unwrap();
    let db = Connection::open(dest.join("state/retrieval.sqlite3")).unwrap();
    let id: String = db
        .query_row("SELECT id FROM nodes WHERE name='ALICE_GRAPH'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let traversal = graph.traverse(&scope(), &id, 1, 10).unwrap();
    assert_eq!(traversal.nodes.len(), 2);
    assert_eq!(traversal.edges.len(), 1);
    assert!(traversal.nodes[0].provenance.contains("sha256:"));
    assert!(graph.get_node(&other, &id).unwrap().is_none());
}

#[test]
fn fts_named_operational_table_is_not_silently_discarded() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    graph(&root.join("memory.db"));
    let db = Connection::open(root.join("memory.db")).unwrap();
    db.execute_batch("CREATE TABLE turns_fts_data(id INTEGER, block BLOB); INSERT INTO turns_fts_data VALUES(1,'not derived');").unwrap();
    drop(db);
    let p = plan(input).unwrap();
    assert!(!p.is_applicable());
    assert!(
        p.issues
            .iter()
            .any(|i| i.code == "unverified_derived_table")
    );
}

#[test]
fn paused_one_shot_rounds_up_and_unknown_schedule_stays_explicitly_unported() {
    let t = tempfile::tempdir().unwrap();
    let (root, input) = export(t.path());
    private(
        input.mapping_path.as_ref().unwrap(),
        &serde_json::to_vec(&serde_json::json!({
            "state:job:once":scope(),"state:job:bad":scope()
        }))
        .unwrap(),
    );
    private(&root.join("jobs.json"),br#"[
        {"id":"once","task":"once","schedule":"every 2 hours","timezone":"UTC","next_run":"2026-11-05T18:00:00.100Z","delete_after_run":true},
        {"id":"bad","task":"bad","schedule":"when the sun feels ready","timezone":"UTC","next_run":"2026-11-05T18:00:00Z"}
    ]"#);
    let dest = t.path().join("native");
    apply(&plan(input).unwrap(), &dest, true).unwrap();
    let db = Connection::open(dest.join("state/automation.sqlite3")).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM automation_jobs", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    let (spec, due): (String, i64) = db
        .query_row(
            "SELECT schedule_spec,next_due FROM automation_jobs",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let spec: missy_schedule_text::ScheduleSpec = serde_json::from_str(&spec).unwrap();
    assert!(spec.is_one_shot());
    let expected = chrono::DateTime::parse_from_rfc3339("2026-11-05T18:00:01Z")
        .unwrap()
        .timestamp();
    assert_eq!(due, expected);
    assert!(
        fs::read_to_string(dest.join("portable-state.json"))
            .unwrap()
            .contains("unported_job")
    );
    verify(dest).unwrap();
}
