//! Native, non-activating portability assets. Explicit mappings alone determine
//! ownership; unsupported operational semantics remain blocking review items.
use super::*;
use chrono::DateTime;
use missy_governance::{ApprovalRegistry, DocumentKind, OperationOutcome};
use missy_retrieval::{GraphEdge, GraphNode, Retrieval};
use missy_schedule_text::ScheduleSpec;
use rusqlite::{Connection, params};

fn provenance(hash: &str, table: &str, id: &str) -> String {
    format!(
        "offline_legacy:sha256:{hash}:table:{table}:id_sha256:{}",
        digest(id.as_bytes())
    )
}
fn graph_scope<'a>(p: &'a MigrationPlan, hash: &str) -> Option<&'a Scope> {
    p.sources
        .iter()
        .filter(|source| source.kind == "database" && source.sha256 == hash)
        .find_map(|source| {
            source
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| p.mappings.get(&format!("state:graph:{name}")))
        })
}
fn value<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn write(p: &MigrationPlan, stage: &Path) -> Result<Vec<String>> {
    let mut artifacts = vec![];
    let mut receipts = vec![];
    let mut governance = None;
    let mut retrieval = None;
    let mut scheduler = None;
    for row in &p.rows {
        let hash = value(&row.row, "_migration_source_sha256");
        if row.table == "entities" {
            let Some(scope) = graph_scope(p, hash) else {
                continue;
            };
            if retrieval.is_none() {
                retrieval = Some(Retrieval::open(stage.join("state/retrieval.sqlite3"))?);
            }
            let id = format!(
                "legacy-{}",
                digest(format!("{hash}:{}", row.source_id).as_bytes())
            );
            let node = GraphNode {
                id: id.clone(),
                name: value(&row.row, "name").into(),
                entity_type: value(&row.row, "entity_type").into(),
                provenance: provenance(hash, "entities", &row.source_id),
            };
            // Invalid native type/name remains exact archived/quarantined data.
            match retrieval.as_ref().unwrap().put_node(scope, &node) {
                Ok(()) => receipts.push(json!({"kind":"native_graph_node","scope":scope,"id":id,"source_sha256":hash})),
                Err(_) => receipts.push(json!({"kind":"unported_graph_node","source_sha256":hash,
                    "source_id_sha256":digest(row.source_id.as_bytes()),"required_resolution":"unsupported native node type, capacity or name"})),
            }
        }
    }
    for row in &p.rows {
        let hash = value(&row.row, "_migration_source_sha256");
        if row.table == "relationships" {
            let Some(scope) = graph_scope(p, hash) else {
                continue;
            };
            let Some(retrieval) = &retrieval else {
                continue;
            };
            let edge = GraphEdge {
                source_id: format!(
                    "legacy-{}",
                    digest(format!("{hash}:{}", value(&row.row, "source_id")).as_bytes())
                ),
                target_id: format!(
                    "legacy-{}",
                    digest(format!("{hash}:{}", value(&row.row, "target_id")).as_bytes())
                ),
                relationship: value(&row.row, "relation_type").into(),
                provenance: provenance(hash, "relationships", &row.source_id),
            };
            match retrieval.put_edge(scope, &edge) {
                Ok(()) => receipts.push(json!({"kind":"native_graph_edge","scope":scope,"edge":edge})),
                Err(_) => receipts.push(json!({"kind":"unported_graph_edge","source_sha256":hash,
                    "source_id_sha256":digest(row.source_id.as_bytes()),"required_resolution":"unsupported relation, capacity or missing exact-scope endpoints"})),
            }
        }
        let Some(record) = row.row.get("legacy_record") else {
            continue;
        };
        if row.table == "legacy_persona" {
            let Some(scope) = p.mappings.get("state:persona") else {
                continue;
            };
            if governance.is_none() {
                governance = Some(ApprovalRegistry::open(
                    stage.join("state/governance.sqlite3"),
                )?);
            }
            // Keep EVERY field including unknown future boundaries, no summarizer.
            let content = format!(
                "Operator-imported legacy persona. Preserve all stated boundaries. Source SHA-256: {hash}\n{}",
                serde_yaml_ng::to_string(record)?
            );
            let proposal = governance.as_ref().unwrap().propose_document(
                scope,
                DocumentKind::Persona,
                0,
                &content,
                &p.fingerprint(),
            )?;
            ensure!(
                governance
                    .as_ref()
                    .unwrap()
                    .document(scope, DocumentKind::Persona)?
                    .is_none(),
                "persona migration must not activate a revision"
            );
            receipts.push(json!({"kind":"pending_native_persona","scope":scope,"proposal_id":proposal.id,
                "proposal_hash":proposal.proposal_hash,"context_fingerprint":proposal.context_fingerprint,
                "source_sha256":hash,"status":"pending"}));
        }
        if row.table == "legacy_trust_devices" {
            let Some(scores) = record.as_object() else {
                continue;
            };
            for (entity, score) in scores {
                let Some(scope) = p.mappings.get(&format!("state:trust:{entity}")) else {
                    continue;
                };
                let Some(score) = score
                    .as_f64()
                    .filter(|score| score.is_finite() && (0.0..=1000.0).contains(score))
                else {
                    continue;
                };
                if governance.is_none() {
                    governance = Some(ApprovalRegistry::open(
                        stage.join("state/governance.sqlite3"),
                    )?);
                }
                let registry = governance.as_ref().unwrap();
                let old = registry.trust(scope)?;
                ensure!(
                    old.version == 0,
                    "multiple legacy trust entities mapped to same native scope; refusing last-writer wins"
                );
                // Neutral legacy 500 -> 0. Negative only is conservative. Positive
                // history is exported as advice, not fabricated successful effects.
                let normalized = (score - 500.0) / 500.0;
                if normalized < 0.0 {
                    registry.update_trust(
                        scope,
                        0,
                        normalized,
                        OperationOutcome::Unknown,
                        "offline-migration",
                    )?;
                }
                receipts.push(json!({"kind":"native_advisory_trust","scope":scope,
                    "legacy_entity_sha256":digest(entity.as_bytes()),"legacy_score":score,
                    "normalized_advisory_score":normalized,"active_score":normalized.min(0.0),
                    "positive_elevation_requires_operator_review":normalized>0.0,"source_sha256":hash}));
            }
        }
        if row.table != "legacy_jobs" {
            continue;
        }
        let id = value(record, "id");
        let Some(scope) = p.mappings.get(&format!("state:job:{id}")) else {
            continue;
        };
        let prompt = value(record, "task");
        let timezone = value(record, "timezone");
        let due = DateTime::parse_from_rfc3339(value(record, "next_run"))
            .ok()
            .and_then(|due| {
                // Scheduler storage is whole-second. Round UP, never dispatch early.
                let seconds = due
                    .timestamp()
                    .checked_add(i64::from(due.timestamp_subsec_nanos() != 0))?;
                DateTime::from_timestamp(seconds, 0)
            });
        let spec = due.and_then(|due| {
            let expression = value(record, "schedule");
            if expression.starts_with("in ")
                || record.get("delete_after_run") == Some(&Value::Bool(true))
            {
                serde_json::from_value::<ScheduleSpec>(
                    json!({"timezone":timezone,"schedule":{"kind":"once","at":due}}),
                )
                .ok()
                .map(|spec| (due, spec))
            } else if expression.starts_with("every ") {
                // Anchor at persisted next_run, NOT next_run + parsed interval.
                let parsed = ScheduleSpec::parse_at(expression, timezone, due).ok()?;
                let mut raw = serde_json::to_value(parsed).ok()?;
                *raw.get_mut("schedule")?.get_mut("anchor")? = json!(due);
                serde_json::from_value::<ScheduleSpec>(raw)
                    .ok()
                    .map(|spec| (due, spec))
            } else {
                ScheduleSpec::parse_at(expression, timezone, due)
                    .ok()
                    .map(|spec| (due, spec))
            }
        });
        let Some((due, spec)) =
            spec.filter(|_| !id.is_empty() && !prompt.trim().is_empty() && prompt.len() <= 65536)
        else {
            receipts.push(json!({"kind":"unported_job","legacy_id_sha256":digest(id.as_bytes()),
                "required_resolution":"explicit valid timezone, RFC3339 next_run, bounded task and native-compatible schedule required"}));
            continue;
        };
        if scheduler.is_none() {
            let path = stage.join("state/automation.sqlite3");
            private_write(&path, b"")?;
            let db = Connection::open(path)?;
            db.execute_batch("PRAGMA application_id=1297307969; PRAGMA user_version=2; PRAGMA synchronous=FULL;
              CREATE TABLE automation_jobs(id TEXT PRIMARY KEY,scope TEXT NOT NULL,prompt TEXT NOT NULL,interval_seconds INTEGER NOT NULL CHECK(interval_seconds>0),next_due INTEGER NOT NULL,enabled INTEGER NOT NULL,removed INTEGER NOT NULL DEFAULT 0,schedule_spec TEXT);")?;
            scheduler = Some(db);
        }
        let native_id = format!("legacy-{}", digest(format!("{hash}:{id}").as_bytes()));
        scheduler.as_ref().unwrap().execute("INSERT INTO automation_jobs(id,scope,prompt,interval_seconds,next_due,enabled,schedule_spec) VALUES(?1,?2,?3,1,?4,0,?5)",
            params![native_id,serde_json::to_string(scope)?,prompt,due.timestamp(),serde_json::to_string(&spec)?])?;
        receipts.push(json!({"kind":"disabled_native_job","scope":scope,"id":native_id,
            "source_sha256":hash,"legacy_id_sha256":digest(id.as_bytes()),"enabled":false,
            "required_resolution":"Review archived provider, capability, active_hours, retry, Nomad execution, overlap and run receipts. Scheduler DB is paused. Do not resume unless equivalent runtime capabilities are configured and external effects reconciled."}));
    }
    drop(retrieval);
    drop(governance);
    drop(scheduler);
    for name in [
        "retrieval.sqlite3",
        "governance.sqlite3",
        "automation.sqlite3",
    ] {
        let path = stage.join("state").join(name);
        if !path.exists() {
            continue;
        }
        let db = Connection::open(&path)?;
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;")?;
        let integrity: String = db.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        ensure!(integrity == "ok", "native portable asset integrity failed");
        drop(db);
        artifacts.push(format!("state/{name}"));
    }
    private_write(
        &stage.join("portable-state.json"),
        &serde_json::to_vec_pretty(&receipts)?,
    )?;
    artifacts.push("portable-state.json".into());
    Ok(artifacts)
}
