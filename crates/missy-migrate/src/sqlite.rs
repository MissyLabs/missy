use super::*;
use rusqlite::{Connection, OpenFlags, types::ValueRef};
use std::os::fd::AsRawFd;

const TABLES: &[(&str, &str)] = &[
    (
        "turns",
        "id:TEXT session_id:TEXT timestamp:TEXT role:TEXT content:TEXT provider:TEXT metadata:TEXT",
    ),
    (
        "learnings",
        "id:TEXT task_type:TEXT outcome:TEXT lesson:TEXT approach:TEXT timestamp:TEXT",
    ),
    (
        "sessions",
        "session_id:TEXT name:TEXT created_at:TEXT updated_at:TEXT turn_count:INTEGER provider:TEXT channel:TEXT",
    ),
    (
        "costs",
        "id:TEXT session_id:TEXT model:TEXT prompt_tokens:INTEGER completion_tokens:INTEGER cost_usd:REAL timestamp:TEXT provider:TEXT account:TEXT",
    ),
    (
        "summaries",
        "id:TEXT session_id:TEXT depth:INTEGER content:TEXT token_estimate:INTEGER source_turn_ids:TEXT source_summary_ids:TEXT parent_id:TEXT time_range_start:TEXT time_range_end:TEXT descendant_count:INTEGER file_refs:TEXT created_at:TEXT",
    ),
    (
        "large_content",
        "id:TEXT session_id:TEXT turn_id:TEXT tool_name:TEXT original_chars:INTEGER content:TEXT summary:TEXT created_at:TEXT",
    ),
    (
        "checkpoints",
        "id:TEXT session_id:TEXT task_id:TEXT prompt:TEXT state:TEXT loop_messages:TEXT tool_names_used:TEXT iteration:INTEGER owner_id:TEXT lease_expires_at:REAL created_at:REAL updated_at:REAL",
    ),
    (
        "entities",
        "id:TEXT name:TEXT entity_type:TEXT properties:TEXT first_seen:TEXT last_seen:TEXT mention_count:INTEGER",
    ),
    (
        "relationships",
        "id:TEXT source_id:TEXT target_id:TEXT relation_type:TEXT weight:REAL context:TEXT created_at:TEXT last_seen:TEXT",
    ),
];

/// Unknown operational databases remain raw quarantined snapshots, but must still
/// be standalone, consistent SQLite backups. No source schema hooks are run.
pub(crate) fn validate_backup(path: &Path, max: usize) -> Result<()> {
    let pinned = source::pin(path, "database_validation", max)?;
    pinned.no_journals()?;
    let uri = format!(
        "file:/proc/self/fd/{}?mode=ro&immutable=1",
        pinned.file.as_raw_fd()
    );
    let db = Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|_| anyhow!("offline SQLite open failed; source details suppressed"))?;
    db.execute_batch("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;")
        .map_err(|_| anyhow!("offline SQLite setup failed; source details suppressed"))?;
    let result: String = db
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .map_err(|_| {
            anyhow!("quarantined SQLite backup validation failed; source details suppressed")
        })?;
    ensure!(result == "ok", "quarantined SQLite backup integrity failed");
    drop(db);
    pinned.no_journals()?;
    ensure!(
        source::pin(path, "database_validation", max)?.source.sha256 == pinned.source.sha256,
        "source database changed during validation"
    );
    Ok(())
}

pub(crate) fn read_backup(
    path: &Path,
    mappings: &BTreeMap<String, Scope>,
    p: &mut MigrationPlan,
) -> Result<()> {
    let pinned = source::pin(path, "database", p.input.limits.max_file_bytes)?;
    pinned.no_journals()?;
    ensure!(
        pinned.bytes.starts_with(b"SQLite format 3\0"),
        "source is not a SQLite backup"
    );
    // /proc/self/fd pins the already checked O_NOFOLLOW descriptor. immutable=1
    // guarantees SQLite never creates journals or takes source write locks.
    let uri = format!(
        "file:/proc/self/fd/{}?mode=ro&immutable=1",
        pinned.file.as_raw_fd()
    );
    let db = Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|_| anyhow!("offline SQLite open failed; source details suppressed"))?;
    db.execute_batch("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;")
        .map_err(|_| anyhow!("offline SQLite setup failed; source details suppressed"))?;
    // Keep malformed schema/resource failures independent of source SQL text.
    let outcome = read_tables(&db, mappings, p, &pinned.source.sha256);
    drop(db);
    pinned.no_journals()?;
    outcome.map_err(|_| anyhow!("offline SQLite schema/data validation failed; expected legacy memory.db schema and bounded UTF-8 rows (source details suppressed)"))?;
    ensure!(
        source::pin(path, "database", p.input.limits.max_file_bytes)?
            .source
            .sha256
            == pinned.source.sha256,
        "source database changed during import planning"
    );
    p.sources.push(pinned.source);
    Ok(())
}

fn read_tables(
    db: &Connection,
    mappings: &BTreeMap<String, Scope>,
    p: &mut MigrationPlan,
    source_sha256: &str,
) -> Result<()> {
    let integrity: String = db.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "legacy SQLite integrity failed");
    ensure!(
        db.prepare("PRAGMA foreign_key_check")?
            .query([])?
            .next()?
            .is_none(),
        "legacy graph/database has unresolved foreign keys"
    );
    let objects: Vec<(String, String, String)> = db.prepare("SELECT name,type,coalesce(sql,'') FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<rusqlite::Result<_>>()?;
    ensure!(
        objects
            .iter()
            .any(|(n, t, _)| TABLES.iter().any(|(known, _)| known == n) && t == "table"),
        "recognized legacy memory, graph or checkpoints table required"
    );
    let mut total_bytes: usize = p
        .rows
        .iter()
        .map(|row| serde_json::to_vec(&row.row).map(|v| v.len()))
        .collect::<serde_json::Result<Vec<_>>>()?
        .iter()
        .sum();
    let mut total_rows = p.rows.len();
    let mut sessions = std::collections::BTreeSet::new();
    let mut identities = std::collections::HashSet::new();
    for (table, kind, sql) in &objects {
        if table.starts_with("sqlite_") {
            continue;
        }
        if matches!(
            table.as_str(),
            "turns_fts"
                | "turns_fts_data"
                | "turns_fts_idx"
                | "turns_fts_docsize"
                | "turns_fts_config"
                | "summaries_fts"
                | "summaries_fts_data"
                | "summaries_fts_idx"
                | "summaries_fts_docsize"
                | "summaries_fts_config"
        ) {
            // Familiar naming alone must not allow ordinary operational tables
            // to be silently classified as disposable derived state.
            let parent = if table.starts_with("turns_fts") {
                "turns_fts"
            } else {
                "summaries_fts"
            };
            let valid_parent = objects.iter().any(|(name, kind, definition)| {
                let normalized = definition.to_ascii_lowercase();
                name == parent
                    && kind == "table"
                    && normalized.contains("create virtual table")
                    && normalized.contains("using fts5")
            });
            if !valid_parent {
                p.error(
                    "unverified_derived_table",
                    "database.fts",
                    "FTS-like table lacks virtual FTS5 parent; refusing silent data loss",
                );
            }
            // Verified FTS derived state is never read or executed.
            continue;
        }
        let Some((_, expected)) = TABLES.iter().find(|(n, _)| n == table) else {
            p.error(
                "unsupported_database_table",
                &format!("database.table_sha256_{}", digest(table.as_bytes())),
                "unknown table/view is not silently dropped; export a reviewed legacy backup",
            );
            continue;
        };
        ensure!(
            kind == "table" && !sql.to_ascii_uppercase().contains("VIRTUAL TABLE"),
            "legacy data must be ordinary tables, not views or virtual tables"
        );
        let columns: Vec<(String, String)> = db
            .prepare(&format!("PRAGMA table_info(\"{table}\")"))?
            .query_map([], |r| Ok((r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let wanted: BTreeMap<_, _> = expected
            .split_whitespace()
            .map(|s| s.split_once(':').unwrap())
            .collect();
        for (column, kind) in &columns {
            ensure!(
                wanted
                    .get(column.as_str())
                    .is_some_and(|ty| *ty == kind.to_ascii_uppercase()),
                "unexpected legacy column or type"
            );
        }
        for name in wanted.keys() {
            // Documented Python migrations added these optional columns over time.
            let optional = (table == "costs" && ["provider", "account"].contains(name))
                || (table == "checkpoints" && ["owner_id", "lease_expires_at"].contains(name));
            ensure!(
                optional || columns.iter().any(|(n, _)| n == name),
                "required legacy column missing"
            );
        }
        let count: i64 = db.query_row(&format!("SELECT count(*) FROM \"{table}\""), [], |r| {
            r.get(0)
        })?;
        total_rows = total_rows
            .checked_add(usize::try_from(count)?)
            .ok_or_else(|| anyhow!("record overflow"))?;
        ensure!(
            total_rows <= p.input.limits.max_records,
            "legacy record limit exceeded"
        );
        for (column, _) in &columns {
            let len: i64 = db.query_row(
                &format!(
                    "SELECT coalesce(max(length(cast(\"{column}\" AS BLOB))),0) FROM \"{table}\""
                ),
                [],
                |r| r.get(0),
            )?;
            ensure!(
                len <= p.input.limits.max_record_bytes as i64,
                "legacy cell exceeds byte bound"
            );
        }
        let order = if table == "sessions" {
            "session_id"
        } else {
            "id"
        };
        let mut statement =
            db.prepare(&format!("SELECT * FROM \"{table}\" ORDER BY \"{order}\""))?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let mut data = serde_json::Map::new();
            for (index, (column, _)) in columns.iter().enumerate() {
                let value = match row.get_ref(index)? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(n) => json!(n),
                    ValueRef::Real(n) => {
                        ensure!(n.is_finite(), "nonfinite source number");
                        json!(n)
                    }
                    ValueRef::Text(bytes) => Value::String(std::str::from_utf8(bytes)?.into()),
                    ValueRef::Blob(_) => return Err(anyhow!("unexpected binary legacy value")),
                };
                data.insert(column.clone(), value);
            }
            let source_id = data
                .get(order)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("legacy identifier missing"))?
                .to_owned();
            ensure!(
                identities.insert((table, source_id.clone())),
                "duplicate legacy source identity"
            );
            let session = data.get("session_id").and_then(Value::as_str).unwrap_or("");
            if !session.is_empty() {
                sessions.insert(session.to_owned());
                if !mappings.contains_key(session) {
                    p.error("missing_scope_mapping", &format!("database.session_sha256_{}", digest(session.as_bytes())), "every legacy session requires an explicit full Scope; no global/default fallback");
                }
            }
            let importable = ["turns", "summaries", "large_content"].contains(&table.as_str());
            let scope = if importable && !session.is_empty() {
                mappings.get(session).cloned()
            } else {
                None
            };
            data.insert("_migration_source_sha256".into(), json!(source_sha256));
            let row = Value::Object(data);
            let size = serde_json::to_vec(&row)?.len();
            ensure!(
                size <= p.input.limits.max_record_bytes,
                "legacy serialized row exceeds byte bound"
            );
            total_bytes = total_bytes
                .checked_add(size)
                .ok_or_else(|| anyhow!("content overflow"))?;
            ensure!(
                total_bytes <= p.input.limits.max_total_content_bytes,
                "legacy content total exceeds bound"
            );
            if scope.is_some() {
                p.counts.imported_memories += 1;
            } else {
                p.counts.quarantined_records += 1;
            }
            p.rows.push(ImportedRow {
                table: table.clone(),
                source_id,
                scope,
                row,
            });
        }
    }
    p.counts.sessions += sessions.len();
    if objects.iter().any(|(name, _, _)| name == "checkpoints") {
        p.resolutions.push(Resolution::new("checkpoint_effect_receipts_missing", "checkpoints",
            "Inspect each checkpoint against audit, tool-side idempotency receipts and external effects. Explicitly abandon or reconstruct a new task after reconciliation; never resume/replay legacy loop_messages.", true));
    }
    if objects
        .iter()
        .any(|(name, _, _)| name == "entities" || name == "relationships")
    {
        p.resolutions.push(Resolution::new("graph_scope_required", "graph",
            "Legacy entities/relationships are global and carry no tenant/user authority. Assign authenticated scopes and review edges before rebuilding native retrieval; raw graph remains quarantined.", true));
    }
    if p.counts.quarantined_records > 0 {
        p.review("quarantined_data", "database", "unscoped rows, global learnings, session/cost metadata and all checkpoints are quarantined outside runtime retrieval, never replayed");
    }
    p.review("inert_history", "database.turns", "raw turns, summaries and large content become untrusted scoped memories, not native conversation turns or executable tool/checkpoint state");
    for session in mappings.keys().filter(|s| !sessions.contains(*s)) {
        p.review(
            "unused_scope_mapping",
            &format!("mapping.session_sha256_{}", digest(session.as_bytes())),
            "scope mapping has no matching legacy session",
        );
    }
    Ok(())
}
