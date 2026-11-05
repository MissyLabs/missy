//! Durable exact-scope immutable snapshots. Capacity never silently evicts receipts.
use super::{Evidence, HistoryConfig};
use anyhow::{Result, anyhow, ensure};
use missy_core::Scope;
use missy_store::hash_text;
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
    time::Duration,
};
use uuid::Uuid;
const APP_ID: i64 = 0x4d534845;
const SCHEMA: &str = "CREATE TABLE evidence(scope TEXT NOT NULL,id TEXT NOT NULL,snapshot_key TEXT NOT NULL,payload TEXT NOT NULL,checksum TEXT NOT NULL,content_bytes INTEGER NOT NULL CHECK(content_bytes>=0),PRIMARY KEY(scope,id),UNIQUE(scope,snapshot_key));";
pub(crate) struct Retained {
    conn: Mutex<Connection>,
    path: PathBuf,
    identity: (u64, u64),
}
type SchemaEntry = (String, String, String, Option<String>);
fn schema(c: &Connection) -> Result<Vec<SchemaEntry>> {
    Ok(
        c.prepare("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?,
    )
}
fn snapshot_key(e: &Evidence) -> Result<String> {
    Ok(hash_text(&serde_json::to_string(&(
        &e.scope, &e.turn_id, &e.tool_id, &e.source, &e.sha256,
    ))?))
}
type Raw = (String, String, String, String, String, usize);
fn raw(r: &rusqlite::Row<'_>) -> rusqlite::Result<Raw> {
    Ok((
        r.get(0)?,
        r.get(1)?,
        r.get(2)?,
        r.get(3)?,
        r.get(4)?,
        r.get(5)?,
    ))
}
fn decode((scope, id, key, payload, checksum, bytes): Raw) -> Result<Evidence> {
    ensure!(
        hash_text(&payload) == checksum,
        "corrupt retained evidence checksum"
    );
    let e: Evidence = serde_json::from_str(&payload)?;
    e.scope.validate()?;
    ensure!(
        serde_json::to_string(&e.scope)? == scope
            && e.id == id
            && snapshot_key(&e)? == key
            && e.text.len() == bytes
            && hash_text(&e.text) == e.sha256,
        "corrupt retained evidence metadata"
    );
    Ok(e)
}
impl Retained {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let _open = super::files::OPEN_LOCK
            .lock()
            .map_err(|_| anyhow!("history open lock poisoned"))?;
        let identity = super::files::private_database(path, true)?;
        let mut conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NOFOLLOW
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        ensure!(
            super::files::private_database(path, false)? == identity,
            "history database changed during open"
        );
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA trusted_schema=OFF; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
        )?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            (app == APP_ID && version == 1) || (app == 0 && version == 0),
            "unrecognized history evidence application or schema version"
        );
        if version == 0 {
            ensure!(
                schema(&tx)?.is_empty(),
                "refusing nonempty unrecognized history database"
            );
            tx.execute_batch(SCHEMA)?;
            tx.pragma_update(None, "application_id", APP_ID)?;
            tx.pragma_update(None, "user_version", 1)?;
        }
        let expected = Connection::open_in_memory()?;
        expected.execute_batch(SCHEMA)?;
        ensure!(
            schema(&tx)? == schema(&expected)?,
            "history evidence schema mismatch"
        );
        tx.commit()?;
        let integrity: String = conn.pragma_query_value(None, "quick_check", |r| r.get(0))?;
        ensure!(integrity == "ok", "history database integrity check failed");
        for row in conn
            .prepare("SELECT scope,id,snapshot_key,payload,checksum,content_bytes FROM evidence")?
            .query_map([], raw)?
        {
            decode(row?)?;
        }
        conn.execute_batch(
            "PRAGMA journal_mode=DELETE; PRAGMA max_page_count=65536; PRAGMA secure_delete=ON;",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
            path: path.to_owned(),
            identity,
        })
    }
    fn connection(&self) -> Result<MutexGuard<'_, Connection>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| anyhow!("history evidence lock poisoned"))?;
        ensure!(
            super::files::private_database(&self.path, false)? == self.identity,
            "history database file identity changed"
        );
        Ok(conn)
    }
    pub(crate) fn get(&self, scope: &Scope, id: &str) -> Result<Evidence> {
        scope.validate()?;
        ensure!(id.len() <= 128, "invalid history receipt ID");
        let scope = serde_json::to_string(scope)?;
        let conn = self.connection()?;
        let row = conn.query_row("SELECT scope,id,snapshot_key,payload,checksum,content_bytes FROM evidence WHERE scope=?1 AND id=?2",params![scope,id],raw).optional()?;
        decode(row.ok_or_else(|| anyhow!("evidence unavailable in this exact scope"))?)
    }
    pub(crate) fn put(&self, mut e: Evidence, config: &HistoryConfig) -> Result<Evidence> {
        e.scope.validate()?;
        ensure!(hash_text(&e.text) == e.sha256, "evidence checksum mismatch");
        let scope = serde_json::to_string(&e.scope)?;
        let key = snapshot_key(&e)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(row) = tx.query_row("SELECT scope,id,snapshot_key,payload,checksum,content_bytes FROM evidence WHERE scope=?1 AND snapshot_key=?2",params![scope,key],raw).optional()? { return decode(row); }
        let (count, bytes): (usize, usize) = tx.query_row(
            "SELECT count(*),coalesce(sum(content_bytes),0) FROM evidence",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        ensure!(
            count < config.max_retained_items
                && bytes.saturating_add(e.text.len()) <= config.max_retained_bytes,
            "retained evidence capacity exhausted; existing receipts remain valid (no automatic expiry)"
        );
        e.id = format!("hist_{}", Uuid::new_v4().simple());
        let payload = serde_json::to_string(&e)?;
        tx.execute("INSERT INTO evidence(scope,id,snapshot_key,payload,checksum,content_bytes) VALUES(?1,?2,?3,?4,?5,?6)",params![scope,e.id,key,payload,hash_text(&payload),e.text.len()])?;
        tx.commit()?;
        Ok(e)
    }
}
