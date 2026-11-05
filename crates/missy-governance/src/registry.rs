use crate::{fingerprint, now};
use anyhow::{Result, ensure};
use missy_core::Scope;
use rusqlite::{Connection, Transaction, params};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

/// Durable registry with independent connections and BEGIN IMMEDIATE writes.
/// Parent directory must be owned and writable only by the service operator.
/// Mutating operator APIs must never be exposed as model-callable tools.
#[derive(Clone)]
pub struct ApprovalRegistry {
    pub(crate) path: Arc<PathBuf>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditEvent {
    pub sequence: i64,
    pub timestamp: i64,
    pub object_id: String,
    pub event: String,
    pub binding_hash: String,
    pub operator_hash: Option<String>,
}
impl ApprovalRegistry {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(
            path != Path::new(":memory:"),
            "governance requires a durable database path"
        );
        ensure!(
            path.parent().is_some_and(Path::is_dir),
            "governance database parent must exist"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(path)?;
            let meta = file.metadata()?;
            ensure!(
                meta.is_file() && meta.nlink() == 1,
                "database must be a regular non-hardlinked file"
            );
            ensure!(
                meta.mode() & 0o077 == 0,
                "governance database requires private permissions (0600)"
            );
            ensure!(
                meta.uid() == unsafe { libc::geteuid() },
                "database must be owned by service user"
            );
        }
        let registry = Self {
            path: Arc::new(path.to_owned()),
        };
        let conn = registry.connect()?;
        conn.execute_batch("PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS governance_meta(version INTEGER NOT NULL);
            INSERT INTO governance_meta(version) SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM governance_meta);
            CREATE TABLE IF NOT EXISTS approval(
              id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, binding_hash TEXT UNIQUE NOT NULL,
              binding TEXT NOT NULL, status TEXT NOT NULL,
              created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
              decided_at INTEGER, consumed_at INTEGER, token_id TEXT);
            CREATE TABLE IF NOT EXISTS audit(
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER NOT NULL,
              scope_key TEXT NOT NULL, object_id TEXT NOT NULL, event TEXT NOT NULL,
              binding_hash TEXT NOT NULL, operator_hash TEXT);
            CREATE TABLE IF NOT EXISTS trust(
              scope_key TEXT PRIMARY KEY, score REAL NOT NULL CHECK(score >= -1 AND score <= 1), version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS document_revision(
              scope_key TEXT NOT NULL, kind TEXT NOT NULL, version INTEGER NOT NULL,
              content TEXT NOT NULL, content_hash TEXT NOT NULL, PRIMARY KEY(scope_key,kind,version));
            CREATE TABLE IF NOT EXISTS document_proposal(
              id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, kind TEXT NOT NULL,
              base_version INTEGER NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
              context_fingerprint TEXT NOT NULL, proposal_hash TEXT NOT NULL,
              status TEXT NOT NULL, created_at INTEGER NOT NULL);")?;
        let version: i64 =
            conn.query_row("SELECT version FROM governance_meta", [], |r| r.get(0))?;
        ensure!(version == 1, "unsupported governance schema version");
        Ok(registry)
    }
    pub(crate) fn connect(&self) -> Result<Connection> {
        let conn = Connection::open_with_flags(
            self.path.as_ref(),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(Duration::from_secs(10))?;
        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")?;
        Ok(conn)
    }
    pub fn audit(
        &self,
        scope: &Scope,
        after_sequence: i64,
        limit: usize,
    ) -> Result<Vec<AuditEvent>> {
        let key = scope_key(scope)?;
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            "SELECT sequence,timestamp,object_id,event,binding_hash,operator_hash
            FROM audit WHERE scope_key=?1 AND sequence>?2 ORDER BY sequence LIMIT ?3",
        )?;
        Ok(stmt
            .query_map(params![key, after_sequence, limit.min(1000) as i64], |r| {
                Ok(AuditEvent {
                    sequence: r.get(0)?,
                    timestamp: r.get(1)?,
                    object_id: r.get(2)?,
                    event: r.get(3)?,
                    binding_hash: r.get(4)?,
                    operator_hash: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }
}
pub(crate) fn scope_key(scope: &Scope) -> Result<String> {
    scope.validate()?;
    fingerprint(scope)
}
pub(crate) fn audit(
    tx: &Transaction<'_>,
    scope_key: &str,
    id: &str,
    event: &str,
    binding_hash: &str,
    operator: Option<&str>,
) -> Result<()> {
    let operator_hash = operator.map(fingerprint).transpose()?;
    tx.execute(
        "INSERT INTO audit(timestamp,scope_key,object_id,event,binding_hash,operator_hash)
        VALUES(?1,?2,?3,?4,?5,?6)",
        params![now()?, scope_key, id, event, binding_hash, operator_hash],
    )?;
    Ok(())
}
