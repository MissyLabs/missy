//! Explicitly enabled PERSONAL memory, separate from exact-scope session memory.
//! The authenticated caller's tenant+user are the ONLY namespace authority. Channel and
//! session are provenance, not lookup keys. Returned content is always untrusted data.
mod files;
mod tools;
pub use tools::PersonalTools;

use anyhow::{Result, anyhow, ensure};
use missy_core::Scope;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
    time::Duration,
};

pub const TRUST: &str = "untrusted_data";
pub const SOURCE: &str = "explicit_user_memory";
pub const MAX_CONTENT_BYTES: usize = 8192;
pub const MAX_QUERY_BYTES: usize = 1024;
pub const MAX_RESULTS: usize = 20;
pub const MAX_OUTPUT_BYTES: usize = 64 * 1024;
pub const MAX_RECORDS: usize = 512;
pub const MAX_NAMESPACE_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_ARGUMENT_BYTES: usize = 64 * 1024;

/// Not deserializable or constructible from model/user fields. Build from a trusted Scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PersonalNamespace {
    tenant: String,
    user: String,
}
impl PersonalNamespace {
    /// Scope MUST be authenticated by the adapter, not supplied by a model tool argument.
    pub fn from_scope(scope: &Scope) -> Result<Self> {
        scope.validate()?;
        Ok(Self {
            tenant: scope.tenant.clone(),
            user: scope.user.clone(),
        })
    }
    pub fn tenant(&self) -> &str {
        &self.tenant
    }
    pub fn user(&self) -> &str {
        &self.user
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PersonalRecord {
    pub namespace: PersonalNamespaceView,
    pub id: String,
    pub content: String,
    /// Full scope of the most recent explicit put, NOT the current recall scope.
    pub source_scope: Scope,
    pub provenance: String,
    pub trust: String,
    /// Whole-record UTF-8 byte citation. No lossy character slicing.
    pub start_byte: usize,
    pub end_byte: usize,
    pub content_sha256: String,
}

/// Serializable output label only; never accepted as an authority by database/tool APIs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PersonalNamespaceView {
    pub tenant: String,
    pub user: String,
}

pub struct PersonalMemory {
    conn: Mutex<Connection>,
    path: PathBuf,
    identity: (u64, u64),
}
impl PersonalMemory {
    /// Explicit operator grant. Requires an absolute path in an existing owned 0700 directory.
    /// Does not create directories, import other databases, or enable any runtime defaults.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let _open = files::OPEN_LOCK
            .lock()
            .map_err(|_| anyhow!("database open lock poisoned"))?;
        let identity = files::private_database(path, true)?;
        let mut conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        ensure!(
            files::private_database(path, false)? == identity,
            "database changed during open"
        );
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA trusted_schema=OFF; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
        )?;
        files::migrate(&mut conn)?;
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
            .map_err(|_| anyhow!("personal memory lock poisoned"))?;
        ensure!(
            files::private_database(&self.path, false)? == self.identity,
            "database file identity changed"
        );
        Ok(conn)
    }

    /// Fingerprint the nonsecret path AND the complete immutable namespace/capability policy.
    pub fn configuration_fingerprint(&self) -> String {
        sha(&serde_json::to_string(&serde_json::json!({
            "path": self.path, "namespace":"personal:tenant+user:v1", "enabled":true,
            "ranking":"namespace-local-fts5-bm25:v1",
            "cross_channel":true, "cross_session":true, "global_fallback":false,
            "max_content_bytes":MAX_CONTENT_BYTES, "max_query_bytes":MAX_QUERY_BYTES,
            "max_results":MAX_RESULTS, "max_output_bytes":MAX_OUTPUT_BYTES,
            "max_records":MAX_RECORDS, "max_namespace_bytes":MAX_NAMESPACE_BYTES,
            "max_argument_bytes":MAX_ARGUMENT_BYTES,"trust":TRUST,"source":SOURCE
        }))
        .expect("serializable personal memory policy"))
    }

    pub fn put(&self, scope: &Scope, id: &str, content: &str) -> Result<PersonalRecord> {
        let ns = PersonalNamespace::from_scope(scope)?;
        identifier(id)?;
        validate_content(content)?;
        let source_scope = serde_json::to_string(scope)?;
        let checksum = checksum(&ns, id, content, &source_scope);
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (count,bytes): (usize,usize) = tx.query_row(
            "SELECT count(*),coalesce(sum(length(CAST(content AS BLOB))),0) FROM memories WHERE tenant=?1 AND user=?2 AND id!=?3",
            params![ns.tenant,ns.user,id], |r| Ok((r.get(0)?,r.get(1)?)))?;
        ensure!(
            count < MAX_RECORDS && bytes + content.len() <= MAX_NAMESPACE_BYTES,
            "personal namespace capacity exceeded"
        );
        // Both the FTS delete and row upsert are constrained by tenant+user, never ID alone.
        tx.execute("DELETE FROM personal_fts WHERE rowid IN (SELECT rowid FROM memories WHERE tenant=?1 AND user=?2 AND id=?3)",
            params![ns.tenant,ns.user,id])?;
        tx.execute("INSERT INTO memories(tenant,user,id,content,source_scope,checksum) VALUES(?1,?2,?3,?4,?5,?6)
            ON CONFLICT(tenant,user,id) DO UPDATE SET content=excluded.content,source_scope=excluded.source_scope,checksum=excluded.checksum
            WHERE memories.tenant=?1 AND memories.user=?2 AND memories.id=?3",
            params![ns.tenant,ns.user,id,content,source_scope,checksum])?;
        tx.execute("INSERT INTO personal_fts(rowid,content) SELECT rowid,content FROM memories WHERE tenant=?1 AND user=?2 AND id=?3",
            params![ns.tenant,ns.user,id])?;
        tx.commit()?;
        record(
            &ns,
            id.to_owned(),
            content.to_owned(),
            source_scope,
            checksum,
        )
    }

    pub fn get(&self, scope: &Scope, id: &str) -> Result<Option<PersonalRecord>> {
        let ns = PersonalNamespace::from_scope(scope)?;
        identifier(id)?;
        let conn = self.connection()?;
        let row = conn.query_row("SELECT id,content,source_scope,checksum FROM memories WHERE tenant=?1 AND user=?2 AND id=?3",
            params![ns.tenant,ns.user,id], raw_record).optional()?;
        row.map(|(id, content, source, sum)| record(&ns, id, content, source, sum))
            .transpose()
    }

    pub fn delete(&self, scope: &Scope, id: &str) -> Result<bool> {
        let ns = PersonalNamespace::from_scope(scope)?;
        identifier(id)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM personal_fts WHERE rowid IN (SELECT rowid FROM memories WHERE tenant=?1 AND user=?2 AND id=?3)",
            params![ns.tenant,ns.user,id])?;
        let deleted = tx.execute(
            "DELETE FROM memories WHERE tenant=?1 AND user=?2 AND id=?3",
            params![ns.tenant, ns.user, id],
        )? != 0;
        tx.commit()?;
        Ok(deleted)
    }

    /// Bounded Unicode lexical search. Every query word is a literal FTS term, not syntax.
    /// Empty/punctuation-only queries return no results; never fall back to another namespace.
    pub fn search(&self, scope: &Scope, query: &str, limit: usize) -> Result<Vec<PersonalRecord>> {
        let ns = PersonalNamespace::from_scope(scope)?;
        query_limit(query, limit)?;
        let terms: Vec<_> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| format!("\"{s}\""))
            .collect();
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let fts = terms.join(" OR ");
        let conn = self.connection()?;
        // FTS5 BM25 uses whole-index term statistics even when a WHERE clause
        // filters returned rows. Rank a bounded, principal-only transient index
        // instead: foreign private writes must not affect even result ordering.
        let mut owned = conn.prepare(
            "SELECT id,content,source_scope,checksum FROM memories
             WHERE tenant=?1 AND user=?2 ORDER BY id LIMIT ?3",
        )?;
        let mut local = Connection::open_in_memory()?;
        local.execute_batch(
            "CREATE VIRTUAL TABLE ranked USING fts5(content,id UNINDEXED,source_scope UNINDEXED,checksum UNINDEXED);",
        )?;
        let tx = local.transaction()?;
        let mut count = 0usize;
        let mut bytes = 0usize;
        for row in owned.query_map(params![ns.tenant, ns.user, MAX_RECORDS + 1], raw_record)? {
            let (id, content, source, sum) = row?;
            count += 1;
            bytes += content.len();
            ensure!(
                count <= MAX_RECORDS && bytes <= MAX_NAMESPACE_BYTES,
                "personal namespace capacity exceeded"
            );
            record(
                &ns,
                id.clone(),
                content.clone(),
                source.clone(),
                sum.clone(),
            )?;
            tx.execute(
                "INSERT INTO ranked(content,id,source_scope,checksum) VALUES(?1,?2,?3,?4)",
                params![content, id, source, sum],
            )?;
        }
        tx.commit()?;
        let mut stmt = local.prepare(
            "SELECT id,content,source_scope,checksum FROM ranked
             WHERE ranked MATCH ?1 ORDER BY bm25(ranked),id LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![fts, limit], raw_record)?;
        let mut result = Vec::new();
        for row in rows {
            let (id, content, source, sum) = row?;
            let candidate = record(&ns, id, content, source, sum)?;
            result.push(candidate);
            // Reserve framing space for tool wrappers. No partial UTF-8 or partial record.
            if serde_json::to_vec(&result)?.len() > MAX_OUTPUT_BYTES - 1024 {
                result.pop();
                break;
            }
        }
        Ok(result)
    }

    /// Read-only recall for a future context provider. Caller must supply authenticated Scope
    /// and frame these records as untrusted after the system-prompt boundary, never as policy.
    pub fn recall(&self, scope: &Scope, query: &str, limit: usize) -> Result<Vec<PersonalRecord>> {
        self.search(scope, query, limit)
    }
}

fn raw_record(r: &rusqlite::Row<'_>) -> rusqlite::Result<(String, String, String, String)> {
    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
}
pub(crate) fn identifier(id: &str) -> Result<()> {
    ensure!(
        !id.trim().is_empty() && id.len() <= 256 && !id.chars().any(char::is_control),
        "id must be 1..256 bytes without control characters"
    );
    Ok(())
}
pub(crate) fn validate_content(content: &str) -> Result<()> {
    ensure!(
        !content.trim().is_empty() && content.len() <= MAX_CONTENT_BYTES && !content.contains('\0'),
        "content must be 1..8192 bytes without NUL"
    );
    Ok(())
}
pub(crate) fn query_limit(query: &str, limit: usize) -> Result<()> {
    ensure!(
        query.len() <= MAX_QUERY_BYTES && !query.contains('\0'),
        "query must be at most 1024 bytes without NUL"
    );
    ensure!((1..=MAX_RESULTS).contains(&limit), "limit must be 1..20");
    Ok(())
}
fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}
fn checksum(ns: &PersonalNamespace, id: &str, content: &str, scope: &str) -> String {
    let mut h = Sha256::new();
    for value in [&ns.tenant, &ns.user, id, content, scope, SOURCE, TRUST] {
        h.update((value.len() as u64).to_le_bytes());
        h.update(value.as_bytes());
    }
    hex::encode(h.finalize())
}
fn record(
    ns: &PersonalNamespace,
    id: String,
    content: String,
    scope: String,
    sum: String,
) -> Result<PersonalRecord> {
    identifier(&id)?;
    validate_content(&content)?;
    ensure!(
        sum == checksum(ns, &id, &content, &scope),
        "personal memory checksum mismatch"
    );
    let source_scope: Scope = serde_json::from_str(&scope)?;
    ensure!(
        PersonalNamespace::from_scope(&source_scope)? == *ns,
        "personal memory source principal mismatch"
    );
    Ok(PersonalRecord {
        namespace: PersonalNamespaceView {
            tenant: ns.tenant.clone(),
            user: ns.user.clone(),
        },
        end_byte: content.len(),
        start_byte: 0,
        content_sha256: sha(&content),
        id,
        content,
        source_scope,
        provenance: SOURCE.into(),
        trust: TRUST.into(),
    })
}
