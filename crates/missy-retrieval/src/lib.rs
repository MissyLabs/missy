//! Persistent scope-isolated retrieval and graph memory. All returned content and provenance
//! are untrusted data, never instructions. Default retrieval is BM25 + deterministic hashed
//! lexical vectors fused by RRF, NOT neural/semantic retrieval. No providers or GPU required.
use anyhow::{Result, anyhow, ensure};
use missy_core::Scope;
use missy_policy::Policy;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};
mod embedding;
mod files;
mod graph;
pub use embedding::{Embedder, HashingEmbedder};
pub use graph::{GraphEdge, GraphNode, GraphTraversal};

pub const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;
pub const MAX_SCOPE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_DOCUMENTS: usize = 256;
pub const MAX_CHUNKS: usize = 8192;
pub const MAX_QUERY_BYTES: usize = 4096;
pub const MAX_RESULTS: usize = 50;
pub const MAX_OUTPUT_BYTES: usize = 128 * 1024;
pub const TRUST: &str = "untrusted_data";
const CHUNK_BYTES: usize = 2048;
const OVERLAP_BYTES: usize = 256;

pub struct Retrieval {
    conn: Mutex<Connection>,
    embedder: Arc<dyn Embedder>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub provenance: String,
    pub hash: String,
    pub trust: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestReport {
    pub document_hash: String,
    pub chunks: usize,
    pub changed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub doc_id: String,
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub chunk_index: usize,
    pub score: f64,
    pub provenance: String,
    pub document_hash: String,
    pub trust: String,
}
pub(crate) fn scope_key(scope: &Scope) -> Result<String> {
    scope.validate()?;
    Ok(serde_json::to_string(scope)?)
}
pub(crate) fn identifier(id: &str) -> Result<()> {
    ensure!(
        !id.trim().is_empty() && id.len() <= 256 && !id.contains('\0'),
        "id must be 1..256 bytes without NUL"
    );
    Ok(())
}
pub(crate) fn provenance(value: &str) -> Result<()> {
    ensure!(
        !value.trim().is_empty() && value.len() <= 2048 && !value.contains('\0'),
        "provenance must be 1..2048 bytes without NUL"
    );
    Ok(())
}
fn hash(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}
fn spans(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + CHUNK_BYTES).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        result.push((start, end));
        if end == text.len() {
            break;
        }
        start = end - OVERLAP_BYTES;
        while !text.is_char_boundary(start) {
            start += 1;
        }
    }
    result
}
fn checksum(
    scope: &str,
    id: &str,
    ordinal: usize,
    start: usize,
    end: usize,
    text: &str,
    vector: &[u8],
) -> String {
    let mut digest = Sha256::new();
    // Length-framed metadata prevents delimiter ambiguity, including untrusted IDs.
    for bytes in [
        scope.as_bytes(),
        id.as_bytes(),
        ordinal.to_string().as_bytes(),
        start.to_string().as_bytes(),
        end.to_string().as_bytes(),
        text.as_bytes(),
        vector,
    ] {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}
impl Retrieval {
    /// Caller supplies an absolute file in an existing owner-only directory. Never opens legacy state.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_embedder(path, Arc::new(HashingEmbedder))
    }
    pub fn open_with_embedder(path: impl AsRef<Path>, embedder: Arc<dyn Embedder>) -> Result<Self> {
        ensure!(
            !embedder.name().is_empty()
                && embedder.name().len() <= 256
                && (1..=4096).contains(&embedder.dimension()),
            "invalid embedder identity/dimension"
        );
        let _guard = files::OPEN_LOCK
            .lock()
            .map_err(|_| anyhow!("database open lock poisoned"))?;
        files::private_database(path.as_ref())?;
        let mut conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF; PRAGMA synchronous=FULL;",
        )?;
        files::migrate(&mut conn, embedder.name(), embedder.dimension())?;
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA max_page_count=65536;")?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder,
        })
    }
    pub(crate) fn connection(&self) -> Result<MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| anyhow!("retrieval lock poisoned"))
    }
    pub fn embedder_name(&self) -> &str {
        self.embedder.name()
    }
    pub fn ingest(
        &self,
        scope: &Scope,
        id: &str,
        text: &str,
        source: &str,
    ) -> Result<IngestReport> {
        let key = scope_key(scope)?;
        identifier(id)?;
        provenance(source)?;
        ensure!(
            text.len() <= MAX_DOCUMENT_BYTES,
            "document exceeds input limit"
        );
        let document_hash = hash(text);
        let ranges = spans(text);
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prior: Option<(String, String)> = tx
            .query_row(
                "SELECT hash,provenance FROM documents WHERE scope=?1 AND id=?2",
                params![key, id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if prior.as_ref() == Some(&(document_hash.clone(), source.to_owned())) {
            // Verify existing data instead of quietly blessing corrupted chunks as unchanged.
            load_chunks(&tx, &key, self.embedder.dimension())?;
            return Ok(IngestReport {
                document_hash,
                chunks: ranges.len(),
                changed: false,
            });
        }
        let (count, bytes): (usize,usize) = tx.query_row("SELECT count(*),coalesce(sum(length(CAST(text AS BLOB))),0) FROM documents WHERE scope=?1 AND id!=?2", params![key,id], |r| Ok((r.get(0)?,r.get(1)?)))?;
        ensure!(
            count < MAX_DOCUMENTS && bytes + text.len() <= MAX_SCOPE_BYTES,
            "scope document capacity exceeded"
        );
        let chunks: usize = tx.query_row(
            "SELECT count(*) FROM chunks WHERE scope=?1 AND doc_id!=?2",
            params![key, id],
            |r| r.get(0),
        )?;
        ensure!(
            chunks + ranges.len() <= MAX_CHUNKS,
            "scope chunk capacity exceeded"
        );
        tx.execute(
            "DELETE FROM documents WHERE scope=?1 AND id=?2",
            params![key, id],
        )?;
        tx.execute(
            "INSERT INTO documents VALUES(?1,?2,?3,?4,?5)",
            params![key, id, text, source, document_hash],
        )?;
        for (ordinal, &(start, end)) in ranges.iter().enumerate() {
            let vector = embedding::normalize(
                self.embedder.embed(&text[start..end])?,
                self.embedder.dimension(),
            )?;
            let bytes = embedding::encode(&vector);
            let sum = checksum(&key, id, ordinal, start, end, &text[start..end], &bytes);
            tx.execute(
                "INSERT INTO chunks VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![key, id, ordinal, start, end, bytes, sum],
            )?;
        }
        tx.commit()?;
        Ok(IngestReport {
            document_hash,
            chunks: ranges.len(),
            changed: true,
        })
    }
    /// File ingestion never bypasses policy roots, bounded reads, or no-symlink traversal.
    pub fn ingest_file(
        &self,
        scope: &Scope,
        id: &str,
        path: impl AsRef<Path>,
        policy: &Policy,
    ) -> Result<IngestReport> {
        scope.validate()?;
        identifier(id)?;
        let path = path.as_ref();
        let source = path
            .to_str()
            .ok_or_else(|| anyhow!("UTF-8 source path required"))?;
        provenance(source)?;
        let bytes = policy.read_file(path)?;
        ensure!(
            bytes.len() <= MAX_DOCUMENT_BYTES,
            "document exceeds input limit"
        );
        self.ingest(scope, id, std::str::from_utf8(&bytes)?, source)
    }
    pub fn get_document(&self, scope: &Scope, id: &str) -> Result<Option<Document>> {
        let key = scope_key(scope)?;
        identifier(id)?;
        let conn = self.connection()?;
        let doc = conn
            .query_row(
                "SELECT id,text,provenance,hash FROM documents WHERE scope=?1 AND id=?2",
                params![key, id],
                document_row,
            )
            .optional()?;
        if let Some(doc) = &doc {
            validate_document(doc)?;
        }
        Ok(doc)
    }
    pub fn drop_document(&self, scope: &Scope, id: &str) -> Result<bool> {
        let key = scope_key(scope)?;
        identifier(id)?;
        Ok(self.connection()?.execute(
            "DELETE FROM documents WHERE scope=?1 AND id=?2",
            params![key, id],
        )? > 0)
    }
    pub fn search(&self, scope: &Scope, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let key = scope_key(scope)?;
        ensure!(
            !query.trim().is_empty() && query.len() <= MAX_QUERY_BYTES,
            "query must be 1..4096 bytes"
        );
        ensure!(
            (1..=MAX_RESULTS).contains(&limit),
            "result limit must be 1..50"
        );
        let qtokens: BTreeSet<_> = embedding::tokens(query).into_iter().collect();
        if qtokens.is_empty() {
            return Ok(vec![]);
        }
        let qvector = embedding::normalize(self.embedder.embed(query)?, self.embedder.dimension())?;
        let mut conn = self.connection()?;
        let tx = conn.transaction()?;
        let chunks = load_chunks(&tx, &key, self.embedder.dimension())?;
        tx.commit()?;
        if chunks.is_empty() {
            return Ok(vec![]);
        }
        let terms: Vec<BTreeMap<String, usize>> = chunks
            .iter()
            .map(|c| {
                let mut counts = BTreeMap::new();
                for token in embedding::tokens(&c.hit.text) {
                    *counts.entry(token).or_default() += 1;
                }
                counts
            })
            .collect();
        let lengths: Vec<usize> = terms.iter().map(|t| t.values().sum()).collect();
        let average = (lengths.iter().sum::<usize>() as f64 / chunks.len() as f64).max(1.0);
        let mut dfs = BTreeMap::new();
        for term in &qtokens {
            dfs.insert(term, terms.iter().filter(|t| t.contains_key(term)).count());
        }
        let mut lexical = Vec::new();
        let mut dense = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let mut bm25 = 0.0;
            for term in &qtokens {
                if let Some(tf) = terms[i].get(term) {
                    let df = dfs[term] as f64;
                    let idf = (1.0 + (chunks.len() as f64 - df + 0.5) / (df + 0.5)).ln();
                    bm25 += idf * (*tf as f64 * 2.5)
                        / (*tf as f64 + 1.5 * (0.25 + 0.75 * lengths[i] as f64 / average));
                }
            }
            if bm25 > 0.0 {
                lexical.push((i, bm25));
            }
            let dot: f64 = chunk
                .vector
                .iter()
                .zip(&qvector)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            if dot > 0.0 {
                dense.push((i, dot));
            }
        }
        let sort = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        lexical.sort_by(sort);
        dense.sort_by(sort);
        let mut fused = BTreeMap::<usize, f64>::new();
        for list in [&lexical, &dense] {
            for (rank, (i, _)) in list.iter().take(200).enumerate() {
                *fused.entry(*i).or_default() += 1.0 / (61 + rank) as f64;
            }
        }
        let mut ranking: Vec<_> = fused.into_iter().collect();
        ranking.sort_by(sort);
        let mut result = Vec::new();
        let mut bytes = 2;
        for (i, score) in ranking.into_iter().take(limit) {
            let mut hit = chunks[i].hit.clone();
            hit.score = score;
            let size = serde_json::to_vec(&hit)?.len() + usize::from(!result.is_empty());
            if bytes + size > MAX_OUTPUT_BYTES {
                break;
            }
            bytes += size;
            result.push(hit);
        }
        Ok(result)
    }
}
fn document_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
    Ok(Document {
        id: r.get(0)?,
        text: r.get(1)?,
        provenance: r.get(2)?,
        hash: r.get(3)?,
        trust: TRUST.into(),
    })
}
fn validate_document(doc: &Document) -> Result<()> {
    identifier(&doc.id)?;
    provenance(&doc.provenance)?;
    ensure!(
        doc.text.len() <= MAX_DOCUMENT_BYTES && hash(&doc.text) == doc.hash,
        "document data/hash corruption"
    );
    Ok(())
}
struct StoredChunk {
    hit: Hit,
    vector: Vec<f32>,
}
fn load_chunks(conn: &Connection, key: &str, dimension: usize) -> Result<Vec<StoredChunk>> {
    let docs: Vec<Document> = conn
        .prepare(
            "SELECT id,text,provenance,hash FROM documents WHERE scope=?1 ORDER BY id LIMIT 257",
        )?
        .query_map([key], document_row)?
        .collect::<rusqlite::Result<_>>()?;
    ensure!(
        docs.len() <= MAX_DOCUMENTS
            && docs.iter().map(|d| d.text.len()).sum::<usize>() <= MAX_SCOPE_BYTES,
        "stored scope capacity corruption"
    );
    let mut result = Vec::new();
    for doc in docs {
        validate_document(&doc)?;
        let ranges = spans(&doc.text);
        let mut statement = conn.prepare("SELECT ordinal,start_byte,end_byte,vector,checksum FROM chunks WHERE scope=?1 AND doc_id=?2 ORDER BY ordinal LIMIT 8193")?;
        let mut rows = statement.query(params![key, doc.id])?;
        let mut ordinal = 0;
        while let Some(row) = rows.next()? {
            let (stored_ordinal, start, end): (usize, usize, usize) =
                (row.get(0)?, row.get(1)?, row.get(2)?);
            ensure!(
                stored_ordinal == ordinal && ranges.get(ordinal) == Some(&(start, end)),
                "chunk span/order corruption"
            );
            let vector: Vec<u8> = row.get(3)?;
            let sum: String = row.get(4)?;
            ensure!(
                checksum(
                    key,
                    &doc.id,
                    ordinal,
                    start,
                    end,
                    &doc.text[start..end],
                    &vector
                ) == sum,
                "chunk checksum corruption"
            );
            let vector = embedding::decode(&vector, dimension)?;
            result.push(StoredChunk {
                hit: Hit {
                    doc_id: doc.id.clone(),
                    text: doc.text[start..end].into(),
                    start_byte: start,
                    end_byte: end,
                    chunk_index: ordinal,
                    score: 0.0,
                    provenance: doc.provenance.clone(),
                    document_hash: doc.hash.clone(),
                    trust: TRUST.into(),
                },
                vector,
            });
            ensure!(
                result.len() <= MAX_CHUNKS,
                "stored chunk capacity corruption"
            );
            ordinal += 1;
        }
        ensure!(ordinal == ranges.len(), "missing document chunks");
    }
    Ok(result)
}
