//! Scope-isolated SQLite persistence with fencing leases and durable tool intents.
//! Audit hashes are tamper-evident chains, NOT signatures or protection against a
//! malicious database owner rewriting the complete chain.
use anyhow::{Result, anyhow, bail, ensure};
use missy_core::{Message, Scope, ToolCall, Usage};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;
mod checkpoint;
mod files;
mod journal;
mod memory;

pub struct Store {
    conn: Mutex<Connection>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Admitted,
    Running,
    Completed,
    Failed,
    Cancelled,
    Interrupted,
}
impl TurnStatus {
    fn text(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "admitted" => Self::Admitted,
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            "interrupted" => Self::Interrupted,
            _ => bail!("invalid stored turn status"),
        })
    }
    fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub id: String,
    pub request_id: String,
    pub prompt: String,
    pub status: TurnStatus,
    pub response: Option<String>,
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Admission {
    pub turn: Turn,
    pub created: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lease {
    pub turn_id: String,
    pub owner: String,
    pub fence: u64,
    pub expires_at: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub transcript: Vec<Message>,
    pub steps: u32,
    pub usage: Usage,
    pub provider_context: Value,
    pub context_hash: String,
    pub system_prompt_hash: String,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Prepared,
    Running,
    Succeeded,
    Failed,
    OutcomeUnknown,
}
impl ToolStatus {
    fn text(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::OutcomeUnknown => "outcome_unknown",
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum ToolBegin {
    Prepared,
    Cached(Value),
    OutcomeUnknown,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolRecord {
    pub call: ToolCall,
    pub status: ToolStatus,
    pub result: Option<Value>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub sequence: i64,
    pub turn_id: String,
    pub kind: String,
    pub payload: Value,
    pub created_at: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub provenance: String,
    pub trust: String,
    pub created_at: i64,
    pub updated_at: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    pub sequence: i64,
    pub kind: String,
    pub payload: Value,
    pub previous_hash: String,
    pub hash: String,
    pub created_at: i64,
}

pub fn hash_text(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_millis()
        .min(i64::MAX as u128) as i64
}
fn scope_key(s: &Scope) -> Result<String> {
    s.validate()?;
    Ok(serde_json::to_string(s)?)
}
fn parse_json<T: serde::de::DeserializeOwned>(s: String) -> Result<T> {
    Ok(serde_json::from_str(&s)?)
}
fn ttl_end(ttl: Duration) -> Result<i64> {
    ensure!(
        ttl >= Duration::from_millis(1) && ttl <= Duration::from_secs(86400),
        "lease TTL must be 1ms..1 day"
    );
    now()
        .checked_add(i64::try_from(ttl.as_millis())?)
        .ok_or_else(|| anyhow!("lease overflow"))
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path != Path::new(":memory:"), "persistent DB path required");
        let _guard = files::OPEN_LOCK
            .lock()
            .map_err(|_| anyhow!("open lock poisoned"))?;
        files::private_database(path)?;
        let mut conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
                | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        conn.busy_timeout(Duration::from_secs(10))?;
        let app: i64 = conn.pragma_query_value(None, "application_id", |r| r.get(0))?;
        ensure!(
            app == 0 || app == 0x4d535953,
            "database belongs to another application"
        );
        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA trusted_schema=OFF;")?;
        files::migrate(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
    fn connection(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn.lock().map_err(|_| anyhow!("store lock poisoned"))
    }
    pub fn admit(&self, scope: &Scope, request_id: &str, prompt: &str) -> Result<Admission> {
        let key = scope_key(scope)?;
        ensure!(
            !request_id.is_empty() && request_id.len() <= 256,
            "invalid request id"
        );
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let prior: Option<(String, String)> = tx
            .query_row(
                "SELECT id,payload_hash FROM turns WHERE scope=?1 AND request_id=?2",
                params![key, request_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((id, hash)) = prior {
            ensure!(hash == hash_text(prompt), "idempotency payload mismatch");
            return Ok(Admission {
                turn: load_turn(&tx, &key, &id)?,
                created: false,
            });
        }
        let id = Uuid::new_v4().to_string();
        let at = now();
        tx.execute("INSERT INTO turns(id,scope,request_id,payload_hash,prompt,status,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,'admitted',?6,?6)",params![id,key,request_id,hash_text(prompt),prompt,at])?;
        record(
            &tx,
            &key,
            &id,
            "admitted",
            &json!({"request_id":request_id,"payload_hash":hash_text(prompt)}),
        )?;
        let turn = load_turn(&tx, &key, &id)?;
        tx.commit()?;
        Ok(Admission {
            turn,
            created: true,
        })
    }
    pub fn get_turn(&self, scope: &Scope, id: &str) -> Result<Turn> {
        let c = self.connection()?;
        load_turn(&c, &scope_key(scope)?, id)
    }
    pub fn list_turns(&self, scope: &Scope, limit: usize) -> Result<Vec<Turn>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        let ids: Vec<String> = {
            let mut q = c.prepare(
                "SELECT id FROM turns WHERE scope=?1 ORDER BY created_at DESC,id LIMIT ?2",
            )?;
            q.query_map(params![key, limit.min(10000) as i64], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        ids.iter().map(|id| load_turn(&c, &key, id)).collect()
    }
    pub fn acquire_lease(
        &self,
        scope: &Scope,
        turn_id: &str,
        owner: &str,
        ttl: Duration,
    ) -> Result<Lease> {
        let key = scope_key(scope)?;
        ensure!(
            !owner.is_empty() && owner.len() <= 256,
            "invalid lease owner"
        );
        ttl_end(ttl)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let turn = load_turn(&tx, &key, turn_id)?;
        ensure!(!turn.status.terminal(), "turn is terminal");
        let prior: Option<(i64, i64, String)> = tx
            .query_row(
                "SELECT fence,expires_at,turn_id FROM leases WHERE scope=?1",
                [&key],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let fence = if let Some((f, e, old)) = prior {
            ensure!(e <= now(), "session already leased");
            mark_uncertain(&tx, &key, &old)?;
            f.checked_add(1).ok_or_else(|| anyhow!("fence exhausted"))?
        } else {
            1
        };
        let expires = ttl_end(ttl)?;
        tx.execute("INSERT INTO leases(scope,turn_id,owner,fence,expires_at) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(scope) DO UPDATE SET turn_id=excluded.turn_id,owner=excluded.owner,fence=excluded.fence,expires_at=excluded.expires_at",params![key,turn_id,owner,fence,expires])?;
        tx.execute(
            "UPDATE turns SET status='running',updated_at=?1 WHERE scope=?2 AND id=?3",
            params![now(), key, turn_id],
        )?;
        record(
            &tx,
            &key,
            turn_id,
            "lease_acquired",
            &json!({"owner":owner,"fence":fence}),
        )?;
        tx.commit()?;
        Ok(Lease {
            turn_id: turn_id.into(),
            owner: owner.into(),
            fence: fence as u64,
            expires_at: expires,
        })
    }
    pub fn heartbeat(&self, scope: &Scope, lease: &Lease, ttl: Duration) -> Result<Lease> {
        let key = scope_key(scope)?;
        ttl_end(ttl)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        let expires = ttl_end(ttl)?;
        tx.execute(
            "UPDATE leases SET expires_at=?1 WHERE scope=?2",
            params![expires, key],
        )?;
        tx.commit()?;
        Ok(Lease {
            expires_at: expires,
            ..lease.clone()
        })
    }
    pub fn release_lease(&self, scope: &Scope, lease: &Lease) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        mark_uncertain(&tx, &key, &lease.turn_id)?;
        tx.execute("UPDATE leases SET expires_at=0 WHERE scope=?1", [&key])?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "lease_released",
            &json!({"fence":lease.fence}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn history(&self, scope: &Scope, limit: usize) -> Result<Vec<Message>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        let mut q=c.prepare("SELECT message FROM(SELECT sequence,message FROM messages WHERE scope=?1 ORDER BY sequence DESC LIMIT ?2) ORDER BY sequence")?;
        q.query_map(params![key, limit.min(10000) as i64], |r| {
            r.get::<_, String>(0)
        })?
        .map(|r| parse_json(r?))
        .collect()
    }
    pub fn append_message(&self, scope: &Scope, lease: &Lease, message: &Message) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        tx.execute(
            "INSERT INTO messages(scope,turn_id,message) VALUES(?1,?2,?3)",
            params![key, lease.turn_id, serde_json::to_string(message)?],
        )?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "message",
            &json!({"role":message.role,"hash":hash_text(&serde_json::to_string(message)?)}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn finish(
        &self,
        scope: &Scope,
        lease: &Lease,
        status: TurnStatus,
        response: Option<&str>,
        error: Option<&str>,
    ) -> Result<Turn> {
        ensure!(
            status.terminal() || status == TurnStatus::Interrupted,
            "invalid final status"
        );
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        // Serialize cancellation against terminal publication. A late worker may
        // still persist a truthful effect receipt, but cannot override cancellation.
        let cancelled: bool = tx.query_row(
            "SELECT cancel_requested FROM turns WHERE scope=?1 AND id=?2",
            params![key, lease.turn_id],
            |r| r.get(0),
        )?;
        let status = if cancelled {
            TurnStatus::Cancelled
        } else {
            status
        };
        let unresolved:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM tools WHERE scope=?1 AND turn_id=?2 AND status IN('prepared','running','outcome_unknown'))",params![key,lease.turn_id],|r|r.get(0))?;
        ensure!(
            status != TurnStatus::Completed || !unresolved,
            "cannot complete a turn with unresolved tool effects"
        );
        mark_uncertain(&tx, &key, &lease.turn_id)?;
        if status == TurnStatus::Completed
            && let Some(cp) = checkpoint::load(&tx, &key, &lease.turn_id)?
        {
            // Checkpoints contain this turn only, never recalled session
            // history. Replace optional staged messages atomically.
            tx.execute(
                "DELETE FROM messages WHERE scope=?1 AND turn_id=?2",
                params![key, lease.turn_id],
            )?;
            for m in cp.transcript.iter().skip(1) {
                tx.execute(
                    "INSERT INTO messages(scope,turn_id,message) VALUES(?1,?2,?3)",
                    params![key, lease.turn_id, serde_json::to_string(m)?],
                )?;
            }
        }
        tx.execute("UPDATE turns SET status=?1,response=?2,error=?3,updated_at=?4 WHERE scope=?5 AND id=?6",params![status.text(),response,error,now(),key,lease.turn_id])?;
        tx.execute("UPDATE leases SET expires_at=0 WHERE scope=?1", [&key])?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "finished",
            &json!({"status":status.text()}),
        )?;
        let turn = load_turn(&tx, &key, &lease.turn_id)?;
        tx.commit()?;
        Ok(turn)
    }
    /// Recovery never executes a tool, never changes live leases, and never
    /// discards uncertainty about an effect that may already have occurred.
    pub fn recover(&self, scope: &Scope) -> Result<Vec<Turn>> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let ids: Vec<String> = {
            let mut q=tx.prepare("SELECT id FROM turns WHERE scope=?1 AND status='running' AND NOT EXISTS(SELECT 1 FROM leases WHERE scope=?1 AND turn_id=turns.id AND expires_at>?2)")?;
            q.query_map(params![key, now()], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        for id in &ids {
            mark_uncertain(&tx, &key, id)?;
            tx.execute(
                "UPDATE turns SET status='interrupted',updated_at=?1 WHERE scope=?2 AND id=?3",
                params![now(), key, id],
            )?;
            record(
                &tx,
                &key,
                id,
                "recovered",
                &json!({"automatic_replay":false}),
            )?;
        }
        let turns = ids
            .iter()
            .map(|id| load_turn(&tx, &key, id))
            .collect::<Result<_>>()?;
        tx.commit()?;
        Ok(turns)
    }
    pub fn request_cancel(&self, scope: &Scope, id: &str) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let t = load_turn(&tx, &key, id)?;
        ensure!(!t.status.terminal(), "turn is terminal");
        tx.execute(
            "UPDATE turns SET cancel_requested=1 WHERE scope=?1 AND id=?2",
            params![key, id],
        )?;
        record(&tx, &key, id, "cancellation_requested", &json!({}))?;
        tx.commit()?;
        Ok(())
    }
    pub fn cancellation_requested(&self, scope: &Scope, id: &str) -> Result<bool> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        c.query_row(
            "SELECT cancel_requested FROM turns WHERE scope=?1 AND id=?2",
            params![key, id],
            |r| r.get(0),
        )
        .optional()?
        .ok_or_else(|| anyhow!("turn not found in scope"))
    }
    pub fn events(&self, scope: &Scope, id: &str) -> Result<Vec<Event>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        load_turn(&c, &key, id)?;
        let mut q=c.prepare("SELECT sequence,kind,payload,created_at FROM events WHERE scope=?1 AND turn_id=?2 ORDER BY sequence")?;
        q.query_map(params![key, id], |r| {
            Ok((
                r.get(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get(3)?,
            ))
        })?
        .map(|r| {
            let (sequence, kind, payload, created_at) = r?;
            Ok(Event {
                sequence,
                turn_id: id.into(),
                kind,
                payload: parse_json(payload)?,
                created_at,
            })
        })
        .collect()
    }
    pub fn audit_entries(&self, scope: &Scope) -> Result<Vec<AuditEntry>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        let mut q=c.prepare("SELECT sequence,kind,payload,previous_hash,hash,created_at FROM audit WHERE scope=?1 ORDER BY sequence")?;
        q.query_map([key], |r| {
            Ok((
                r.get(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            ))
        })?
        .map(|r| {
            let (sequence, kind, payload, previous_hash, hash, created_at) = r?;
            Ok(AuditEntry {
                sequence,
                kind,
                payload: parse_json(payload)?,
                previous_hash,
                hash,
                created_at,
            })
        })
        .collect()
    }
    pub fn verify_audit(&self, scope: &Scope) -> Result<usize> {
        let key = scope_key(scope)?;
        let entries = self.audit_entries(scope)?;
        let mut previous = String::new();
        for e in &entries {
            ensure!(e.previous_hash == previous, "audit previous hash mismatch");
            ensure!(
                audit_hash(
                    &key,
                    e.created_at,
                    &e.kind,
                    &serde_json::to_string(&e.payload)?,
                    &previous
                ) == e.hash,
                "audit hash mismatch"
            );
            previous = e.hash.clone();
        }
        Ok(entries.len())
    }
    /// Retain turn tombstones and payload bindings forever to prevent old
    /// deliveries executing again. Purge content only, never audit chain rows.
    pub fn retention(&self, scope: &Scope, before: i64) -> Result<usize> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM memory_fts WHERE rowid IN(SELECT rowid FROM memories WHERE scope=?1 AND updated_at<?2)",params![key,before])?;
        let memories = tx.execute(
            "DELETE FROM memories WHERE scope=?1 AND updated_at<?2",
            params![key, before],
        )?;
        for table in ["messages", "checkpoints", "events", "tools"] {
            tx.execute(&format!("DELETE FROM {table} WHERE scope=?1 AND turn_id IN(SELECT id FROM turns WHERE scope=?1 AND updated_at<?2 AND status IN('completed','failed','cancelled'))"),params![key,before])?;
        }
        let turns=tx.execute("UPDATE turns SET prompt='',response=NULL,error='content removed by retention',retained=1 WHERE scope=?1 AND updated_at<?2 AND status IN('completed','failed','cancelled') AND retained=0",params![key,before])?;
        audit(
            &tx,
            &key,
            "retention",
            &json!({"before":before,"turns":turns,"memories":memories}),
        )?;
        tx.commit()?;
        Ok(turns + memories)
    }
}
fn load_turn(c: &Connection, key: &str, id: &str) -> Result<Turn> {
    let row=c.query_row("SELECT id,request_id,prompt,status,response,error,created_at,updated_at FROM turns WHERE scope=?1 AND id=?2",params![key,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get::<_,String>(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?))).optional()?;
    let (id, request_id, prompt, status, response, error, created_at, updated_at) =
        row.ok_or_else(|| anyhow!("turn not found in scope"))?;
    Ok(Turn {
        id,
        request_id,
        prompt,
        status: TurnStatus::parse(&status)?,
        response,
        error,
        created_at,
        updated_at,
    })
}
fn check_fence(tx: &Transaction<'_>, key: &str, lease: &Lease) -> Result<()> {
    let active:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM leases l JOIN turns t ON t.id=l.turn_id AND t.scope=l.scope WHERE l.scope=?1 AND l.turn_id=?2 AND l.owner=?3 AND l.fence=?4 AND l.expires_at>?5 AND t.status='running')",params![key,lease.turn_id,lease.owner,lease.fence,now()],|r|r.get(0))?;
    ensure!(active, "stale, expired, or wrong-scope lease");
    Ok(())
}
fn mark_uncertain(tx: &Transaction<'_>, key: &str, id: &str) -> Result<()> {
    let n=tx.execute("UPDATE tools SET status='outcome_unknown',error='executor lease ended before durable settlement' WHERE scope=?1 AND turn_id=?2 AND status IN('prepared','running')",params![key,id])?;
    if n > 0 {
        record(
            tx,
            key,
            id,
            "tools_outcome_unknown",
            &json!({"count":n,"automatic_replay":false}),
        )?;
    }
    Ok(())
}
fn record(tx: &Transaction<'_>, key: &str, id: &str, kind: &str, payload: &Value) -> Result<()> {
    tx.execute(
        "INSERT INTO events(scope,turn_id,kind,payload,created_at) VALUES(?1,?2,?3,?4,?5)",
        params![key, id, kind, serde_json::to_string(payload)?, now()],
    )?;
    audit(tx, key, kind, &json!({"turn_id":id,"data":payload}))
}
fn audit_hash(key: &str, at: i64, kind: &str, payload: &str, previous: &str) -> String {
    hash_text(
        &serde_json::to_string(&(key, at, kind, payload, previous))
            .expect("serializable audit tuple"),
    )
}
fn audit(tx: &Transaction<'_>, key: &str, kind: &str, payload: &Value) -> Result<()> {
    let previous: String = tx
        .query_row(
            "SELECT hash FROM audit WHERE scope=?1 ORDER BY sequence DESC LIMIT 1",
            [key],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or_default();
    let at = now();
    let payload = serde_json::to_string(payload)?;
    let hash = audit_hash(key, at, kind, &payload, &previous);
    tx.execute("INSERT INTO audit(scope,kind,payload,previous_hash,hash,created_at) VALUES(?1,?2,?3,?4,?5,?6)",params![key,kind,payload,previous,hash,at])?;
    Ok(())
}
