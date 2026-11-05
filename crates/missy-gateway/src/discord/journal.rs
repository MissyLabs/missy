//! Single-owner private SQLite intake, checkpoint, governor and outbound journal.
use super::*;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use std::{
    fs::{File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    sync::Mutex,
};
pub(super) const QUEUE_LIMIT: i64 = 256;
pub(super) fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}
pub(super) struct Journal {
    conn: Mutex<Connection>,
    _lock: File,
}
pub(super) struct Delivery {
    pub message: Incoming,
    pub nonce: String,
    pub text: String,
    pub state: String,
}
#[derive(Serialize)]
pub struct OutboxReview {
    pub nonce: String,
    pub request_id: String,
    pub state: String,
    pub binding_sha256: String,
}
fn review_hash(nonce: &str, request: &str, state: &str, text: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&json!([
                "discord-outbox-review-v1",
                nonce,
                request,
                state,
                text
            ]))
            .expect("JSON strings")
        )
    )
}
impl Journal {
    pub fn open(dir: &Path) -> Result<Self> {
        ensure!(
            dir.is_absolute() && std::fs::canonicalize(dir)? == dir,
            "Discord state directory must be an existing canonical absolute path"
        );
        let meta = std::fs::symlink_metadata(dir)?;
        // SAFETY: geteuid has no preconditions or side effects.
        let uid = unsafe { libc::geteuid() };
        ensure!(
            meta.is_dir() && meta.uid() == uid && meta.mode() & 0o077 == 0,
            "Discord state directory must be owned by caller and private (0700)"
        );
        let private_file = |name: &str| -> Result<File> {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(dir.join(name))?;
            let meta = file.metadata()?;
            ensure!(
                meta.is_file()
                    && meta.uid() == uid
                    && meta.mode() & 0o077 == 0
                    && meta.nlink() == 1,
                "Discord state files must be private, regular, single-link caller-owned files"
            );
            Ok(file)
        };
        let lock = private_file("discord.lock")?;
        // SAFETY: live owned descriptor, released automatically on drop.
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "Discord state directory already in use"
        );
        drop(private_file("discord.sqlite")?);
        for name in [
            "discord.sqlite-wal",
            "discord.sqlite-shm",
            "discord.sqlite-journal",
        ] {
            if std::fs::symlink_metadata(dir.join(name)).is_ok() {
                drop(private_file(name)?);
            }
        }
        let conn = Connection::open(dir.join("discord.sqlite"))?;
        conn.busy_timeout(Duration::from_secs(2))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON; PRAGMA max_page_count=65536;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS intake(id TEXT PRIMARY KEY,payload TEXT NOT NULL,hash TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'pending',denied INTEGER NOT NULL,ack INTEGER NOT NULL DEFAULT 0,interaction INTEGER NOT NULL,checked INTEGER NOT NULL DEFAULT 0,resume_started INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS governor(principal TEXT PRIMARY KEY,start INTEGER NOT NULL,count INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS prepared_context(id TEXT PRIMARY KEY REFERENCES intake(id),prompt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inbound_evidence(id TEXT NOT NULL REFERENCES intake(id),source TEXT NOT NULL,data TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(id,source));
            CREATE TABLE IF NOT EXISTS result_bindings(id TEXT PRIMARY KEY REFERENCES intake(id),sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operator_resolutions(nonce TEXT PRIMARY KEY,decision TEXT NOT NULL,binding TEXT NOT NULL,operator TEXT NOT NULL,proof_sha256 TEXT NOT NULL,at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(nonce TEXT PRIMARY KEY,request TEXT NOT NULL REFERENCES intake(id),chunk INTEGER NOT NULL,text TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'ready',receipt TEXT,attempts INTEGER NOT NULL DEFAULT 0,not_before INTEGER NOT NULL DEFAULT 0,UNIQUE(request,chunk));
            CREATE INDEX IF NOT EXISTS intake_work ON intake(state,checked);
            CREATE INDEX IF NOT EXISTS outbox_work ON outbox(state,not_before);")?;
        Ok(Self {
            conn: Mutex::new(conn),
            _lock: lock,
        })
    }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Discord journal lock poisoned"))
    }
    pub fn bind_bot(&self, bot: &str) -> Result<()> {
        let conn = self.lock()?;
        conn.execute("INSERT OR IGNORE INTO meta VALUES('bot',?1)", [bot])?;
        let existing: String =
            conn.query_row("SELECT value FROM meta WHERE key='bot'", [], |r| r.get(0))?;
        ensure!(
            existing == bot,
            "Discord state directory belongs to a different bot"
        );
        Ok(())
    }
    pub fn resume(&self) -> Result<Resume> {
        let value: Option<String> = self
            .lock()?
            .query_row("SELECT value FROM meta WHERE key='resume'", [], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(match value {
            Some(s) => serde_json::from_str(&s)?,
            None => Resume::default(),
        })
    }
    pub fn session_budget(&self, remaining: u64, reset_after: u64) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO meta VALUES('session_remaining',?1)",
            [remaining.to_string()],
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO meta VALUES('session_reset_after_ms',?1)",
            [reset_after.to_string()],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn reserve_identify(&self, now: i64) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let last: Option<String> = tx
            .query_row("SELECT value FROM meta WHERE key='identify'", [], |r| {
                r.get(0)
            })
            .optional()?;
        let remaining: Option<String> = tx
            .query_row(
                "SELECT value FROM meta WHERE key='session_remaining'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(remaining) = remaining {
            let remaining = remaining.parse::<u64>()?;
            ensure!(
                remaining > 0,
                "Discord session-start budget exhausted; authenticated preflight refresh required"
            );
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='session_remaining'",
                [(remaining - 1).to_string()],
            )?;
        }
        // Single shard, conservative persistent throttle, including failed writes.
        ensure!(
            last.is_none_or(|s| s
                .parse::<i64>()
                .is_ok_and(|t| now.saturating_sub(t) >= 60_000)),
            "Discord identify governor waiting"
        );
        tx.execute(
            "INSERT OR REPLACE INTO meta VALUES('identify',?1)",
            [now.to_string()],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn intake(&self, message: Option<&Incoming>, resume: &Resume, now: i64) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(message) = message {
            let payload = serde_json::to_string(message)?;
            let hash = format!("{:x}", Sha256::digest(payload.as_bytes()));
            let old: Option<String> = tx
                .query_row(
                    "SELECT hash FROM intake WHERE id=?1",
                    [&message.request_id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(old) = old {
                ensure!(
                    old == hash,
                    "Discord occurrence payload changed; checkpoint refused"
                );
            } else {
                let queued: i64 = tx.query_row(
                    "SELECT count(*) FROM intake WHERE state NOT IN ('done','blocked')",
                    [],
                    |r| r.get(0),
                )?;
                ensure!(
                    queued < QUEUE_LIMIT,
                    "Discord durable queue full; checkpoint refused for backpressure"
                );
                let rate: Option<(i64, i64)> = tx
                    .query_row(
                        "SELECT start,count FROM governor WHERE principal=?1",
                        [&message.scope.user],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                let (start, count) = match rate {
                    Some((start, count)) if now.saturating_sub(start) < 60_000 => (start, count),
                    _ => (now, 0),
                };
                let denied = count >= 30;
                tx.execute(
                    "INSERT OR REPLACE INTO governor VALUES(?1,?2,?3)",
                    params![message.scope.user, start, (count + 1).min(31)],
                )?;
                tx.execute(
                    "INSERT INTO intake(id,payload,hash,denied,interaction) VALUES(?1,?2,?3,?4,?5)",
                    params![
                        message.request_id,
                        payload,
                        hash,
                        denied,
                        message.interaction.is_some()
                    ],
                )?;
            }
        }
        tx.execute(
            "INSERT OR REPLACE INTO meta VALUES('resume',?1)",
            [serde_json::to_string(resume)?],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn pending(&self) -> Result<Option<Incoming>> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        // Preserve admission order within one conversation. Control operations
        // bypass this queue so status/cancel stay responsive during a long ask.
        let row: Option<(String,String)> = tx.query_row("SELECT i.id,i.payload FROM intake i WHERE i.state='pending' AND (json_type(i.payload,'$.action.Ask') IS NULL AND json_type(i.payload,'$.action.Resume') IS NULL OR NOT EXISTS(SELECT 1 FROM intake p WHERE p.state='pending' AND p.rowid<i.rowid AND json_extract(p.payload,'$.scope')=json_extract(i.payload,'$.scope') AND (json_type(p.payload,'$.action.Ask') IS NOT NULL OR json_type(p.payload,'$.action.Resume') IS NOT NULL))) ORDER BY i.checked,i.rowid LIMIT 1", [], |r| Ok((r.get(0)?,r.get(1)?))).optional()?;
        let result = if let Some((id, payload)) = row {
            tx.execute(
                "UPDATE intake SET checked=?2 WHERE id=?1",
                params![id, now_ms()],
            )?;
            Some(serde_json::from_str(&payload)?)
        } else {
            None
        };
        tx.commit()?;
        Ok(result)
    }
    pub fn claim_resume(&self, id: &str) -> Result<bool> {
        Ok(self.lock()?.execute(
            "UPDATE intake SET resume_started=1 WHERE id=?1 AND resume_started=0",
            [id],
        )? == 1)
    }
    pub fn prepared_prompt(&self, id: &str) -> Result<Option<String>> {
        Ok(self
            .lock()?
            .query_row(
                "SELECT prompt FROM prepared_context WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?)
    }
    pub fn retain_inbound(&self, id: &str, source: &str, data: &Value) -> Result<()> {
        let encoded = serde_json::to_string(data)?;
        ensure!(
            encoded.len() <= 8_388_608,
            "inbound evidence bound exceeded"
        );
        let hash = format!("{:x}", Sha256::digest(encoded.as_bytes()));
        let mut conn = self.lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT OR IGNORE INTO inbound_evidence VALUES(?1,?2,?3,?4)",
            params![id, source, encoded, hash],
        )?;
        let old: String = tx.query_row(
            "SELECT sha256 FROM inbound_evidence WHERE id=?1 AND source=?2",
            params![id, source],
            |r| r.get(0),
        )?;
        ensure!(
            old == hash,
            "inbound evidence changed; refusing a divergent source snapshot"
        );
        tx.commit()?;
        Ok(())
    }
    pub fn prepare_prompt(&self, id: &str, prompt: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT OR IGNORE INTO prepared_context(id,prompt) VALUES(?1,?2)",
            params![id, prompt],
        )?;
        let stored: String = tx.query_row(
            "SELECT prompt FROM prepared_context WHERE id=?1",
            [id],
            |r| r.get(0),
        )?;
        ensure!(stored == prompt, "Discord prepared context mismatch");
        tx.commit()?;
        Ok(())
    }
    pub fn pending_ack(&self) -> Result<Option<Incoming>> {
        let conn = self.lock()?;
        let row: Option<String> = conn.query_row("SELECT payload FROM intake WHERE interaction=1 AND ack<2 AND state!='blocked' ORDER BY rowid LIMIT 1", [], |r| r.get(0)).optional()?;
        row.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }
    pub fn acking(&self, id: &str) -> Result<()> {
        self.lock()?
            .execute("UPDATE intake SET ack=1 WHERE id=?1", [id])?;
        Ok(())
    }
    pub fn acked(&self, id: &str) -> Result<()> {
        self.lock()?
            .execute("UPDATE intake SET ack=2 WHERE id=?1", [id])?;
        Ok(())
    }
    pub fn rate_denied(&self, id: &str) -> Result<bool> {
        Ok(self
            .lock()?
            .query_row("SELECT denied FROM intake WHERE id=?1", [id], |r| r.get(0))?)
    }
    pub fn block(&self, id: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute("UPDATE intake SET state='blocked' WHERE id=?1", [id])?;
        tx.execute(
            "UPDATE outbox SET state='blocked' WHERE request=?1 AND state!='sent'",
            [id],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn result(&self, message: &Incoming, text: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
        tx.execute(
            "INSERT OR IGNORE INTO result_bindings VALUES(?1,?2)",
            params![message.request_id, hash],
        )?;
        let old: String = tx.query_row(
            "SELECT sha256 FROM result_bindings WHERE id=?1",
            [&message.request_id],
            |r| r.get(0),
        )?;
        ensure!(
            old == hash,
            "Discord result changed after outbox preparation; divergent delivery refused"
        );
        let mut output = chunks(text);
        if message.interaction.is_some() && output.len() > 1 {
            output.truncate(1);
            output[0].push_str("\n[Truncated; inspect scoped history.]");
        }
        if output.is_empty() {
            output.push("Turn finished without text.".into());
        }
        for (index, chunk) in output.iter().enumerate() {
            tx.execute(
                "INSERT OR IGNORE INTO outbox(nonce,request,chunk,text) VALUES(?1,?2,?3,?4)",
                params![
                    nonce(&message.request_id, index),
                    message.request_id,
                    index,
                    chunk
                ],
            )?;
        }
        tx.execute(
            "UPDATE intake SET state='result' WHERE id=?1 AND state='pending'",
            [&message.request_id],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn next_delivery(&self) -> Result<Option<Delivery>> {
        let conn = self.lock()?;
        let row: Option<(String,String,String,String)> = conn.query_row("SELECT i.payload,o.nonce,o.text,o.state FROM outbox o JOIN intake i ON i.id=o.request WHERE i.state='result' AND o.state IN ('ready','sending') AND o.not_before<=?1 AND (i.interaction=0 OR i.ack=2) AND NOT EXISTS(SELECT 1 FROM outbox prev WHERE prev.request=o.request AND prev.chunk<o.chunk AND prev.state!='sent') ORDER BY o.rowid LIMIT 1", [now_ms()], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?;
        row.map(|(payload, nonce, text, state)| {
            Ok(Delivery {
                message: serde_json::from_str(&payload)?,
                nonce,
                text,
                state,
            })
        })
        .transpose()
    }
    pub fn sending(&self, nonce: &str) -> Result<()> {
        self.lock()?.execute("UPDATE outbox SET state='sending',attempts=attempts+1 WHERE nonce=?1 AND state IN ('ready','sending')", [nonce])?;
        Ok(())
    }
    pub fn retry(&self, nonce: &str, after: i64) -> Result<()> {
        self.lock()?.execute("UPDATE outbox SET state=CASE WHEN attempts<3 THEN 'ready' ELSE 'uncertain' END,not_before=?2 WHERE nonce=?1", params![nonce,after])?;
        Ok(())
    }
    pub fn uncertain(&self, nonce: &str) -> Result<()> {
        self.lock()?.execute(
            "UPDATE outbox SET state='uncertain' WHERE nonce=?1",
            [nonce],
        )?;
        Ok(())
    }
    pub fn receipt(&self, nonce: &str, receipt: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute(
            "UPDATE outbox SET state='sent',receipt=?2 WHERE nonce=?1",
            params![nonce, receipt],
        )?;
        tx.execute("UPDATE intake SET state='done' WHERE id=(SELECT request FROM outbox WHERE nonce=?1) AND NOT EXISTS(SELECT 1 FROM outbox o WHERE o.request=intake.id AND o.state!='sent')", [nonce])?;
        tx.commit()?;
        Ok(())
    }
    pub fn review_outbox(&self) -> Result<Vec<OutboxReview>> {
        let conn = self.lock()?;
        let mut q=conn.prepare("SELECT nonce,request,state,text FROM outbox WHERE state IN ('sending','uncertain') ORDER BY rowid LIMIT 1000")?;
        let rows = q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        rows.map(|row| {
            let (nonce, request, state, text) = row?;
            let binding_sha256 = review_hash(&nonce, &request, &state, &text);
            Ok(OutboxReview {
                nonce,
                request_id: request,
                state,
                binding_sha256,
            })
        })
        .collect()
    }
    pub fn resolve_outbox(
        &self,
        nonce: &str,
        binding: &str,
        operator: &str,
        receipt: Option<&str>,
        proof: &str,
    ) -> Result<()> {
        ensure!(
            !operator.is_empty()
                && operator.len() <= 128
                && operator
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_:-.".contains(&b)),
            "invalid operator identity"
        );
        ensure!(
            !proof.trim().is_empty() && proof.len() <= 16_384,
            "operator evidence required"
        );
        if let Some(receipt) = receipt {
            ensure!(snowflake(receipt), "proven Discord message ID required");
        }
        let mut conn = self.lock()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (request, state, text): (String, String, String) = tx.query_row(
            "SELECT request,state,text FROM outbox WHERE nonce=?1",
            [nonce],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        ensure!(
            matches!(state.as_str(), "sending" | "uncertain"),
            "outbox is not awaiting reconciliation"
        );
        ensure!(
            review_hash(nonce, &request, &state, &text) == binding,
            "outbox review binding changed"
        );
        tx.execute(
            "INSERT INTO operator_resolutions VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                nonce,
                if receipt.is_some() {
                    "proven_delivered"
                } else {
                    "abandon_without_retry"
                },
                binding,
                operator,
                format!("{:x}", Sha256::digest(proof.as_bytes())),
                now_ms()
            ],
        )?;
        if let Some(receipt) = receipt {
            tx.execute(
                "UPDATE outbox SET state='sent',receipt=?2 WHERE nonce=?1",
                params![nonce, receipt],
            )?;
            tx.execute("UPDATE intake SET state='done' WHERE id=?1 AND NOT EXISTS(SELECT 1 FROM outbox WHERE request=?1 AND state!='sent')",[&request])?;
        } else {
            tx.execute("UPDATE intake SET state='blocked' WHERE id=?1", [&request])?;
            tx.execute(
                "UPDATE outbox SET state='blocked' WHERE request=?1 AND state!='sent'",
                [&request],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn has_uncertain(&self) -> Result<bool> {
        Ok(self.lock()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM outbox WHERE state='uncertain')",
            [],
            |r| r.get(0),
        )?)
    }
}
