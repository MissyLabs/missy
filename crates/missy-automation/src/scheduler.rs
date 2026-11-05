use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use futures_util::{StreamExt, stream::FuturesUnordered};
use missy_core::Scope;
use missy_runtime::Runtime;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const LEASE_SECONDS: i64 = 30;
static OPEN_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewJob {
    pub prompt: String,
    pub interval_seconds: u64,
    pub first_due: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub scope: Scope,
    pub prompt: String,
    pub interval_seconds: u64,
    pub next_due: DateTime<Utc>,
    pub enabled: bool,
    pub schedule_spec: Option<missy_schedule_text::ScheduleSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OccurrenceStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Occurrence {
    pub request_id: String,
    pub job_id: String,
    pub due: DateTime<Utc>,
    pub status: OccurrenceStatus,
    pub turn_id: Option<String>,
    /// Bounded public diagnostic code, never raw provider/tool error text.
    pub error: Option<String>,
}

/// Durable scheduler backed by its own private SQLite file. The caller supplies
/// a SERVER-TRUSTED Scope, not a user-supplied claim. Opening never launches work,
/// sweeps leases, resumes old turns, or changes the Runtime's Store.
#[derive(Clone)]
pub struct Scheduler {
    db: Arc<Mutex<Connection>>,
    runtime: Arc<Runtime>,
}

#[derive(Clone)]
struct Claim {
    occurrence: Occurrence,
    prompt: String,
    owner: String,
    fence: i64,
}

fn scope_key(scope: &Scope) -> Result<String> {
    scope.validate()?;
    Ok(serde_json::to_string(scope)?)
}
fn timestamp(t: i64) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp(t, 0).context("invalid persisted timestamp")
}
// SQLite due dates have whole-second resolution. Spec dates must round up,
// never down: flooring a fractional one-shot/anchor can run early and leave
// next_occurrence_after(due) pointing at the occurrence that just completed.
fn ceil_second(date: DateTime<Utc>) -> Result<DateTime<Utc>> {
    let seconds = date
        .timestamp()
        .checked_add(i64::from(date.timestamp_subsec_nanos() != 0))
        .context("schedule date exceeds UTC range")?;
    timestamp(seconds)
}
fn status(s: &str) -> Result<OccurrenceStatus> {
    Ok(match s {
        "running" => OccurrenceStatus::Running,
        "completed" => OccurrenceStatus::Completed,
        "failed" => OccurrenceStatus::Failed,
        "cancelled" => OccurrenceStatus::Cancelled,
        "unknown" => OccurrenceStatus::Unknown,
        _ => bail!("invalid persisted occurrence status"),
    })
}
fn status_str(s: &OccurrenceStatus) -> &'static str {
    match s {
        OccurrenceStatus::Running => "running",
        OccurrenceStatus::Completed => "completed",
        OccurrenceStatus::Failed => "failed",
        OccurrenceStatus::Cancelled => "cancelled",
        OccurrenceStatus::Unknown => "unknown",
    }
}

impl Scheduler {
    /// Parent directory must already exist, be canonical, owned by this account,
    /// and private (0700). Existing DBs must be regular, owned, private and not links.
    pub fn open(path: impl AsRef<Path>, runtime: Arc<Runtime>) -> Result<Self> {
        let _guard = OPEN_LOCK
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler open mutex poisoned"))?;
        let path = path.as_ref();
        if !path.is_absolute() || path.file_name().is_none() {
            bail!("absolute scheduler database path required");
        }
        let parent = path.parent().context("missing database parent")?;
        crate::skills::reject_symlink_components(parent)?;
        if fs::canonicalize(parent)? != parent {
            bail!("canonical database parent required");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
            let m = fs::metadata(parent)?;
            if m.uid() != unsafe { libc::geteuid() } || m.permissions().mode() & 0o077 != 0 {
                bail!("database parent must be owned and private");
            }
            match fs::symlink_metadata(path) {
                Ok(m) => {
                    if !m.is_file()
                        || m.file_type().is_symlink()
                        || m.uid() != unsafe { libc::geteuid() }
                        || m.permissions().mode() & 0o077 != 0
                        || m.nlink() != 1
                    {
                        bail!("existing database must be owned private regular file without links");
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Never open/close an existing DB descriptor: doing so could drop
                    // SQLite's process-wide POSIX locks. create_new is mandatory.
                    match fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .open(path)
                    {
                        Ok(file) => drop(file),
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                        Err(e) => return Err(e.into()),
                    }
                    let m = fs::symlink_metadata(path)?;
                    if !m.is_file()
                        || m.file_type().is_symlink()
                        || m.uid() != unsafe { libc::geteuid() }
                        || m.permissions().mode() & 0o077 != 0
                        || m.nlink() != 1
                    {
                        bail!("unsafe database creation race");
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        #[cfg(not(unix))]
        bail!("private scheduler database requires Unix ownership validation");
        let mut db = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
                | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        db.busy_timeout(Duration::from_secs(5))?;
        // Refuse unrelated databases before changing their persistent journal
        // mode. This is only a preflight; authoritative checks repeat under lock.
        let preflight = db.transaction()?;
        let application_id: i64 = preflight.query_row("PRAGMA application_id", [], |r| r.get(0))?;
        let tables: i64 = preflight.query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        let version: i64 = preflight.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if application_id != 0x4d535941 && (application_id != 0 || tables != 0) {
            bail!("scheduler requires its own database, not a runtime or unrelated database");
        }
        if !(0..=2).contains(&version) {
            bail!("scheduler schema version unsupported");
        }
        preflight.commit()?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")?;
        // The process-local OPEN_LOCK cannot serialize other processes. Inspect
        // and migrate under the same SQLite write lock, including on new DBs.
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let application_id: i64 = tx.query_row("PRAGMA application_id", [], |r| r.get(0))?;
        let tables: i64 = tx.query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        if application_id != 0x4d535941 && (application_id != 0 || tables != 0) {
            bail!("scheduler requires its own database, not a runtime or unrelated database");
        }
        let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if !(0..=2).contains(&version) {
            bail!("scheduler schema version unsupported");
        }
        tx.execute_batch("
          PRAGMA application_id=1297307969;
          CREATE TABLE IF NOT EXISTS automation_jobs(
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, prompt TEXT NOT NULL,
            interval_seconds INTEGER NOT NULL CHECK(interval_seconds > 0),
            next_due INTEGER NOT NULL, enabled INTEGER NOT NULL, removed INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS automation_due ON automation_jobs(scope,enabled,next_due);
          CREATE TABLE IF NOT EXISTS automation_occurrences(
            request_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES automation_jobs(id),
            due INTEGER NOT NULL, status TEXT NOT NULL, owner TEXT NOT NULL,
            fence INTEGER NOT NULL, lease_until INTEGER NOT NULL, turn_id TEXT, error TEXT,
            UNIQUE(job_id,due));")?;
        if version < 2 {
            tx.execute_batch(
                "ALTER TABLE automation_jobs ADD COLUMN schedule_spec TEXT; PRAGMA user_version=2;",
            )?;
        }
        let migrated_version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if migrated_version != 2 {
            bail!("scheduler schema migration failed");
        }
        tx.commit()?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            runtime,
        })
    }

    pub fn add(&self, scope: &Scope, job: NewJob) -> Result<Job> {
        let key = scope_key(scope)?;
        if job.prompt.trim().is_empty()
            || job.prompt.len() > 65_536
            || job.interval_seconds == 0
            || job.interval_seconds > 315_576_000
        {
            bail!("job requires bounded prompt and interval 1..315576000 seconds");
        }
        if job
            .first_due
            .timestamp()
            .checked_add(job.interval_seconds as i64)
            .and_then(|t| DateTime::from_timestamp(t, 0))
            .is_none()
        {
            bail!("job date exceeds UTC range");
        }
        let id = Uuid::new_v4().to_string();
        let due = timestamp(job.first_due.timestamp())?;
        self.db.lock().map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?.execute("INSERT INTO automation_jobs(id,scope,prompt,interval_seconds,next_due,enabled) VALUES(?1,?2,?3,?4,?5,1)",params![id,key,job.prompt,job.interval_seconds as i64,due.timestamp()])?;
        Ok(Job {
            id,
            scope: scope.clone(),
            prompt: job.prompt,
            interval_seconds: job.interval_seconds,
            next_due: due,
            enabled: true,
            schedule_spec: None,
        })
    }

    pub fn add_spec(
        &self,
        scope: &Scope,
        prompt: String,
        spec: missy_schedule_text::ScheduleSpec,
        after: DateTime<Utc>,
    ) -> Result<Job> {
        let key = scope_key(scope)?;
        if prompt.trim().is_empty() || prompt.len() > 65536 {
            bail!("bounded prompt required");
        }
        let next = ceil_second(spec.next_after(after)?)?;
        let id = Uuid::new_v4().to_string();
        self.db.lock().map_err(|_|anyhow::anyhow!("scheduler mutex poisoned"))?.execute("INSERT INTO automation_jobs(id,scope,prompt,interval_seconds,next_due,enabled,schedule_spec) VALUES(?1,?2,?3,1,?4,1,?5)",params![id,key,prompt,next.timestamp(),serde_json::to_string(&spec)?])?;
        Ok(Job {
            id,
            scope: scope.clone(),
            prompt,
            interval_seconds: 0,
            next_due: next,
            enabled: true,
            schedule_spec: Some(spec),
        })
    }

    pub fn list(&self, scope: &Scope) -> Result<Vec<Job>> {
        let key = scope_key(scope)?;
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let mut q=db.prepare("SELECT id,prompt,interval_seconds,next_due,enabled,schedule_spec FROM automation_jobs WHERE scope=?1 AND removed=0 ORDER BY next_due,id")?;
        let rows = q.query_map([key], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, u64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, bool>(4)?,
                r.get::<_, Option<String>>(5)?,
            ))
        })?;
        rows.map(|r| {
            let (id, prompt, interval_seconds, due, enabled, spec) = r?;
            let schedule_spec = spec.map(|s| serde_json::from_str(&s)).transpose()?;
            Ok(Job {
                id,
                scope: scope.clone(),
                prompt,
                interval_seconds: if schedule_spec.is_some() {
                    0
                } else {
                    interval_seconds
                },
                next_due: timestamp(due)?,
                enabled,
                schedule_spec,
            })
        })
        .collect()
    }

    pub fn pause(&self, scope: &Scope, id: &str, paused: bool) -> Result<()> {
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if !paused {
            let unknown:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM automation_occurrences o JOIN automation_jobs j ON j.id=o.job_id WHERE j.id=?1 AND j.scope=?2 AND o.status='unknown')",params![id,scope_key(scope)?],|r|r.get(0))?;
            if unknown {
                bail!(
                    "job has an uncertain occurrence; remove and create a new job only after operator reconciliation"
                );
            }
            let spec: Option<String> = tx.query_row(
                "SELECT schedule_spec FROM automation_jobs WHERE id=?1 AND scope=?2 AND removed=0",
                params![id, scope_key(scope)?],
                |r| r.get(0),
            ).optional()?.flatten();
            if let Some(spec) = spec {
                let spec: missy_schedule_text::ScheduleSpec = serde_json::from_str(&spec)?;
                let terminal: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM automation_occurrences WHERE job_id=?1 AND status!='running')",
                    [id],
                    |r| r.get(0),
                )?;
                if spec.is_one_shot() && terminal {
                    bail!(
                        "one-shot job is exhausted; create a new job to schedule another occurrence"
                    );
                }
            }
        }
        let n = tx.execute(
            "UPDATE automation_jobs SET enabled=?1 WHERE id=?2 AND scope=?3 AND removed=0",
            params![!paused, id, scope_key(scope)?],
        )?;
        if n != 1 {
            bail!("job not found in scope");
        }
        tx.commit()?;
        Ok(())
    }

    /// Tombstone preserves audit/occurrence history. Already claimed work drains;
    /// removal does not pretend that an executing effect can be undone.
    pub fn remove(&self, scope: &Scope, id: &str) -> Result<()> {
        let n=self.db.lock().map_err(|_|anyhow::anyhow!("scheduler mutex poisoned"))?.execute("UPDATE automation_jobs SET removed=1,enabled=0 WHERE id=?1 AND scope=?2 AND removed=0",params![id,scope_key(scope)?])?;
        if n != 1 {
            bail!("job not found in scope");
        }
        Ok(())
    }

    pub fn occurrences(&self, scope: &Scope, limit: usize) -> Result<Vec<Occurrence>> {
        if !(1..=1000).contains(&limit) {
            bail!("occurrence limit must be 1..1000");
        }
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let mut q=db.prepare("SELECT o.request_id,o.job_id,o.due,o.status,o.turn_id,o.error FROM automation_occurrences o JOIN automation_jobs j ON j.id=o.job_id WHERE j.scope=?1 ORDER BY o.due DESC,o.request_id LIMIT ?2")?;
        let rows = q.query_map(params![scope_key(scope)?, limit as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
            ))
        })?;
        rows.map(|r| {
            let (request_id, job_id, due, s, turn_id, error) = r?;
            Ok(Occurrence {
                request_id,
                job_id,
                due: timestamp(due)?,
                status: status(&s)?,
                turn_id,
                error,
            })
        })
        .collect()
    }

    fn claim(&self, scope: &Scope, now: i64) -> Result<Option<Claim>> {
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Runtime sessions serialize their durable transcript. Do not create
        // competing runtime leases for different jobs in the same scope.
        let row=tx.query_row("SELECT j.id,j.prompt,j.next_due,o.request_id,o.fence FROM automation_jobs j LEFT JOIN automation_occurrences o ON o.job_id=j.id AND o.due=j.next_due WHERE j.scope=?1 AND j.enabled=1 AND j.removed=0 AND j.next_due<=?2 AND (o.request_id IS NULL OR (o.status='running' AND o.lease_until<=?2)) AND NOT EXISTS (SELECT 1 FROM automation_occurrences active JOIN automation_jobs aj ON aj.id=active.job_id WHERE aj.scope=j.scope AND active.status='running' AND active.lease_until>?2) ORDER BY j.next_due,j.id LIMIT 1",params![scope_key(scope)?,now],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,i64>(2)?,r.get::<_,Option<String>>(3)?,r.get::<_,Option<i64>>(4)?))).optional()?;
        let Some((job_id, prompt, due, existing, old_fence)) = row else {
            tx.commit()?;
            return Ok(None);
        };
        let request_id = existing.unwrap_or_else(|| format!("automation:{}", Uuid::new_v4()));
        let fence = old_fence
            .unwrap_or(0)
            .checked_add(1)
            .context("claim fence overflow")?;
        let owner = Uuid::new_v4().to_string();
        let changed=tx.execute("INSERT INTO automation_occurrences(request_id,job_id,due,status,owner,fence,lease_until) VALUES(?1,?2,?3,'running',?4,?5,?6) ON CONFLICT(job_id,due) DO UPDATE SET owner=excluded.owner,fence=excluded.fence,lease_until=excluded.lease_until WHERE automation_occurrences.status='running' AND automation_occurrences.lease_until<=?7",params![request_id,job_id,due,owner,fence,now+LEASE_SECONDS,now])?;
        if changed != 1 {
            bail!("claim conflict");
        }
        tx.commit()?;
        Ok(Some(Claim {
            occurrence: Occurrence {
                request_id,
                job_id,
                due: timestamp(due)?,
                status: OccurrenceStatus::Running,
                turn_id: None,
                error: None,
            },
            prompt,
            owner,
            fence,
        }))
    }

    fn heartbeat(&self, c: &Claim) -> Result<()> {
        let changed=self.db.lock().map_err(|_|anyhow::anyhow!("scheduler mutex poisoned"))?.execute("UPDATE automation_occurrences SET lease_until=?1 WHERE request_id=?2 AND owner=?3 AND fence=?4 AND status='running' AND lease_until>?5",params![Utc::now().timestamp()+LEASE_SECONDS,c.occurrence.request_id,c.owner,c.fence,Utc::now().timestamp()])?;
        if changed != 1 {
            bail!("scheduler claim lease lost");
        }
        Ok(())
    }

    fn finish(&self, c: &Claim, result: &Occurrence) -> Result<()> {
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed=tx.execute("UPDATE automation_occurrences SET status=?1,turn_id=?2,error=?3,lease_until=0 WHERE request_id=?4 AND owner=?5 AND fence=?6 AND status='running' AND lease_until>?7",params![status_str(&result.status),result.turn_id,result.error,c.occurrence.request_id,c.owner,c.fence,Utc::now().timestamp()])?;
        if changed != 1 {
            bail!("scheduler claim lost before completion; runtime request remains durable");
        }
        let (interval, spec): (i64, Option<String>) = tx.query_row(
            "SELECT interval_seconds,schedule_spec FROM automation_jobs WHERE id=?1",
            [&c.occurrence.job_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let next = if let Some(spec) = spec {
            let spec: missy_schedule_text::ScheduleSpec = serde_json::from_str(&spec)?;
            spec.next_occurrence_after(c.occurrence.due)?
                .map(ceil_second)
                .transpose()?
                .map(|next| next.timestamp())
        } else {
            c.occurrence
                .due
                .timestamp()
                .checked_add(interval)
                .filter(|t| DateTime::from_timestamp(*t, 0).is_some())
        };
        if let Some(next) = next {
            tx.execute(
                "UPDATE automation_jobs SET next_due=?1 WHERE id=?2 AND next_due=?3",
                params![next, c.occurrence.job_id, c.occurrence.due.timestamp()],
            )?;
        } else {
            tx.execute(
                "UPDATE automation_jobs SET enabled=0 WHERE id=?1",
                [&c.occurrence.job_id],
            )?;
        }
        if result.status == OccurrenceStatus::Unknown {
            tx.execute(
                "UPDATE automation_jobs SET enabled=0 WHERE id=?1",
                [&c.occurrence.job_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// At most 128 occurrences per invocation, workers 1..32 (a trusted scope's
    /// runtime transcript is serialized, even across scheduler instances). Cancellation stops
    /// admission immediately and gracefully drains already running turns. Dropping
    /// runtime futures could strand an effect, so active work is never aborted here.
    /// Expired claims only are retried, always with the SAME durable request id.
    pub async fn run_due(
        &self,
        scope: &Scope,
        max_workers: usize,
        cancellation: CancellationToken,
    ) -> Result<Vec<Occurrence>> {
        scope.validate()?;
        if !(1..=32).contains(&max_workers) {
            bail!("scheduler workers must be 1..32");
        }
        let mut pending = FuturesUnordered::new();
        let mut admitted = 0;
        let mut results = Vec::new();
        let mut first_error = None;
        loop {
            while pending.len() < max_workers
                && admitted < 128
                && !cancellation.is_cancelled()
                && first_error.is_none()
            {
                match self.claim(scope, Utc::now().timestamp()) {
                    Ok(Some(c)) => {
                        admitted += 1;
                        pending.push(self.execute(scope.clone(), c));
                    }
                    Ok(None) => break,
                    Err(e) => {
                        first_error = Some(e);
                        break;
                    }
                }
            }
            let Some(result) = pending.next().await else {
                break;
            };
            match result {
                Ok(r) => results.push(r),
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e)
                    }
                }
            }
        }
        if let Some(e) = first_error {
            return Err(e);
        }
        Ok(results)
    }

    async fn execute(&self, scope: Scope, c: Claim) -> Result<Occurrence> {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        interval.tick().await;
        // Runtime alone owns tool dispatch, policy enforcement and side-effect
        // journaling. Reclaims are never dispatched directly to ToolExecutor.
        let run = self.runtime.run(
            scope.clone(),
            c.prompt.clone(),
            c.occurrence.request_id.clone(),
        );
        tokio::pin!(run);
        let mut lease_error = None;
        let runtime_result = loop {
            tokio::select! {
                result=&mut run=>break result,
                _=interval.tick()=>{ if lease_error.is_none() && let Err(e)=self.heartbeat(&c){lease_error=Some(e);} },
            }
        };
        if let Some(e) = lease_error {
            return Err(e);
        }
        let mut result = c.occurrence.clone();
        match runtime_result {
            Ok(turn) => {
                result.status = match turn.status.as_str() {
                    "completed" => OccurrenceStatus::Completed,
                    "cancelled" => OccurrenceStatus::Cancelled,
                    "failed" => OccurrenceStatus::Failed,
                    _ => OccurrenceStatus::Unknown,
                };
                // A terminal label alone cannot erase an unsettled tool intent.
                if !self
                    .runtime
                    .store()
                    .unresolved_tools(&scope, &turn.turn_id)?
                    .is_empty()
                {
                    result.status = OccurrenceStatus::Unknown;
                }
                result.turn_id = Some(turn.turn_id);
                if result.status != OccurrenceStatus::Completed {
                    result.error = Some(
                        match result.status {
                            OccurrenceStatus::Failed => "runtime_failed",
                            OccurrenceStatus::Cancelled => "runtime_cancelled",
                            _ => "runtime_outcome_unconfirmed",
                        }
                        .into(),
                    );
                }
            }
            // An infrastructure error is not proof that nothing executed. Only
            // a durable terminal Runtime status can establish Failed/Completed.
            Err(_) => {
                result.status = OccurrenceStatus::Unknown;
                result.error = Some("runtime_error_outcome_unconfirmed".into());
            }
        }
        self.finish(&c, &result)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::ceil_second;
    use chrono::{DateTime, Timelike, Utc};

    #[test]
    fn schedule_date_ceiling_preserves_integral_dates_and_rejects_overflow() {
        let integral = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        assert_eq!(ceil_second(integral).unwrap(), integral);
        let fractional = integral.with_nanosecond(1).unwrap();
        assert_eq!(
            ceil_second(fractional).unwrap().timestamp(),
            integral.timestamp() + 1
        );
        assert!(ceil_second(DateTime::<Utc>::MAX_UTC).is_err());
    }
}
