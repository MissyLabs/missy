//! Explicit consent-bound triggers. No background work starts at database open.
use crate::{Scheduler, execution::fingerprint};
use anyhow::{Result, ensure};
use chrono::{DateTime, Timelike, Utc};
use missy_core::Scope;
use missy_runtime::RunResult;
use missy_store::TurnStatus;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerConfig {
    pub name: String,
    #[serde(default)]
    pub enabled: bool,
    /// Operator-selected consent. Never deserialize this from model arguments.
    #[serde(default)]
    pub operator_consent: bool,
    pub kind: TriggerKind,
    pub prompt: String,
    pub cooldown_seconds: u64,
    /// UTC minutes since midnight, half-open, supports overnight windows.
    #[serde(default)]
    pub active_utc: Option<(u16, u16)>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TriggerKind {
    Heartbeat,
    Schedule,
    FileChange,
    DiskAbove { percent: f64 },
    LoadAbove { normalized_load: f64 },
    Idle { seconds: u64 },
    Watchdog,
}
/// Trusted adapters supply only observations they are authorized to collect.
#[derive(Clone, Debug, Default)]
pub struct TriggerObservation {
    pub file_revision: Option<String>,
    pub disk_percent: Option<f64>,
    pub normalized_load: Option<f64>,
    pub idle_seconds: Option<u64>,
    pub unhealthy: bool,
    pub suppress_heartbeat: bool,
}
/// Read-only operator receipt. A missing turn is an uncertain reservation, NOT
/// permission to admit another turn. `status` is the actual scoped Store status
/// when a turn exists, otherwise `outcome_unknown`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriggerReceipt {
    pub operation: String,
    pub request_id: String,
    pub turn_scope: Scope,
    pub turn_id: Option<String>,
    pub status: String,
    pub blocked: bool,
}

/// Exact child scope for authenticated operator inspection/resume. This grants
/// no approval and never changes tenant/user/channel.
pub fn trigger_scope(scope: &Scope, name: &str) -> Result<Scope> {
    scope.validate()?;
    let mut child = scope.clone();
    child.session = format!("trigger:{}", &fingerprint(&(scope, name))?[..64]);
    Ok(child)
}

impl TriggerReceipt {
    fn poll_result(&self) -> RunResult {
        RunResult {
            turn_id: self.turn_id.clone().unwrap_or_default(),
            status: if matches!(self.status.as_str(), "admitted" | "running") {
                "inflight"
            } else {
                "paused"
            }
            .into(),
            content: format!(
                "Trigger occurrence {} is {}; no execution or replay attempted. Operator receipt inspection required.",
                self.operation, self.status
            ),
            // Poll receipts do not claim runtime usage or resumability.
            usage: missy_core::Usage {
                cost_known: false,
                ..Default::default()
            },
            resumable: false,
            deduplicated: true,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthReport {
    pub name: String,
    pub healthy: bool,
    pub consecutive_failures: u64,
    pub checked_at: i64,
    pub threshold_exceeded: bool,
    pub recovered: bool,
    pub audited: bool,
}
impl Scheduler {
    /// Durable watchdog state; healthy unchanged audit events are hourly.
    /// Observations never grant process restart or repair authority.
    pub fn record_health(
        &self,
        scope: &Scope,
        name: &str,
        healthy: bool,
        now: DateTime<Utc>,
        failure_threshold: u64,
    ) -> Result<HealthReport> {
        scope.validate()?;
        ensure!(
            !name.is_empty()
                && name.len() <= 128
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "invalid health name"
        );
        ensure!(
            (1..=1000).contains(&failure_threshold),
            "invalid health failure threshold"
        );
        let key = serde_json::to_string(scope)?;
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let old: Option<(bool, u64, i64)> = tx
            .query_row(
                "SELECT healthy,failures,audited FROM automation_health WHERE scope=?1 AND name=?2",
                params![key, name],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let failures = if healthy {
            0
        } else {
            old.as_ref()
                .map_or(1, |(_, f, _)| f.saturating_add(1).min(i64::MAX as u64))
        };
        let recovered = healthy && old.as_ref().is_some_and(|(h, _, _)| !h);
        let audited = !healthy
            || old
                .as_ref()
                .is_none_or(|(h, _, at)| !h || now.timestamp() >= at.saturating_add(3600));
        let audit_at = if audited {
            now.timestamp()
        } else {
            old.unwrap().2
        };
        tx.execute("INSERT INTO automation_health(scope,name,healthy,failures,checked,audited) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(scope,name) DO UPDATE SET healthy=excluded.healthy,failures=excluded.failures,checked=excluded.checked,audited=excluded.audited",params![key,name,healthy,failures,now.timestamp(),audit_at])?;
        if audited {
            tx.execute("INSERT INTO automation_health_events(scope,name,healthy,failures,checked) VALUES(?1,?2,?3,?4,?5)",params![key,name,healthy,failures,now.timestamp()])?;
        }
        tx.commit()?;
        Ok(HealthReport {
            name: name.into(),
            healthy,
            consecutive_failures: failures,
            checked_at: now.timestamp(),
            threshold_exceeded: failures >= failure_threshold,
            recovered,
            audited,
        })
    }
}
impl TriggerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.is_empty()
                && self.name.len() <= 128
                && self
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "invalid trigger name"
        );
        ensure!(
            !self.prompt.trim().is_empty() && self.prompt.len() <= 65536,
            "bounded trigger prompt required"
        );
        ensure!(
            (30..=315576000).contains(&self.cooldown_seconds),
            "trigger cooldown must be 30..315576000 seconds"
        );
        if let Some((start, end)) = self.active_utc {
            ensure!(
                start < 1440 && end < 1440 && start != end,
                "invalid UTC active window"
            );
        }
        match self.kind {
            TriggerKind::DiskAbove { percent } => ensure!(
                percent.is_finite() && (0.0..=100.0).contains(&percent),
                "invalid disk threshold"
            ),
            TriggerKind::LoadAbove { normalized_load } => ensure!(
                normalized_load.is_finite() && normalized_load > 0.0,
                "invalid load threshold"
            ),
            TriggerKind::Idle { seconds } => {
                ensure!(seconds >= 30, "idle threshold must be at least 30 seconds")
            }
            _ => (),
        }
        Ok(())
    }
}
impl Scheduler {
    fn trigger_receipt(
        &self,
        scope: &Scope,
        name: &str,
        operation: String,
    ) -> Result<TriggerReceipt> {
        let child = trigger_scope(scope, name)?;
        let request_id = format!("trigger:{operation}");
        let turn = self.runtime.store().turn_by_request(&child, &request_id)?;
        let status = turn
            .as_ref()
            .map(|t| serde_json::to_value(t.status))
            .transpose()?
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "outcome_unknown".into());
        Ok(TriggerReceipt {
            operation,
            request_id,
            turn_scope: child,
            turn_id: turn.map(|t| t.id),
            blocked: status != "completed",
            status,
        })
    }

    /// Exact scoped read-only snapshots, including uncertain reservations.
    /// Never starts or resumes a runtime turn and never invokes a provider.
    pub fn trigger_receipts(&self, scope: &Scope, name: &str) -> Result<Vec<TriggerReceipt>> {
        scope.validate()?;
        let key = serde_json::to_string(scope)?;
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let mut q = db.prepare("SELECT operation FROM automation_trigger_runs WHERE scope=?1 AND name=?2 ORDER BY operation")?;
        let ops = q
            .query_map(params![key, name], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ops.into_iter()
            .map(|op| self.trigger_receipt(scope, name, op))
            .collect()
    }

    /// Atomically admits at most one occurrence per cooldown. Every turn uses
    /// the existing Runtime policy, approval registry and cost bounds unchanged.
    pub async fn fire_trigger(
        &self,
        scope: &Scope,
        c: &TriggerConfig,
        o: &TriggerObservation,
        now: DateTime<Utc>,
    ) -> Result<Option<RunResult>> {
        scope.validate()?;
        c.validate()?;
        if !c.enabled || !c.operator_consent {
            return Ok(None);
        }
        if let Some((start, end)) = c.active_utc {
            let m = (now.hour() * 60 + now.minute()) as u16;
            if !(if start < end {
                m >= start && m < end
            } else {
                m >= start || m < end
            }) {
                return Ok(None);
            }
        }
        let ready = match c.kind {
            TriggerKind::Heartbeat => !o.suppress_heartbeat,
            TriggerKind::Schedule => true,
            TriggerKind::FileChange => o
                .file_revision
                .as_ref()
                .is_some_and(|s| !s.is_empty() && s.len() <= 256),
            TriggerKind::DiskAbove { percent } => o
                .disk_percent
                .is_some_and(|v| v.is_finite() && (0.0..=100.0).contains(&v) && v >= percent),
            TriggerKind::LoadAbove { normalized_load } => o
                .normalized_load
                .is_some_and(|v| v.is_finite() && v >= normalized_load),
            TriggerKind::Idle { seconds } => o.idle_seconds.is_some_and(|s| s >= seconds),
            TriggerKind::Watchdog => o.unhealthy,
        };
        if !ready {
            return Ok(None);
        }
        let key = serde_json::to_string(scope)?;
        let binding = fingerprint(&(c, self.runtime.configuration_fingerprint()?))?;
        let revision = o.file_revision.as_deref();
        let stamp = now.timestamp();
        {
            let mut db = self
                .db
                .lock()
                .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            // Refresh from exact scoped request identity, not the last runtime
            // return value. Human resume can complete after that value was saved.
            // The reservation remains blocking even if runtime admission never
            // happened, so a crash cannot silently replay an uncertain effect.
            let ops = {
                let mut q = tx.prepare("SELECT operation FROM automation_trigger_runs WHERE scope=?1 AND name=?2 AND status!='completed'")?;
                q.query_map(params![key, c.name], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?
            };
            let mut blocked = None;
            for op in ops {
                let receipt = self.trigger_receipt(scope, &c.name, op)?;
                tx.execute("UPDATE automation_trigger_runs SET status=?1 WHERE scope=?2 AND name=?3 AND operation=?4", params![receipt.status, key, c.name, receipt.operation])?;
                if receipt.blocked && blocked.is_none() {
                    blocked = Some(receipt);
                }
            }
            if let Some(receipt) = blocked {
                tx.commit()?;
                return Ok(Some(receipt.poll_result()));
            }
            let old:Option<(String,i64,Option<String>)>=tx.query_row("SELECT binding,last_fired,observation FROM automation_triggers WHERE scope=?1 AND name=?2",params![key,c.name],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            if let Some((b, last, previous)) = old {
                ensure!(
                    b == binding,
                    "trigger config/runtime changed; operator must reset consent state"
                );
                if stamp < last.saturating_add(c.cooldown_seconds as i64) {
                    tx.commit()?;
                    return Ok(None);
                }
                if matches!(c.kind, TriggerKind::FileChange) && previous.as_deref() == revision {
                    tx.commit()?;
                    return Ok(None);
                }
                tx.execute("UPDATE automation_triggers SET last_fired=?1,observation=?2 WHERE scope=?3 AND name=?4",params![stamp,revision,key,c.name])?;
            } else {
                tx.execute("INSERT INTO automation_triggers(scope,name,binding,last_fired,observation) VALUES(?1,?2,?3,?4,?5)",params![key,c.name,binding,stamp,revision])?;
                if matches!(c.kind, TriggerKind::FileChange) {
                    tx.commit()?;
                    return Ok(None);
                }
            }
            let operation = fingerprint(&(scope, &c.name, stamp, &binding))?;
            tx.execute("INSERT INTO automation_trigger_runs(scope,name,operation,status) VALUES(?1,?2,?3,'running')",params![key,c.name,operation])?;
            tx.commit()?;
        }
        let operation = fingerprint(&(scope, &c.name, stamp, &binding))?;
        let child = trigger_scope(scope, &c.name)?;
        let result = self
            .runtime
            .run(
                child,
                format!("[AUTOMATION: {}]\n{}", c.name, c.prompt),
                format!("trigger:{operation}"),
            )
            .await;
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let receipt = self.trigger_receipt(scope, &c.name, operation.clone())?;
        db.execute("UPDATE automation_trigger_runs SET status=?1 WHERE scope=?2 AND name=?3 AND operation=?4",params![receipt.status,key,c.name,operation])?;
        // A runtime admission/execution failure is an occurrence pause, not a
        // supervisor outage. The durable reservation prohibits automatic retry.
        Ok(Some(result.unwrap_or_else(|_| receipt.poll_result())))
    }
    /// Authenticated operator only. Not a model-callable tool.
    pub fn reset_trigger(&self, scope: &Scope, name: &str) -> Result<()> {
        scope.validate()?;
        let key = serde_json::to_string(scope)?;
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let ops = {
            let mut q = tx.prepare(
                "SELECT operation FROM automation_trigger_runs WHERE scope=?1 AND name=?2",
            )?;
            q.query_map(params![key, name], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for op in ops {
            let receipt = self.trigger_receipt(scope, name, op)?;
            let id = receipt.turn_id.as_ref().ok_or_else(|| {
                anyhow::anyhow!("uncertain trigger reservation cannot reset without reconciliation")
            })?;
            let turn = self.runtime.status(&receipt.turn_scope, id)?;
            ensure!(
                matches!(
                    turn.status,
                    TurnStatus::Completed | TurnStatus::Failed | TurnStatus::Cancelled
                ),
                "active or interrupted trigger cannot reset; explicitly complete or cancel original turn first"
            );
            ensure!(
                self.runtime
                    .store()
                    .unresolved_tools(&receipt.turn_scope, id)?
                    .is_empty(),
                "uncertain tool effects require operator reconciliation before reset"
            );
        }
        tx.execute(
            "DELETE FROM automation_triggers WHERE scope=?1 AND name=?2",
            params![key, name],
        )?;
        tx.execute(
            "DELETE FROM automation_trigger_runs WHERE scope=?1 AND name=?2",
            params![key, name],
        )?;
        tx.commit()?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unsafe_trigger_settings() {
        let mut c = TriggerConfig {
            name: "x".into(),
            enabled: false,
            operator_consent: false,
            kind: TriggerKind::Heartbeat,
            prompt: "check".into(),
            cooldown_seconds: 30,
            active_utc: None,
        };
        c.validate().unwrap();
        c.cooldown_seconds = 0;
        assert!(c.validate().is_err());
        c.cooldown_seconds = 30;
        c.kind = TriggerKind::DiskAbove { percent: f64::NAN };
        assert!(c.validate().is_err());
        c.kind = TriggerKind::Heartbeat;
        c.active_utc = Some((1440, 10));
        assert!(c.validate().is_err());
    }
}
