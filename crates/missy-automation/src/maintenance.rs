//! Provider-free sleeptime baseline: deterministic bounded conversation digests
//! and tool outcome learnings. Never promotes memories into instructions.
use crate::{Scheduler, execution::fingerprint};
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use missy_core::Scope;
use missy_store::{ToolStatus, TurnStatus};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::json;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaintenanceConfig {
    pub enabled: bool,
    pub operator_consent: bool,
    pub idle_seconds: u64,
    pub min_unprocessed_turns: usize,
    pub batch_size: usize,
    pub max_excerpt_bytes: usize,
}
impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            operator_consent: false,
            idle_seconds: 300,
            min_unprocessed_turns: 5,
            batch_size: 20,
            max_excerpt_bytes: 2048,
        }
    }
}
impl MaintenanceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (30..=86400).contains(&self.idle_seconds),
            "maintenance idle bound must be 30..86400"
        );
        ensure!(
            (1..=100).contains(&self.batch_size)
                && self.min_unprocessed_turns > 0
                && self.min_unprocessed_turns <= self.batch_size,
            "maintenance batch must be 1..100 and cover minimum turns"
        );
        ensure!(
            (128..=8192).contains(&self.max_excerpt_bytes),
            "maintenance excerpt bound must be 128..8192"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MaintenanceReceipt {
    pub processed_turns: usize,
    pub summaries_created: usize,
    pub learnings_created: usize,
    pub recovered_writes: usize,
    pub skipped: Option<String>,
}
fn clip(s: &str, max: usize) -> String {
    let mut end = s.len().min(max);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].into()
}
impl Scheduler {
    /// Writes deterministic IDs then marks each source complete. Interrupted
    /// writes are recovered by checking immutable exact memory content, never
    /// provider re-execution. No models, network, commands or desktop access.
    pub fn run_maintenance(
        &self,
        scope: &Scope,
        c: &MaintenanceConfig,
        now: DateTime<Utc>,
    ) -> Result<MaintenanceReceipt> {
        scope.validate()?;
        c.validate()?;
        let mut r = MaintenanceReceipt::default();
        if !c.enabled || !c.operator_consent {
            r.skipped = Some("not_enabled_or_consented".into());
            return Ok(r);
        }
        let turns = self.runtime.store().list_turns(scope, 1000)?;
        if turns.iter().any(|t| {
            matches!(
                t.status,
                TurnStatus::Admitted | TurnStatus::Running | TurnStatus::Interrupted
            )
        }) {
            r.skipped = Some("scope_has_unfinished_turns".into());
            return Ok(r);
        }
        let Some(last) = turns.iter().map(|t| t.updated_at.max(t.created_at)).max() else {
            r.skipped = Some("no_turns".into());
            return Ok(r);
        };
        if now.timestamp_millis().saturating_sub(last) < (c.idle_seconds as i64) * 1000 {
            r.skipped = Some("not_idle".into());
            return Ok(r);
        }
        let key = serde_json::to_string(scope)?;
        let binding = fingerprint(c)?;
        // Hold the private automation write lock across the bounded batch to
        // serialize cross-process ownership. Store uses a separate database.
        let mut db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler mutex poisoned"))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut candidates = Vec::new();
        for turn in turns.into_iter().filter(|t| {
            matches!(
                t.status,
                TurnStatus::Completed | TurnStatus::Failed | TurnStatus::Cancelled
            )
        }) {
            let state:Option<(String,String)>=tx.query_row("SELECT binding,status FROM automation_maintenance WHERE scope=?1 AND turn_id=?2",params![key,turn.id],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
            if let Some((old, status)) = state {
                ensure!(
                    old == binding,
                    "maintenance config changed after processing; explicit migration required"
                );
                if status == "completed" {
                    continue;
                }
            }
            candidates.push(turn);
        }
        candidates.sort_by_key(|t| (t.created_at, t.id.clone()));
        if candidates.len() < c.min_unprocessed_turns {
            r.skipped = Some("below_minimum_turns".into());
            return Ok(r);
        }
        for turn in candidates.into_iter().take(c.batch_size) {
            let suffix = fingerprint(&(scope, &turn.id))?;
            let summary_id = format!("sleeptime:summary:{suffix}");
            let provenance = format!("native-sleeptime-v1:turn:{}", turn.id);
            let summary = serde_json::to_string(
                &json!({"kind":"conversation_digest","method":"bounded_extractive_v1","untrusted":true,"turn_id":turn.id,"status":turn.status,"prompt_excerpt":clip(&turn.prompt,c.max_excerpt_bytes),"response_excerpt":turn.response.as_ref().map(|s|clip(s,c.max_excerpt_bytes)),"prompt_sha256":fingerprint(&turn.prompt)?,"response_sha256":turn.response.as_ref().map(fingerprint).transpose()?}),
            )?;
            let journal = self.runtime.store().tool_journal(scope, &turn.id)?;
            let learning_id = if journal.is_empty() {
                None
            } else {
                Some(format!("sleeptime:learning:{suffix}"))
            };
            tx.execute("INSERT INTO automation_maintenance(scope,turn_id,binding,status,summary_id,learning_id) VALUES(?1,?2,?3,'writing',?4,?5) ON CONFLICT(scope,turn_id) DO UPDATE SET status='writing'",params![key,turn.id,binding,summary_id,learning_id])?;
            if let Some(old) = self.runtime.store().memory_get(scope, &summary_id)? {
                ensure!(
                    old.content == summary
                        && old.provenance == provenance
                        && old.trust == "untrusted",
                    "sleeptime summary memory collision"
                );
                r.recovered_writes += 1;
            } else {
                self.runtime.store().memory_put(
                    scope,
                    &summary_id,
                    &summary,
                    &provenance,
                    "untrusted",
                )?;
                r.summaries_created += 1;
            }
            if let Some(id) = &learning_id {
                // Outcome facts, not speculative advice or automatic instruction.
                let outcomes:Vec<_>=journal.iter().take(128).map(|t|json!({"tool":t.call.name,"call_id":t.call.id,"status":t.status,"confirmed_success":t.status==ToolStatus::Succeeded,"requires_reconciliation":matches!(t.status,ToolStatus::Prepared|ToolStatus::Running|ToolStatus::OutcomeUnknown)})).collect();
                let learning = serde_json::to_string(
                    &json!({"kind":"tool_outcome_learning","method":"observed_journal_v1","untrusted":true,"turn_id":turn.id,"outcomes":outcomes,"truncated":journal.len()>128}),
                )?;
                if let Some(old) = self.runtime.store().memory_get(scope, id)? {
                    ensure!(
                        old.content == learning
                            && old.provenance == provenance
                            && old.trust == "untrusted",
                        "sleeptime learning memory collision"
                    );
                    r.recovered_writes += 1;
                } else {
                    self.runtime.store().memory_put(
                        scope,
                        id,
                        &learning,
                        &provenance,
                        "untrusted",
                    )?;
                    r.learnings_created += 1;
                }
            }
            tx.execute("UPDATE automation_maintenance SET status='completed' WHERE scope=?1 AND turn_id=?2",params![key,turn.id])?;
            r.processed_turns += 1;
        }
        tx.commit()?;
        Ok(r)
    }
}
