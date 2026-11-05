//! Explicit local operator actions. Not exposed to model tools or chat payloads.
use super::*;
use anyhow::Context;
fn idle(tx: &Transaction<'_>, key: &str) -> Result<()> {
    let active: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM leases WHERE scope=?1 AND expires_at>?2)",
        params![key, now()],
        |r| r.get(0),
    )?;
    ensure!(!active, "operator mutation refused while session is leased");
    Ok(())
}
impl Store {
    /// Reconcile billing and terminate, never regenerate an unknown completion.
    /// Exact prompt hash and evidence bind this local operator-only decision.
    pub fn reconcile_billing_and_abandon(
        &self,
        scope: &Scope,
        turn_id: &str,
        prompt_sha256: &str,
        usage: Usage,
        evidence: &str,
    ) -> Result<()> {
        ensure!(
            usage.cost_known && !evidence.trim().is_empty() && evidence.len() <= 16384,
            "known billing and operator evidence required"
        );
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        idle(&tx, &key)?;
        let turn = load_turn(&tx, &key, turn_id)?;
        ensure!(
            !turn.status.terminal() && hash_text(&turn.prompt) == prompt_sha256,
            "turn disposition or prompt hash mismatch"
        );
        let unresolved:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM tools WHERE scope=?1 AND turn_id=?2 AND status IN('prepared','running','outcome_unknown'))",params![key,turn_id],|r|r.get(0))?;
        ensure!(
            !unresolved,
            "reconcile all tool effects before abandoning turn"
        );
        let tokens = usage
            .input_tokens
            .checked_add(usage.output_tokens)
            .context("token overflow")?;
        let previous:Option<(i64,i64)>=tx.query_row("SELECT used_cost,used_tokens FROM budget_reservations WHERE scope=?1 AND turn_id=?2",params![key,turn_id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        if let Some((cost, count)) = previous {
            ensure!(
                usage.cost_microusd >= cost as u64 && tokens >= count as u64,
                "operator billing cannot decrease known usage"
            );
            tx.execute("UPDATE budget_reservations SET cost=?1,tokens=?2,used_cost=?1,used_tokens=?2,uncertain=0,settled=1 WHERE scope=?3 AND turn_id=?4",params![i64::try_from(usage.cost_microusd)?,i64::try_from(tokens)?,key,turn_id])?;
        }
        tx.execute("UPDATE turns SET status='failed',error='operator reconciled billing and abandoned turn',updated_at=?1 WHERE scope=?2 AND id=?3",params![now(),key,turn_id])?;
        record(
            &tx,
            &key,
            turn_id,
            "operator_billing_disposition",
            &json!({"usage":usage,"evidence_sha256":hash_text(evidence),"evidence":evidence,"regeneration":false}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn session_display_name(&self, scope: &Scope) -> Result<Option<String>> {
        let key = scope_key(scope)?;
        Ok(self
            .connection()?
            .query_row(
                "SELECT NULLIF(display_name,'') FROM session_metadata WHERE scope=?1",
                [key],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten())
    }
    /// Rename changes presentation metadata, never authorization or dedup scope.
    pub fn set_session_display_name(&self, scope: &Scope, name: &str) -> Result<()> {
        ensure!(
            !name.trim().is_empty() && name.len() <= 256 && !name.chars().any(char::is_control),
            "invalid session display name"
        );
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("INSERT INTO session_metadata(scope,display_name) VALUES(?1,?2) ON CONFLICT(scope) DO UPDATE SET display_name=excluded.display_name",params![key,name])?;
        audit(
            &tx,
            &key,
            "session_renamed",
            &json!({"name_sha256":hash_text(name)}),
        )?;
        tx.commit()?;
        Ok(())
    }
    /// Logical clear preserves turns, request binding, effect and audit evidence.
    pub fn session_clear(&self, scope: &Scope) -> Result<i64> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        idle(&tx, &key)?;
        let pending: bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM turns WHERE scope=?1 AND status IN('admitted','running','interrupted')) OR EXISTS(SELECT 1 FROM tools WHERE scope=?1 AND status IN('prepared','running','outcome_unknown'))",[&key],|r|r.get(0))?;
        ensure!(!pending, "session has nonterminal or uncertain work");
        let cutoff: i64 = tx.query_row(
            "SELECT COALESCE(MAX(sequence),0) FROM messages WHERE scope=?1",
            [&key],
            |r| r.get(0),
        )?;
        tx.execute("INSERT INTO session_metadata(scope,history_cutoff) VALUES(?1,?2) ON CONFLICT(scope) DO UPDATE SET history_cutoff=MAX(history_cutoff,excluded.history_cutoff)",params![key,cutoff])?;
        audit(
            &tx,
            &key,
            "session_history_cleared",
            &json!({"cutoff":cutoff}),
        )?;
        tx.commit()?;
        Ok(cutoff)
    }
    /// Exact external disposition. Failed is not permission to retry this ID.
    #[allow(clippy::too_many_arguments)]
    pub fn reconcile_tool(
        &self,
        scope: &Scope,
        turn_id: &str,
        call_id: &str,
        arguments_sha256: &str,
        result: &Value,
        succeeded: bool,
        evidence: &str,
    ) -> Result<()> {
        ensure!(
            !evidence.trim().is_empty() && evidence.len() <= 16384,
            "operator evidence required"
        );
        let encoded = serde_json::to_string(result)?;
        ensure!(
            encoded.len() <= 4 * 1024 * 1024,
            "reconciled result exceeds bound"
        );
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        idle(&tx, &key)?;
        let turn = load_turn(&tx, &key, turn_id)?;
        ensure!(
            turn.status != TurnStatus::Completed,
            "completed turn cannot reconcile"
        );
        let (hash, status): (String, String) = tx.query_row(
            "SELECT args_hash,status FROM tools WHERE scope=?1 AND turn_id=?2 AND call_id=?3",
            params![key, turn_id, call_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        ensure!(hash == arguments_sha256, "operator argument hash mismatch");
        ensure!(
            matches!(status.as_str(), "prepared" | "running" | "outcome_unknown"),
            "tool already has final disposition"
        );
        tx.execute("UPDATE tools SET status=?1,result=?2,error=?3 WHERE scope=?4 AND turn_id=?5 AND call_id=?6",params![if succeeded{"succeeded"}else{"failed"},encoded,if succeeded{None}else{Some("operator attested non-success")},key,turn_id,call_id])?;
        record(
            &tx,
            &key,
            turn_id,
            "operator_tool_reconciliation",
            &json!({"id":call_id,"arguments_sha256":hash,"result_sha256":hash_text(&encoded),"succeeded":succeeded,"evidence_sha256":hash_text(evidence),"evidence":evidence}),
        )?;
        tx.commit()?;
        Ok(())
    }
}
