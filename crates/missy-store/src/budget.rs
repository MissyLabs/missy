//! Database-global cumulative reservations. Unknown billing never frees capacity.
use super::*;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetTotals {
    pub reserved_cost_microusd: u64,
    pub reserved_tokens: u64,
    pub used_cost_microusd: u64,
    pub used_tokens: u64,
    pub uncertain_turns: u64,
}
fn integer(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| anyhow!("budget exceeds SQLite integer bound"))
}
fn totals(c: &Connection) -> Result<BudgetTotals> {
    c.query_row("SELECT COALESCE(SUM(cost),0),COALESCE(SUM(tokens),0),COALESCE(SUM(used_cost),0),COALESCE(SUM(used_tokens),0),COALESCE(SUM(uncertain),0) FROM budget_reservations",[],|r|Ok(BudgetTotals{reserved_cost_microusd:r.get::<_,i64>(0)? as u64,reserved_tokens:r.get::<_,i64>(1)? as u64,used_cost_microusd:r.get::<_,i64>(2)? as u64,used_tokens:r.get::<_,i64>(3)? as u64,uncertain_turns:r.get::<_,i64>(4)? as u64})).map_err(Into::into)
}
impl Store {
    pub fn budget_totals(&self) -> Result<BudgetTotals> {
        let c = self.connection()?;
        totals(&c)
    }
    pub fn reserve_turn_budget(
        &self,
        scope: &Scope,
        lease: &Lease,
        reservation: Usage,
        max_cost: u64,
        max_tokens: u64,
    ) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        let policy: Option<(i64, i64)> = tx
            .query_row(
                "SELECT max_cost,max_tokens FROM budget_policy WHERE id=1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((cost, tokens)) = policy {
            ensure!(
                cost == integer(max_cost)? && tokens == integer(max_tokens)?,
                "database global budget policy mismatch; explicit operator migration required"
            );
        } else if max_cost > 0 || max_tokens > 0 {
            tx.execute(
                "INSERT INTO budget_policy(id,max_cost,max_tokens) VALUES(1,?1,?2)",
                params![integer(max_cost)?, integer(max_tokens)?],
            )?;
        } else {
            return Ok(());
        }
        let prior: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM budget_reservations WHERE scope=?1 AND turn_id=?2)",
            params![key, lease.turn_id],
            |r| r.get(0),
        )?;
        if prior {
            return Ok(());
        }
        let totals = totals(&tx)?;
        ensure!(
            totals.uncertain_turns == 0,
            "global budget has unresolved provider billing"
        );
        let cost = reservation.cost_microusd;
        let tokens = reservation
            .input_tokens
            .saturating_add(reservation.output_tokens);
        ensure!(
            max_cost == 0
                || (cost > 0
                    && totals
                        .reserved_cost_microusd
                        .checked_add(cost)
                        .is_some_and(|sum| sum <= max_cost)),
            "global cost budget reservation exhausted"
        );
        ensure!(
            max_tokens == 0
                || totals
                    .reserved_tokens
                    .checked_add(tokens)
                    .is_some_and(|sum| sum <= max_tokens),
            "global token budget reservation exhausted"
        );
        tx.execute(
            "INSERT INTO budget_reservations(scope,turn_id,cost,tokens) VALUES(?1,?2,?3,?4)",
            params![key, lease.turn_id, integer(cost)?, integer(tokens)?],
        )?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "global_budget_reserved",
            &json!({"cost":cost,"tokens":tokens}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn update_turn_budget(
        &self,
        scope: &Scope,
        lease: &Lease,
        usage: Usage,
        release: bool,
    ) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        let tokens = usage.input_tokens.saturating_add(usage.output_tokens);
        let prior:Option<(i64,i64,i64,i64)>=tx.query_row("SELECT cost,tokens,used_cost,used_tokens FROM budget_reservations WHERE scope=?1 AND turn_id=?2",params![key,lease.turn_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?;
        if let Some((cost, reserved_tokens, used_cost, used_tokens)) = prior {
            ensure!(
                integer(usage.cost_microusd)? >= used_cost && integer(tokens)? >= used_tokens,
                "budget usage cannot decrease"
            );
            tx.execute("UPDATE budget_reservations SET used_cost=?1,used_tokens=?2,uncertain=?3,cost=?4,tokens=?5,settled=?6 WHERE scope=?7 AND turn_id=?8",params![integer(usage.cost_microusd)?,integer(tokens)?,!usage.cost_known,if release && usage.cost_known {integer(usage.cost_microusd)?}else{cost.max(integer(usage.cost_microusd)?)},if release && usage.cost_known {integer(tokens)?}else{reserved_tokens.max(integer(tokens)?)},release && usage.cost_known,key,lease.turn_id])?;
        }
        tx.commit()?;
        Ok(())
    }
}
