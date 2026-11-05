use super::*;
pub(super) fn validate(cp: &Checkpoint) -> Result<()> {
    ensure!(
        !cp.context_hash.is_empty(),
        "checkpoint context binding required"
    );
    let system = cp
        .transcript
        .first()
        .ok_or_else(|| anyhow!("empty checkpoint transcript"))?;
    ensure!(
        system.role == "system" && hash_text(&system.content) == cp.system_prompt_hash,
        "checkpoint must preserve original system prompt lineage"
    );
    ensure!(
        cp.transcript
            .iter()
            .skip(1)
            .all(|m| m.role != "system" && m.role != "developer"),
        "recalled/tool instructions must not be promoted"
    );
    Ok(())
}
pub(super) fn digest(key: &str, id: &str, data: &str) -> String {
    hash_text(
        &serde_json::to_string(&(1, key, id, data)).expect("serializable checkpoint checksum"),
    )
}
pub(super) fn load(c: &Connection, key: &str, id: &str) -> Result<Option<Checkpoint>> {
    let row: Option<(String, i64, String)> = c
        .query_row(
            "SELECT data,format_version,digest FROM checkpoints WHERE scope=?1 AND turn_id=?2",
            params![key, id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    row.map(|(data, version, checksum)| {
        ensure!(version == 1, "unsupported checkpoint format version");
        ensure!(
            digest(key, id, &data) == checksum,
            "checkpoint checksum mismatch"
        );
        let cp: Checkpoint = parse_json(data)?;
        validate(&cp)?;
        Ok(cp)
    })
    .transpose()
}
impl Store {
    /// One serialized runtime writer per lease is required. Counters cannot go
    /// backwards; scope leases fence other workers/processes transactionally.
    pub fn save_checkpoint(&self, scope: &Scope, lease: &Lease, cp: &Checkpoint) -> Result<()> {
        validate(cp)?;
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        if let Some(old) = load(&tx, &key, &lease.turn_id)? {
            ensure!(
                old.context_hash == cp.context_hash
                    && old.system_prompt_hash == cp.system_prompt_hash,
                "checkpoint context/system prompt lineage mismatch"
            );
            ensure!(
                cp.steps >= old.steps
                    && cp.usage.input_tokens >= old.usage.input_tokens
                    && cp.usage.output_tokens >= old.usage.output_tokens
                    && cp.usage.cost_microusd >= old.usage.cost_microusd,
                "checkpoint budget counters cannot decrease"
            );
            ensure!(
                old.usage.cost_known || !cp.usage.cost_known,
                "unknown cost cannot become known on continuation"
            );
        }
        let data = serde_json::to_string(cp)?;
        let checksum = digest(&key, &lease.turn_id, &data);
        tx.execute("INSERT INTO checkpoints(scope,turn_id,context_hash,system_prompt_hash,data,format_version,digest) VALUES(?1,?2,?3,?4,?5,1,?6) ON CONFLICT(scope,turn_id) DO UPDATE SET data=excluded.data,digest=excluded.digest",params![key,lease.turn_id,cp.context_hash,cp.system_prompt_hash,data,checksum])?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "checkpoint",
            &json!({"steps":cp.steps,"context_hash":cp.context_hash}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn checkpoint(
        &self,
        scope: &Scope,
        turn_id: &str,
        expected_context_hash: &str,
    ) -> Result<Option<Checkpoint>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        load_turn(&c, &key, turn_id)?;
        load(&c, &key, turn_id)?
            .map(|cp| {
                ensure!(
                    cp.context_hash == expected_context_hash,
                    "continuation context mismatch"
                );
                Ok(cp)
            })
            .transpose()
    }
    pub fn continuation(
        &self,
        scope: &Scope,
        turn_id: &str,
        context_hash: &str,
    ) -> Result<Option<Checkpoint>> {
        self.checkpoint(scope, turn_id, context_hash)
    }
}
