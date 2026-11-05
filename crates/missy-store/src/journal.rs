use super::*;
impl Store {
    /// Atomically bind metadata to a new request. Dedup retains original metadata.
    pub fn admit_with_metadata(
        &self,
        scope: &Scope,
        request_id: &str,
        prompt: &str,
        metadata: &Value,
    ) -> Result<Admission> {
        self.admit_inner(scope, request_id, prompt, None, Some(metadata))
    }
    pub fn admit_tools_with_metadata(
        &self,
        scope: &Scope,
        request_id: &str,
        calls: &[ToolCall],
        metadata: &Value,
    ) -> Result<Admission> {
        let prompt = serde_json::to_string(calls)?;
        self.admit_inner(scope, request_id, &prompt, Some(calls), Some(metadata))
    }
    /// Read only the original admission record in exactly the supplied scope.
    pub fn admission_metadata(&self, scope: &Scope, turn_id: &str) -> Result<Option<Value>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        load_turn(&c, &key, turn_id)?;
        let raw: String = c.query_row("SELECT payload FROM events WHERE scope=?1 AND turn_id=?2 AND kind='admitted' ORDER BY sequence LIMIT 1", params![key, turn_id], |r| r.get(0))?;
        let payload: Value = parse_json(raw)?;
        Ok(payload.get("metadata").filter(|v| !v.is_null()).cloned())
    }
    /// All tools are conservatively effect-capable. Persist running before dispatch.
    pub fn tool_begin(&self, scope: &Scope, lease: &Lease, call: &ToolCall) -> Result<ToolBegin> {
        ensure!(
            !call.id.is_empty() && !call.name.is_empty(),
            "tool id and name required"
        );
        let key = scope_key(scope)?;
        let args_hash = hash_text(&serde_json::to_string(&call.arguments)?);
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        ensure!(
            !tx.query_row(
                "SELECT cancel_requested FROM turns WHERE scope=?1 AND id=?2",
                params![key, lease.turn_id],
                |r| r.get::<_, bool>(0)
            )?,
            "cancelled turn cannot prepare new effects"
        );
        let prior:Option<(String,String,String,Option<String>)>=tx.query_row("SELECT name,args_hash,status,result FROM tools WHERE scope=?1 AND turn_id=?2 AND call_id=?3",params![key,lease.turn_id,call.id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?;
        if let Some((name, hash, status, result)) = prior {
            ensure!(
                name == call.name && hash == args_hash,
                "tool call id arguments/name mismatch"
            );
            return Ok(match status.as_str() {
                "succeeded" => ToolBegin::Cached(parse_json(
                    result.ok_or_else(|| anyhow!("missing cached result"))?,
                )?),
                _ => ToolBegin::OutcomeUnknown,
            });
        }
        let unresolved:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM tools WHERE scope=?1 AND turn_id=?2 AND status IN('running','outcome_unknown'))",params![key,lease.turn_id],|r|r.get(0))?;
        ensure!(
            !unresolved,
            "unresolved effect blocks new tool intents, including new call IDs"
        );
        tx.execute("INSERT INTO tools(scope,turn_id,call_id,name,args_hash,args,status,fence) VALUES(?1,?2,?3,?4,?5,?6,'prepared',?7)",params![key,lease.turn_id,call.id,call.name,args_hash,serde_json::to_string(&call.arguments)?,lease.fence])?;
        record(
            &tx,
            &key,
            &lease.turn_id,
            "tool_prepared",
            &json!({"id":call.id,"name":call.name,"args_hash":args_hash,"fence":lease.fence,"effect_capable":true}),
        )?;
        tx.commit()?;
        Ok(ToolBegin::Prepared)
    }
    pub fn tool_running(&self, scope: &Scope, lease: &Lease, call_id: &str) -> Result<()> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        ensure!(
            !tx.query_row(
                "SELECT cancel_requested FROM turns WHERE scope=?1 AND id=?2",
                params![key, lease.turn_id],
                |r| r.get::<_, bool>(0)
            )?,
            "cancelled turn cannot dispatch new effects"
        );
        let n=tx.execute("UPDATE tools SET status='running' WHERE scope=?1 AND turn_id=?2 AND call_id=?3 AND status='prepared' AND fence=?4",params![key,lease.turn_id,call_id,lease.fence])?;
        ensure!(
            n == 1,
            "tool not prepared under current fence; replay refused"
        );
        record(
            &tx,
            &key,
            &lease.turn_id,
            "tool_running",
            &json!({"id":call_id,"fence":lease.fence}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn tool_settle(
        &self,
        scope: &Scope,
        lease: &Lease,
        call_id: &str,
        status: ToolStatus,
        result: Option<&Value>,
        error: Option<&str>,
    ) -> Result<()> {
        self.settle_inner(scope, lease, call_id, status, (result, error, None))
    }
    /// Atomically persist complete bounded raw evidence and its model receipt.
    pub fn tool_succeed_with_evidence(
        &self,
        scope: &Scope,
        lease: &Lease,
        call_id: &str,
        result: &Value,
        raw: &Value,
    ) -> Result<()> {
        let encoded = serde_json::to_string(raw)?;
        ensure!(
            encoded.len() <= 4 * 1024 * 1024,
            "raw tool evidence exceeds storage bound"
        );
        self.settle_inner(
            scope,
            lease,
            call_id,
            ToolStatus::Succeeded,
            (Some(result), None, Some(&encoded)),
        )
    }
    pub fn tool_evidence(
        &self,
        scope: &Scope,
        turn_id: &str,
        call_id: &str,
    ) -> Result<Option<Value>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        load_turn(&c, &key, turn_id)?;
        let raw: Option<(String,String)> = c.query_row("SELECT data,sha256 FROM raw_tool_evidence WHERE scope=?1 AND turn_id=?2 AND call_id=?3",params![key,turn_id,call_id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        raw.map(|(data, hash)| {
            ensure!(
                hash_text(&data) == hash,
                "raw tool evidence integrity mismatch"
            );
            parse_json(data)
        })
        .transpose()
    }
    fn settle_inner(
        &self,
        scope: &Scope,
        lease: &Lease,
        call_id: &str,
        status: ToolStatus,
        data: (Option<&Value>, Option<&str>, Option<&str>),
    ) -> Result<()> {
        let (result, error, evidence) = data;
        ensure!(
            matches!(
                status,
                ToolStatus::Succeeded | ToolStatus::Failed | ToolStatus::OutcomeUnknown
            ),
            "invalid terminal tool status"
        );
        ensure!(
            status != ToolStatus::Succeeded || result.is_some(),
            "successful result required"
        );
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_fence(&tx, &key, lease)?;
        let n=tx.execute("UPDATE tools SET status=?1,result=?2,error=?3 WHERE scope=?4 AND turn_id=?5 AND call_id=?6 AND status='running' AND fence=?7",params![status.text(),result.map(serde_json::to_string).transpose()?,error,key,lease.turn_id,call_id,lease.fence])?;
        ensure!(n == 1, "tool not running under current fence");
        if let Some(raw) = evidence {
            tx.execute("INSERT INTO raw_tool_evidence(scope,turn_id,call_id,data,sha256) VALUES(?1,?2,?3,?4,?5)",params![key,lease.turn_id,call_id,raw,hash_text(raw)])?;
        }
        record(
            &tx,
            &key,
            &lease.turn_id,
            "tool_settled",
            &json!({"id":call_id,"status":status.text()}),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn tool_journal(&self, scope: &Scope, turn_id: &str) -> Result<Vec<ToolRecord>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        load_turn(&c, &key, turn_id)?;
        let mut q=c.prepare("SELECT call_id,name,args,status,result,error FROM tools WHERE scope=?1 AND turn_id=?2 ORDER BY rowid")?;
        q.query_map(params![key, turn_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
            ))
        })?
        .map(|r| {
            let (id, name, args, status, result, error) = r?;
            Ok(ToolRecord {
                call: ToolCall {
                    id,
                    name,
                    arguments: parse_json(args)?,
                },
                status: parse_json(format!("\"{status}\""))?,
                result: result.map(parse_json).transpose()?,
                error,
            })
        })
        .collect()
    }
    pub fn tools(&self, scope: &Scope, turn_id: &str) -> Result<Vec<ToolRecord>> {
        self.tool_journal(scope, turn_id)
    }
    /// Runtime must block ALL resumed generation if this is nonempty.
    pub fn unresolved_tools(&self, scope: &Scope, turn_id: &str) -> Result<Vec<ToolRecord>> {
        Ok(self
            .tool_journal(scope, turn_id)?
            .into_iter()
            .filter(|r| {
                matches!(
                    r.status,
                    ToolStatus::Prepared | ToolStatus::Running | ToolStatus::OutcomeUnknown
                )
            })
            .collect())
    }
    pub fn unresolved_effects(&self, scope: &Scope, turn_id: &str) -> Result<Vec<ToolRecord>> {
        self.unresolved_tools(scope, turn_id)
    }
}
