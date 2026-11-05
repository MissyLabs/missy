use super::*;
fn row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: r.get(0)?,
        content: r.get(1)?,
        provenance: r.get(2)?,
        trust: r.get(3)?,
        created_at: r.get(4)?,
        updated_at: r.get(5)?,
    })
}
fn load(c: &Connection, key: &str, id: &str) -> Result<Option<Memory>> {
    Ok(c.query_row("SELECT id,content,provenance,trust,created_at,updated_at FROM memories WHERE scope=?1 AND id=?2",params![key,id],row).optional()?)
}
impl Store {
    pub fn memory_put(
        &self,
        scope: &Scope,
        id: &str,
        content: &str,
        provenance: &str,
        trust: &str,
    ) -> Result<Memory> {
        ensure!(
            !id.is_empty() && id.len() <= 256 && !provenance.is_empty(),
            "memory id and provenance required"
        );
        ensure!(
            ["untrusted", "user_provided", "tool_output"].contains(&trust),
            "memory trust must remain data; instruction promotion forbidden"
        );
        let key = scope_key(scope)?;
        let at = now();
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("INSERT INTO memories(scope,id,content,provenance,trust,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?6) ON CONFLICT(scope,id) DO UPDATE SET content=excluded.content,provenance=excluded.provenance,trust=excluded.trust,updated_at=excluded.updated_at",params![key,id,content,provenance,trust,at])?;
        let rowid: i64 = tx.query_row(
            "SELECT rowid FROM memories WHERE scope=?1 AND id=?2",
            params![key, id],
            |r| r.get(0),
        )?;
        tx.execute("DELETE FROM memory_fts WHERE rowid=?1", [rowid])?;
        tx.execute(
            "INSERT INTO memory_fts(rowid,content) VALUES(?1,?2)",
            params![rowid, content],
        )?;
        audit(
            &tx,
            &key,
            "memory_put",
            &json!({"id":id,"hash":hash_text(content),"provenance":provenance,"trust":trust}),
        )?;
        let memory = load(&tx, &key, id)?.ok_or_else(|| anyhow!("memory disappeared"))?;
        tx.commit()?;
        Ok(memory)
    }
    pub fn memory_get(&self, scope: &Scope, id: &str) -> Result<Option<Memory>> {
        let c = self.connection()?;
        load(&c, &scope_key(scope)?, id)
    }
    pub fn memory_delete(&self, scope: &Scope, id: &str) -> Result<bool> {
        let key = scope_key(scope)?;
        let mut c = self.connection()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM memory_fts WHERE rowid IN(SELECT rowid FROM memories WHERE scope=?1 AND id=?2)",params![key,id])?;
        let n = tx.execute(
            "DELETE FROM memories WHERE scope=?1 AND id=?2",
            params![key, id],
        )?;
        audit(&tx, &key, "memory_delete", &json!({"id":id,"deleted":n}))?;
        tx.commit()?;
        Ok(n > 0)
    }
    pub fn memory_search(&self, scope: &Scope, query: &str, limit: usize) -> Result<Vec<Memory>> {
        let key = scope_key(scope)?;
        ensure!(query.len() <= 4096, "memory query too long");
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        // Literal FTS phrase, escaped and bound. No caller operator syntax.
        let literal = format!("\"{}\"", query.replace('"', "\"\""));
        let c = self.connection()?;
        let mut q=c.prepare("SELECT m.id,m.content,m.provenance,m.trust,m.created_at,m.updated_at FROM memories m JOIN memory_fts f ON f.rowid=m.rowid WHERE m.scope=?1 AND memory_fts MATCH ?2 ORDER BY bm25(memory_fts),m.rowid LIMIT ?3")?;
        Ok(
            q.query_map(params![key, literal, limit.min(1000) as i64], row)?
                .collect::<rusqlite::Result<_>>()?,
        )
    }
}
