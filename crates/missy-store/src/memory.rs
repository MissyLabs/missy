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
    /// Newest bounded shortlist belonging to the entire exact Scope. No FTS
    /// corpus statistics, cross-user ranking, or implicit channel/session widening.
    pub fn memory_candidates(&self, scope: &Scope, limit: usize) -> Result<Vec<Memory>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        let mut q = c.prepare("SELECT id,content,provenance,trust,created_at,updated_at FROM memories WHERE scope=?1 ORDER BY updated_at DESC,created_at DESC,id ASC LIMIT ?2")?;
        Ok(q.query_map(params![key, limit.min(256) as i64], row)?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn memory_search(&self, scope: &Scope, query: &str, limit: usize) -> Result<Vec<Memory>> {
        ensure!(query.len() <= 4096, "memory query too long");
        let tokens = search_tokens(query);
        if tokens.is_empty() || limit == 0 {
            scope.validate()?;
            return Ok(vec![]);
        }
        // Literal contiguous token phrase, not FTS operator syntax. All ranking
        // inputs are principal-local, so foreign writes cannot change results.
        let mut matches: Vec<_> = self
            .memory_candidates(scope, 256)?
            .into_iter()
            .filter_map(|memory| {
                let content = search_tokens(&memory.content);
                let count = content
                    .windows(tokens.len())
                    .filter(|part| *part == tokens)
                    .count();
                (count > 0).then_some((count, memory))
            })
            .collect();
        // Stable sort preserves the deterministic newest-first candidate tie order.
        matches.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        Ok(matches
            .into_iter()
            .take(limit.min(256))
            .map(|(_, memory)| memory)
            .collect())
    }
}

fn search_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod scoped_memory_tests {
    use super::*;

    fn scope() -> Scope {
        Scope {
            tenant: "tenant".into(),
            user: "alice".into(),
            channel: "channel".into(),
            session: "session".into(),
        }
    }

    #[test]
    fn scoped_memory_candidates_have_explicit_recency_and_id_ties() {
        let dir = tempfile::tempdir().unwrap();
        let db = Store::open(dir.path().join("memory.db")).unwrap();
        let scope = scope();
        for id in ["b", "a", "c"] {
            db.memory_put(&scope, id, "literal café 世界", "user", "untrusted")
                .unwrap();
        }
        {
            let c = db.connection().unwrap();
            c.execute("UPDATE memories SET updated_at=2,created_at=1", [])
                .unwrap();
            c.execute("UPDATE memories SET updated_at=3 WHERE id='c'", [])
                .unwrap();
        }
        assert_eq!(
            db.memory_candidates(&scope, 2)
                .unwrap()
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "a"]
        );
        assert_eq!(
            db.memory_search(&scope, "café 世界", 3)
                .unwrap()
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "a", "b"]
        );
        assert!(
            db.memory_search(&scope, "café OR 世界", 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn scoped_memory_foreign_corpus_cannot_change_rank_or_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let db = Store::open(dir.path().join("memory.db")).unwrap();
        let scope = scope();
        db.memory_put(&scope, "a", "café café database", "user", "untrusted")
            .unwrap();
        db.memory_put(&scope, "b", "café archive", "user", "user_provided")
            .unwrap();
        let before = db.memory_search(&scope, "café", 20).unwrap();
        let candidates = serde_json::to_value(db.memory_candidates(&scope, 20).unwrap()).unwrap();
        for field in ["tenant", "user", "channel", "session"] {
            let mut foreign = scope.clone();
            match field {
                "tenant" => foreign.tenant = "other".into(),
                "user" => foreign.user = "other".into(),
                "channel" => foreign.channel = "other".into(),
                _ => foreign.session = "other".into(),
            }
            for n in 0..16 {
                db.memory_put(
                    &foreign,
                    &format!("foreign-{n}"),
                    &"café ".repeat(100),
                    "foreign",
                    "untrusted",
                )
                .unwrap();
            }
        }
        assert_eq!(
            before.iter().map(|m| &m.id).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(
            serde_json::to_value(before).unwrap(),
            serde_json::to_value(db.memory_search(&scope, "café", 20).unwrap()).unwrap()
        );
        assert_eq!(
            candidates,
            serde_json::to_value(db.memory_candidates(&scope, 20).unwrap()).unwrap()
        );
        assert_eq!(db.memory_candidates(&scope, 1).unwrap().len(), 1);
        assert!(db.memory_candidates(&scope, 0).unwrap().is_empty());
    }
}
