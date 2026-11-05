use super::*;

impl Store {
    /// Inventory sessions backed by turns or session metadata, within exactly this
    /// tenant/user/channel. Memories are intentionally not consulted.
    pub fn scoped_sessions(
        &self,
        scope: &Scope,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        scope.validate()?;
        ensure!(
            prefix.len() <= 128
                && prefix
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_:.".contains(&b)),
            "invalid session prefix"
        );
        ensure!((1..=1000).contains(&limit), "limit must be 1..=1000");
        let conn = self.connection()?;
        let mut stmt =
            conn.prepare("SELECT scope FROM turns UNION SELECT scope FROM session_metadata")?;
        let mut sessions = std::collections::BTreeSet::new();
        for value in stmt.query_map([], |r| r.get::<_, String>(0))? {
            let parsed: Scope = serde_json::from_str(&value?)?;
            parsed.validate()?;
            if parsed.tenant == scope.tenant
                && parsed.user == scope.user
                && parsed.channel == scope.channel
                && parsed.session.starts_with(prefix)
            {
                sessions.insert(parsed.session);
            }
        }
        Ok(sessions.into_iter().take(limit).collect())
    }
}
