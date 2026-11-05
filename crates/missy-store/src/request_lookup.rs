use crate::*;

impl Store {
    /// Read-only exact scoped lookup. Unlike admission, this never creates a turn.
    pub fn turn_by_request(&self, scope: &Scope, request_id: &str) -> Result<Option<Turn>> {
        let key = scope_key(scope)?;
        let c = self.connection()?;
        let id: Option<String> = c
            .query_row(
                "SELECT id FROM turns WHERE scope=?1 AND request_id=?2",
                params![key, request_id],
                |r| r.get(0),
            )
            .optional()?;
        id.map(|id| load_turn(&c, &key, &id)).transpose()
    }
}
