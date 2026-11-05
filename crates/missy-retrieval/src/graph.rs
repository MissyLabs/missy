use super::*;
use std::collections::VecDeque;

pub const ENTITY_TYPES: &[&str] = &[
    "person",
    "tool",
    "file",
    "project",
    "concept",
    "location",
    "organization",
];
pub const RELATIONSHIPS: &[&str] = &[
    "uses",
    "creates",
    "modifies",
    "depends_on",
    "related_to",
    "owns",
    "triggers",
];
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub provenance: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEdge {
    pub source_id: String,
    pub target_id: String,
    pub relationship: String,
    pub provenance: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphTraversal {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub truncated: bool,
    pub trust: String,
}
fn validate_node(node: &GraphNode) -> Result<()> {
    identifier(&node.id)?;
    provenance(&node.provenance)?;
    ensure!(
        !node.name.trim().is_empty() && node.name.len() <= 512 && !node.name.contains('\0'),
        "invalid graph node name"
    );
    ensure!(
        ENTITY_TYPES.contains(&node.entity_type.as_str()),
        "invalid entity type"
    );
    Ok(())
}
fn validate_edge(edge: &GraphEdge) -> Result<()> {
    identifier(&edge.source_id)?;
    identifier(&edge.target_id)?;
    provenance(&edge.provenance)?;
    ensure!(
        RELATIONSHIPS.contains(&edge.relationship.as_str()),
        "invalid relationship type"
    );
    Ok(())
}
fn node_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<GraphNode> {
    Ok(GraphNode {
        id: r.get(0)?,
        name: r.get(1)?,
        entity_type: r.get(2)?,
        provenance: r.get(3)?,
    })
}
fn edge_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<GraphEdge> {
    Ok(GraphEdge {
        source_id: r.get(0)?,
        target_id: r.get(1)?,
        relationship: r.get(2)?,
        provenance: r.get(3)?,
    })
}
fn load_node(conn: &Connection, key: &str, id: &str) -> Result<Option<GraphNode>> {
    let node = conn
        .query_row(
            "SELECT id,name,entity_type,provenance FROM nodes WHERE scope=?1 AND id=?2",
            params![key, id],
            node_row,
        )
        .optional()?;
    if let Some(node) = &node {
        validate_node(node)?;
    }
    Ok(node)
}
impl Retrieval {
    pub fn put_node(&self, scope: &Scope, node: &GraphNode) -> Result<()> {
        let key = scope_key(scope)?;
        validate_node(node)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let count: usize = tx.query_row(
            "SELECT count(*) FROM nodes WHERE scope=?1 AND id!=?2",
            params![key, node.id],
            |r| r.get(0),
        )?;
        ensure!(count < 2048, "scope graph node capacity exceeded");
        tx.execute("INSERT INTO nodes VALUES(?1,?2,?3,?4,?5) ON CONFLICT(scope,id) DO UPDATE SET name=excluded.name,entity_type=excluded.entity_type,provenance=excluded.provenance",params![key,node.id,node.name,node.entity_type,node.provenance])?;
        tx.commit()?;
        Ok(())
    }
    pub fn get_node(&self, scope: &Scope, id: &str) -> Result<Option<GraphNode>> {
        let key = scope_key(scope)?;
        identifier(id)?;
        let conn = self.connection()?;
        load_node(&conn, &key, id)
    }
    pub fn drop_node(&self, scope: &Scope, id: &str) -> Result<bool> {
        let key = scope_key(scope)?;
        identifier(id)?;
        Ok(self.connection()?.execute(
            "DELETE FROM nodes WHERE scope=?1 AND id=?2",
            params![key, id],
        )? > 0)
    }
    pub fn put_edge(&self, scope: &Scope, edge: &GraphEdge) -> Result<()> {
        let key = scope_key(scope)?;
        validate_edge(edge)?;
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        ensure!(
            load_node(&tx, &key, &edge.source_id)?.is_some()
                && load_node(&tx, &key, &edge.target_id)?.is_some(),
            "edge endpoints must exist in the exact scope"
        );
        let count: usize = tx.query_row("SELECT count(*) FROM edges WHERE scope=?1 AND NOT (source_id=?2 AND target_id=?3 AND relationship=?4)", params![key,edge.source_id,edge.target_id,edge.relationship], |r|r.get(0))?;
        ensure!(count < 4096, "scope graph edge capacity exceeded");
        tx.execute("INSERT INTO edges VALUES(?1,?2,?3,?4,?5) ON CONFLICT(scope,source_id,target_id,relationship) DO UPDATE SET provenance=excluded.provenance",params![key,edge.source_id,edge.target_id,edge.relationship,edge.provenance])?;
        tx.commit()?;
        Ok(())
    }
    pub fn drop_edge(
        &self,
        scope: &Scope,
        source_id: &str,
        target_id: &str,
        relationship: &str,
    ) -> Result<bool> {
        let key = scope_key(scope)?;
        identifier(source_id)?;
        identifier(target_id)?;
        ensure!(
            RELATIONSHIPS.contains(&relationship),
            "invalid relationship type"
        );
        Ok(self.connection()?.execute("DELETE FROM edges WHERE scope=?1 AND source_id=?2 AND target_id=?3 AND relationship=?4",params![key,source_id,target_id,relationship])? > 0)
    }
    /// Directed breadth-first traversal. Depth 0 returns only the seed. Nodes AND edges each
    /// have a hard row limit of 100; serialized total output is capped at 128 KiB.
    /// A depth horizon is intentional, not reported as row/output truncation.
    pub fn traverse(
        &self,
        scope: &Scope,
        start_id: &str,
        depth: usize,
        limit: usize,
    ) -> Result<GraphTraversal> {
        let key = scope_key(scope)?;
        identifier(start_id)?;
        ensure!(
            depth <= 8 && (1..=100).contains(&limit),
            "graph depth must be 0..8 and limit 1..100"
        );
        let mut conn = self.connection()?;
        let tx = conn.transaction()?;
        let mut result = GraphTraversal {
            nodes: vec![],
            edges: vec![],
            truncated: false,
            trust: TRUST.into(),
        };
        let Some(seed) = load_node(&tx, &key, start_id)? else {
            return Ok(result);
        };
        let mut bytes = serde_json::to_vec(&seed)?.len() + 128;
        let mut queue = VecDeque::from([(seed.id.clone(), 0)]);
        let mut seen = BTreeSet::from([seed.id.clone()]);
        result.nodes.push(seed);
        'bfs: while let Some((id, level)) = queue.pop_front() {
            if level == depth {
                continue;
            }
            let remaining = limit - result.edges.len();
            let mut statement = tx.prepare("SELECT source_id,target_id,relationship,provenance FROM edges WHERE scope=?1 AND source_id=?2 ORDER BY target_id,relationship LIMIT ?3")?;
            let edges: Vec<_> = statement
                .query_map(params![key, id, remaining + 1], edge_row)?
                .collect::<rusqlite::Result<_>>()?;
            for edge in edges {
                validate_edge(&edge)?;
                let target = load_node(&tx, &key, &edge.target_id)?
                    .ok_or_else(|| anyhow!("graph endpoint corruption"))?;
                let new_node = !seen.contains(&target.id);
                let added_bytes = serde_json::to_vec(&edge)?.len()
                    + if new_node {
                        serde_json::to_vec(&target)?.len()
                    } else {
                        0
                    }
                    + 2;
                if result.edges.len() == limit
                    || (new_node && result.nodes.len() == limit)
                    || bytes + added_bytes > MAX_OUTPUT_BYTES
                {
                    result.truncated = true;
                    break 'bfs;
                }
                bytes += added_bytes;
                result.edges.push(edge);
                if new_node {
                    seen.insert(target.id.clone());
                    queue.push_back((target.id.clone(), level + 1));
                    result.nodes.push(target);
                }
            }
        }
        tx.commit()?;
        Ok(result)
    }
}
