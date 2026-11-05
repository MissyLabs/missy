use anyhow::{Result, bail};
use missy_core::ToolCall;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A scoped, literal DAG, not an execution permission or a substitution language.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub scope: missy_core::Scope,
    pub nodes: Vec<PlanNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanNode {
    pub id: String,
    pub dependencies: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompiledPlan {
    pub scope: missy_core::Scope,
    /// Stable topological order, with only literal tool arguments.
    pub nodes: Vec<PlanNode>,
}

impl Plan {
    /// Bound externally supplied JSON before deserialization. Validation still
    /// requires compile with the caller's independently authenticated scope.
    pub fn parse_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1_048_576 {
            bail!("plan exceeds byte bound");
        }
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Validate against a server-trusted scope. There is deliberately no execute method:
    /// direct ToolExecutor dispatch would bypass the runtime's durable effect journal.
    pub fn compile(&self, trusted_scope: &missy_core::Scope) -> Result<CompiledPlan> {
        trusted_scope.validate()?;
        if &self.scope != trusted_scope {
            bail!("plan scope differs from trusted scope");
        }
        if self.nodes.is_empty() || self.nodes.len() > 128 {
            bail!("plan must contain 1..128 nodes");
        }
        let valid_id = |s: &str| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-:.".contains(&b))
        };
        let mut nodes = BTreeMap::new();
        let mut call_ids = BTreeSet::new();
        let mut total_calls = 0;
        for n in &self.nodes {
            if !valid_id(&n.id) || nodes.insert(n.id.clone(), n).is_some() {
                bail!("invalid or duplicate node id");
            }
            if n.dependencies.len() > 128 || n.tool_calls.len() > 32 {
                bail!("node bounds exceeded");
            }
            let mut deps = BTreeSet::new();
            for dep in &n.dependencies {
                if dep == &n.id || !deps.insert(dep) {
                    bail!("self dependency or duplicate dependency");
                }
            }
            total_calls += n.tool_calls.len();
            for call in &n.tool_calls {
                if !valid_id(&call.id)
                    || !valid_id(&call.name)
                    || !call_ids.insert(call.id.clone())
                    || !call.arguments.is_object()
                {
                    bail!("invalid or duplicate tool call");
                }
            }
        }
        if total_calls > 512 {
            bail!("plan exceeds tool call bound");
        }
        // Bound output without allocating an attacker-sized serialization buffer.
        let mut sink = BoundedSink { bytes: 0 };
        serde_json::to_writer(&mut sink, self)?;
        for node in nodes.values() {
            if node.dependencies.iter().any(|d| !nodes.contains_key(d)) {
                bail!("missing dependency");
            }
        }
        let mut ordered = Vec::new();
        let mut done = BTreeSet::new();
        while !nodes.is_empty() {
            let ready = nodes
                .iter()
                .find(|(_, n)| n.dependencies.iter().all(|d| done.contains(d)))
                .map(|(id, _)| id.clone());
            let Some(id) = ready else {
                bail!("cyclic plan");
            };
            ordered.push(nodes.remove(&id).unwrap().clone());
            done.insert(id);
        }
        Ok(CompiledPlan {
            scope: trusted_scope.clone(),
            nodes: ordered,
        })
    }
}

struct BoundedSink {
    bytes: usize,
}
impl std::io::Write for BoundedSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self.bytes.saturating_add(bytes.len());
        if self.bytes > 1_048_576 {
            return Err(std::io::Error::other("plan exceeds byte bound"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scope() -> missy_core::Scope {
        missy_core::Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        }
    }
    #[test]
    fn rejects_cycles_scope_spoof_and_duplicate_ids() {
        let mut p = Plan {
            scope: scope(),
            nodes: vec![
                PlanNode {
                    id: "a".into(),
                    dependencies: vec!["b".into()],
                    tool_calls: vec![],
                },
                PlanNode {
                    id: "b".into(),
                    dependencies: vec!["a".into()],
                    tool_calls: vec![],
                },
            ],
        };
        assert!(p.compile(&scope()).is_err());
        p.nodes[1].dependencies.clear();
        assert_eq!(p.compile(&scope()).unwrap().nodes[0].id, "b");
        p.nodes[1].id = "a".into();
        assert!(p.compile(&scope()).is_err());
        p.scope.user = "other".into();
        assert!(p.compile(&scope()).is_err());
    }
    #[test]
    fn arguments_are_literal_not_substituted() {
        let p = Plan {
            scope: scope(),
            nodes: vec![PlanNode {
                id: "a".into(),
                dependencies: vec![],
                tool_calls: vec![ToolCall {
                    id: "c".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path":"${previous.result}"}),
                }],
            }],
        };
        assert_eq!(
            p.compile(&scope()).unwrap().nodes[0].tool_calls[0].arguments["path"],
            "${previous.result}"
        );
    }
    #[test]
    fn rejects_missing_dependency_and_bounds() {
        let mut p = Plan {
            scope: scope(),
            nodes: vec![PlanNode {
                id: "a".into(),
                dependencies: vec!["missing".into()],
                tool_calls: vec![],
            }],
        };
        assert!(p.compile(&scope()).is_err());
        p.nodes[0].dependencies.clear();
        p.nodes[0].tool_calls = vec![ToolCall {
            id: "c".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"data":"x".repeat(1_048_577)}),
        }];
        assert!(p.compile(&scope()).is_err());
        assert!(Plan::parse_json(&vec![b' '; 1_048_577]).is_err());
    }
}
