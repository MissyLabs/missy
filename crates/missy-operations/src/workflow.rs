use crate::SharedRuntime;
use anyhow::{Result, bail, ensure};
use missy_core::{Scope, Usage};
use missy_runtime::RunResult;
use missy_store::TurnStatus;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

/// Portable literal task DAG. No tools, policy, credentials, principal or runtime
/// settings are accepted in node JSON. Subtasks share the trusted Runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub id: String,
    pub nodes: Vec<WorkflowNode>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowNode {
    pub id: String,
    pub dependencies: Vec<String>,
    pub prompt: String,
}
#[derive(Clone, Debug)]
pub struct WorkflowBounds {
    pub max_nodes: usize,
    pub max_seconds: u64,
    pub max_recorded_tokens: u64,
    pub max_recorded_cost_microusd: u64,
}
impl Default for WorkflowBounds {
    fn default() -> Self {
        Self {
            max_nodes: 16,
            max_seconds: 600,
            max_recorded_tokens: 200_000,
            max_recorded_cost_microusd: 1_000_000,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct WorkflowResult {
    pub workflow_id: String,
    pub turns: BTreeMap<String, RunResult>,
    pub recorded_usage: Usage,
}
impl Workflow {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() <= 65_536, "workflow byte bound exceeded");
        Ok(serde_json::from_slice(bytes)?)
    }
    pub fn ordered_nodes(&self, bounds: &WorkflowBounds) -> Result<Vec<WorkflowNode>> {
        ensure!(
            (1..=128).contains(&bounds.max_nodes)
                && (1..=3600).contains(&bounds.max_seconds)
                && bounds.max_recorded_tokens > 0
                && bounds.max_recorded_cost_microusd > 0,
            "invalid workflow bounds"
        );
        let valid_id = |s: &str| {
            !s.is_empty()
                && s.len() <= 64
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
        };
        ensure!(
            valid_id(&self.id) && !self.nodes.is_empty() && self.nodes.len() <= bounds.max_nodes,
            "invalid workflow identity or node count"
        );
        ensure!(
            serde_json::to_vec(self)?.len() <= 65_536,
            "workflow byte bound exceeded"
        );
        let mut pending = BTreeMap::new();
        for n in &self.nodes {
            ensure!(
                valid_id(&n.id)
                    && !n.prompt.trim().is_empty()
                    && n.prompt.len() <= 64_000
                    && n.dependencies.len() <= bounds.max_nodes,
                "invalid workflow node"
            );
            ensure!(
                pending.insert(n.id.clone(), n.clone()).is_none(),
                "duplicate node"
            );
            ensure!(
                n.dependencies.iter().collect::<BTreeSet<_>>().len() == n.dependencies.len(),
                "duplicate dependency"
            );
        }
        for n in pending.values() {
            ensure!(
                n.dependencies
                    .iter()
                    .all(|d| d != &n.id && pending.contains_key(d)),
                "unknown or self dependency"
            );
        }
        let mut done = BTreeSet::new();
        let mut out = Vec::new();
        while !pending.is_empty() {
            let Some(id) = pending
                .iter()
                .find(|(_, n)| n.dependencies.iter().all(|d| done.contains(d)))
                .map(|(id, _)| id.clone())
            else {
                bail!("cyclic workflow");
            };
            out.push(pending.remove(&id).unwrap());
            done.insert(id);
        }
        Ok(out)
    }
    /// Sequential, explicit execution, no recursion or automatic continuation.
    /// Workflow limits gate each new turn; in-flight turn limits remain Runtime's
    /// original immutable limits. Aggregate usage may overshoot by ONE turn.
    /// Unknown cost stops further nodes instead of treating it as free.
    pub async fn execute(
        &self,
        runtime: SharedRuntime,
        trusted_scope: Scope,
        bounds: WorkflowBounds,
    ) -> Result<WorkflowResult> {
        trusted_scope.validate()?;
        let ordered = self.ordered_nodes(&bounds)?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(bounds.max_seconds);
        let mut result = WorkflowResult {
            workflow_id: self.id.clone(),
            turns: BTreeMap::new(),
            recorded_usage: Usage::default(),
        };
        for node in ordered {
            ensure!(
                tokio::time::Instant::now() < deadline,
                "workflow deadline exceeded"
            );
            ensure!(
                result.recorded_usage.cost_known
                    && result
                        .recorded_usage
                        .input_tokens
                        .saturating_add(result.recorded_usage.output_tokens)
                        < bounds.max_recorded_tokens
                    && result.recorded_usage.cost_microusd < bounds.max_recorded_cost_microusd,
                "workflow cumulative admission budget exhausted or unknown"
            );
            let request_id = format!("workflow:{}:{}", self.id, node.id);
            let admitted = runtime
                .start(
                    trusted_scope.clone(),
                    node.prompt.clone(),
                    request_id.clone(),
                )
                .await?;
            loop {
                let status = runtime.status(&trusted_scope, &admitted.turn_id)?.status;
                if matches!(
                    status,
                    TurnStatus::Completed
                        | TurnStatus::Failed
                        | TurnStatus::Cancelled
                        | TurnStatus::Interrupted
                ) {
                    break;
                }
                if tokio::time::Instant::now() >= deadline {
                    runtime.cancel(&trusted_scope, &admitted.turn_id)?;
                    bail!("workflow deadline exceeded; active turn cancellation requested");
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let receipt = runtime
                .run(trusted_scope.clone(), node.prompt, request_id)
                .await?;
            ensure!(
                receipt.status == "completed",
                "workflow node did not complete; no dependent nodes launched"
            );
            result.recorded_usage.add(receipt.usage);
            result.turns.insert(node.id, receipt);
        }
        Ok(result)
    }
}
