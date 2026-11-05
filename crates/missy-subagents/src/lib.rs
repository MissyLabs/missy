//! Bounded operator DAGs. Model output cannot spawn agents or grant capabilities.
mod pool;
mod registry;
use anyhow::{Result, ensure};
use missy_core::{Scope, Usage};
use missy_runtime::RunResult;
pub use pool::AgentPool;
pub use registry::AgentRegistry;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub max_depth: u8,
    pub max_agents: usize,
    pub max_workers: usize,
    pub deadline_seconds: u64,
    pub max_steps: u64,
    pub max_tokens: u64,
    pub max_cost_microusd: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_agents: 16,
            max_workers: 4,
            deadline_seconds: 600,
            max_steps: 128,
            max_tokens: 800_000,
            max_cost_microusd: 4_000_000,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=4).contains(&self.max_depth)
                && (1..=16).contains(&self.max_agents)
                && (1..=4).contains(&self.max_workers)
                && (1..=600).contains(&self.deadline_seconds)
                && self.max_steps > 0
                && self.max_tokens > 0
                && self.max_cost_microusd > 0,
            "limits must be finite: depth<=4, agents<=16, workers<=4, deadline<=600"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DagSpec {
    pub agents: Vec<AgentSpec>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentSpec {
    pub id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub depth: u8,
    pub provider: String,
    pub prompt: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reservation {
    pub steps: u64,
    pub tokens: u64,
    pub cost_microusd: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChildReceipt {
    pub id: String,
    pub parent_id: Option<String>,
    pub depth: u8,
    pub provider: String,
    pub scope: Scope,
    pub request_id: String,
    pub turn_id: Option<String>,
    pub status: String,
    pub untrusted: bool,
    pub result: Option<RunResult>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootReceipt {
    pub root_id: String,
    pub request_id: String,
    pub scope: Scope,
    pub immutable_hash: String,
    pub deadline_ms: u64,
    pub status: String,
    pub reserved: Reservation,
    pub usage: Usage,
    pub children: BTreeMap<String, ChildReceipt>,
    pub deduplicated: bool,
}
fn hash(value: &impl Serialize) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
#[cfg(test)]
mod tests;
