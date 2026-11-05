use crate::Scope;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Explicit long-lived production composition. No worker runs merely by loading config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatewayConfig {
    pub http_enabled: bool,
    pub startup_timeout_seconds: u64,
    pub shutdown_timeout_seconds: u64,
    pub scheduler: SchedulerWorkerConfig,
}
impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            http_enabled: false,
            startup_timeout_seconds: 60,
            shutdown_timeout_seconds: 30,
            scheduler: SchedulerWorkerConfig::default(),
        }
    }
}
impl GatewayConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=600).contains(&self.startup_timeout_seconds)
            || !(1..=300).contains(&self.shutdown_timeout_seconds)
        {
            bail!("gateway startup/shutdown bounds must be 1..600/1..300 seconds");
        }
        let s = &self.scheduler;
        if !(1..=3600).contains(&s.poll_seconds)
            || !(1..=32).contains(&s.workers)
            || s.scopes.len() > 128
            || (s.enabled && s.scopes.is_empty())
        {
            bail!(
                "scheduler requires 1..128 explicit scopes when enabled, 1..32 workers and 1..3600 second polling"
            );
        }
        let mut scopes = std::collections::HashSet::new();
        for scope in &s.scopes {
            scope.validate()?;
            if !scopes.insert(scope) {
                bail!("duplicate scheduler worker scope");
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerWorkerConfig {
    pub enabled: bool,
    pub poll_seconds: u64,
    pub workers: usize,
    pub scopes: Vec<Scope>,
}
impl Default for SchedulerWorkerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_seconds: 5,
            workers: 1,
            scopes: vec![],
        }
    }
}
