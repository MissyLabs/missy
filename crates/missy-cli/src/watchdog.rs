//! Trusted, opt-in health observations. Never repairs or expands policy.
use anyhow::{Result, ensure};
use missy_automation::{Scheduler, TriggerConfig, TriggerKind, TriggerObservation};
use missy_core::{PolicyConfig, Scope};
use missy_policy::PolicyHttpClient;
use missy_runtime::Runtime;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WatchdogConfig {
    #[serde(default)]
    pub enabled: bool,
    pub scope: Option<Scope>,
    pub poll_seconds: Option<u64>,
    pub failure_threshold: Option<u64>,
    #[serde(default)]
    pub probes: Vec<Probe>,
    #[serde(default)]
    pub trigger: Option<TriggerConfig>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Probe {
    Database { name: String },
    Http { name: String, url: String },
}
impl Probe {
    fn name(&self) -> &str {
        match self {
            Self::Database { name } | Self::Http { name, .. } => name,
        }
    }
}

pub fn validate(config: &WatchdogConfig, policy: &PolicyConfig) -> Result<()> {
    PolicyHttpClient::new(policy.clone())?;
    if !config.enabled {
        return Ok(());
    }
    config
        .scope
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("watchdog scope required"))?
        .validate()?;
    ensure!(
        (1..=86400).contains(&config.poll_seconds.unwrap_or(0)),
        "watchdog poll_seconds required (1..86400)"
    );
    ensure!(
        (1..=1000).contains(&config.failure_threshold.unwrap_or(0)),
        "watchdog failure_threshold required (1..1000)"
    );
    ensure!(
        !config.probes.is_empty() && config.probes.len() <= 64,
        "watchdog requires 1..64 probes"
    );
    let mut names = HashSet::new();
    for p in &config.probes {
        let name = p.name();
        ensure!(
            !name.is_empty()
                && name.len() <= 128
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                && names.insert(name),
            "invalid or duplicate watchdog probe name"
        );
        if let Probe::Http { url, .. } = p {
            ensure!(
                url.len() <= 2048
                    && (url.starts_with("http://") || url.starts_with("https://"))
                    && !url.contains('#')
                    && !url.contains('?')
                    && !url.contains('@'),
                "watchdog URL must be explicit HTTP(S) without credentials, query or fragment"
            );
        }
    }
    if let Some(trigger) = &config.trigger {
        trigger.validate()?;
        ensure!(
            matches!(trigger.kind, TriggerKind::Watchdog),
            "watchdog trigger must have watchdog kind"
        );
    }
    Ok(())
}

/// Runs until cancelled. Results are persisted only as bounded health metadata.
pub async fn run(
    config: WatchdogConfig,
    policy: PolicyConfig,
    state_dir: PathBuf,
    runtime: Arc<Runtime>,
    cancel: CancellationToken,
) -> Result<()> {
    validate(&config, &policy)?;
    if !config.enabled {
        return Ok(());
    }
    let scheduler = Scheduler::open(state_dir.join("scheduler.sqlite3"), runtime.clone())?;
    let client = PolicyHttpClient::new(policy)?.with_timeout(Duration::from_secs(5))?;
    let scope = config.scope.as_ref().unwrap();
    let mut interval = tokio::time::interval(Duration::from_secs(config.poll_seconds.unwrap()));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! { biased; _ = cancel.cancelled() => return Ok(()), _ = interval.tick() => {} }
        let mut unhealthy = false;
        for probe in &config.probes {
            let check = async {
                match probe {
                    Probe::Database { .. } => runtime.store().quick_check().unwrap_or(false),
                    Probe::Http { url, .. } => client
                        .get(url)
                        .await
                        .is_ok_and(|r| (200..300).contains(&r.status)),
                }
            };
            let healthy = tokio::select! { biased; _ = cancel.cancelled() => return Ok(()), result = check => result };
            let report = scheduler.record_health(
                scope,
                probe.name(),
                healthy,
                chrono::Utc::now(),
                config.failure_threshold.unwrap(),
            )?;
            unhealthy |= report.threshold_exceeded;
        }
        if let Some(trigger) = &config.trigger {
            let observation = TriggerObservation {
                unhealthy,
                ..Default::default()
            };
            tokio::select! { biased; _ = cancel.cancelled() => return Ok(()), result = scheduler.fire_trigger(scope, trigger, &observation, chrono::Utc::now()) => { result?; } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_configuration_and_explicit_consent() {
        let policy = PolicyConfig::default();
        let disabled: WatchdogConfig = serde_json::from_str("{}").unwrap();
        assert!(!disabled.enabled);
        assert!(validate(&disabled, &policy).is_ok());
        assert!(
            serde_json::from_str::<WatchdogConfig>(r#"{"enabled":true,"unexpected":1}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Probe>(
                r#"{"kind":"http","name":"a","url":"http://x","unexpected":1}"#
            )
            .is_err()
        );
        let mut config: WatchdogConfig = serde_json::from_str(r#"{"enabled":true,"scope":{"tenant":"t","user":"u","channel":"c","session":"s"},"poll_seconds":1,"failure_threshold":2,"probes":[{"kind":"http","name":"h","url":"http://example.com/health"}]}"#).unwrap();
        assert!(validate(&config, &policy).is_ok());
        config.probes.push(config.probes[0].clone());
        assert!(validate(&config, &policy).is_err());
    }
}
