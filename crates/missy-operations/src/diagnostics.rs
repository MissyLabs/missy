use anyhow::{Result, ensure};
use missy_core::{Scope, Usage};
use missy_store::{Store, TurnStatus};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub const DIAGNOSTIC_SAMPLE_LIMIT: usize = 256;
#[derive(Clone, Debug, Serialize)]
pub struct Diagnostics {
    pub observed_at_ms: i64,
    pub sample_limit: usize,
    pub sampled_turns: usize,
    /// True means more turns exist. Counts are deliberately not lifetime totals.
    pub sample_truncated: bool,
    pub states: BTreeMap<String, u64>,
    pub recorded_usage: Usage,
    pub turns_without_usage_checkpoint: usize,
    pub subsystem_observations: BTreeMap<String, LifecycleState>,
}
/// No probes or exports happen here. Reads exactly the trusted scope, never
/// includes scope labels, text, error details, paths, model names or credentials.
pub fn snapshot(store: &Store, scope: &Scope) -> Result<Diagnostics> {
    scope.validate()?;
    let mut turns = store.list_turns(scope, DIAGNOSTIC_SAMPLE_LIMIT + 1)?;
    let truncated = turns.len() > DIAGNOSTIC_SAMPLE_LIMIT;
    turns.truncate(DIAGNOSTIC_SAMPLE_LIMIT);
    let mut report = Diagnostics {
        observed_at_ms: chrono::Utc::now().timestamp_millis(),
        sample_limit: DIAGNOSTIC_SAMPLE_LIMIT,
        sampled_turns: turns.len(),
        sample_truncated: truncated,
        states: BTreeMap::new(),
        recorded_usage: Usage::default(),
        turns_without_usage_checkpoint: 0,
        subsystem_observations: BTreeMap::from([
            ("store".into(), LifecycleState::Ready),
            ("provider".into(), LifecycleState::Unknown),
            ("tools".into(), LifecycleState::Unknown),
            ("gateway".into(), LifecycleState::Unknown),
            ("scheduler".into(), LifecycleState::Unknown),
            ("telemetry_exporter".into(), LifecycleState::Unknown),
        ]),
    };
    for turn in turns {
        let state = match turn.status {
            TurnStatus::Admitted => "admitted",
            TurnStatus::Running => "running",
            TurnStatus::Completed => "completed",
            TurnStatus::Failed => "failed",
            TurnStatus::Cancelled => "cancelled",
            TurnStatus::Interrupted => "interrupted",
        };
        *report.states.entry(state.into()).or_default() += 1;
        // Public Store checkpoint API requires the context fingerprint. Obtain
        // the latest fingerprint from its scoped journal, never bypass SQLite.
        let events = store.events(scope, &turn.id)?;
        let context = events
            .iter()
            .rev()
            .find(|e| e.kind == "checkpoint")
            .and_then(|e| e.payload.get("context_hash"))
            .and_then(|s| s.as_str());
        if let Some(context) = context
            && let Some(cp) = store.checkpoint(scope, &turn.id, context)?
        {
            report.recorded_usage.add(cp.usage);
        } else {
            report.turns_without_usage_checkpoint += 1;
            report.recorded_usage.cost_known = false;
        }
    }
    Ok(report)
}
impl Diagnostics {
    pub fn prometheus(&self) -> String {
        let mut out = format!(
            "# HELP missy_sample_turns Turns in the bounded scoped diagnostic sample.\n# TYPE missy_sample_turns gauge\nmissy_sample_turns {}\n# TYPE missy_sample_truncated gauge\nmissy_sample_truncated {}\n# TYPE missy_sample_usage_missing gauge\nmissy_sample_usage_missing {}\n# TYPE missy_sample_input_tokens gauge\nmissy_sample_input_tokens {}\n# TYPE missy_sample_output_tokens gauge\nmissy_sample_output_tokens {}\n# TYPE missy_sample_cost_microusd gauge\nmissy_sample_cost_microusd {}\n# TYPE missy_sample_cost_known gauge\nmissy_sample_cost_known {}\n",
            self.sampled_turns,
            u8::from(self.sample_truncated),
            self.turns_without_usage_checkpoint,
            self.recorded_usage.input_tokens,
            self.recorded_usage.output_tokens,
            self.recorded_usage.cost_microusd,
            u8::from(self.recorded_usage.cost_known)
        );
        out.push_str("# TYPE missy_sample_turn_states gauge\n");
        // Use only this closed label list, even if a deserialized report changes.
        for state in [
            "admitted",
            "running",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        ] {
            out.push_str(&format!(
                "missy_sample_turn_states{{state=\"{state}\"}} {}\n",
                self.states.get(state).copied().unwrap_or(0)
            ));
        }
        out
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Unknown,
    Disabled,
    Starting,
    Ready,
    Degraded,
    Stopped,
    Failed,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Subsystem {
    Runtime,
    Store,
    Provider,
    Tools,
    Gateway,
    Scheduler,
    Webhook,
    TelemetryExporter,
}
#[derive(Clone, Debug, Serialize)]
pub struct LifecycleObservation {
    pub state: LifecycleState,
    pub observed_at_ms: i64,
}
/// Explicit owner-reported status, never "healthy" simply because configured.
#[derive(Clone, Default)]
pub struct LifecycleRegistry(Arc<Mutex<BTreeMap<Subsystem, LifecycleObservation>>>);
impl LifecycleRegistry {
    pub fn record(&self, subsystem: Subsystem, state: LifecycleState) -> Result<()> {
        let mut map = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("lifecycle lock poisoned"))?;
        if let Some(old) = map.get(&subsystem) {
            ensure!(
                old.state != LifecycleState::Stopped
                    || matches!(
                        state,
                        LifecycleState::Starting
                            | LifecycleState::Disabled
                            | LifecycleState::Stopped
                    ),
                "restart must pass through starting"
            );
        }
        map.insert(
            subsystem,
            LifecycleObservation {
                state,
                observed_at_ms: chrono::Utc::now().timestamp_millis(),
            },
        );
        Ok(())
    }
    pub fn observations(&self) -> Result<BTreeMap<Subsystem, LifecycleObservation>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("lifecycle lock poisoned"))?
            .clone())
    }
}
