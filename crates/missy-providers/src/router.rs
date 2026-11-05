//! Deterministic operator routing and conservative, process-local admission.
//! No credential resolution, content classification, model calls, or policy grants.
use anyhow::{Result, ensure};
use missy_core::{Message, ProviderConfig, RoutingConfig, RoutingMode, ToolDefinition};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct Lane {
    window: Option<Instant>,
    requests: u64,
    tokens: u64,
    active: u32,
    score: i64,
}
struct State {
    lanes: Mutex<Vec<Lane>>,
    changed: Notify,
}
pub(crate) struct Router {
    pub config: RoutingConfig,
    profiles: Vec<ProviderConfig>,
    state: Arc<State>,
}
pub(crate) struct Reservation {
    state: Arc<State>,
    index: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        if let Ok(mut lanes) = self.state.lanes.lock() {
            lanes[self.index].active = lanes[self.index].active.saturating_sub(1);
        }
        self.state.changed.notify_waiters();
    }
}
impl Router {
    pub fn new(profiles: &[ProviderConfig], config: RoutingConfig) -> Result<Self> {
        config.validate(profiles)?;
        Ok(Self {
            config,
            profiles: profiles.to_vec(),
            state: Arc::new(State {
                lanes: Mutex::new((0..profiles.len()).map(|_| Lane::default()).collect()),
                changed: Notify::new(),
            }),
        })
    }
    pub fn plan(
        &self,
        task: Option<&str>,
        tier: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Vec<usize>> {
        // Only immutable user-input size and operator-supplied tool definitions.
        // Assistant/tool output never selects a more expensive tier on followups.
        let chars: usize = messages
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.chars().count())
            .sum();
        let tier = tier
            .or_else(|| task.and_then(|t| self.config.task_tiers.get(t).map(String::as_str)))
            .or_else(|| {
                self.config
                    .rules
                    .iter()
                    .find(|r| chars >= r.min_input_chars && tools.len() >= r.min_tools)
                    .map(|r| r.tier.as_str())
            })
            .or(self.config.default_tier.as_deref());
        let mut eligible: Vec<_> = self
            .profiles
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                (p.routing.tasks.is_empty()
                    || task.is_some_and(|t| p.routing.tasks.iter().any(|x| x == t)))
                    && (p.routing.tiers.is_empty()
                        || tier.is_some_and(|t| p.routing.tiers.iter().any(|x| x == t)))
            })
            .map(|(i, _)| i)
            .collect();
        ensure!(
            !eligible.is_empty(),
            "no provider eligible for requested task/tier"
        );
        eligible.sort_by_key(|i| (self.profiles[*i].routing.priority, *i));
        if self.config.mode == RoutingMode::Weighted {
            let best = self.profiles[eligible[0]].routing.priority;
            let peers: Vec<_> = eligible
                .iter()
                .copied()
                .take_while(|i| self.profiles[*i].routing.priority == best)
                .collect();
            let mut lanes = self
                .state
                .lanes
                .lock()
                .map_err(|_| anyhow::anyhow!("routing lock poisoned"))?;
            let total: i64 = peers
                .iter()
                .map(|i| self.profiles[*i].routing.weight as i64)
                .sum();
            let mut selected = peers[0];
            for i in peers {
                lanes[i].score += self.profiles[i].routing.weight as i64;
                if lanes[i].score > lanes[selected].score {
                    selected = i;
                }
            }
            lanes[selected].score -= total;
            eligible.retain(|i| *i != selected);
            eligible.insert(0, selected);
        }
        Ok(eligible)
    }
    pub async fn reserve(
        &self,
        index: usize,
        tokens: u64,
        requests: u64,
        deadline: tokio::time::Instant,
        cancel: &CancellationToken,
    ) -> Result<Option<Reservation>> {
        let profile = &self.profiles[index].routing;
        if profile.tokens_per_minute.is_some_and(|n| tokens > n)
            || profile.requests_per_minute.is_some_and(|n| requests > n)
        {
            return Ok(None);
        }
        loop {
            ensure!(!cancel.is_cancelled(), "provider request cancelled");
            {
                let mut lanes = self
                    .state
                    .lanes
                    .lock()
                    .map_err(|_| anyhow::anyhow!("routing lock poisoned"))?;
                let lane = &mut lanes[index];
                let now = Instant::now();
                if lane
                    .window
                    .is_none_or(|start| now.duration_since(start) >= Duration::from_secs(60))
                {
                    lane.window = Some(now);
                    lane.requests = 0;
                    lane.tokens = 0;
                }
                if lane.active < profile.max_concurrent
                    && profile
                        .requests_per_minute
                        .is_none_or(|n| lane.requests <= n.saturating_sub(requests))
                    && profile
                        .tokens_per_minute
                        .is_none_or(|n| lane.tokens <= n.saturating_sub(tokens))
                {
                    lane.active += 1;
                    lane.requests = lane.requests.saturating_add(requests);
                    lane.tokens = lane.tokens.saturating_add(tokens);
                    return Ok(Some(Reservation {
                        state: self.state.clone(),
                        index,
                    }));
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(None);
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => anyhow::bail!("provider request cancelled"),
                _ = tokio::time::sleep_until(deadline.min(tokio::time::Instant::now() + Duration::from_millis(20))) => {},
                _ = self.state.changed.notified() => {},
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn configs() -> Vec<ProviderConfig> {
        ["a", "b"]
            .into_iter()
            .map(|name| ProviderConfig {
                name: name.into(),
                ..Default::default()
            })
            .collect()
    }
    #[test]
    fn ordered_weighted_and_priority_are_stable() {
        let mut p = configs();
        let ordered = Router::new(&p, RoutingConfig::default()).unwrap();
        assert_eq!(ordered.plan(None, None, &[], &[]).unwrap(), vec![0, 1]);
        p[1].routing.weight = 3;
        let r = Router::new(
            &p,
            RoutingConfig {
                mode: RoutingMode::Weighted,
                ..Default::default()
            },
        )
        .unwrap();
        let sequence: Vec<_> = (0..8)
            .map(|_| r.plan(None, None, &[], &[]).unwrap()[0])
            .collect();
        assert_eq!(sequence, vec![1, 0, 1, 1, 1, 0, 1, 1]);
        p[0].routing.priority = -1;
        let r = Router::new(
            &p,
            RoutingConfig {
                mode: RoutingMode::Weighted,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((0..10).all(|_| r.plan(None, None, &[], &[]).unwrap()[0] == 0));
    }
    #[test]
    fn tier_task_and_model_output_isolation() {
        let mut p = configs();
        p[0].routing.tiers = vec!["fast".into()];
        p[1].routing.tiers = vec!["premium".into()];
        p[1].routing.tasks = vec!["worker".into()];
        let r = Router::new(
            &p,
            RoutingConfig {
                default_tier: Some("fast".into()),
                task_tiers: [("worker".into(), "premium".into())].into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            r.plan(
                None,
                None,
                &[Message::text("assistant", "premium".repeat(1000))],
                &[]
            )
            .unwrap(),
            vec![0]
        );
        assert_eq!(r.plan(Some("worker"), None, &[], &[]).unwrap(), vec![1]);
        assert!(r.plan(None, Some("premium"), &[], &[]).is_err());
        assert!(r.plan(Some("worker"), Some("missing"), &[], &[]).is_err());
    }
    #[tokio::test]
    async fn rate_backpressure_cancellation_and_release() {
        let mut p = configs();
        p[0].routing.max_concurrent = 1;
        p[0].routing.requests_per_minute = Some(2);
        p[0].routing.tokens_per_minute = Some(20);
        let r = Router::new(&p, RoutingConfig::default()).unwrap();
        let cancel = CancellationToken::new();
        let now = tokio::time::Instant::now();
        let permit = r.reserve(0, 10, 1, now, &cancel).await.unwrap().unwrap();
        assert!(r.reserve(0, 10, 1, now, &cancel).await.unwrap().is_none());
        let waiting = r.reserve(0, 10, 1, now + Duration::from_secs(2), &cancel);
        let cancelling = async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            cancel.cancel();
        };
        let (result, _) = tokio::join!(waiting, cancelling);
        assert!(result.is_err());
        drop(permit);
        let cancel = CancellationToken::new();
        assert!(r.reserve(0, 10, 1, now, &cancel).await.unwrap().is_some());
        assert!(r.reserve(0, 1, 1, now, &cancel).await.unwrap().is_none());
    }
    #[test]
    fn invalid_operator_config_is_rejected() {
        let mut p = configs();
        p[0].routing.weight = 0;
        assert!(Router::new(&p, RoutingConfig::default()).is_err());
        p[0].routing.weight = 1;
        p[0].routing.requests_per_minute = Some(0);
        assert!(Router::new(&p, RoutingConfig::default()).is_err());
        p[0].routing.requests_per_minute = None;
        p[0].routing.max_concurrent = 0;
        assert!(Router::new(&p, RoutingConfig::default()).is_err());
        p[0].routing.max_concurrent = 1;
        p[0].routing.circuit_failure_threshold = 0;
        assert!(Router::new(&p, RoutingConfig::default()).is_err());
        p[0].routing.circuit_failure_threshold = 1;
        assert!(
            Router::new(
                &p,
                RoutingConfig {
                    max_wait_ms: 60001,
                    ..Default::default()
                }
            )
            .is_err()
        );
        for x in &mut p {
            x.routing.tiers = vec!["fast".into()];
        }
        assert!(
            Router::new(
                &p,
                RoutingConfig {
                    default_tier: Some("premium".into()),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn omitted_routing_fields_preserve_existing_config_order() {
        let config: missy_core::Config = serde_json::from_value(serde_json::json!({"providers":[
            {"name":"original-first","kind":"ollama","model":"model-a","base_url":"http://127.0.0.1:11434"},
            {"name":"original-second","kind":"ollama","model":"model-b","base_url":"http://127.0.0.1:11435"}
        ]})).unwrap();
        config.validate().unwrap();
        let r = Router::new(&config.providers, config.routing).unwrap();
        for _ in 0..5 {
            assert_eq!(r.plan(None, None, &[], &[]).unwrap(), vec![0, 1]);
        }
        assert_eq!(r.config.max_wait_ms, 0);
    }
    #[tokio::test]
    async fn limiter_deadline_and_released_concurrency() {
        let mut p = configs();
        p[0].routing.max_concurrent = 1;
        let r = Router::new(&p, RoutingConfig::default()).unwrap();
        let cancel = CancellationToken::new();
        let now = tokio::time::Instant::now();
        let permit = r.reserve(0, 1, 1, now, &cancel).await.unwrap().unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(250),
            r.reserve(0, 1, 1, now + Duration::from_millis(10), &cancel),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(result.is_none());
        let waiting = r.reserve(
            0,
            1,
            1,
            tokio::time::Instant::now() + Duration::from_secs(1),
            &cancel,
        );
        let releasing = async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(permit);
        };
        let (result, _) = tokio::join!(waiting, releasing);
        assert!(result.unwrap().is_some());
    }
    #[test]
    fn rules_only_use_user_size_and_operator_tool_count() {
        let mut p = configs();
        p[0].routing.tiers = vec!["fast".into()];
        p[1].routing.tiers = vec!["premium".into()];
        let r = Router::new(
            &p,
            RoutingConfig {
                default_tier: Some("fast".into()),
                rules: vec![missy_core::RoutingRule {
                    min_input_chars: 5,
                    min_tools: 0,
                    tier: "premium".into(),
                }],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            r.plan(
                None,
                None,
                &[
                    Message::text("assistant", "x".repeat(100)),
                    Message::text("tool", "x".repeat(100))
                ],
                &[]
            )
            .unwrap(),
            vec![0]
        );
        assert_eq!(
            r.plan(None, None, &[Message::text("user", "12345")], &[])
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            r.plan(None, Some("fast"), &[Message::text("user", "12345")], &[])
                .unwrap(),
            vec![0]
        );
    }
}
