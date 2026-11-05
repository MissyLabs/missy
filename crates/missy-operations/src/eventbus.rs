use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::broadcast;
/// Closed metadata vocabulary prevents prompt/credential leakage. Not durable.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationalKind {
    WebhookAccepted,
    WebhookRejected,
    ExportSucceeded,
    ExportFailed,
    WorkflowCompleted,
}
#[derive(Clone, Debug, Serialize)]
pub struct OperationalEvent {
    pub sequence: u64,
    pub timestamp_ms: i64,
    pub kind: OperationalKind,
}
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<OperationalEvent>,
    sequence: Arc<AtomicU64>,
}
pub struct EventSubscription {
    receiver: broadcast::Receiver<OperationalEvent>,
}
#[derive(Debug)]
pub enum EventRead {
    Event(OperationalEvent),
    Lagged { skipped: u64 },
    Closed,
}
impl EventBus {
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            (1..=65_536).contains(&capacity),
            "event bus capacity must be 1..65536"
        );
        let (sender, _) = broadcast::channel(capacity);
        Ok(Self {
            sender,
            sequence: Arc::new(AtomicU64::new(0)),
        })
    }
    pub fn subscribe(&self) -> EventSubscription {
        EventSubscription {
            receiver: self.sender.subscribe(),
        }
    }
    pub fn publish(&self, kind: OperationalKind) -> u64 {
        let sequence = self
            .sequence
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        let _ = self.sender.send(OperationalEvent {
            sequence,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            kind,
        });
        sequence
    }
}
impl EventSubscription {
    pub async fn recv(&mut self) -> EventRead {
        match self.receiver.recv().await {
            Ok(e) => EventRead::Event(e),
            Err(broadcast::error::RecvError::Lagged(skipped)) => EventRead::Lagged { skipped },
            Err(broadcast::error::RecvError::Closed) => EventRead::Closed,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn lag_and_close_are_explicit() {
        let bus = EventBus::new(2).unwrap();
        let mut sub = bus.subscribe();
        for _ in 0..5 {
            bus.publish(OperationalKind::WebhookAccepted);
        }
        assert!(matches!(sub.recv().await, EventRead::Lagged { skipped: 3 }));
        assert!(matches!(sub.recv().await, EventRead::Event(_)));
        assert!(matches!(sub.recv().await, EventRead::Event(_)));
        drop(bus);
        assert!(matches!(sub.recv().await, EventRead::Closed));
    }
}
