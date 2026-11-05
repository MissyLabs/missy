//! Bounded, lossy, payload-free runtime telemetry. Never retries uncertain POSTs.
use crate::{OperationalKind, OperationalSpan, OtlpExporter};
use anyhow::{Result, ensure};
use missy_core::PolicyConfig;
use missy_policy::PolicyHttpClient;
use missy_runtime::{RuntimeEvent, RuntimeEventKind, RuntimeObserver};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{Notify, mpsc},
    task::JoinHandle,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TelemetryConfig {
    /// None disables export entirely. Policy must independently allow this origin.
    pub endpoint: Option<String>,
    pub queue_capacity: usize,
    pub max_batch: usize,
    pub max_bytes: usize,
    pub flush_interval_ms: u64,
    pub request_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
}
impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            queue_capacity: 1024,
            max_batch: 64,
            max_bytes: 131_072,
            flush_interval_ms: 1000,
            request_timeout_ms: 5000,
            shutdown_timeout_ms: 10_000,
        }
    }
}
impl TelemetryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=65_536).contains(&self.queue_capacity),
            "invalid telemetry queue capacity"
        );
        ensure!(
            (1..=1024).contains(&self.max_batch),
            "invalid telemetry batch size"
        );
        // Conservative reservation ensures one full batch always fits, including parent IDs.
        ensure!(
            self.max_bytes >= 512 + self.max_batch * 768 && self.max_bytes <= 1_048_576,
            "invalid telemetry encoded byte limit"
        );
        ensure!(
            (10..=60_000).contains(&self.flush_interval_ms),
            "invalid telemetry flush interval"
        );
        ensure!(
            (1..=60_000).contains(&self.request_timeout_ms),
            "invalid telemetry request timeout"
        );
        ensure!(
            (1..=60_000).contains(&self.shutdown_timeout_ms),
            "invalid telemetry shutdown timeout"
        );
        if let Some(endpoint) = &self.endpoint {
            ensure!(
                endpoint.len() <= 2048
                    && endpoint.ends_with("/v1/traces")
                    && (endpoint.starts_with("https://") || endpoint.starts_with("http://")),
                "invalid telemetry endpoint"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct TelemetryMetrics {
    pub accepted_spans: u64,
    pub pending_spans: u64,
    pub exported_spans: u64,
    pub failed_spans: u64,
    pub dropped_spans: u64,
    pub successful_batches: u64,
    pub failed_batches: u64,
}
impl TelemetryMetrics {
    /// Fixed names only; no high-cardinality or user-controlled metric labels.
    pub fn prometheus(&self) -> String {
        format!(
            "missy_telemetry_accepted_spans_total {}\nmissy_telemetry_pending_spans {}\nmissy_telemetry_exported_spans_total {}\nmissy_telemetry_failed_spans_total {}\nmissy_telemetry_dropped_spans_total {}\nmissy_telemetry_successful_batches_total {}\nmissy_telemetry_failed_batches_total {}\n",
            self.accepted_spans,
            self.pending_spans,
            self.exported_spans,
            self.failed_spans,
            self.dropped_spans,
            self.successful_batches,
            self.failed_batches
        )
    }
}
/// Independent read-only metrics handle. Does not keep worker/channel alive.
#[derive(Clone)]
pub struct TelemetryMonitor(Arc<Counters>);
impl TelemetryMonitor {
    pub fn metrics(&self) -> TelemetryMetrics {
        self.0.snapshot()
    }
}
#[derive(Default)]
struct Counters {
    accepted: AtomicU64,
    pending: AtomicU64,
    exported: AtomicU64,
    failed: AtomicU64,
    dropped: AtomicU64,
    good_batches: AtomicU64,
    bad_batches: AtomicU64,
}
impl Counters {
    fn snapshot(&self) -> TelemetryMetrics {
        TelemetryMetrics {
            accepted_spans: self.accepted.load(Ordering::Relaxed),
            pending_spans: self.pending.load(Ordering::Relaxed),
            exported_spans: self.exported.load(Ordering::Relaxed),
            failed_spans: self.failed.load(Ordering::Relaxed),
            dropped_spans: self.dropped.load(Ordering::Relaxed),
            successful_batches: self.good_batches.load(Ordering::Relaxed),
            failed_batches: self.bad_batches.load(Ordering::Relaxed),
        }
    }
}
struct Observer {
    sender: Option<mpsc::Sender<Frame>>,
    counters: Arc<Counters>,
}
// Accounting follows ownership, including receiver cancellation racing a producer
// that already reserved a slot. No global pending swap can lose that race.
struct Frame {
    event: RuntimeEvent,
    counters: Arc<Counters>,
    completed: bool,
}
impl Drop for Frame {
    fn drop(&mut self) {
        self.counters.pending.fetch_sub(1, Ordering::Relaxed);
        if !self.completed {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl RuntimeObserver for Observer {
    fn event(&self, event: RuntimeEvent) {
        let Some(sender) = &self.sender else {
            return;
        };
        match sender.try_reserve() {
            Ok(permit) => {
                self.counters.pending.fetch_add(1, Ordering::Relaxed);
                self.counters.accepted.fetch_add(1, Ordering::Relaxed);
                permit.send(Frame {
                    event,
                    counters: self.counters.clone(),
                    completed: false,
                });
            }
            Err(_) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Dropping this handle does not stop a surviving runtime observer. Last sender
/// drop triggers bounded final drain. Keep handle for metrics/explicit shutdown.
pub struct TelemetryRuntime {
    observer: Arc<Observer>,
    worker: Option<JoinHandle<Result<()>>>,
    stop: Arc<Notify>,
    shutdown_timeout: Duration,
}
impl TelemetryRuntime {
    pub fn build_client(policy: PolicyConfig, config: TelemetryConfig) -> Result<Self> {
        config.validate()?;
        let counters = Arc::new(Counters::default());
        let stop = Arc::new(Notify::new());
        let shutdown_timeout = Duration::from_millis(config.shutdown_timeout_ms);
        let Some(endpoint) = &config.endpoint else {
            return Ok(Self {
                observer: Arc::new(Observer {
                    sender: None,
                    counters,
                }),
                worker: None,
                stop,
                shutdown_timeout,
            });
        };
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("enabled telemetry requires a Tokio runtime"))?;
        let exporter = OtlpExporter::new(
            PolicyHttpClient::new(policy)?,
            endpoint.clone(),
            config.max_batch,
            config.max_bytes,
        )?;
        let (sender, receiver) = mpsc::channel(config.queue_capacity);
        let observer = Arc::new(Observer {
            sender: Some(sender),
            counters: counters.clone(),
        });
        let worker = handle.spawn(worker(exporter, receiver, counters, stop.clone(), config));
        Ok(Self {
            observer,
            worker: Some(worker),
            stop,
            shutdown_timeout,
        })
    }
    pub fn observer(&self) -> Arc<dyn RuntimeObserver> {
        self.observer.clone()
    }
    pub fn metrics(&self) -> TelemetryMetrics {
        self.observer.counters.snapshot()
    }
    pub fn monitor(&self) -> TelemetryMonitor {
        TelemetryMonitor(self.observer.counters.clone())
    }
    pub async fn shutdown(mut self) -> Result<TelemetryMetrics> {
        self.stop.notify_one();
        if let Some(mut worker) = self.worker.take() {
            match tokio::time::timeout(self.shutdown_timeout, &mut worker).await {
                Ok(result) => result.map_err(|_| anyhow::anyhow!("telemetry worker failed"))??,
                Err(_) => {
                    worker.abort();
                    let _ = worker.await;
                    anyhow::bail!("telemetry shutdown timed out; unexported spans dropped");
                }
            }
        }
        Ok(self.metrics())
    }
}

fn span(event: RuntimeEvent) -> OperationalSpan {
    OperationalSpan {
        trace_id: hex::encode(event.trace_id),
        span_id: hex::encode(event.span_id),
        parent_span_id: event.parent_span_id.map(hex::encode),
        start_time_unix_nano: event.start_time_unix_nano,
        end_time_unix_nano: event.end_time_unix_nano,
        succeeded: event.succeeded,
        kind: match event.kind {
            RuntimeEventKind::Turn => OperationalKind::RuntimeTurn,
            RuntimeEventKind::Provider => OperationalKind::RuntimeProvider,
            RuntimeEventKind::Tool => OperationalKind::RuntimeTool,
            RuntimeEventKind::Settlement => OperationalKind::RuntimeSettlement,
        },
    }
}
async fn export(
    exporter: &OtlpExporter,
    batch: &mut Vec<Frame>,
    counters: &Counters,
    timeout: Duration,
) {
    if batch.is_empty() {
        return;
    }
    let count = batch.len() as u64;
    let spans: Vec<OperationalSpan> = batch.iter().map(|frame| span(frame.event)).collect();
    let success = matches!(
        tokio::time::timeout(timeout, exporter.export(&spans)).await,
        Ok(Ok(()))
    );
    if success {
        counters.exported.fetch_add(count, Ordering::Relaxed);
        counters.good_batches.fetch_add(1, Ordering::Relaxed);
    } else {
        counters.failed.fetch_add(count, Ordering::Relaxed);
        counters.bad_batches.fetch_add(1, Ordering::Relaxed);
    }
    for frame in batch.iter_mut() {
        frame.completed = true;
    }
    batch.clear();
}
async fn worker(
    exporter: OtlpExporter,
    mut receiver: mpsc::Receiver<Frame>,
    counters: Arc<Counters>,
    stop: Arc<Notify>,
    config: TelemetryConfig,
) -> Result<()> {
    let mut batch = Vec::with_capacity(config.max_batch);
    let timeout = Duration::from_millis(config.request_timeout_ms);
    let mut tick = tokio::time::interval(Duration::from_millis(config.flush_interval_ms));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if receiver.is_closed() {
            break;
        }
        tokio::select! {
            biased;
            _ = stop.notified() => { receiver.close(); break; }
            event = receiver.recv() => match event {
                Some(event) => { batch.push(event); if batch.len() >= config.max_batch { export(&exporter, &mut batch, &counters, timeout).await; } }
                None => break,
            },
            _ = tick.tick() => export(&exporter, &mut batch, &counters, timeout).await,
        }
    }
    receiver.close();
    tokio::time::timeout(Duration::from_millis(config.shutdown_timeout_ms), async {
        while let Some(event) = receiver.recv().await {
            batch.push(event);
            if batch.len() >= config.max_batch {
                export(&exporter, &mut batch, &counters, timeout).await;
            }
        }
        export(&exporter, &mut batch, &counters, timeout).await;
    })
    .await
    .map_err(|_| anyhow::anyhow!("telemetry drain timed out; unexported spans dropped"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
