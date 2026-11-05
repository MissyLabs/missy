//! Opt-in local model HTTP embeddings. No hash features, fallback vectors, model downloads,
//! process launches, environment credentials, or remote default. The operator starts a local
//! model server and declares its weights SHA256; HTTP alone cannot attest the loaded weights.
use anyhow::{Result, anyhow, ensure};
use missy_core::PolicyConfig;
use missy_policy::PolicyHttpClient;
use missy_retrieval::Embedder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc as async_mpsc, oneshot, watch};
use url::{Host, Url};

pub const MAX_INPUT_BYTES: usize = 16 * 1024;
const MAX_DIMENSIONS: usize = 4096;

/// All fields are explicit. `model_sha256` is the operator-declared digest of the exact
/// loaded weights, not a claim that an untrusted HTTP server has been attested.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalEmbeddingConfig {
    pub endpoint: String,
    pub allowed_origin: String,
    pub allow_private_network: bool,
    pub model: String,
    pub model_sha256: String,
    pub dimensions: usize,
    pub timeout_seconds: u64,
}
impl fmt::Debug for LocalEmbeddingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEmbeddingConfig")
            .field("endpoint", &"[redacted]")
            .field("model", &"[redacted]")
            .field("dimensions", &self.dimensions)
            .field("timeout_seconds", &self.timeout_seconds)
            .finish_non_exhaustive()
    }
}

struct Job {
    text: String,
    deadline: Instant,
    reply: mpsc::SyncSender<Result<Vec<f32>>>,
    cancel: oneshot::Receiver<()>,
}

/// Synchronous retrieval adapter backed by one bounded, dedicated async worker.
/// Works inside or outside an existing Tokio runtime without nesting block_on.
/// Only one active and one queued request fit. Overload fails closed instead of
/// allocating an unbounded queue. Each call's deadline includes time in the queue.
pub struct LocalHttpEmbedder {
    identity: String,
    dimensions: usize,
    timeout: Duration,
    jobs: async_mpsc::Sender<Job>,
    stop: watch::Sender<bool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl fmt::Debug for LocalHttpEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalHttpEmbedder")
            .field("identity", &self.identity)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl LocalHttpEmbedder {
    /// Construction validates configuration but never contacts the endpoint.
    pub fn new(mut config: LocalEmbeddingConfig) -> Result<Self> {
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|_| anyhow!("invalid local embedding endpoint"))?;
        ensure!(
            endpoint.scheme() == "http" || endpoint.scheme() == "https",
            "embedding scheme denied"
        );
        ensure!(
            endpoint.username().is_empty()
                && endpoint.password().is_none()
                && endpoint.query().is_none()
                && endpoint.fragment().is_none(),
            "embedding endpoint must not contain credentials, query or fragment"
        );
        ensure!(
            matches!(endpoint.host(), Some(Host::Ipv4(ip)) if ip.is_loopback())
                || matches!(endpoint.host(), Some(Host::Ipv6(ip)) if ip.is_loopback()),
            "embedding endpoint must use an explicit loopback IP"
        );
        ensure!(
            matches!(endpoint.path(), "/v1/embeddings" | "/embeddings"),
            "embedding endpoint path denied"
        );
        ensure!(
            config.allowed_origin == endpoint.origin().ascii_serialization(),
            "embedding exact origin opt-in required"
        );
        ensure!(
            config.allow_private_network,
            "embedding private-network opt-in required"
        );
        ensure!(
            !config.model.trim().is_empty()
                && config.model.len() <= 256
                && !config.model.chars().any(char::is_control),
            "invalid embedding model name"
        );
        ensure!(
            config.model_sha256.len() == 64
                && config.model_sha256.bytes().all(|v| v.is_ascii_hexdigit()),
            "embedding weights SHA256 required"
        );
        ensure!(
            (1..=MAX_DIMENSIONS).contains(&config.dimensions),
            "embedding dimension must be 1..4096"
        );
        ensure!(
            (1..=120).contains(&config.timeout_seconds),
            "embedding timeout must be 1..120 seconds"
        );
        config.model_sha256.make_ascii_lowercase();
        config.endpoint = endpoint.to_string();
        let mut digest = Sha256::new();
        for field in [
            &config.endpoint,
            &config.model,
            &config.model_sha256,
            &config.dimensions.to_string(),
        ] {
            digest.update((field.len() as u64).to_le_bytes());
            digest.update(field.as_bytes());
        }
        let identity = format!(
            "local-http-embeddings-v1:sha256:{}",
            hex::encode(digest.finalize())
        );
        let timeout = Duration::from_secs(config.timeout_seconds);
        let policy = PolicyConfig {
            allowed_origins: vec![config.allowed_origin.clone()],
            allow_private_network: config.allow_private_network,
            ..PolicyConfig::default()
        };
        let client = PolicyHttpClient::new(policy)?.with_timeout(timeout)?;
        let dimensions = config.dimensions;
        let (jobs, mut rx) = async_mpsc::channel::<Job>(1);
        let (stop, mut stopping) = watch::channel(false);
        // Build before spawning so a construction failure cannot leave a silently dead worker.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let worker = thread::Builder::new().name("missy-local-embeddings".into()).spawn(move || {
            runtime.block_on(async move {
                loop {
                    let job = tokio::select! {
                        biased;
                        _ = stopping.changed() => break,
                        job = rx.recv() => match job { Some(job) => job, None => break },
                    };
                    let result = tokio::select! {
                        biased;
                        _ = stopping.changed() => break,
                        _ = job.cancel => continue,
                        result = tokio::time::timeout_at(tokio::time::Instant::from_std(job.deadline),
                            request_embedding(&client, &config, &job.text)) =>
                            result.unwrap_or_else(|_| Err(anyhow!("local embedding deadline exceeded"))),
                    };
                    let _ = job.reply.try_send(result);
                }
            });
            runtime.shutdown_timeout(Duration::from_millis(100));
        }).map_err(|_| anyhow!("local embedding worker unavailable"))?;
        Ok(Self {
            identity,
            dimensions,
            timeout,
            jobs,
            stop,
            worker: Some(worker),
        })
    }
}
impl Embedder for LocalHttpEmbedder {
    fn name(&self) -> &str {
        &self.identity
    }
    fn dimension(&self) -> usize {
        self.dimensions
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        ensure!(
            !text.trim().is_empty() && text.len() <= MAX_INPUT_BYTES,
            "embedding input must be 1..16384 nonblank UTF-8 bytes"
        );
        let deadline = Instant::now() + self.timeout;
        let (reply, result) = mpsc::sync_channel(1);
        let (cancel, cancelled) = oneshot::channel();
        self.jobs
            .try_send(Job {
                text: text.to_owned(),
                deadline,
                reply,
                cancel: cancelled,
            })
            .map_err(|_| anyhow!("local embedding worker busy or stopped"))?;
        let response = result
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|_| anyhow!("local embedding deadline exceeded or worker stopped"));
        // Cancel in-flight HTTP as soon as the blocking caller times out.
        let _ = cancel.send(());
        response?
    }
}
impl Drop for LocalHttpEmbedder {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    model: String,
    data: Vec<EmbeddingData>,
}
#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

async fn request_embedding(
    client: &PolicyHttpClient,
    config: &LocalEmbeddingConfig,
    text: &str,
) -> Result<Vec<f32>> {
    let response = client.request("POST", &config.endpoint,
        vec![("content-type".into(), "application/json".into())],
        Some(serde_json::to_vec(&serde_json::json!({"model": config.model, "input": [text], "encoding_format": "float"}))?)
    ).await?;
    ensure!(
        response.status == 200,
        "local embedding server returned status {}",
        response.status
    );
    let mut response: EmbeddingResponse = serde_json::from_slice(&response.body)
        .map_err(|_| anyhow!("invalid local embedding response"))?;
    ensure!(
        response.model == config.model,
        "local embedding model identity mismatch"
    );
    ensure!(
        response.data.len() == 1 && response.data[0].index == 0,
        "local embedding response count/index mismatch"
    );
    let mut vector = response.data.remove(0).embedding;
    ensure!(
        vector.len() == config.dimensions && vector.iter().all(|v| v.is_finite()),
        "local embedding dimension or finite-value mismatch"
    );
    let norm = vector
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "local embedding returned zero or invalid vector"
    );
    for value in &mut vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(vector)
}
