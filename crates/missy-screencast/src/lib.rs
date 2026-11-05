//! A native browser screencast CHANNEL, not an alias for local desktop capture.
//! All session issuance is a trusted Rust API behind an injected approval verifier.
//! The browser must separately click Start and grant getDisplayMedia permission.
mod image_frame;
mod server;
mod vision;

pub use server::{ServerHandle, router, start};
pub use vision::{FrameGovernor, GovernedVisionAnalyzer};

use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{Scope, ToolContext};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Deliberately disabled by default. Only a literal loopback socket is supported.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScreencastConfig {
    pub enabled: bool,
    pub bind: SocketAddr,
    pub max_sessions: usize,
    pub max_connections: usize,
    pub max_parallel_frames: usize,
    pub max_frame_bytes: usize,
    pub max_pixels: u64,
    pub max_dimension: u32,
    pub max_frames_per_session: u64,
    pub session_ttl_seconds: u64,
    pub handshake_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub min_frame_interval_ms: u64,
    pub analysis_cooldown_ms: u64,
    /// Mean absolute grayscale thumbnail difference, 0..255. Zero still skips exact duplicates.
    pub change_threshold: u8,
    pub analysis_timeout_seconds: u64,
    pub max_analysis_bytes: usize,
    pub allow_analysis: bool,
    pub allow_analysis_delivery: bool,
    pub allow_frame_artifacts: bool,
}
impl Default for ScreencastConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind: "127.0.0.1:8780".parse().unwrap(),
            max_sessions: 20,
            max_connections: 20,
            max_parallel_frames: 2,
            max_frame_bytes: 2 * 1024 * 1024,
            max_pixels: 4_000_000,
            max_dimension: 4096,
            max_frames_per_session: 3600,
            session_ttl_seconds: 900,
            handshake_timeout_seconds: 5,
            idle_timeout_seconds: 60,
            min_frame_interval_ms: 2000,
            analysis_cooldown_ms: 10000,
            change_threshold: 3,
            analysis_timeout_seconds: 120,
            max_analysis_bytes: 16384,
            allow_analysis: false,
            allow_analysis_delivery: false,
            allow_frame_artifacts: false,
        }
    }
}
impl ScreencastConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.bind.ip().is_loopback() || self.bind.port() == 0 {
            bail!("screencast requires a fixed literal loopback address and port");
        }
        if !(1..=100).contains(&self.max_sessions)
            || !(1..=100).contains(&self.max_connections)
            || !(1..=8).contains(&self.max_parallel_frames)
            || !(1024..=5 * 1024 * 1024).contains(&self.max_frame_bytes)
            || !(1..=8_000_000).contains(&self.max_pixels)
            || !(1..=8192).contains(&self.max_dimension)
            || !(1..=100_000).contains(&self.max_frames_per_session)
            || !(1..=3600).contains(&self.session_ttl_seconds)
            || !(1..=10).contains(&self.handshake_timeout_seconds)
            || !(1..=300).contains(&self.idle_timeout_seconds)
            || !(100..=60_000).contains(&self.min_frame_interval_ms)
            || self.analysis_cooldown_ms > 300_000
            || !(1..=180).contains(&self.analysis_timeout_seconds)
            || !(64..=65536).contains(&self.max_analysis_bytes)
        {
            bail!("invalid bounded screencast limits");
        }
        if self.allow_analysis_delivery && !self.allow_analysis {
            bail!("analysis delivery requires analysis");
        }
        Ok(())
    }
    pub fn origin(&self) -> String {
        format!("http://{}", self.bind)
    }
}

/// Exact operator-approved disclosure policy. Never constructed from browser input.
/// `approval_receipt` must refer to a durable, exact scope/config-bound parent approval.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionApproval {
    pub scope: Scope,
    pub approval_receipt: String,
    pub ttl_seconds: u64,
    pub analyze: bool,
    pub deliver_analysis: bool,
    pub deliver_frame_artifacts: bool,
}
#[async_trait]
pub trait ApprovalVerifier: Send + Sync {
    /// Verify a durable human approval, its expiry, exact scope and disclosure flags.
    /// Implementations must bind the service config fingerprint in the approval.
    async fn verify(&self, approval: &SessionApproval, config_fingerprint: &str) -> Result<()>;
}

/// Returned exactly once; intentionally does NOT implement Debug or Serialize.
/// Send privately to the approved owner, never into logs, URLs, or shared Discord channels.
pub struct Pairing {
    pub session_id: String,
    pub token: String,
    pub scope: Scope,
    pub capture_url: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionPhase {
    Paired,
    Connected,
    Closed,
}
#[derive(Clone, Debug, Serialize)]
pub struct SessionStatus {
    pub session_id: String,
    pub scope: Scope,
    pub phase: SessionPhase,
    pub frames_received: u64,
    pub analyses_completed: u64,
    pub latest_analysis: Option<String>,
}

/// The stable context turn ID is unique per session/frame. Pixels are untrusted data.
/// No tool-capable agent is ever dispatched by this crate.
pub struct FrameRequest {
    pub context: ToolContext,
    pub approval_receipt: String,
    pub session_id: String,
    pub frame_number: u64,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    pub png: Vec<u8>,
}
#[async_trait]
pub trait FrameAnalyzer: Send + Sync {
    /// Must reserve durable billing/disclosure authorization before a real provider call.
    /// Use GovernedVisionAnalyzer for the existing configured NoTools vision backend.
    async fn analyze(&self, frame: &FrameRequest) -> Result<String>;
}

/// Sink destination is ONLY the approval's exact scope, never a browser-provided target.
/// This is an untrusted observation. Sinks must not execute it or permit mentions.
pub struct ChannelEvent {
    pub request_id: String,
    pub scope: Scope,
    pub session_id: String,
    pub frame_number: u64,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    pub analysis: Option<String>,
    /// Only present with BOTH operator config and exact-session artifact approval.
    pub png_artifact: Option<Vec<u8>>,
}
#[async_trait]
pub trait EventSink: Send + Sync {
    /// Implement durable idempotency on request_id; no automatic retries are performed here.
    async fn deliver(&self, event: ChannelEvent) -> Result<()>;
}

struct Session {
    approval: SessionApproval,
    token_hash: Option<[u8; 32]>,
    expires: Instant,
    cancel: CancellationToken,
    status: SessionStatus,
    last_frame: Option<Instant>,
    last_analysis: Option<Instant>,
    sample: Option<Vec<u8>>,
}
struct Inner {
    config: ScreencastConfig,
    verifier: Arc<dyn ApprovalVerifier>,
    analyzer: Option<Arc<dyn FrameAnalyzer>>,
    sink: Option<Arc<dyn EventSink>>,
    sessions: Mutex<HashMap<String, Session>>,
    shutdown: CancellationToken,
    connections: Arc<Semaphore>,
    frames: Arc<Semaphore>,
}
#[derive(Clone)]
pub struct ScreencastChannel {
    inner: Arc<Inner>,
}
impl ScreencastChannel {
    pub fn new(
        config: ScreencastConfig,
        verifier: Arc<dyn ApprovalVerifier>,
        analyzer: Option<Arc<dyn FrameAnalyzer>>,
        sink: Option<Arc<dyn EventSink>>,
    ) -> Result<Self> {
        config.validate()?;
        if config.allow_analysis && analyzer.is_none() {
            bail!("screencast analysis requires a governed analyzer");
        }
        if (config.allow_analysis_delivery || config.allow_frame_artifacts) && sink.is_none() {
            bail!("screencast delivery requires an explicit event sink");
        }
        Ok(Self {
            inner: Arc::new(Inner {
                connections: Arc::new(Semaphore::new(config.max_connections)),
                frames: Arc::new(Semaphore::new(config.max_parallel_frames)),
                config,
                verifier,
                analyzer,
                sink,
                sessions: Mutex::new(HashMap::new()),
                shutdown: CancellationToken::new(),
            }),
        })
    }
    pub fn config(&self) -> &ScreencastConfig {
        &self.inner.config
    }
    pub fn configuration_fingerprint(&self) -> String {
        format!(
            "screencast-v1:{}",
            hex::encode(Sha256::digest(
                serde_json::to_vec(self.config()).expect("config serializes")
            ))
        )
    }
    pub async fn create_session(&self, approval: SessionApproval) -> Result<Pairing> {
        if !self.config().enabled || self.inner.shutdown.is_cancelled() {
            bail!("screencast disabled or stopped");
        }
        approval.scope.validate()?;
        if approval.approval_receipt.is_empty()
            || approval.approval_receipt.len() > 256
            || approval.ttl_seconds == 0
            || approval.ttl_seconds > self.config().session_ttl_seconds
            || (approval.analyze && !self.config().allow_analysis)
            || (approval.deliver_analysis
                && (!approval.analyze || !self.config().allow_analysis_delivery))
            || (approval.deliver_frame_artifacts && !self.config().allow_frame_artifacts)
        {
            bail!("screencast session exceeds approved operator policy");
        }
        self.inner
            .verifier
            .verify(&approval, &self.configuration_fingerprint())
            .await?;
        let mut sessions = self.inner.sessions.lock().unwrap();
        // Closed/expired sessions are bounded and do not accumulate for process lifetime.
        sessions.retain(|_, s| {
            let keep = s.expires > Instant::now() && !s.cancel.is_cancelled();
            if !keep {
                s.cancel.cancel();
            }
            keep
        });
        if sessions.len() >= self.config().max_sessions {
            bail!("screencast session capacity reached");
        }
        if sessions
            .values()
            .any(|s| s.approval.scope == approval.scope)
        {
            bail!("scope already owns a live screencast session");
        }
        let id = Uuid::new_v4().to_string();
        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        let token = hex::encode(secret);
        let status = SessionStatus {
            session_id: id.clone(),
            scope: approval.scope.clone(),
            phase: SessionPhase::Paired,
            frames_received: 0,
            analyses_completed: 0,
            latest_analysis: None,
        };
        let pairing = Pairing {
            session_id: id.clone(),
            token: token.clone(),
            scope: approval.scope.clone(),
            capture_url: self.config().origin(),
        };
        sessions.insert(
            id.clone(),
            Session {
                expires: Instant::now() + Duration::from_secs(approval.ttl_seconds),
                approval,
                token_hash: Some(token_hash(&id, &token)),
                status,
                cancel: self.inner.shutdown.child_token(),
                last_frame: None,
                last_analysis: None,
                sample: None,
            },
        );
        Ok(pairing)
    }
    pub fn status(&self, owner: &Scope, id: &str) -> Result<SessionStatus> {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let session = sessions
            .get_mut(id)
            .filter(|s| &s.approval.scope == owner)
            .ok_or_else(|| anyhow::anyhow!("session unavailable"))?;
        if session.expires <= Instant::now() {
            close(session);
        }
        Ok(session.status.clone())
    }
    /// Owner-only cancellation invalidates the token and cancels pending work immediately.
    pub fn cancel_session(&self, owner: &Scope, id: &str) -> Result<()> {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let session = sessions
            .get_mut(id)
            .filter(|s| &s.approval.scope == owner)
            .ok_or_else(|| anyhow::anyhow!("session unavailable"))?;
        close(session);
        Ok(())
    }
    /// Terminal for this channel instance. Restart with a fresh channel; tokens never survive.
    pub fn shutdown(&self) {
        self.inner.shutdown.cancel();
        for session in self.inner.sessions.lock().unwrap().values_mut() {
            close(session);
        }
    }
    fn authenticate(&self, auth: server::Authenticate) -> Result<(CancellationToken, Instant)> {
        if auth.token.len() != 64 || auth.session_id.len() > 64 {
            bail!("authentication failed");
        }
        let mut sessions = self.inner.sessions.lock().unwrap();
        let session = sessions
            .get_mut(&auth.session_id)
            .ok_or_else(|| anyhow::anyhow!("authentication failed"))?;
        if session.expires <= Instant::now()
            || session.cancel.is_cancelled()
            || session.status.phase != SessionPhase::Paired
            || session.approval.scope != auth.scope
            || !session.token_hash.is_some_and(|hash| {
                bool::from(hash.ct_eq(&token_hash(&auth.session_id, &auth.token)))
            })
        {
            bail!("authentication failed");
        }
        // Consume under the owner lock. Concurrent sockets and reconnects cannot replay.
        session.token_hash = None;
        session.status.phase = SessionPhase::Connected;
        Ok((session.cancel.clone(), session.expires))
    }
    fn disconnect(&self, id: &str) {
        if let Some(session) = self.inner.sessions.lock().unwrap().get_mut(id) {
            close(session);
        }
    }
}
fn close(session: &mut Session) {
    session.token_hash = None;
    session.status.phase = SessionPhase::Closed;
    session.cancel.cancel();
}
fn token_hash(id: &str, token: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(id.as_bytes());
    hash.update([0]);
    hash.update(token.as_bytes());
    hash.finalize().into()
}
