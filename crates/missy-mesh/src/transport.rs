use crate::{
    MAX_ENVELOPE_BYTES, MeshNode, MeshRuntime, SignedEnvelope, VerifiedEnvelope, unix_now,
};
use anyhow::{Context, Result, ensure};
use axum::{
    Router,
    body::Bytes,
    extract::{ConnectInfo, DefaultBodyLimit, State},
    http::StatusCode,
    routing::post,
};
use missy_policy::PolicyHttpClient;
use std::net::SocketAddr;
use tokio::sync::mpsc;

#[derive(Clone)]
struct Ingress {
    node: MeshNode,
    inbox: mpsc::Sender<VerifiedEnvelope>,
}
impl MeshNode {
    /// Bounded signed ingress only. The receiver must explicitly dispatch
    /// admitted messages to apply_memory/delegation/Quorum. No remote tool API.
    /// Serve with into_make_service_with_connect_info::<SocketAddr>().
    pub fn router(&self, inbox: mpsc::Sender<VerifiedEnvelope>) -> Result<Router> {
        ensure!(
            inbox.max_capacity() > 0 && inbox.max_capacity() <= 256,
            "mesh inbox capacity must be 1..256"
        );
        Ok(Router::new()
            .route("/mesh/envelopes", post(receive))
            .layer(DefaultBodyLimit::max(MAX_ENVELOPE_BYTES))
            .with_state(Ingress {
                node: self.clone(),
                inbox,
            }))
    }
    /// Does not bind or launch until explicitly called by an operator-facing CLI.
    pub async fn serve(&self, inbox: mpsc::Sender<VerifiedEnvelope>) -> Result<()> {
        ensure!(
            self.inner.config.listen.ip().is_loopback(),
            "non-loopback mesh bind denied"
        );
        let listener = tokio::net::TcpListener::bind(self.inner.config.listen).await?;
        axum::serve(
            listener,
            self.router(inbox)?
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await?;
        Ok(())
    }
    /// Sequential bounded worker through the ordinary Runtime. Admitted
    /// receipts survive crashes; no transcript or tool output goes to peers.
    pub async fn serve_with_runtime(&self, runtime: std::sync::Arc<MeshRuntime>) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(32);
        let server = self.serve(tx);
        let worker = async {
            while let Some(envelope) = rx.recv().await {
                let _ = self.dispatch(envelope, &runtime).await;
            }
            Ok::<(), anyhow::Error>(())
        };
        tokio::try_join!(server, worker)?;
        Ok(())
    }
    /// Address comes exclusively from the configured, persisted peer record.
    /// No caller URL, broadcast discovery, redirects, proxy environment or
    /// peer-asserted endpoint. PolicyHttpClient independently enforces origins,
    /// DNS pinning, private-IP policy, TLS, timeouts and response size bounds.
    pub async fn send(&self, envelope: &SignedEnvelope) -> Result<u16> {
        envelope.body.validate(unix_now()?)?;
        ensure!(
            envelope.body.sender == self.peer_id(),
            "outbound sender must be local identity"
        );
        envelope.authenticate(&self.inner.identity.public_key())?;
        let endpoint = {
            let state = self
                .inner
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
            let peer = self.authorize(
                &state,
                &envelope.body.recipient,
                &envelope.body.scope,
                envelope.body.purpose,
            )?;
            ensure!(
                self.inner.config.permits(peer)?,
                "outbound peer differs from configured allowlist"
            );
            peer.endpoint
                .clone()
                .context("peer has no configured endpoint")?
        };
        let response = PolicyHttpClient::new(self.inner.config.http_policy.clone())?
            .with_timeout(std::time::Duration::from_secs(15))?
            .request(
                "POST",
                &endpoint,
                vec![("content-type".into(), "application/json".into())],
                Some(envelope.to_json()?),
            )
            .await?;
        ensure!(
            response.status == 202,
            "mesh upstream rejected admission: {}",
            response.status
        );
        Ok(response.status)
    }
}
async fn receive(
    State(state): State<Ingress>,
    ConnectInfo(remote): ConnectInfo<SocketAddr>,
    bytes: Bytes,
) -> StatusCode {
    if !remote.ip().is_loopback() {
        return StatusCode::FORBIDDEN;
    }
    // Reserve capacity before admitting a durable nonce. Backpressure never
    // causes memory growth or silently drops an already admitted message.
    let Ok(permit) = state.inbox.try_reserve() else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };
    let Ok(now) = unix_now() else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };
    match state.node.verify_json(&bytes, now) {
        Ok(verified) => {
            permit.send(verified);
            StatusCode::ACCEPTED
        }
        Err(_) => StatusCode::UNAUTHORIZED,
    }
}

// No unauthenticated status, read-memory, peer-add, grant, arbitrary URL or tool
// endpoint is intentionally exposed. Purpose-specific dispatch is local only.
