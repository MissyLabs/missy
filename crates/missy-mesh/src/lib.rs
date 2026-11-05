//! Native, deny-by-default mesh primitives. No Python bridge, peer discovery,
//! implicit trust, direct delegated tool dispatch, or live-state defaults.
mod config;
mod crdt;
mod envelope;
mod identity;
mod json;
mod private;
mod quorum;
mod transport;

pub use config::{Grant, MeshConfig, Peer};
pub use crdt::{Entry, LwwMap, MemoryUpdate};
pub use envelope::{
    Capability, EnvelopeBody, MAX_ENVELOPE_BYTES, Purpose, SignedEnvelope, VerifiedEnvelope,
};
pub use identity::{Identity, peer_id};
pub use quorum::{Quorum, QuorumResult, Vote};

use anyhow::{Context, Result, ensure};
use missy_core::{Provider, RuntimeConfig, Scope, ToolExecutor};
use missy_runtime::{RunResult, Runtime};
use private::{PrivateDir, STATE_LIMIT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_REPLAY_ENTRIES: usize = 8192;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    version: u32,
    local_id: String,
    peers: BTreeMap<String, Peer>,
    /// Expiry is persisted before any admitted message is returned to caller.
    replay: BTreeMap<String, u64>,
    high_water: u64,
    memory: LwwMap,
    #[serde(default)]
    receipts: BTreeMap<String, Receipt>,
}
struct Inner {
    identity: Identity,
    config: MeshConfig,
    dir: PrivateDir,
    _lock: File,
    state: Mutex<State>,
}
#[derive(Clone)]
pub struct MeshNode {
    inner: Arc<Inner>,
}
#[derive(Debug, Serialize)]
pub struct MeshStatus {
    pub peer_id: String,
    pub public_key: String,
    pub peers: usize,
    pub replay_entries: usize,
    pub memory_entries: usize,
    pub listen: std::net::SocketAddr,
}

impl MeshNode {
    /// Loads an existing identity. Only the explicit private state_dir may be
    /// created. Acquires an exclusive nonblocking lock for process lifetime.
    pub fn open(config: MeshConfig) -> Result<Self> {
        config.validate()?;
        let identity = Identity::load(&config.identity_path)?;
        let local_id = identity.peer_id();
        let dir = PrivateDir::open(&config.state_dir, true)?;
        let lock = dir.lock()?;
        let state = match dir.read("mesh.json", STATE_LIMIT)? {
            Some(bytes) => serde_json::from_value::<State>(json::parse_private(&bytes)?)?,
            None => {
                let peers = config
                    .authorized_peers
                    .iter()
                    .map(|p| Ok((p.id()?, p.clone())))
                    .collect::<Result<_>>()?;
                State {
                    version: 1,
                    local_id: local_id.clone(),
                    peers,
                    replay: BTreeMap::new(),
                    high_water: 0,
                    memory: LwwMap::default(),
                    receipts: BTreeMap::new(),
                }
            }
        };
        ensure!(
            state.version == 1 && state.local_id == local_id,
            "mesh state identity/version mismatch"
        );
        ensure!(
            state.peers.len() <= 256
                && state.replay.len() <= MAX_REPLAY_ENTRIES
                && state.receipts.len() <= MAX_REPLAY_ENTRIES,
            "persisted mesh limits exceeded"
        );
        for (id, peer) in &state.peers {
            peer.validate()?;
            ensure!(
                id == &peer.id()? && id != &local_id,
                "invalid persisted peer identity"
            );
        }
        ensure!(
            !config
                .authorized_peers
                .iter()
                .any(|p| p.id().ok().as_ref() == Some(&local_id)),
            "local identity must not be configured as peer"
        );
        state.memory.validate()?;
        dir.replace("mesh.json", &serde_json::to_vec(&state)?)?;
        Ok(Self {
            inner: Arc::new(Inner {
                identity,
                config,
                dir,
                _lock: lock,
                state: Mutex::new(state),
            }),
        })
    }
    pub fn peer_id(&self) -> String {
        self.inner.identity.peer_id()
    }
    pub fn status(&self) -> Result<MeshStatus> {
        let s = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
        Ok(MeshStatus {
            peer_id: self.peer_id(),
            public_key: self.inner.identity.public_key(),
            peers: s.peers.len(),
            replay_entries: s.replay.len(),
            memory_entries: s.memory.entries().count(),
            listen: self.inner.config.listen,
        })
    }
    pub fn peers(&self) -> Result<Vec<Peer>> {
        Ok(self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?
            .peers
            .values()
            .cloned()
            .collect())
    }
    /// Local operator API only. New or widened grants must already be present
    /// in private MeshConfig's authorized upper bound. Never an inbound action.
    pub fn peer_add(&self, peer: Peer) -> Result<String> {
        peer.validate()?;
        let id = peer.id()?;
        ensure!(
            id != self.peer_id() && self.inner.config.permits(&peer)?,
            "peer or grants outside local allowlist"
        );
        self.mutate(|s| {
            ensure!(
                s.peers.contains_key(&id) || s.peers.len() < 256,
                "peer capacity exceeded"
            );
            s.peers.insert(id.clone(), peer);
            Ok(())
        })?;
        Ok(id)
    }
    pub fn peer_remove(&self, id: &str) -> Result<bool> {
        self.mutate(|s| Ok(s.peers.remove(id).is_some()))
    }
    pub fn sign(
        &self,
        recipient: String,
        scope: Scope,
        purpose: Purpose,
        payload: Value,
        now: u64,
        ttl: u64,
    ) -> Result<SignedEnvelope> {
        let s = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
        self.authorize(&s, &recipient, &scope, purpose)?;
        SignedEnvelope::sign(
            &self.inner.identity,
            recipient,
            scope,
            purpose,
            payload,
            now,
            ttl,
        )
    }
    /// Authentication AND durable one-shot admission, not a side-effect-free
    /// diagnostic. Never returns an admitted envelope before nonce fsync.
    pub fn verify_json(&self, bytes: &[u8], now: u64) -> Result<VerifiedEnvelope> {
        let envelope = SignedEnvelope::from_json(bytes)?;
        envelope.body.validate(now)?;
        ensure!(
            envelope.body.recipient == self.peer_id(),
            "wrong mesh recipient"
        );
        self.mutate(|s| {
            ensure!(
                now >= s.high_water,
                "clock moved backwards; fail closed until restored"
            );
            let peer = self.authorize(
                s,
                &envelope.body.sender,
                &envelope.body.scope,
                envelope.body.purpose,
            )?;
            envelope.authenticate(&peer.public_key)?;
            // Keyed by sender AND nonce, independent of scope/purpose. Reusing a
            // nonce in another scope is still replay, not a second authority.
            let key = format!("{}:{}", envelope.body.sender, envelope.body.nonce);
            s.replay.retain(|_, expiry| *expiry > now);
            ensure!(!s.replay.contains_key(&key), "mesh replay denied");
            ensure!(
                s.replay.len() < MAX_REPLAY_ENTRIES,
                "replay capacity exhausted; no live nonce eviction"
            );
            s.replay.insert(key, envelope.body.expiry);
            ensure!(
                s.receipts.len() < MAX_REPLAY_ENTRIES,
                "receipt capacity exhausted; operator archival required"
            );
            s.receipts.insert(
                receipt_id(&envelope)?,
                Receipt {
                    id: receipt_id(&envelope)?,
                    scope: envelope.body.scope.clone(),
                    sender: envelope.body.sender.clone(),
                    purpose: envelope.body.purpose,
                    status: "admitted".into(),
                    turn_id: None,
                },
            );
            s.high_water = now;
            Ok(())
        })?;
        Ok(VerifiedEnvelope { envelope })
    }
    fn authorize<'a>(
        &self,
        s: &'a State,
        id: &str,
        scope: &Scope,
        purpose: Purpose,
    ) -> Result<&'a Peer> {
        let peer = s.peers.get(id).context("unknown mesh peer")?;
        ensure!(
            peer.allows(scope, purpose.capability()),
            "peer lacks exact-scope capability grant"
        );
        let allowed = self
            .inner
            .config
            .authorized_peers
            .iter()
            .find(|p| p.id().ok().as_deref() == Some(id))
            .context("peer removed from configuration allowlist")?;
        ensure!(
            allowed.allows(scope, purpose.capability()),
            "configuration denies capability"
        );
        Ok(peer)
    }
    fn mutate<T>(&self, update: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
        let mut candidate = state.clone();
        let result = update(&mut candidate)?;
        self.inner
            .dir
            .replace("mesh.json", &serde_json::to_vec(&candidate)?)?;
        *state = candidate;
        Ok(result)
    }
    fn reauthorize(&self, s: &State, verified: &VerifiedEnvelope, now: u64) -> Result<()> {
        let body = verified.body();
        body.validate(now)?;
        ensure!(
            now >= s.high_water,
            "clock moved backwards before execution"
        );
        ensure!(
            body.recipient == self.peer_id(),
            "verified token belongs to another node"
        );
        self.authorize(s, &body.sender, &body.scope, body.purpose)?;
        Ok(())
    }
    pub fn apply_memory(&self, verified: VerifiedEnvelope, now: u64) -> Result<bool> {
        self.mutate(|s| {
            self.reauthorize(s, &verified, now)?;
            s.memory.apply(&verified, now)
        })
    }
    pub fn memory_get(&self, scope: &Scope, key: &str) -> Result<Option<Value>> {
        scope.validate()?;
        Ok(self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?
            .memory
            .get(scope, key)
            .cloned())
    }
    /// Local operator API. Never exposed by the inbound HTTP router. An
    /// admitted receipt is not proof of execution and must not be redispatched.
    pub fn receipt(&self, scope: &Scope, id: &str) -> Result<Option<Receipt>> {
        scope.validate()?;
        Ok(self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?
            .receipts
            .get(id)
            .filter(|r| &r.scope == scope)
            .cloned())
    }
    /// Full inbound dispatch through the ordinary bounded Runtime. No response
    /// content is returned to peers. Other purposes require a local consumer.
    pub async fn dispatch(
        &self,
        verified: VerifiedEnvelope,
        runtime: &MeshRuntime,
    ) -> Result<Receipt> {
        let id = receipt_id(&verified.envelope)?;
        let now = unix_now()?;
        let outcome: Result<(String, Option<String>)> = match verified.body().purpose {
            Purpose::MemoryUpdate => self
                .apply_memory(verified, now)
                .map(|_| ("completed".into(), None)),
            Purpose::DelegateRequest => match self.delegation(verified, now) {
                Ok(request) => self
                    .run_delegation(request, runtime)
                    .await
                    .map(|r| (r.status, Some(r.turn_id))),
                Err(e) => Err(e),
            },
            _ => Err(anyhow::anyhow!("purpose requires explicit local consumer")),
        };
        let success = outcome.is_ok();
        let (status, turn_id) = outcome.unwrap_or_else(|_| ("not_completed".into(), None));
        let receipt = self.mutate(|s| {
            let receipt = s
                .receipts
                .get_mut(&id)
                .context("missing admitted receipt")?;
            receipt.status = status;
            receipt.turn_id = turn_id;
            Ok(receipt.clone())
        })?;
        ensure!(
            success,
            "mesh dispatch not completed; reconcile local runtime journal before retry"
        );
        Ok(receipt)
    }
    /// Builds an opaque request using the exact locally granted Scope. A peer
    /// cannot select tools, runtime policy, a different scope, or a budget.
    pub fn delegation(&self, verified: VerifiedEnvelope, now: u64) -> Result<DelegationRequest> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
        self.reauthorize(&state, &verified, now)?;
        let body = verified.body();
        ensure!(
            body.purpose == Purpose::DelegateRequest,
            "not a delegation request"
        );
        let task: DelegatedTask = serde_json::from_value(body.payload.clone())?;
        ensure!(
            !task.task.trim().is_empty() && task.task.len() <= 16 * 1024,
            "delegated task input exceeds bounds"
        );
        ensure!(
            !task.task_id.is_empty()
                && task.task_id.len() <= 128
                && task
                    .task_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "invalid delegated task id"
        );
        let digest = hex::encode(Sha256::digest(serde_json::to_vec(&(
            &body.sender,
            &body.recipient,
            &body.scope,
            &task.task_id,
        ))?));
        Ok(DelegationRequest {
            scope: body.scope.clone(),
            task: task.task,
            request_id: format!("mesh_{digest}"),
            sender: body.sender.clone(),
            expires: body.expiry,
            recipient: body.recipient.clone(),
        })
    }
    pub async fn run_delegation(
        &self,
        request: DelegationRequest,
        runtime: &MeshRuntime,
    ) -> Result<RunResult> {
        {
            let s = self
                .inner
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("mesh lock poisoned"))?;
            ensure!(
                request.recipient == self.peer_id() && unix_now()? < request.expires,
                "delegation expired or recipient mismatch"
            );
            self.authorize(
                &s,
                &request.sender,
                &request.scope,
                Purpose::DelegateRequest,
            )?;
        }
        runtime
            .0
            .run(request.scope, request.task, request.request_id)
            .await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelegatedTask {
    pub task_id: String,
    pub task: String,
}
/// Fields are private so only a locally authenticated admission can create one.
pub struct DelegationRequest {
    scope: Scope,
    task: String,
    request_id: String,
    sender: String,
    expires: u64,
    recipient: String,
}
impl DelegationRequest {
    pub fn scope(&self) -> &Scope {
        &self.scope
    }
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
    pub fn task(&self) -> &str {
        &self.task
    }
}
/// A dedicated locally budgeted runtime. Cannot wrap an arbitrary existing
/// unlimited runtime or dispatch ToolExecutor directly.
pub struct MeshRuntime(Arc<Runtime>);
impl MeshRuntime {
    pub fn new(
        store: Arc<missy_store::Store>,
        provider: Arc<dyn Provider>,
        tools: Arc<dyn ToolExecutor>,
        system: String,
        budget: RuntimeConfig,
    ) -> Result<Self> {
        validate_budget(&budget)?;
        Ok(Self(Arc::new(Runtime::new(
            store, provider, tools, system, budget,
        )?)))
    }
    pub fn try_from_runtime(runtime: Arc<Runtime>) -> Result<Self> {
        validate_budget(runtime.configuration())?;
        Ok(Self(runtime))
    }
}
fn validate_budget(budget: &RuntimeConfig) -> Result<()> {
    budget.validate()?;
    ensure!(
        budget.max_steps <= 32
            && budget.max_tokens <= 100_000
            && budget.max_turn_seconds <= 300
            && budget.max_cost_microusd > 0
            && budget.max_cost_microusd <= 1_000_000
            && budget.max_input_bytes <= 16 * 1024
            && budget.max_concurrent_turns <= 8,
        "mesh runtime requires explicit bounded local budget"
    );
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub id: String,
    pub scope: Scope,
    pub sender: String,
    pub purpose: Purpose,
    pub status: String,
    pub turn_id: Option<String>,
}
pub fn receipt_id(envelope: &SignedEnvelope) -> Result<String> {
    Ok(hex::encode(Sha256::digest(envelope.body.signing_bytes()?)))
}
pub fn unix_now() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

#[cfg(test)]
mod tests;
