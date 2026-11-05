//! Durable bounded agent runtime.
mod context;
mod guards;
mod memory_context;
#[cfg(test)]
#[path = "tests/observability.rs"]
mod observability_tests;
mod observer;
mod prompts;
use observer::Span;
pub use observer::{RuntimeEvent, RuntimeEventKind, RuntimeObserver};
pub use prompts::ScopedPromptResolver;
use prompts::{MemorySnapshot, PromptSnapshot};
#[cfg(test)]
mod tests;
use anyhow::{Context, Result, anyhow, bail, ensure};
use context::{Lineage, clip, compose, digest, frame_history};
pub use missy_core::RuntimeConfig;
use missy_core::{
    ApprovalRequired, Message, Provider, Scope, ToolCall, ToolContext, ToolDefinition,
    ToolDispatchRefused, ToolExecutor, Usage,
};
use missy_store::{Checkpoint, Lease, Store, ToolBegin, ToolStatus, Turn, TurnStatus};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub turn_id: String,
    pub status: String,
    pub content: String,
    pub usage: Usage,
    pub resumable: bool,
    pub deduplicated: bool,
}
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}
struct Inner {
    store: Arc<Store>,
    provider: Arc<dyn Provider>,
    tools: Arc<dyn ToolExecutor>,
    system: String,
    config: RuntimeConfig,
    context_hash: String,
    definitions: Vec<ToolDefinition>,
    schemas: HashMap<String, jsonschema::Validator>,
    permits: Arc<Semaphore>,
    active: Mutex<HashMap<String, CancellationToken>>,
    closing: AtomicBool,
    observer: Option<Arc<dyn RuntimeObserver>>,
    prompt_resolver: Option<Arc<dyn ScopedPromptResolver>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    version: u32,
    current_turn_start: usize,
    request_hash: String,
    deadline_ms: i64,
    history: String,
    transcript: Vec<Message>,
    steps: u32,
    continuations: u32,
    usage: Usage,
    seen_ids: HashSet<String>,
    lineage: Vec<Lineage>,
    partial: String,
    final_content: Option<String>,
    provider_pending: bool,
    #[serde(default)]
    guard_retries: u32,
    #[serde(default)]
    waiting_tools: Option<WaitingTools>,
    #[serde(default)]
    prompt_snapshot: Option<PromptSnapshot>,
    #[serde(default)]
    memory_snapshot: Option<MemorySnapshot>,
}
/// The only journal-free tool batch eligible for explicit resume. Original calls,
/// IDs and arguments are retained, never reconstructed by another provider call.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WaitingTools {
    calls: Vec<ToolCall>,
    approval_request_id: Option<String>,
}
struct AdmittedTask {
    result: RunResult,
    task: Option<JoinHandle<Result<RunResult>>>,
}
struct StopOnDrop(CancellationToken);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl Runtime {
    pub fn new(
        store: Arc<Store>,
        provider: Arc<dyn Provider>,
        tools: Arc<dyn ToolExecutor>,
        system_prompt: String,
        config: RuntimeConfig,
    ) -> Result<Self> {
        Self::construct(store, provider, tools, system_prompt, config, None)
    }
    pub fn new_observed(
        store: Arc<Store>,
        provider: Arc<dyn Provider>,
        tools: Arc<dyn ToolExecutor>,
        system_prompt: String,
        config: RuntimeConfig,
        observer: Arc<dyn RuntimeObserver>,
    ) -> Result<Self> {
        Self::construct(
            store,
            provider,
            tools,
            system_prompt,
            config,
            Some(observer),
        )
    }
    fn construct(
        store: Arc<Store>,
        provider: Arc<dyn Provider>,
        tools: Arc<dyn ToolExecutor>,
        system_prompt: String,
        config: RuntimeConfig,
        observer: Option<Arc<dyn RuntimeObserver>>,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(!system_prompt.is_empty(), "empty system prompt");
        let definitions = tools.definitions();
        let mut schemas = HashMap::new();
        for d in &definitions {
            ensure!(
                !d.name.is_empty()
                    && d.name.len() <= 128
                    && d.name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
                "invalid tool name"
            );
            ensure!(!schemas.contains_key(&d.name), "duplicate tool definition");
            reject_remote_refs(&d.parameters)?;
            schemas.insert(
                d.name.clone(),
                jsonschema::validator_for(&d.parameters)
                    .map_err(|e| anyhow!("invalid tool schema: {e}"))?,
            );
        }
        let context_hash = digest(&serde_json::to_string(
            &json!({"system":system_prompt,"config":config,"provider":provider.configuration_fingerprint(),"tools":definitions,"tool_configuration":tools.configuration_fingerprint()}),
        )?);
        let permits = Arc::new(Semaphore::new(config.max_concurrent_turns));
        Ok(Self {
            inner: Arc::new(Inner {
                store,
                provider,
                tools,
                system: system_prompt,
                config,
                context_hash,
                definitions,
                schemas,
                permits,
                active: Mutex::new(HashMap::new()),
                closing: AtomicBool::new(false),
                observer,
                prompt_resolver: None,
            }),
        })
    }
    pub fn store(&self) -> &Arc<Store> {
        &self.inner.store
    }
    /// Attach before cloning. Scopes share provider, store and capacity paths.
    pub fn with_prompt_resolver(mut self, resolver: Arc<dyn ScopedPromptResolver>) -> Result<Self> {
        let inner = Arc::get_mut(&mut self.inner)
            .context("attach prompt resolver before cloning runtime")?;
        inner.prompt_resolver = Some(resolver);
        let hash = self.current_configuration_hash()?;
        Arc::get_mut(&mut self.inner)
            .context("runtime unexpectedly shared")?
            .context_hash = hash;
        Ok(self)
    }
    pub fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.inner.definitions
    }
    /// Separate immutable NoTools runtime sharing provider, state and budgets.
    pub fn no_tools(&self) -> Result<Self> {
        self.check_configuration()?;
        let runtime = Self::construct(
            self.inner.store.clone(),
            self.inner.provider.clone(),
            Arc::new(NoTools),
            self.inner.system.clone(),
            self.inner.config.clone(),
            self.inner.observer.clone(),
        )?;
        match &self.inner.prompt_resolver {
            Some(resolver) => runtime.with_prompt_resolver(resolver.clone()),
            None => Ok(runtime),
        }
    }
    /// Close admissions before cancelling/draining. Uncertain effects stay journaled.
    pub async fn shutdown(&self, timeout: Duration) -> Result<()> {
        self.inner.closing.store(true, Ordering::SeqCst);
        self.inner.permits.close();
        {
            let active = self
                .inner
                .active
                .lock()
                .map_err(|_| anyhow!("active map poisoned"))?;
            for cancellation in active.values() {
                cancellation.cancel();
            }
        }
        tokio::time::timeout(timeout, async {
            loop {
                if self
                    .inner
                    .active
                    .lock()
                    .map_err(|_| anyhow!("active map poisoned"))?
                    .is_empty()
                {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .context("runtime shutdown drain timed out")?
    }
    /// Literal durable dispatch, never a provider prompt imitating exact calls.
    pub async fn run_tools(
        &self,
        scope: Scope,
        calls: Vec<ToolCall>,
        request_id: String,
    ) -> Result<RunResult> {
        ensure!(
            !self.inner.closing.load(Ordering::SeqCst),
            "runtime shutting down"
        );
        ensure!(
            !calls.is_empty() && calls.len() <= 32,
            "exact tools require 1..32 calls"
        );
        ensure!(
            serde_json::to_vec(&calls)?.len() <= self.inner.config.max_input_bytes,
            "exact tool request exceeds input bound"
        );
        let metadata = self.admission_context(&scope, &serde_json::to_string(&calls)?)?;
        let a = self
            .store()
            .admit_tools_with_metadata(&scope, &request_id, &calls, &metadata)?;
        if !a.created {
            return self.result(&scope, &a.turn, true);
        }
        match self.launch(scope.clone(), a.turn.clone()) {
            Ok(task) => task.await.context("runtime task failed")?,
            Err(e)
                if e.to_string().contains("capacity exhausted")
                    || e.to_string().contains("session already leased") =>
            {
                self.result(&scope, &a.turn, false)
            }
            Err(e) => Err(e),
        }
    }
    /// A cached construction hash is not proof that a shared executor/provider
    /// has remained unchanged. Refuse observed drift; never rebind an old turn.
    fn check_configuration(&self) -> Result<()> {
        let current = self.current_configuration_hash()?;
        ensure!(
            current == self.inner.context_hash,
            "runtime configuration changed; rebuild runtime before execution"
        );
        Ok(())
    }
    fn current_configuration_hash(&self) -> Result<String> {
        let mut value = json!({
            "system": self.inner.system,
            "config": self.inner.config,
            "provider": self.inner.provider.configuration_fingerprint(),
            "tools": self.inner.tools.definitions(),
            "tool_configuration": self.inner.tools.configuration_fingerprint(),
        });
        if let Some(resolver) = &self.inner.prompt_resolver {
            value["prompt_resolver"] = json!(resolver.configuration_fingerprint());
        }
        Ok(digest(&serde_json::to_string(&value)?))
    }
    /// Immutable local bounds for admission adapters; never exposes credentials.
    pub fn configuration(&self) -> &RuntimeConfig {
        &self.inner.config
    }
    /// Checked opaque configuration identity, never configuration contents.
    pub fn configuration_fingerprint(&self) -> Result<String> {
        self.check_configuration()?;
        Ok(self.inner.context_hash.clone())
    }
    pub fn status(&self, scope: &Scope, turn_id: &str) -> Result<Turn> {
        self.store().get_turn(scope, turn_id)
    }
    pub fn cancel(&self, scope: &Scope, turn_id: &str) -> Result<()> {
        self.store().request_cancel(scope, turn_id)?;
        if let Some(token) = self
            .inner
            .active
            .lock()
            .map_err(|_| anyhow!("active map poisoned"))?
            .get(turn_id)
        {
            token.cancel();
        }
        Ok(())
    }
    /// Owned task survives caller disconnect. Returns actual durable turn identity.
    pub async fn start(
        &self,
        scope: Scope,
        prompt: String,
        request_id: String,
    ) -> Result<RunResult> {
        Ok(self.admit_task(scope, prompt, request_id)?.result)
    }
    pub async fn run(&self, scope: Scope, prompt: String, request_id: String) -> Result<RunResult> {
        let a = self.admit_task(scope, prompt, request_id)?;
        match a.task {
            Some(t) => t.await.context("runtime task failed")?,
            None => Ok(a.result),
        }
    }
    pub async fn resume(&self, scope: Scope, turn_id: impl AsRef<str>) -> Result<RunResult> {
        scope.validate()?;
        let turn = self.store().get_turn(&scope, turn_id.as_ref())?;
        ensure!(
            matches!(
                turn.status,
                TurnStatus::Interrupted | TurnStatus::Admitted | TurnStatus::Running
            ),
            "terminal turn cannot resume"
        );
        ensure!(
            self.store().unresolved_tools(&scope, &turn.id)?.is_empty(),
            "unresolved tool outcome requires human reconciliation; generation and replay blocked"
        );
        self.launch(scope, turn)?
            .await
            .context("runtime task failed")?
    }
    fn admit_task(&self, scope: Scope, prompt: String, request_id: String) -> Result<AdmittedTask> {
        ensure!(
            !self.inner.closing.load(Ordering::SeqCst),
            "runtime shutting down"
        );
        scope.validate()?;
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= self.inner.config.max_input_bytes,
            "prompt empty or exceeds input bound"
        );
        let metadata = self.admission_context(&scope, &prompt)?;
        let a = self
            .store()
            .admit_with_metadata(&scope, &request_id, &prompt, &metadata)?;
        if !a.created {
            return Ok(AdmittedTask {
                result: self.result(&scope, &a.turn, true)?,
                task: None,
            });
        }
        let result = RunResult {
            turn_id: a.turn.id.clone(),
            status: "running".into(),
            content: String::new(),
            usage: Usage::default(),
            resumable: false,
            deduplicated: false,
        };
        match self.launch(scope.clone(), a.turn.clone()) {
            Ok(task) => Ok(AdmittedTask {
                result,
                task: Some(task),
            }),
            Err(e)
                if e.to_string().contains("capacity exhausted")
                    || e.to_string().contains("session already leased") =>
            {
                Ok(AdmittedTask {
                    result: self.result(&scope, &a.turn, false)?,
                    task: None,
                })
            }
            Err(e) => Err(e),
        }
    }
    fn fresh_state(&self, scope: &Scope, turn: &Turn) -> Result<State> {
        let snapshot = self.admitted_prompt(scope, &turn.id)?;
        ensure!(
            snapshot.is_some() || self.inner.prompt_resolver.is_none(),
            "missing admission prompt snapshot; cannot safely resolve an old turn"
        );
        let memory_snapshot = self.admitted_memory(scope, &turn.id)?;
        ensure!(
            memory_snapshot.is_some() || !self.inner.config.memory_context_enabled,
            "missing admission memory snapshot"
        );
        let mut transcript = vec![Message::text("user", turn.prompt.clone())];
        if let Some(memory) = &memory_snapshot
            && !memory.data.is_empty()
        {
            transcript.push(Message::text("user", memory.data.clone()));
        }
        Ok(State {
            version: 1,
            current_turn_start: 1,
            request_hash: digest(&turn.prompt),
            deadline_ms: turn.created_at.saturating_add(
                self.inner
                    .config
                    .max_turn_seconds
                    .saturating_mul(1000)
                    .min(i64::MAX as u64) as i64,
            ),
            history: frame_history(
                &self.store().history(scope, 100)?,
                self.inner.config.max_context_chars / 4,
            )?,
            transcript,
            steps: 0,
            continuations: 0,
            usage: Usage::default(),
            seen_ids: HashSet::new(),
            lineage: vec![],
            partial: String::new(),
            final_content: None,
            provider_pending: false,
            guard_retries: 0,
            waiting_tools: None,
            prompt_snapshot: snapshot,
            memory_snapshot,
        })
    }
    fn launch(&self, scope: Scope, turn: Turn) -> Result<JoinHandle<Result<RunResult>>> {
        // Serialize registration against shutdown's empty-map drain.
        let mut active = self
            .inner
            .active
            .lock()
            .map_err(|_| anyhow!("active map poisoned"))?;
        ensure!(
            !self.inner.closing.load(Ordering::SeqCst),
            "runtime shutting down"
        );
        self.check_configuration()?;
        let permit = self
            .inner
            .permits
            .clone()
            .try_acquire_owned()
            .context("runtime capacity exhausted; admitted request requires explicit resume")?;
        let lease = self.store().acquire_lease_bounded(
            &scope,
            &turn.id,
            &uuid::Uuid::new_v4().to_string(),
            Duration::from_secs(self.inner.config.lease_seconds),
            self.inner.config.max_concurrent_turns,
        )?;
        let initialize = (|| -> Result<State> {
            self.store().reserve_turn_budget(
                &scope,
                &lease,
                Usage {
                    input_tokens: self.inner.config.max_tokens,
                    output_tokens: 0,
                    cost_microusd: self.inner.config.max_cost_microusd,
                    cost_known: true,
                },
                self.inner.config.global_max_cost_microusd,
                self.inner.config.global_max_tokens,
            )?;
            ensure!(
                self.store().unresolved_tools(&scope, &turn.id)?.is_empty(),
                "unresolved tool outcome blocks resume under new fence"
            );
            let state = if self
                .store()
                .checkpoint(&scope, &turn.id, &self.inner.context_hash)?
                .is_some()
            {
                self.load(&scope, &turn)?
            } else {
                ensure!(
                    turn.status == TurnStatus::Admitted
                        && self.store().tools(&scope, &turn.id)?.is_empty(),
                    "missing checkpoint for previously executing turn"
                );
                self.fresh_state(&scope, &turn)?
            };
            self.save(&scope, &lease, &state)?;
            Ok(state)
        })();
        let state = match initialize {
            Ok(state) => state,
            Err(e) => {
                let _ = self.store().release_lease(&scope, &lease);
                return Err(e);
            }
        };
        let token = CancellationToken::new();
        active.insert(turn.id.clone(), token.clone());
        drop(active);
        let runtime = self.clone();
        Ok(tokio::spawn(async move {
            observer::observe_run(
                runtime.inner.observer.clone(),
                runtime.drive(scope, turn, lease, state, token, permit),
            )
            .await
        }))
    }
    fn save(&self, scope: &Scope, lease: &Lease, state: &State) -> Result<()> {
        let mut reserved_usage = state.usage;
        reserved_usage.cost_known &= !state.provider_pending;
        self.store()
            .update_turn_budget(scope, lease, reserved_usage, false)?;
        let system = state
            .prompt_snapshot
            .as_ref()
            .map(|p| p.text.as_str())
            .unwrap_or(&self.inner.system);
        let mut transcript = vec![Message::text("system", system)];
        transcript.extend_from_slice(&state.transcript);
        self.store().save_checkpoint(
            scope,
            lease,
            &Checkpoint {
                transcript,
                steps: state.steps,
                usage: state.usage,
                provider_context: serde_json::to_value(state)?,
                context_hash: self.inner.context_hash.clone(),
                system_prompt_hash: digest(system),
            },
        )
    }
    fn load(&self, scope: &Scope, turn: &Turn) -> Result<State> {
        self.check_configuration()?;
        let cp = self
            .store()
            .checkpoint(scope, &turn.id, &self.inner.context_hash)?
            .context("no checkpoint: cannot safely resume")?;
        let mut state: State = serde_json::from_value(cp.provider_context)?;
        let memory = self.admitted_memory(scope, &turn.id)?;
        ensure!(
            state.memory_snapshot == memory,
            "checkpoint memory differs from admission snapshot"
        );
        if let Some(memory) = &memory {
            memory.validate(scope)?;
            if !memory.data.is_empty() {
                ensure!(
                    state.transcript.get(1) == Some(&Message::text("user", memory.data.clone())),
                    "checkpoint memory transcript mismatch"
                );
            }
        } else {
            ensure!(
                !self.inner.config.memory_context_enabled,
                "missing checkpoint memory snapshot"
            );
        }
        let admitted = self.admitted_prompt(scope, &turn.id)?;
        if let Some(snapshot) = &state.prompt_snapshot {
            snapshot.validate(scope, &self.inner.context_hash)?;
            ensure!(
                Some(snapshot) == admitted.as_ref(),
                "checkpoint prompt differs from admission snapshot"
            );
        } else {
            ensure!(
                self.inner.prompt_resolver.is_none() && admitted.is_none(),
                "missing checkpoint prompt snapshot"
            );
        }
        let system = state
            .prompt_snapshot
            .as_ref()
            .map(|p| p.text.as_str())
            .unwrap_or(&self.inner.system);
        ensure!(
            cp.system_prompt_hash == digest(system),
            "system prompt integrity mismatch"
        );
        ensure!(
            state.version == 1
                && state.current_turn_start == 1
                && state.request_hash == digest(&turn.prompt),
            "checkpoint request integrity mismatch"
        );
        ensure!(
            state.transcript.first() == Some(&Message::text("user", turn.prompt.clone())),
            "original current request changed"
        );
        let mut expected = vec![Message::text("system", system)];
        expected.extend_from_slice(&state.transcript);
        ensure!(
            cp.transcript == expected && cp.steps == state.steps && cp.usage == state.usage,
            "checkpoint transcript integrity mismatch"
        );
        ensure!(
            state.steps <= self.inner.config.max_steps
                && state.continuations <= self.inner.config.max_continuations.saturating_add(1),
            "invalid checkpoint counters"
        );
        ensure!(
            state.deadline_ms
                == turn.created_at.saturating_add(
                    self.inner
                        .config
                        .max_turn_seconds
                        .saturating_mul(1000)
                        .min(i64::MAX as u64) as i64
                ),
            "checkpoint deadline changed"
        );
        if state.provider_pending {
            state.usage.cost_known = false;
            // The provider may already have returned calls before a crash. Keep
            // this marker until human reconciliation; never regenerate unknown
            // completion content, including when billing limits are disabled.
        }
        self.hydrate(scope, turn, &mut state)?;
        Ok(state)
    }
    fn result(&self, scope: &Scope, turn: &Turn, deduplicated: bool) -> Result<RunResult> {
        let usage = self
            .store()
            .checkpoint(scope, &turn.id, &self.inner.context_hash)?
            .map(|c| c.usage)
            .unwrap_or_default();
        let resumable = turn.status == TurnStatus::Interrupted
            && self.store().unresolved_tools(scope, &turn.id)?.is_empty()
            && self.load(scope, turn).is_ok_and(|state| {
                !state.provider_pending
                    && (state.final_content.is_some()
                        || if state.waiting_tools.is_some() {
                            self.consumed_budget(&state).is_ok()
                        } else {
                            self.budget(&state).is_ok()
                        })
            });
        Ok(RunResult {
            turn_id: turn.id.clone(),
            status: serde_json::to_value(turn.status)?
                .as_str()
                .unwrap_or("unknown")
                .into(),
            content: missy_security::redact_output(&turn.response.clone().unwrap_or_default()),
            usage,
            resumable,
            deduplicated,
        })
    }
}
fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}
struct NoTools;
#[async_trait::async_trait]
impl ToolExecutor for NoTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        bail!("NoTools runtime cannot execute tools")
    }
}
impl Runtime {
    fn hydrate(&self, scope: &Scope, turn: &Turn, state: &mut State) -> Result<()> {
        let records = self.store().tools(scope, &turn.id)?;
        if let Some(waiting) = &state.waiting_tools {
            let last = state
                .transcript
                .last()
                .context("waiting batch missing assistant")?;
            ensure!(
                !waiting.calls.is_empty()
                    && last.role == "assistant"
                    && last.tool_calls == waiting.calls
                    && state.final_content.is_none()
                    && !state.provider_pending,
                "waiting tool batch does not match current assistant calls"
            );
            ensure!(
                waiting
                    .calls
                    .iter()
                    .all(|call| !records.iter().any(|r| r.call.id == call.id)),
                "waiting tool batch already has a journal; replay blocked"
            );
            let mut validation = state.clone();
            for call in &waiting.calls {
                validation.seen_ids.remove(&call.id);
            }
            self.validate_calls(&waiting.calls, &mut validation)?;
        }
        let mut rebuilt = vec![state.transcript.first().context("missing request")?.clone()];
        let mut ids = HashSet::new();
        let mut i = 1;
        while i < state.transcript.len() {
            let message = &state.transcript[i];
            i += 1;
            ensure!(
                matches!(message.role.as_str(), "user" | "assistant")
                    && message.tool_call_id.is_none(),
                "invalid checkpoint role or orphan tool result"
            );
            ensure!(
                message.role == "assistant" || message.tool_calls.is_empty(),
                "user tool calls in checkpoint"
            );
            rebuilt.push(message.clone());
            let waiting = state
                .waiting_tools
                .as_ref()
                .filter(|_| i == state.transcript.len());
            for call in &message.tool_calls {
                ensure!(
                    ids.insert(call.id.clone()) && state.seen_ids.contains(&call.id),
                    "checkpoint duplicate or untracked call ID"
                );
                if waiting.is_some() {
                    continue;
                }
                let record = records
                    .iter()
                    .find(|r| r.call.id == call.id)
                    .context("checkpoint tool intent has no settled journal entry; no replay")?;
                ensure!(
                    record.call == *call && record.status == ToolStatus::Succeeded,
                    "checkpoint tool journal mismatch or non-success outcome; replay blocked"
                );
                let content = serde_json::to_string(
                    record
                        .result
                        .as_ref()
                        .context("succeeded call missing result")?,
                )?;
                if i < state.transcript.len() && state.transcript[i].role == "tool" {
                    ensure!(
                        state.transcript[i] == Message::tool(&call.id, content.clone()),
                        "checkpoint result does not match journal"
                    );
                    i += 1;
                }
                rebuilt.push(Message::tool(&call.id, content));
            }
        }
        ensure!(
            records.iter().all(|r| ids.contains(&r.call.id)),
            "journal call missing from checkpoint"
        );
        ensure!(
            ids == state.seen_ids,
            "checkpoint seen IDs do not match assistant calls"
        );
        state.transcript = rebuilt;
        Ok(())
    }
}
impl Runtime {
    async fn dispatch(
        &self,
        scope: &Scope,
        lease: &Lease,
        state: &mut State,
        token: &CancellationToken,
        calls: &[ToolCall],
        deadline: tokio::time::Instant,
    ) -> Result<()> {
        // Preflight the WHOLE batch before the first journal intent or effect.
        // A preflight error is a definite non-execution, never UNKNOWN.
        let mut pending = None;
        for call in calls {
            self.check_configuration()?;
            self.consumed_budget(state)?;
            ensure!(
                !token.is_cancelled()
                    && !self.store().cancellation_requested(scope, &lease.turn_id)?,
                "cancelled before preflight"
            );
            let ctx = ToolContext {
                scope: scope.clone(),
                turn_id: lease.turn_id.clone(),
                cancellation: token.child_token(),
            };
            let timeout = Duration::from_secs(self.inner.config.tool_timeout_seconds)
                .min(deadline.saturating_duration_since(tokio::time::Instant::now()));
            let result = tokio::select! {
                result = tokio::time::timeout(timeout, self.inner.tools.preflight(&ctx, call)) => result.context("tool preflight timeout").and_then(|r| r),
                _ = token.cancelled() => Err(anyhow!("tool preflight cancelled")),
            };
            ctx.cancellation.cancel();
            self.check_configuration()?;
            if let Err(error) = result {
                if let Some(approval) = error.downcast_ref::<ApprovalRequired>() {
                    pending.get_or_insert_with(|| approval.request_id.clone());
                } else {
                    state.waiting_tools = None;
                    self.save(scope, lease, state)?;
                    return Err(
                        error.context("tool preflight refused; no batch effects dispatched")
                    );
                }
            }
        }
        if let Some(request_id) = pending {
            state.waiting_tools = Some(WaitingTools {
                calls: calls.to_vec(),
                approval_request_id: Some(request_id.clone()),
            });
            self.save(scope, lease, state)?;
            return Err(ApprovalRequired { request_id }.into());
        }
        self.consumed_budget(state)?;
        ensure!(
            tokio::time::Instant::now() < deadline,
            "monotonic time budget exhausted before batch prepare"
        );
        ensure!(
            !token.is_cancelled() && !self.store().cancellation_requested(scope, &lease.turn_id)?,
            "cancelled before batch prepare"
        );
        // Store has no atomic checkpoint+batch-intent API. Clear waiting FIRST:
        // a crash in this gap fails closed on missing journal, never replays.
        state.waiting_tools = None;
        self.save(scope, lease, state)?;
        for call in calls {
            ensure!(
                matches!(
                    self.store().tool_begin(scope, lease, call)?,
                    ToolBegin::Prepared
                ),
                "replayed tool identity rejected"
            );
        }
        for call in calls {
            self.check_configuration()?;
            self.consumed_budget(state)?;
            ensure!(
                tokio::time::Instant::now() < deadline,
                "monotonic time budget exhausted before dispatch"
            );
            ensure!(
                !token.is_cancelled()
                    && !self.store().cancellation_requested(scope, &lease.turn_id)?,
                "cancelled before dispatch"
            );
            self.store().heartbeat(
                scope,
                lease,
                Duration::from_secs(self.inner.config.lease_seconds),
            )?;
            self.store().tool_running(scope, lease, &call.id)?;
            let ctx = ToolContext {
                scope: scope.clone(),
                turn_id: lease.turn_id.clone(),
                cancellation: token.child_token(),
            };
            let timeout = Duration::from_secs(self.inner.config.tool_timeout_seconds)
                .min(Duration::from_millis(
                    state.deadline_ms.saturating_sub(now_ms()).max(0) as u64,
                ))
                .min(deadline.saturating_duration_since(tokio::time::Instant::now()));
            let tool_span = Span::child(RuntimeEventKind::Tool);
            let mut execute = Box::pin(self.inner.tools.execute(&ctx, call));
            let result = tokio::select! {result=&mut execute=>result,_=token.cancelled()=>{ctx.cancellation.cancel();let _=tokio::time::timeout(Duration::from_secs(5),&mut execute).await;Err(anyhow!("tool cancelled; outcome unknown"))},_=tokio::time::sleep(timeout)=>{ctx.cancellation.cancel();let _=tokio::time::timeout(Duration::from_secs(5),&mut execute).await;Err(anyhow!("tool timed out; outcome unknown"))}};
            // BuiltinTools Drop synchronously joins process worker after killing/waiting children.
            drop(execute);
            tool_span.finish(result.is_ok());
            let settlement_span = Span::child(RuntimeEventKind::Settlement);
            match result {
                Ok(value) => {
                    let bounded = bound_tool_result(
                        &context::project_json(&value),
                        self.inner.config.max_tool_output_bytes,
                    )?;
                    self.store()
                        .tool_succeed_with_evidence(scope, lease, &call.id, &bounded, &value)?;
                    settlement_span.finish(true);
                    state
                        .transcript
                        .push(Message::tool(&call.id, serde_json::to_string(&bounded)?));
                    self.save(scope, lease, state)?;
                }
                Err(e) => {
                    let error = missy_security::redact_output(&format!("{e:#}"));
                    if e.downcast_ref::<ToolDispatchRefused>().is_some() {
                        let receipt = json!({"untrusted":true,"refused":true,"executed":false});
                        self.store().tool_settle(
                            scope,
                            lease,
                            &call.id,
                            ToolStatus::Failed,
                            Some(&receipt),
                            Some(&error),
                        )?;
                        settlement_span.finish(true);
                        state
                            .transcript
                            .push(Message::tool(&call.id, serde_json::to_string(&receipt)?));
                        self.save(scope, lease, state)?;
                        bail!(
                            "tool {} dispatch refused without execution; halted: {error}",
                            call.id
                        );
                    }
                    self.store().tool_settle(
                        scope,
                        lease,
                        &call.id,
                        ToolStatus::OutcomeUnknown,
                        None,
                        Some(&error),
                    )?;
                    settlement_span.finish(true);
                    bail!(
                        "tool {} outcome unknown; halted without retry: {error}",
                        call.id
                    );
                }
            }
        }
        Ok(())
    }
}
impl Runtime {
    async fn agent_loop(
        &self,
        scope: &Scope,
        lease: &Lease,
        state: &mut State,
        token: &CancellationToken,
    ) -> Result<String> {
        ensure!(
            !state.provider_pending,
            "provider completion not durably checkpointed; explicit reconciliation required"
        );
        ensure!(
            !self.store().cancellation_requested(scope, &lease.turn_id)?,
            "turn cancelled before completion"
        );
        if let Some(content) = &state.final_content {
            return Ok(content.clone());
        }
        let monotonic_deadline = tokio::time::Instant::now()
            + Duration::from_millis(state.deadline_ms.saturating_sub(now_ms()).max(0) as u64);
        if let Some(waiting) = &state.waiting_tools {
            let calls = waiting.calls.clone();
            self.consumed_budget(state)?;
            self.dispatch(scope, lease, state, token, &calls, monotonic_deadline)
                .await?;
        }
        if let Some(calls) = self.store().exact_tools(scope, &lease.turn_id)? {
            let initial_messages = 1 + usize::from(
                state
                    .memory_snapshot
                    .as_ref()
                    .is_some_and(|m| !m.data.is_empty()),
            );
            if state.transcript.len() == initial_messages {
                self.budget(state)?;
                self.validate_calls(&calls, state)?;
                state.steps = state.steps.saturating_add(1);
                state.transcript.push(Message {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: calls.clone(),
                    tool_call_id: None,
                });
                state.waiting_tools = Some(WaitingTools {
                    calls: calls.clone(),
                    approval_request_id: None,
                });
                self.save(scope, lease, state)?;
                self.dispatch(scope, lease, state, token, &calls, monotonic_deadline)
                    .await?;
            }
            let records = self.store().tools(scope, &lease.turn_id)?;
            let mut results = Vec::new();
            for call in &calls {
                let record = records
                    .iter()
                    .find(|r| r.call.id == call.id)
                    .context("missing exact tool receipt")?;
                ensure!(
                    record.status == ToolStatus::Succeeded,
                    "exact tool did not succeed; no automatic retry"
                );
                results.push(json!({"id":call.id,"result":record.result}));
            }
            let content = if results.len() == 1 {
                serde_json::to_string(&results[0]["result"])?
            } else {
                serde_json::to_string(&results)?
            };
            state.final_content = Some(content.clone());
            self.save(scope, lease, state)?;
            return Ok(content);
        }
        loop {
            self.check_configuration()?;
            self.budget(state)?;
            ensure!(
                tokio::time::Instant::now() < monotonic_deadline,
                "monotonic turn time budget exhausted"
            );
            ensure!(
                !token.is_cancelled()
                    && !self.store().cancellation_requested(scope, &lease.turn_id)?,
                "turn cancelled or lease lost"
            );
            ensure!(
                self.store()
                    .unresolved_tools(scope, &lease.turn_id)?
                    .is_empty(),
                "unresolved tool outcome blocks generation"
            );
            let safe_transcript = context::project_messages(&state.transcript, true);
            let safe_history = context::project_history(&state.history);
            let (messages, lineage) = compose(
                state
                    .prompt_snapshot
                    .as_ref()
                    .map(|p| p.text.as_str())
                    .unwrap_or(&self.inner.system),
                &safe_history,
                &safe_transcript,
                self.inner.config.max_context_chars,
            )?;
            if let Some(l) = lineage
                && state
                    .lineage
                    .last()
                    .is_none_or(|last| last.source_sha256 != l.source_sha256)
            {
                state.lineage.push(l);
            }
            state.steps = state.steps.saturating_add(1);
            state.provider_pending = true;
            self.save(scope, lease, state)?;
            let remaining =
                Duration::from_millis(state.deadline_ms.saturating_sub(now_ms()).max(0) as u64)
                    .min(monotonic_deadline.saturating_duration_since(tokio::time::Instant::now()));
            let provider_span = Span::child(RuntimeEventKind::Provider);
            let attempt = tokio::select! {_=token.cancelled()=>Err(anyhow!("generation cancelled")),result=tokio::time::timeout(remaining,self.inner.provider.complete(&messages,&self.inner.definitions,token.child_token()))=>result.context("turn timeout during generation").and_then(|r|r)};
            provider_span.finish(attempt.is_ok());
            state.provider_pending = false;
            let completion = match attempt {
                Ok(completion) => completion,
                Err(e) => {
                    state.usage.cost_known = false;
                    self.save(scope, lease, state)?;
                    return Err(e);
                }
            };
            state.usage.add(completion.usage);
            self.check_configuration()?;
            ensure!(
                !token.is_cancelled()
                    && !self.store().cancellation_requested(scope, &lease.turn_id)?,
                "cancelled after generation"
            );
            ensure!(
                completion.message.role == "assistant" && completion.message.tool_call_id.is_none(),
                "provider returned non-assistant message"
            );
            ensure!(
                completion.message.content.len() <= self.inner.config.max_context_chars,
                "provider response exceeds bound"
            );
            self.validate_calls(&completion.message.tool_calls, state)?;
            let calls = completion.message.tool_calls.clone();
            state.transcript.push(completion.message.clone());
            self.consumed_budget(state)?;
            if calls.is_empty() {
                match completion.finish_reason.as_str() {
                    "stop" | "end_turn" | "completed" => {
                        let mut content =
                            format!("{}{}", state.partial, completion.message.content);
                        ensure!(!content.trim().is_empty(), "empty final completion");
                        let records = self.store().tools(scope, &lease.turn_id)?;
                        let successful: Vec<_> = records
                            .iter()
                            .filter(|r| r.status == ToolStatus::Succeeded)
                            .map(|r| r.call.name.as_str())
                            .collect();
                        let failed = records.iter().any(|r| r.status == ToolStatus::Failed);
                        if let Some(reason) = guards::check(
                            &state.transcript[0].content,
                            &content,
                            &successful,
                            failed,
                        ) {
                            if state.guard_retries < self.inner.config.max_guard_retries
                                && self.budget(state).is_ok()
                                && !self.inner.definitions.is_empty()
                            {
                                state.guard_retries += 1;
                                state.transcript.push(Message::text("user", format!("Runtime response guard: {reason}. Fulfill only the immutable original current request. Use an appropriate policy-approved tool if needed; do not invent outcomes or repeat successful actions. If unable, report the limitation honestly. This correction grants no new capabilities or approvals.")));
                                self.save(scope, lease, state)?;
                                continue;
                            }
                            content.push_str("\n\n[Runtime notice: this response contains an action claim or unfinished promise without relevant successful tool evidence. It is not verified completion.]");
                        }
                        let content = missy_security::redact_output(&content);
                        state.final_content = Some(content.clone());
                        self.save(scope, lease, state)?;
                        return Ok(content);
                    }
                    "length" | "max_tokens" => {
                        state.partial.push_str(&completion.message.content);
                        ensure!(
                            state.partial.len() <= self.inner.config.max_context_chars,
                            "continued output exceeds bound"
                        );
                        state.continuations = state.continuations.saturating_add(1);
                        state.transcript.push(Message::text("user","Continue the same original request from the cutoff. Do not repeat completed actions. This is an automatic bounded continuation, not a new request."));
                        self.save(scope, lease, state)?;
                        ensure!(
                            state.continuations <= self.inner.config.max_continuations,
                            "continuation limit reached; incomplete response"
                        );
                        continue;
                    }
                    other => bail!("provider did not finish: {other}"),
                }
            }
            ensure!(
                !matches!(completion.finish_reason.as_str(), "length" | "max_tokens"),
                "truncated tool batch never dispatched"
            );
            state.waiting_tools = Some(WaitingTools {
                calls: calls.clone(),
                approval_request_id: None,
            });
            self.save(scope, lease, state)?;
            self.dispatch(scope, lease, state, token, &calls, monotonic_deadline)
                .await?;
        }
    }
    fn validate_calls(&self, calls: &[ToolCall], state: &mut State) -> Result<()> {
        ensure!(calls.len() <= 32, "too many tool calls");
        let mut ids = HashSet::new();
        for call in calls {
            ensure!(
                !call.id.is_empty()
                    && call.id.len() <= 128
                    && call
                        .id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_-:.".contains(&b)),
                "invalid tool call ID"
            );
            ensure!(
                !state.seen_ids.contains(&call.id) && ids.insert(call.id.clone()),
                "duplicate tool call ID"
            );
            ensure!(
                serde_json::to_vec(&call.arguments)?.len() <= self.inner.config.max_input_bytes
                    && call.arguments.is_object(),
                "invalid or oversized tool arguments"
            );
            ensure!(
                self.inner
                    .schemas
                    .get(&call.name)
                    .context("unknown tool")?
                    .is_valid(&call.arguments),
                "tool arguments fail schema for {}",
                call.name
            );
        }
        state.seen_ids.extend(ids);
        Ok(())
    }
}
impl Runtime {
    async fn drive(
        &self,
        scope: Scope,
        turn: Turn,
        lease: Lease,
        mut state: State,
        token: CancellationToken,
        _permit: OwnedSemaphorePermit,
    ) -> Result<RunResult> {
        let heartbeat_stop = CancellationToken::new();
        let _heartbeat_guard = StopOnDrop(heartbeat_stop.clone());
        let hb = {
            let runtime = self.clone();
            let scope = scope.clone();
            let lease = lease.clone();
            let token = token.clone();
            let stop = heartbeat_stop.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_millis(100));
                let mut beat_at = tokio::time::Instant::now();
                loop {
                    tokio::select! {_=stop.cancelled()=>break,_=tick.tick()=>{
                        if runtime.store().cancellation_requested(&scope,&lease.turn_id).unwrap_or(true){token.cancel();}
                        if beat_at.elapsed()>=Duration::from_secs(runtime.inner.config.lease_seconds/3){if runtime.store().heartbeat(&scope,&lease,Duration::from_secs(runtime.inner.config.lease_seconds)).is_err(){token.cancel();break;}beat_at=tokio::time::Instant::now();}
                    }}
                }
            })
        };
        let result = self.agent_loop(&scope, &lease, &mut state, &token).await;
        let (status, content, error) = match result {
            Ok(content) => (TurnStatus::Completed, Some(content), None),
            Err(e) => {
                let cancelled = self
                    .store()
                    .cancellation_requested(&scope, &turn.id)
                    .unwrap_or(false);
                (
                    if cancelled {
                        TurnStatus::Cancelled
                    } else {
                        TurnStatus::Interrupted
                    },
                    if state.partial.is_empty() {
                        None
                    } else {
                        Some(missy_security::redact_output(&state.partial))
                    },
                    Some(missy_security::redact_output(&format!("{e:#}"))),
                )
            }
        };
        let settlement_span = Span::child(RuntimeEventKind::Settlement);
        let final_result = match self.save(&scope, &lease, &state).and_then(|()| {
            self.store().update_turn_budget(
                &scope,
                &lease,
                state.usage,
                status == TurnStatus::Completed || status == TurnStatus::Cancelled,
            )
        }) {
            Ok(()) => {
                self.store()
                    .finish(&scope, &lease, status, content.as_deref(), error.as_deref())
            }
            Err(e) => Err(e),
        };
        settlement_span.finish(final_result.is_ok());
        heartbeat_stop.cancel();
        let _ = hb.await;
        self.inner
            .active
            .lock()
            .map_err(|_| anyhow!("active map poisoned"))?
            .remove(&turn.id);
        self.result(&scope, &final_result?, false)
    }
    fn budget(&self, state: &State) -> Result<()> {
        self.consumed_budget(state)?;
        ensure!(
            state.steps < self.inner.config.max_steps,
            "step budget exhausted"
        );
        if self.inner.config.max_cost_microusd > 0 {
            ensure!(
                state.usage.cost_microusd < self.inner.config.max_cost_microusd,
                "cost budget exhausted"
            );
        }
        ensure!(
            state
                .usage
                .input_tokens
                .saturating_add(state.usage.output_tokens)
                < self.inner.config.max_tokens,
            "token budget exhausted"
        );
        ensure!(
            state.continuations <= self.inner.config.max_continuations,
            "continuation budget exhausted"
        );
        Ok(())
    }
    fn consumed_budget(&self, state: &State) -> Result<()> {
        ensure!(now_ms() < state.deadline_ms, "turn time budget exhausted");
        ensure!(
            state
                .usage
                .input_tokens
                .saturating_add(state.usage.output_tokens)
                <= self.inner.config.max_tokens,
            "token budget exceeded"
        );
        if self.inner.config.max_cost_microusd > 0 {
            ensure!(state.usage.cost_known, "provider cost unknown");
            ensure!(
                state.usage.cost_microusd <= self.inner.config.max_cost_microusd,
                "cost budget exceeded"
            );
        }
        Ok(())
    }
}
fn reject_remote_refs(v: &Value) -> Result<()> {
    match v {
        Value::Object(o) => {
            for (k, v) in o {
                if k == "$ref" {
                    ensure!(
                        v.as_str().is_some_and(|s| s.starts_with('#')),
                        "remote schema refs disabled"
                    );
                }
                reject_remote_refs(v)?;
            }
        }
        Value::Array(a) => {
            for v in a {
                reject_remote_refs(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn bound_tool_result(value: &Value, max: usize) -> Result<Value> {
    let raw = serde_json::to_string(value)?;
    let hash = digest(&raw);
    let mut budget = max;
    loop {
        let bounded = json!({"untrusted":true,"sha256":hash,"truncated":raw.len()>budget,"data":clip(&raw,budget)});
        if serde_json::to_vec(&bounded)?.len() <= max {
            return Ok(bounded);
        }
        ensure!(budget > 0, "tool result envelope cannot fit output bound");
        budget /= 2;
    }
}
