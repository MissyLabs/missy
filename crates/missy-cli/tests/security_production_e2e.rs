//! Independent production-boundary fixtures. Disposable state; no real provider,
//! account, device, network request or deployment. Known-bug reproductions assert
//! the observed defect, not the desired security property.
#[allow(dead_code)]
#[path = "../src/extensions.rs"]
mod extensions;

use anyhow::Result;
use async_trait::async_trait;
use missy_core::{
    Completion, Message, Provider, RuntimeConfig, Scope, ToolCall, ToolContext, ToolDefinition,
    ToolExecutor,
};
use missy_runtime::Runtime;
use missy_store::Store;
use serde_json::json;
use sha2::Digest;
use std::{collections::BTreeSet, sync::Arc};
use tokio_util::sync::CancellationToken;

struct NeverProvider;
struct BlockingProvider {
    entered: std::sync::atomic::AtomicBool,
    cancelled: std::sync::atomic::AtomicBool,
}
#[async_trait]
impl Provider for BlockingProvider {
    fn name(&self) -> &str {
        "independent-blocking-fixture"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        cancel: CancellationToken,
    ) -> Result<Completion> {
        self.entered
            .store(true, std::sync::atomic::Ordering::SeqCst);
        cancel.cancelled().await;
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
        anyhow::bail!("cancelled disposable provider")
    }
}
#[async_trait]
impl Provider for NeverProvider {
    fn name(&self) -> &str {
        "independent-no-provider"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        anyhow::bail!("fixture must never contact a provider")
    }
}
fn scope(session: &str) -> Scope {
    Scope {
        tenant: "fixture".into(),
        user: "operator".into(),
        channel: "test".into(),
        session: session.into(),
    }
}
fn runtime(store: Arc<Store>, tools: Arc<dyn ToolExecutor>, config: RuntimeConfig) -> Arc<Runtime> {
    Arc::new(
        Runtime::new(
            store,
            Arc::new(NeverProvider),
            tools,
            "Immutable independent fixture".into(),
            config,
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn real_builtin_raw_evidence_survives_projection_scope_and_deduplication() {
    let dir = tempfile::tempdir().unwrap();
    let credential = format!("ghp_{}", "x".repeat(36));
    let original =
        format!("<h1>Fixture title</h1><script>RAW_ONLY_CANARY</script><p>{credential}</p>");
    let source = dir.path().join("document.html");
    std::fs::write(&source, &original).unwrap();
    let store = Arc::new(Store::open(dir.path().join("state.sqlite")).unwrap());
    let cfg = RuntimeConfig::default();
    let tools = Arc::new(
        missy_tools::BuiltinTools::new(
            missy_core::PolicyConfig {
                read_roots: vec![dir.path().into()],
                max_file_bytes: 1_048_576,
                ..Default::default()
            },
            cfg.clone(),
            store.clone(),
        )
        .unwrap(),
    );
    let rt = runtime(store.clone(), tools, cfg);
    let call = ToolCall {
        id: "document".into(),
        name: "read_document".into(),
        arguments: json!({"path":source,"format":"html"}),
    };
    let done = rt
        .run_tools(
            scope("document"),
            vec![call.clone()],
            "document-request".into(),
        )
        .await
        .unwrap();
    assert_eq!(done.status, "completed");
    for forbidden in [&credential, "RAW_ONLY_CANARY", "raw_evidence"] {
        assert!(!done.content.contains(forbidden));
    }
    assert!(done.content.contains("Fixture title"));
    let evidence = store
        .tool_evidence(&scope("document"), &done.turn_id, "document")
        .unwrap()
        .unwrap();
    assert_eq!(evidence["data"]["raw_evidence"]["source_utf8"], original);
    assert!(
        store
            .tool_evidence(&scope("wrong"), &done.turn_id, "document")
            .is_err()
    );
    let duplicate = rt
        .run_tools(scope("document"), vec![call], "document-request".into())
        .await
        .unwrap();
    assert!(duplicate.deduplicated);
    assert_eq!(done.content, duplicate.content);
    assert_eq!(
        store
            .tool_evidence(&scope("document"), &done.turn_id, "document")
            .unwrap()
            .unwrap(),
        evidence
    );
}

#[tokio::test]
async fn no_tools_runtime_rejects_literal_tool_dispatch_before_execution() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(dir.path().join("state.sqlite")).unwrap());
    let rt = runtime(
        store,
        Arc::new(extensions::NoTools),
        RuntimeConfig::default(),
    );
    assert!(rt.tool_definitions().is_empty());
    let output = dir.path().join("must-not-exist");
    let result = rt
        .run_tools(
            scope("safe"),
            vec![ToolCall {
                id: "write".into(),
                name: "file_write".into(),
                arguments: json!({"path":output,"content":"forbidden"}),
            }],
            "safe-request".into(),
        )
        .await;
    assert!(result.is_err() || result.as_ref().unwrap().status != "completed");
    assert!(!output.exists());
}

#[tokio::test]
async fn reproduces_plan_child_starvation_at_valid_single_turn_global_capacity() {
    let dir = tempfile::tempdir().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Arc::new(Store::open(dir.path().join("state.sqlite")).unwrap());
    let config = RuntimeConfig {
        max_concurrent_turns: 1,
        ..Default::default()
    };
    let builtin = Arc::new(
        missy_tools::BuiltinTools::new(Default::default(), config.clone(), store.clone()).unwrap(),
    );
    let base = runtime(store.clone(), builtin.clone(), config.clone());
    let scheduler =
        missy_automation::Scheduler::open(dir.path().join("scheduler.sqlite"), base).unwrap();
    let tools = Arc::new(missy_automation::AutomationTools::new(builtin, scheduler).unwrap());
    let registry =
        missy_governance::ApprovalRegistry::open(dir.path().join("approvals.sqlite")).unwrap();
    let guarded = Arc::new(
        missy_governance::GuardedTools::new(
            tools,
            registry.clone(),
            missy_governance::GuardConfig {
                sensitive_tool_names: BTreeSet::from(["execute_plan".into()]),
                context_fingerprint: "0".repeat(64),
                approval_ttl_seconds: 300,
            },
        )
        .unwrap(),
    );
    let parent = runtime(store.clone(), guarded, config);
    let call = ToolCall {
        id: "plan".into(),
        name: "execute_plan".into(),
        arguments: json!({"operation_id":"one","plan":{"nodes":[{"id":"list","tool":"list_files","arguments":{"path":"/not-accessed"}}]}}),
    };
    let paused = parent
        .run_tools(scope("parent"), vec![call], "plan-request".into())
        .await
        .unwrap();
    assert_eq!(paused.status, "interrupted");
    let review = registry
        .pending_reviews(&scope("parent"), 10)
        .unwrap()
        .pop()
        .unwrap();
    registry
        .decide(
            &scope("parent"),
            &review.id,
            &review.binding_hash,
            missy_governance::Decision::Approve,
            "fixture",
        )
        .unwrap();
    let done = parent
        .resume(scope("parent"), &paused.turn_id)
        .await
        .unwrap();
    let evidence = store
        .tool_evidence(&scope("parent"), &done.turn_id, "plan")
        .unwrap()
        .unwrap();
    assert_eq!(
        evidence["nodes"]["list"]["status"], "admitted",
        "single slot is held by waiting parent: {evidence}"
    );
    let child_scope = missy_automation::model_plan_scope(&scope("parent"), "one").unwrap();
    let children = store.list_turns(&child_scope, 10).unwrap();
    assert_eq!(children.len(), 1);
    assert!(
        store
            .tools(&child_scope, &children[0].id)
            .unwrap()
            .is_empty()
    );
}

struct CountTools(std::sync::atomic::AtomicUsize);
#[async_trait]
impl ToolExecutor for CountTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "effect".into(),
            description: "fixture effect".into(),
            parameters: json!({"type":"object"}),
            side_effect: true,
        }]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<serde_json::Value> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(json!({"effect":true}))
    }
}

#[tokio::test]
async fn outer_guard_exact_scope_turn_id_tool_id_and_argument_binding_is_single_use() {
    let dir = tempfile::tempdir().unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let registry =
        missy_governance::ApprovalRegistry::open(dir.path().join("approvals.sqlite")).unwrap();
    let inner = Arc::new(CountTools(std::sync::atomic::AtomicUsize::new(0)));
    let guard = missy_governance::GuardedTools::new(
        inner.clone(),
        registry.clone(),
        missy_governance::GuardConfig {
            sensitive_tool_names: BTreeSet::from(["effect".into()]),
            context_fingerprint: "0".repeat(64),
            approval_ttl_seconds: 300,
        },
    )
    .unwrap();
    let ctx = ToolContext {
        scope: scope("binding"),
        turn_id: "turn".into(),
        cancellation: CancellationToken::new(),
    };
    let call = ToolCall {
        id: "call".into(),
        name: "effect".into(),
        arguments: json!({"value":"reviewed"}),
    };
    assert!(
        guard
            .preflight(&ctx, &call)
            .await
            .unwrap_err()
            .is::<missy_core::ApprovalRequired>()
    );
    let review = registry
        .pending_reviews(&ctx.scope, 10)
        .unwrap()
        .pop()
        .unwrap();
    registry
        .decide(
            &ctx.scope,
            &review.id,
            &review.binding_hash,
            missy_governance::Decision::Approve,
            "fixture",
        )
        .unwrap();
    let mut changed = call.clone();
    changed.arguments["value"] = json!("changed");
    assert!(guard.execute(&ctx, &changed).await.is_err());
    changed = call.clone();
    changed.id = "other".into();
    assert!(guard.execute(&ctx, &changed).await.is_err());
    let mut changed_ctx = ctx.clone();
    changed_ctx.scope.user = "other".into();
    assert!(guard.execute(&changed_ctx, &call).await.is_err());
    changed_ctx = ctx.clone();
    changed_ctx.turn_id = "other".into();
    assert!(guard.execute(&changed_ctx, &call).await.is_err());
    assert_eq!(inner.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    guard.execute(&ctx, &call).await.unwrap();
    assert!(guard.execute(&ctx, &call).await.is_err());
    assert_eq!(inner.0.load(std::sync::atomic::Ordering::SeqCst), 1);
}

struct Gate;
#[async_trait]
impl missy_discord_voice::VoiceApprovalGate for Gate {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> Result<bool> {
        Ok(true)
    }
}
struct FakeGateway;
#[async_trait]
impl missy_discord_voice::GatewayVoiceAdapter for FakeGateway {
    async fn negotiate(
        &self,
        _: u64,
        _: u64,
        _: u64,
    ) -> Result<missy_discord_voice::GatewayVoiceCredentials> {
        anyhow::bail!("no network in fixture")
    }
    async fn leave(&self, _: u64, _: u64) -> Result<()> {
        anyhow::bail!("no network in fixture")
    }
}
struct FakeSpeech(&'static str);
#[async_trait]
impl missy_voice::Stt for FakeSpeech {
    async fn transcribe(
        &self,
        _: &[u8],
        _: missy_voice::AudioFormat,
        _: CancellationToken,
    ) -> Result<String> {
        Ok(self.0.into())
    }
}
#[async_trait]
impl missy_voice::Tts for FakeSpeech {
    async fn synthesize(&self, _: &str, _: CancellationToken) -> Result<missy_voice::SpeechAudio> {
        Ok(missy_voice::SpeechAudio {
            wav: self.0.as_bytes().to_vec(),
            sample_rate: 48_000,
        })
    }
}
struct FakeAgent;
#[async_trait]
impl missy_discord_voice::DiscordVoiceAgent for FakeAgent {
    async fn respond(
        &self,
        _: &missy_discord_voice::VoiceBinding,
        _: &str,
        _: CancellationToken,
    ) -> Result<String> {
        anyhow::bail!("no provider in fixture")
    }
}
fn voice_config() -> missy_discord_voice::DiscordVoiceConfig {
    serde_json::from_value(json!({
        "native_dave":{"library_path":"/does/not/get/loaded.so","sha256":"0".repeat(64),"version":"1.2.2/cpp","source_commit":"6396c90aedbb9a6ee117ea9ce1ce9832af6a3375"},
        "enabled":true,"bot_user_id":1,"bindings":[{"guild_id":2,"channel_id":3,"user_id":4,"consent":true,"mode":"no-tools","scope":scope("voice")}],
        "endpoint_hosts":["voice.invalid"],"udp_port_min":5000,"udp_port_max":5001,"allow_private_udp":false,"max_utterance_seconds":10,"silence_ms":500,"max_users":1,"max_reply_seconds":10
    })).unwrap()
}
fn service(
    config: missy_discord_voice::DiscordVoiceConfig,
    speech: &'static str,
) -> Arc<missy_discord_voice::DiscordVoiceService> {
    Arc::new(
        missy_discord_voice::DiscordVoiceService::new(
            config,
            Arc::new(missy_policy::PolicyHttpClient::new(Default::default()).unwrap()),
            Arc::new(FakeGateway),
            Arc::new(FakeSpeech(speech)),
            Arc::new(FakeSpeech(speech)),
            Arc::new(FakeAgent),
        )
        .unwrap(),
    )
}

#[test]
fn reproduces_late_voice_accepting_distinct_speech_backend_with_identical_fingerprint() {
    let config = voice_config();
    let a =
        missy_discord_voice::DiscordVoiceSayTools::pending(config.clone(), Arc::new(Gate)).unwrap();
    let b =
        missy_discord_voice::DiscordVoiceSayTools::pending(config.clone(), Arc::new(Gate)).unwrap();
    let old = service(config.clone(), "old-reviewed-backend");
    let changed = service(config, "changed-unreviewed-backend");
    let before = a.configuration_fingerprint();
    a.attach(old).unwrap();
    b.attach(changed).unwrap();
    assert_eq!(before, a.configuration_fingerprint());
    assert_eq!(
        a.configuration_fingerprint(),
        b.configuration_fingerprint(),
        "speech dependency lineage is absent"
    );
}

fn private(path: &std::path::Path, value: &impl serde::Serialize) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, serde_yaml_ng::to_string(value).unwrap()).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
}
fn command(root: &std::path::Path, args: &[&str]) -> std::process::Command {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_missy-rs"));
    cmd.env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", root)
        .arg("--config")
        .arg(root.join("config.yaml"))
        .args(args);
    cmd
}
fn production_config(root: &std::path::Path) -> missy_core::Config {
    let mut cfg = missy_core::Config {
        state_dir: root.join("state"),
        ..Default::default()
    };
    cfg.runtime.max_concurrent_turns = 2;
    cfg.providers.push(missy_core::ProviderConfig {
        model: "never-contacted".into(),
        input_microusd_per_million: Some(0),
        output_microusd_per_million: Some(0),
        ..Default::default()
    });
    cfg.policy
        .allowed_origins
        .push("http://127.0.0.1:11434".into());
    cfg.policy.allow_private_network = true;
    cfg.gateway.scheduler.enabled = true;
    cfg.gateway.scheduler.poll_seconds = 1;
    cfg.gateway.scheduler.scopes.push(scope("scheduler"));
    let automation = root.join("automation.yaml");
    private(&automation, &json!({"enabled":false}));
    cfg.extensions.automation_config = Some(automation);
    let approvals = root.join("approvals.yaml");
    private(
        &approvals,
        &json!({"sensitive_tool_names":[],"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":300}),
    );
    cfg.extensions.approvals_config = Some(approvals);
    cfg
}

#[test]
fn production_binary_rejects_single_slot_orchestration_before_any_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = production_config(dir.path());
    cfg.runtime.max_concurrent_turns = 1;
    private(&dir.path().join("config.yaml"), &cfg);
    let result = command(dir.path(), &["gateway", "start"]).output().unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("native gateway startup failed"));
    let status: serde_json::Value = serde_json::from_slice(
        &std::fs::read(cfg.state_dir.join("run/gateway_status.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(status["phase"], "failed");
    assert_eq!(status["ready"], false);
    assert_eq!(status["subsystems"]["runtime"], "startup_failed");
}

struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn production_reload_fingerprint_detects_semantic_extension_change_at_same_path() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = production_config(dir.path());
    private(&dir.path().join("config.yaml"), &cfg);
    let _child = ChildGuard(
        command(dir.path(), &["gateway", "start"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let start = std::time::Instant::now();
    loop {
        if command(dir.path(), &["gateway", "status"])
            .output()
            .unwrap()
            .status
            .success()
        {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "disposable supervisor never became ready"
        );
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    assert!(
        command(dir.path(), &["gateway", "check-reload"])
            .output()
            .unwrap()
            .status
            .success()
    );
    private(
        cfg.extensions.automation_config.as_deref().unwrap(),
        &json!({"enabled":true}),
    );
    let reload = command(dir.path(), &["gateway", "check-reload"])
        .output()
        .unwrap();
    assert!(!reload.status.success());
    let result: serde_json::Value = serde_json::from_slice(&reload.stdout).unwrap();
    assert_eq!(result["restart_required"], true);
}

#[tokio::test]
async fn regression_probe_planner_cancellation_must_not_leave_detached_running_turn() {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let store = Arc::new(Store::open(dir.path().join("runtime.sqlite")).unwrap());
    let provider = Arc::new(BlockingProvider {
        entered: false.into(),
        cancelled: false.into(),
    });
    let rt = Arc::new(
        Runtime::new(
            store.clone(),
            provider.clone(),
            Arc::new(extensions::NoTools),
            "Planner fixture".into(),
            RuntimeConfig {
                max_cost_microusd: 100,
                max_tokens: 100,
                max_steps: 2,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let pool = Arc::new(
        missy_subagents::AgentPool::new(
            std::collections::BTreeMap::from([("fixture".into(), rt.clone())]),
            Default::default(),
        )
        .unwrap(),
    );
    let tools = Arc::new(
        missy_subagents::DelegationTools::new(
            Arc::new(extensions::NoTools),
            pool,
            Arc::new(
                missy_subagents::AgentRegistry::open(dir.path().join("agents.sqlite")).unwrap(),
            ),
            missy_subagents::DelegationPolicy {
                enabled: true,
                provider: "fixture".into(),
                max_tasks: 2,
                allowed_providers: vec![],
                failure_policy: Default::default(),
            },
        )
        .unwrap(),
    );
    let ctx = ToolContext {
        scope: scope("planner-cancel"),
        turn_id: "parent".into(),
        cancellation: CancellationToken::new(),
    };
    let call = ToolCall {
        id: "delegate".into(),
        name: "delegate_task".into(),
        arguments: json!({"operation_id":"cancel","prompt":"Plan a bounded fixture"}),
    };
    let worker = tokio::spawn({
        let tools = tools.clone();
        let ctx = ctx.clone();
        async move { tools.execute(&ctx, &call).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !provider.entered.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    ctx.cancellation.cancel();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    let hash = hex::encode(sha2::Sha256::digest(
        serde_json::to_vec(&("model-delegation-v2", &ctx.scope, "cancel")).unwrap(),
    ));
    let mut planner_scope = ctx.scope.clone();
    planner_scope.session = format!("planner:{hash}");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let turn = store.list_turns(&planner_scope, 10).unwrap().pop().unwrap();
    let stopped = store
        .cancellation_requested(&planner_scope, &turn.id)
        .unwrap()
        || !matches!(turn.status, missy_store::TurnStatus::Running);
    // Always clean up the detached disposable turn, even when the assertion fails.
    let _ = rt.cancel(&planner_scope, &turn.id);
    rt.shutdown(std::time::Duration::from_secs(2))
        .await
        .unwrap();
    assert!(
        stopped,
        "parent cancelled but planner remained Running without a cancellation request"
    );
}
