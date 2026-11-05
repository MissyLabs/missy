//! Source-derived fixtures: missy/config/settings.py (MissyConfig/ProviderConfig),
//! missy/channels/discord/config.py, examples/config.yaml and discord-configs.yaml.
//! These are sanitized offline values, never the operator's actual config.
use missy_migrate::{MigrationInput, MigrationPlan, apply, plan, verify};
use std::{fs, os::unix::fs::PermissionsExt};

fn fixture(config: &str) -> (tempfile::TempDir, MigrationPlan) {
    let t = tempfile::tempdir().unwrap();
    let path = t.path().join("exported-config.yaml");
    fs::write(&path, config).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let p = plan(MigrationInput {
        config_path: Some(path),
        ..Default::default()
    })
    .unwrap();
    (t, p)
}

// Full current production schema, including provider operational fields that a
// dataclass/writer export includes even when they equal defaults.
const MODERN: &str = r#"
config_version: 8
network:
  default_deny: true
  allowed_cidrs: [192.168.0.0/16]
  allowed_domains: ['*.github.com', api.openai.com]
  allowed_hosts: ['localhost:11434']
  provider_allowed_hosts: [api.anthropic.com]
  tool_allowed_hosts: [github.com]
  discord_allowed_hosts: [discord.com]
  presets: [anthropic]
  rest_policies: [{host: github.com, methods: [GET], paths: ['/repos/*']}]
filesystem:
  allowed_read_paths: [/srv/assistant/workspace]
  allowed_write_paths: [/srv/assistant/workspace/output]
  protected_write_paths: []
shell: {enabled: false, allowed_commands: [git], unrestricted: true, allowed_env_vars: [BUILD_TOKEN]}
plugins: {enabled: false, allowed_plugins: [old-python-plugin]}
providers:
  anthropic:
    name: anthropic
    model: claude-sonnet-4-6
    api_key: '${ANTHROPIC_API_KEY}'
    base_url: null
    timeout: 90
    enabled: true
    api_keys: []
    key_rotation_strategy: failover
    oauth_accounts: []
    fast_model: claude-haiku-4-5
    premium_model: ''
    context_worker_provider: ''
    context_worker_model: ''
    requests_per_minute: 60
    tokens_per_minute: 100000
    max_wait_seconds: 30.0
    circuit_breaker_threshold: 5
    circuit_breaker_cooldown_seconds: 60.0
    weight: 1.0
    account_weights: []
  ollama: {name: ollama, model: llama3.2, base_url: 'http://localhost:11434', timeout: 60}
default_provider: anthropic
workspace_path: '~/workspace'
audit_log_path: '~/.missy/audit.jsonl'
max_iterations: 22
temperature: 0.7
max_sub_agents: 10
max_concurrent_agents: 3
max_sub_agent_depth: 2
max_spend_usd: 2.5
global_max_spend_usd: 50
global_budget_period: daily
landlock_enabled: true
tools: {profile: full, allow: [], deny: [], also_allow: [], by_provider: {}, by_model: {}, groups: {}, disabled_tools: []}
agents: {}
scheduling: {enabled: true, max_jobs: 4, active_hours: '09:00-17:00', misfire_grace_seconds: 300}
nomad: {enabled: false, address: 'https://nomad.example.com', allowed_namespaces: [], require_image_digest: true, max_parallel_jobs: 4}
heartbeat: {enabled: false, interval_seconds: 1800, workspace: '~/workspace', active_hours: ''}
observability: {otel_enabled: false, otel_endpoint: 'http://localhost:4317', otel_protocol: grpc, otel_service_name: missy, log_level: warning, log_file_path: '~/.missy/missy.log'}
vault: {enabled: false, vault_dir: '~/.missy/secrets'}
proactive: {enabled: false, triggers: []}
sleeptime: {enabled: false, provider: ''}
features: {graph_memory_enabled: false, semantic_memory_enabled: false, prompt_patch_proposals_enabled: false, model_routing_enabled: false, condenser_pipeline_enabled: false, condenser_min_messages: 30}
retention: {enabled: true, memory_days: 0, checkpoints_days: 7, captures_days: 0, inbound_attachments_days: 0, request_tracker_days: 30, graph_memory_days: 0, pending_pairing_hours: 24}
sandbox: null
container: null
vision: {enabled: true, preferred_device: '', capture_width: 1920, capture_height: 1080, warmup_frames: 5, max_retries: 3, auto_activate_threshold: 0.8, scene_memory_max_frames: 20, scene_memory_max_sessions: 5}
repoeval_foundry: {enabled: false, api_available: false, base_url: '', project_id: '', allowed_hosts: [], token_file: ''}
tool_intelligence: {candidate_generation_enabled: false, min_pattern_count: 3, allow_shell: false, provider_gating_enabled: false}
obs: {enabled: false, host: 127.0.0.1, port: 4455, password: null, scene_allowlist: [], rate_limit_per_minute: 30}
vtube: {enabled: false, host: 127.0.0.1, port: 8001, auth_token: null, plugin_name: Missy, plugin_developer: MissyLabs, rate_limit_per_minute: 30}
desktop: {enabled: false, app_allowlist: [], unrestricted: false, window_allowlist: [], rate_limit_per_minute: 30, allow_software_install: false, auto_approve_software_install: false}
discord:
  enabled: true
  auto_approve_uploads: false
  max_concurrent_conversations: 3
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      token: null
      account_id: null
      application_id: '123456789012345678'
      dm_policy: disabled
      dm_allowlist: []
      owner_ids: ['223456789012345678']
      ack_reaction: ''
      ignore_bots: true
      allow_bots_if_mention_only: false
      allowed_bot_author_ids: []
      auto_thread_threshold: 0
      rate_limit_per_minute: 10
      guild_policies:
        '323456789012345678':
          enabled: true
          require_mention: true
          allowed_channels: ['423456789012345678']
          allowed_roles: []
          allowed_role_ids: []
          allowed_users: ['523456789012345678']
          mode: full
"#;

#[test]
fn full_source_derived_modern_schema_publishes_usable_denied_candidate_and_review_templates() {
    let (t, p) = fixture(MODERN);
    assert!(p.is_applicable(), "{p:?}");
    let cfg = p.converted_config.as_ref().unwrap();
    cfg.validate().unwrap();
    assert_eq!(cfg.runtime.max_steps, 22);
    assert_eq!(cfg.runtime.max_concurrent_turns, 3);
    assert_eq!(cfg.runtime.global_max_cost_microusd, 2_500_000);
    assert_eq!(cfg.providers.len(), 2);
    assert_eq!(
        cfg.providers[0].api_key_env.as_deref(),
        Some("ANTHROPIC_API_KEY")
    );
    assert!(cfg.policy.allowed_origins.is_empty());
    assert!(!cfg.policy.allow_private_network && !cfg.policy.shell_enabled);
    assert!(cfg.policy.read_roots.is_empty() && cfg.policy.write_roots.is_empty());
    assert_eq!(cfg.discord.allowed_users, ["523456789012345678"]);
    assert!(!cfg.discord.enabled && !cfg.discord.allow_dm);
    assert_eq!(
        serde_json::to_value(&cfg.extensions).unwrap(),
        serde_json::to_value(missy_core::ExtensionConfig::default()).unwrap()
    );
    for section in [
        "network",
        "scheduling",
        "nomad",
        "vision",
        "desktop",
        "delegation_limits",
        "landlock_enabled",
    ] {
        assert!(
            p.config_templates
                .iter()
                .any(|v| v["legacy_section"] == section
                    && v["enabled"] == false
                    && v["reviewed"] == false)
        );
    }
    let dest = t.path().join("native");
    let manifest = apply(&p, &dest, true).unwrap();
    assert!(
        manifest
            .artifacts
            .iter()
            .any(|a| a.path == "config-review.json")
    );
    verify(&dest).unwrap();
    let loaded: missy_core::Config =
        serde_yaml_ng::from_slice(&fs::read(dest.join("config.yaml")).unwrap()).unwrap();
    loaded.validate().unwrap();
    assert_eq!(loaded.runtime.max_steps, 22);
    assert!(!loaded.gateway.scheduler.enabled);
}

#[test]
fn explicit_runtime_prompt_and_native_optins_convert_without_inferred_authority() {
    let (_, p) = fixture(
        r#"
system_prompt: 'Be concise. Never expose credentials.'
max_turn_seconds: 90
tool_timeout_seconds: 12
max_tokens: 9000
max_guard_retries: 4
max_iterations: 8
max_spend_usd: 0
global_max_spend_usd: 0.000001
global_budget_period: total
network: {default_deny: true, allowed_origins: ['https://api.example.test:443']}
shell: {enabled: true, exact_argv: [[/usr/bin/printf, hello]]}
"#,
    );
    assert!(p.is_applicable(), "{p:?}");
    let cfg = p.converted_config.unwrap();
    assert_eq!(cfg.system_prompt, "Be concise. Never expose credentials.");
    assert_eq!(cfg.runtime.max_turn_seconds, 90);
    assert_eq!(cfg.runtime.tool_timeout_seconds, 12);
    assert_eq!(cfg.runtime.max_tokens, 9000);
    assert_eq!(cfg.runtime.max_guard_retries, 4);
    assert_eq!(cfg.runtime.global_max_cost_microusd, 1);
    assert_eq!(cfg.policy.allowed_origins, ["https://api.example.test:443"]);
    assert_eq!(
        cfg.policy.allowed_commands,
        [vec!["/usr/bin/printf", "hello"]]
    );
}

#[test]
fn unsafe_restriction_drops_are_blocking_but_optional_disabled_settings_are_not() {
    for config in [
        "network: {default_deny: false}",
        "network: {allowed_origins: ['https://example.test'], rest_policies: [{host: example.test, methods: [GET]}]}",
        "shell: {enabled: true, unrestricted: true}",
        "shell: {enabled: true, allowed_commands: [git]}",
        "filesystem: {allowed_write_paths: [/srv], protected_write_paths: [/srv/secrets]}",
        "tools: {deny: [shell_exec]}",
        "agents: {helper: {tools: {profile: minimal}}}",
        "max_spend_usd: 0.0000001",
        "max_iterations: 0",
    ] {
        let (_, p) = fixture(config);
        assert!(!p.is_applicable(), "unexpectedly allowed {config}: {p:?}");
    }
    let (_, p) = fixture(
        r#"
shell: {enabled: false, unrestricted: true, allowed_commands: [bash], allowed_env_vars: [TOKEN]}
plugins: {enabled: true, allowed_plugins: [legacy-plugin]}
providers: {unused: {enabled: false, name: openai-codex, model: '', api_key: INACTIVE_TOKEN_CANARY}}
discord: {enabled: false, accounts: [{token: INACTIVE_TOKEN_CANARY}]}
desktop: {enabled: true, unrestricted: true, auto_approve_software_install: true}
"#,
    );
    assert!(p.is_applicable(), "{p:?}");
    let report = serde_json::to_string(&p).unwrap();
    assert!(!report.contains("INACTIVE_TOKEN_CANARY") && !report.contains("legacy-plugin"));
    assert!(!p.converted_config.unwrap().policy.shell_enabled);
}

#[test]
fn unknown_and_secret_values_never_appear_in_report_or_emitted_candidate() {
    let (_, p) = fixture(
        r#"
unknown_secret_key: UNKNOWN_SECRET_CANARY
obs: {password: OBS_SECRET_CANARY, enabled: false}
providers: {openai: {model: model, api_key: RAW_SECRET_CANARY}}
"#,
    );
    assert!(!p.is_applicable());
    let rendered = format!("{p:?} {}", serde_json::to_string(&p).unwrap());
    for secret in [
        "unknown_secret_key",
        "UNKNOWN_SECRET_CANARY",
        "OBS_SECRET_CANARY",
        "RAW_SECRET_CANARY",
    ] {
        assert!(!rendered.contains(secret));
    }
    assert!(p.issues.iter().any(|i| i.code == "unsupported_key"));
    let (_, p) = fixture(
        "obs: {password: DUPLICATED_SECRET_CANARY}\nsystem_prompt: DUPLICATED_SECRET_CANARY\n",
    );
    assert!(!p.is_applicable());
    assert!(p.converted_config.is_none());
    assert!(!format!("{p:?}").contains("DUPLICATED_SECRET_CANARY"));
}

#[test]
fn discord_roles_pairing_names_multi_guild_and_multi_account_never_form_flat_grants() {
    for config in [
        MODERN.replace("allowed_roles: []", "allowed_roles: [administrator]"),
        MODERN.replace("mode: full", "mode: no_tools"),
        MODERN.replace("'423456789012345678'", "general"),
        MODERN.replace("allowed_users: ['523456789012345678']", "allowed_users: []"),
    ] {
        let (_, p) = fixture(&config);
        assert!(p.is_applicable(), "{p:?}");
        let d = &p.converted_config.unwrap().discord;
        assert!(!d.enabled && !d.allow_dm);
        assert!(d.allowed_guilds.is_empty() && d.allowed_users.is_empty());
    }
    let (_, p) =
        fixture("discord: {enabled: true, accounts: [{token_env_var: A}, {token_env_var: B}]}\n");
    assert!(p.is_applicable());
    assert!(p.converted_config.unwrap().discord.allowed_users.is_empty());
}

#[test]
fn unsupported_selected_provider_disables_pool_instead_of_substituting_an_account() {
    let (_, p) = fixture(
        r#"
providers:
  local: {name: ollama, model: llama3.2}
  codex: {name: openai-codex, model: codex, oauth_accounts: [work]}
default_provider: codex
"#,
    );
    assert!(p.is_applicable(), "{p:?}");
    assert!(p.converted_config.unwrap().providers.is_empty());
    assert!(
        p.config_templates
            .iter()
            .any(|v| v["legacy_section"] == "default_provider")
    );
}

#[test]
fn actual_repository_example_converts_with_denied_grants_not_archive_only_refusal() {
    let (_, p) = fixture(include_str!("../../../examples/config.yaml"));
    assert!(p.is_applicable(), "{p:?}");
    let cfg = p.converted_config.unwrap();
    assert_eq!(cfg.providers.len(), 3);
    assert!(cfg.policy.allowed_origins.is_empty());
    assert!(cfg.policy.write_roots.is_empty());
    assert_eq!(cfg.policy.read_roots, [std::path::PathBuf::from("/tmp")]);
    assert!(!cfg.policy.shell_enabled && !cfg.discord.enabled);
}

#[test]
fn implicit_unsupported_primary_never_falls_back_to_a_different_account() {
    let (_, p) = fixture(
        "providers:\n  codex: {name: openai-codex, model: codex}\n  local: {name: ollama, model: llama3.2}\n",
    );
    assert!(p.is_applicable(), "{p:?}");
    assert!(p.converted_config.unwrap().providers.is_empty());
}
