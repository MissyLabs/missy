# Configuration schema reference

Annotated `~/.missy/config.yaml` (moved verbatim from `CLAUDE.md`).

```yaml
config_version: 2                    # schema version (auto-migrated on startup)

network:
  default_deny: true
  presets:                           # auto-expand to hosts/domains/CIDRs
    - anthropic
    - github
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts: []               # host:port pairs
  provider_allowed_hosts: []      # per-category overrides
  tool_allowed_hosts: []
  discord_allowed_hosts: []
  rest_policies:                  # L7 HTTP method + path controls
    - host: "api.github.com"
      method: "GET"
      path: "/repos/**"
      action: "allow"
    - host: "api.github.com"
      method: "DELETE"
      path: "/**"
      action: "deny"

filesystem:
  allowed_write_paths: []
  allowed_read_paths: []

shell:
  enabled: false
  allowed_commands: []
  unrestricted: false          # true = skip allow-list matching entirely (still needs enabled: true)

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
    api_key: null                  # or set ANTHROPIC_API_KEY env var
    api_keys: []                   # multiple keys for rotation/balancing
    key_rotation_strategy: failover  # "failover" | "round_robin"
    fast_model: ""                 # e.g. claude-haiku-4-5
    premium_model: ""              # e.g. claude-opus-4-6
    enabled: true
  openai:
    name: openai
    model: "gpt-5.5"
    api_keys: ["sk-account-1-...", "sk-account-2-..."]  # two OpenAI accounts
    key_rotation_strategy: round_robin  # balance every call across both
  openai-codex:
    name: openai-codex
    model: "gpt-5.2"
    oauth_accounts: ["work", "personal"]  # 2+ named `missy providers auth openai-codex --method oauth --account <name>` logins
    key_rotation_strategy: round_robin    # balance every call across both signed-in ChatGPT accounts

scheduling:
  enabled: true
  max_jobs: 0                      # 0 = unlimited
  active_hours: ""                 # e.g. "08:00-22:00"; applies to jobs without their own window
  misfire_grace_seconds: 300       # a run missed by up to this long (e.g. restart) still fires once

features:                          # opt-in agent features, applied to every AgentConfig
  graph_memory_enabled: false
  semantic_memory_enabled: false
  prompt_patch_proposals_enabled: false
  model_routing_enabled: false
  condenser_pipeline_enabled: false
  condenser_min_messages: 30

retention:                         # daily gateway maintenance job; 0 disables a pruner
  enabled: true
  # Pruners that delete operator content are opt-in (default 0); the gateway
  # logs a preview of what an enabled file pruner will delete at startup.
  memory_days: 0                   # conversation turns (pinned kept) + orphaned summaries
  captures_days: 0                 # ~/.missy/captures (vision)
  inbound_attachments_days: 0      # Discord inbound attachments / extracted zips
  graph_memory_days: 0
  # Internal bookkeeping, pruned by default:
  checkpoints_days: 7
  request_tracker_days: 30
  pending_pairing_hours: 24        # unapproved voice pairing requests

global_max_spend_usd: 0.0           # process-wide cap across sessions (0 = off); installed by every entry point
global_budget_period: total         # total | daily | monthly

heartbeat:
  enabled: false
  interval_seconds: 1800
  workspace: "~/workspace"
  active_hours: ""

observability:
  otel_enabled: false
  otel_endpoint: "http://localhost:4317"
  otel_protocol: "grpc"           # "grpc" | "http/protobuf"
  otel_service_name: "missy"
  log_level: "warning"

vault:
  enabled: false
  vault_dir: "~/.missy/secrets"

# Voice channel (read from raw YAML, not a dataclass)
voice:
  host: "127.0.0.1"
  port: 8765
  stt:
    engine: "faster-whisper"
    model: "base.en"
  tts:
    engine: "piper"
    voice: "en_US-lessac-medium"

discord:
  # See missy/channels/discord/config.py for full schema

# Web TUI / REST API — auto-started by `missy gateway start` (read from raw
# YAML, not a dataclass). api_key defaults to the persisted key at
# ~/.missy/secrets/web_console.key when unset. Browser login sessions are
# persisted to ~/.missy/secrets/web_sessions.json (0600) so a gateway
# restart doesn't log the operator out; session_ttl_seconds is a sliding
# window refreshed on every authenticated request (default 30 days).
api:
  enabled: true
  host: "127.0.0.1"
  port: 8080
  api_key: ""
  session_ttl_seconds: 2592000

container:
  enabled: false
  image: "python:3.12-slim"
  memory_limit: "256m"
  cpu_quota: 0.5
  network_mode: "none"              # no network in sandbox by default

vision:
  enabled: true
  preferred_device: ""               # auto-detect if empty
  capture_width: 1920
  capture_height: 1080
  warmup_frames: 5
  max_retries: 3
  auto_activate_threshold: 0.80     # audio intent confidence for auto-activation
  scene_memory_max_frames: 20
  scene_memory_max_sessions: 5

workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
max_spend_usd: 0.0                  # per-session budget cap; 0 = unlimited
```
