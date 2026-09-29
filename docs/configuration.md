# Configuration Reference

Missy is configured through a single YAML file, by default located at
`~/.missy/config.yaml`.  The path can be overridden with the `--config` CLI
flag or the `MISSY_CONFIG` environment variable.

Run `missy init` to create a default configuration file with a
secure-by-default posture.

All configuration is loaded by `missy.config.settings.load_config()` and
parsed into a `MissyConfig` dataclass hierarchy.

---

## Table of Contents

1. [config_version](#config_version)
2. [network](#network)
3. [filesystem](#filesystem)
4. [shell](#shell)
5. [plugins](#plugins)
6. [tools](#tools)
7. [repoeval_foundry](#repoeval_foundry)
8. [scheduling](#scheduling)
9. [nomad](#nomad)
10. [providers](#providers)
11. [discord](#discord)
12. [heartbeat](#heartbeat)
13. [observability](#observability)
14. [vault](#vault)
15. [voice](#voice)
16. [container](#container)
17. [vision](#vision)
18. [workspace_path](#workspace_path)
19. [audit_log_path](#audit_log_path)
20. [max_spend_usd](#max_spend_usd)
21. [Full Annotated Example](#full-annotated-example)

---

## `config_version`

| Key | Type | Default |
|---|---|---|
| `config_version` | int | `2` |

Schema version for the configuration file.  Old configs (version 1 or unversioned) are
automatically migrated on startup.  The migration detects manually listed hosts that match
built-in presets and replaces them with `presets: [...]`.  A backup is created before
modification.

---

## `network`

Controls all outbound network access.  When `default_deny` is `true` (the
default), every outbound request is blocked unless the destination matches an
entry in one of the allowlists.

| Key | Type | Default | Description |
|---|---|---|---|
| `default_deny` | bool | `true` | When `true`, block all outbound traffic not explicitly allowed. Set to `false` to allow all traffic (not recommended). |
| `allowed_cidrs` | list of strings | `[]` | CIDR blocks that are reachable (e.g. `"10.0.0.0/8"`, `"192.168.1.0/24"`). |
| `allowed_domains` | list of strings | `[]` | Fully-qualified domain names or suffix patterns (e.g. `"*.github.com"`) that are reachable. |
| `allowed_hosts` | list of strings | `[]` | Explicit hostname or `host:port` strings that are reachable (e.g. `"api.anthropic.com"`, `"localhost:11434"`). |
| `provider_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for provider API traffic. Merged (union) with `allowed_hosts`. |
| `tool_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for tool HTTP requests. Merged (union) with `allowed_hosts`. |
| `discord_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for Discord API traffic. Merged (union) with `allowed_hosts`. |
| `presets` | list of strings | `[]` | Named preset bundles that auto-expand to correct hosts/domains/CIDRs. Available presets: `anthropic`, `github`, `openai`, `ollama`, `discord`. Use `missy presets list` to see all. |
| `rest_policies` | list of objects | `[]` | L7 HTTP method + path glob rules per host. Each entry has `host`, `method` (GET/POST/PUT/DELETE/PATCH/*), `path` (glob pattern like `/repos/**`), and `action` (allow/deny). |

The per-category `*_allowed_hosts` lists are unioned with the global
`allowed_hosts` and `allowed_domains` lists during policy evaluation.  They
exist so that operators can grant narrow access to specific subsystems without
opening the allowlist globally.

---

## `filesystem`

Controls which directories the agent may read from and write to.  Paths
support tilde (`~`) expansion.

| Key | Type | Default | Description |
|---|---|---|---|
| `allowed_read_paths` | list of strings | `[]` | Directories the agent may read from.  Access to any path outside these trees raises `PolicyViolationError`. |
| `allowed_write_paths` | list of strings | `[]` | Directories the agent may write to.  Access to any path outside these trees raises `PolicyViolationError`. |

Paths are matched by prefix after resolution.  For example, if
`allowed_read_paths` contains `"~/workspace"`, the agent may read any file
under `~/workspace/` and its subdirectories.

---

## `shell`

Controls shell command execution.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for shell execution.  When `false`, all shell commands are denied regardless of `allowed_commands`. |
| `allowed_commands` | list of strings | `[]` | Whitelist of command names (not full paths) that may be executed when `enabled` is `true`.  An empty list means no commands are allowed even when the shell is enabled. |
| `allowed_env_vars` | list of strings | `[]` | Extra environment-variable names to inherit into `shell_exec` children. Credential variables remain stripped unless explicitly listed; grant only variables required by a trusted child process. |

Example:

```yaml
shell:
  enabled: true
  allowed_commands:
    - "git"
    - "python3"
    - "ls"
  allowed_env_vars: []
```

---

## `plugins`

Controls the plugin loading system.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for the plugin system.  When `false`, no plugins may be loaded. |
| `allowed_plugins` | list of strings | `[]` | Whitelist of plugin name strings that may be loaded when `enabled` is `true`.  A plugin whose name does not appear in this list is denied with a `PolicyViolationError`. |

To disable all third-party plugins globally, set `enabled: false` (which is
the default).  Even when `enabled: true`, only plugins explicitly named in
`allowed_plugins` may load.

---

## `tools`

Controls the tools exposed to the model before runtime execution policy checks.
Execution still fails closed in the tool registry; this section only shapes
which tool schemas are visible on a turn.

| Key | Type | Default | Description |
|---|---|---|---|
| `profile` | string | `full` | One of `minimal`, `coding`, `messaging`, or `full`. |
| `allow` | list of strings | `[]` | Tool names, glob patterns, or `group:<name>` entries to keep visible. |
| `deny` | list of strings | `[]` | Tool names or glob patterns to remove. |
| `alsoAllow` | list of strings | `[]` | Adds tools back after a restrictive layer. |
| `byProvider` | mapping | `{}` | Provider-specific `allow`/`deny`/`alsoAllow`, with optional nested `byModel`. |
| `groups` | mapping | `{}` | Custom tool groups used by `group:<name>` references. |

Per-agent overrides live under `agents.<id>.tools`; subagent-specific filters
can be declared under `agents.<id>.subagents.tools`. Layers are applied as
profile, provider, global, agent, group, sandbox, then subagent.

```yaml
tools:
  profile: coding
  deny: ["browser_*"]
  byProvider:
    anthropic:
      deny: ["vision_*"]
      byModel:
        claude-haiku-*:
          allow: ["calculator", "file_read"]
  groups:
    project: ["file_read", "file_write"]

agents:
  default:
    tools:
      allow: ["group:project", "-shell_exec"]
    subagents:
      tools:
        deny: ["sessions_*"]
```

## `tool_intelligence`

Controls repeated-request tracking, candidate synthesis, provider benchmark
gating, and candidate runtime loading. All mutating/execution-affecting
features default to disabled.

| Key | Type | Default | Description |
|---|---|---|---|
| `candidate_generation.enabled` | bool | `false` | Propose tool candidates from frequent request patterns. Candidates start as `proposed` and are not executable. |
| `candidate_generation.min_pattern_count` | int | `3` | Minimum repeated requests before a pattern can be proposed as a candidate. |
| `candidate_generation.allow_shell` | bool | `false` | Allow generated candidates to request shell permission. Approval and execution policy still apply. |
| `candidate_generation.check_every_n_requests` | int | `5` | Throttle for pattern scans. |
| `provider_gating.enabled` | bool | `false` | Hide weak tool/provider pairings from a turn based on benchmark evidence. |
| `provider_gating.min_samples` | int | `3` | Minimum benchmark runs before provider gating trusts a score. |
| `provider_gating.min_composite` | float | `0.4` | Composite score below which a provider is considered weak for a tool. |
| `candidate_runtime.enabled` | bool | `false` | Register enabled candidates at runtime only when they have valid provenance, schema, provider flags, permissions, and explicit implementation metadata. |

```yaml
tool_intelligence:
  candidate_generation:
    enabled: true
    min_pattern_count: 5
  provider_gating:
    enabled: true
    min_samples: 3
    min_composite: 0.55
  candidate_runtime:
    enabled: true
```

---

## `repoeval_foundry`

Optional project-pinned RepoEval Foundry HTTP **client**. **Disabled by default.**
`MissyLabs/missy` is the first repository being evaluated; Foundry core, server,
workload catalog and deployment belong in `MissyLabs/repoeval-foundry`, not here.
This change only provides opt-in client code and does not deploy or enable a service.
`api_available` is an explicit operator assertion that a compatible authenticated
API is available; it is not a health check and does not establish that a service
exists. No real API availability or deployment is claimed here.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Explicit operator opt-in. |
| `api_available` | bool | `false` | Explicit acknowledgement that a real compatible API is deployed; not an automatic health check. |
| `base_url` | string | empty | Exact operator-provided API base URL; never supplied by Missy. No embedded credentials, query, fragment, or redirects. Plain HTTP is allowed only on loopback. |
| `project_id` | string | empty | Required project ID, pinned in every tool request. |
| `allowed_hosts` | list of strings | `[]` | Exact hostname allowlist for the endpoint, not a global network-policy grant. |
| `token_file` | string | empty | Normalized absolute path to the bearer-token file. Required when both switches are `true`; the token itself never belongs in YAML or tool arguments. |

`enabled` and `api_available` accept actual YAML booleans only. The endpoint
must be an exact root or `/api` URL (optionally with the trailing slash), use a
valid nonzero port, and have a syntactically valid hostname. Plain HTTP is
loopback-only. The project ID is limited to letters, digits, `.`, `_`, and `-`.
The token path must be absolute and normalized: no `~`, `.`, `..`, repeated
slashes, whitespace/control characters, or backslashes. Config parsing checks syntax only;
the registration security helper separately verifies that the token path is
safe and usable.

That helper requires a current-UID regular file, mode `0600`, one hard link,
no symlink at the file or any path component, and a size from 1 through 4096 bytes.
Registration checks metadata without reading bytes. At request time the file is
reopened securely and revalidated; contents must match the ASCII bearer-token
grammar `[A-Za-z0-9._~+/-]+=*`, optionally followed by one LF. The token is loaded per
request and is never put in YAML or a raw tool argument. Keep the credential
separate from configuration and restrict access to its parent directories.

Network policy must independently allow the exact endpoint through
`network.tool_allowed_hosts` (including its port where specified); the
Foundry-specific `allowed_hosts` does not grant network access. Requests use
the `tool` policy category and the policy-aware HTTP client, which accepts
Authorization headers and does not follow redirects by default. Do not relax
global policy to enable this integration. Registration exposes two separately
grantable tool-policy identities: `repoeval_foundry_read` (list, plan, status,
compare, artifacts, draft report) and `repoeval_foundry_mutate` (snapshot,
start, cancel). Merely enabling the read tool cannot call mutation actions.
`writes_state` metadata does not enforce authorization. The `capabilities`
action is not offered until Foundry implements it. Configuration does
not create or deploy an API, prove availability, or authorize production work.

```yaml
# Defaults only. Enabling requires an operator-reviewed service, protected token,
# and an independent network.tool_allowed_hosts grant for the exact endpoint.
repoeval_foundry:
  enabled: false
  api_available: false
  base_url: ""
  project_id: ""
  allowed_hosts: []
  token_file: ""
```

---

## `scheduling`

Controls the job scheduler subsystem.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch for scheduled job execution.  When `false`, no new jobs may be added or run. |
| `max_jobs` | int | `0` | Maximum number of concurrent scheduled jobs.  `0` means unlimited. |

---

## `nomad`

Controls secure Nomad workload orchestration. It is disabled by default and
requires a file-backed TLS/ACL bundle plus explicit owner-authorized scope.
Secrets must not appear in YAML.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. |
| `address` | string | `https://nomad.example.com` | HTTPS Nomad endpoint; embedded credentials are rejected. |
| `bundle_dir` | string | `~/.missy/nomad` | Private directory containing CA, client certificate/key, and token files. |
| `state_dir` | string | `~/.missy/nomad-state` | Private atomic operation journal. |
| `binary` | string | `nomad` | Fixed Nomad CLI executable. |
| `identity_cn` | string | `missy` | Required client-certificate common name. |
| `owner` | string | `missy-bot` | Ownership label applied to managed jobs. |
| `allowed_namespaces` | list | `[]` | Owner-authorized mutation scope; empty denies mutation. |
| `allowed_node_pools` | list | `[]` | Owner-authorized pools; empty denies mutation. |
| `allowed_datacenters` | list | `[]` | Owner-authorized datacenters; empty denies mutation. |
| `default_namespace`, `default_node_pool`, `default_datacenter` | string | `""` | Optional defaults, each of which must also be allowlisted. |
| `approved_registries` | list | `[]` | Permitted container registries. |
| `allowed_job_commands` | list | `[]` | Explicit container commands permitted in reviewed job requests; empty permits only image entrypoints. |
| `approved_artifact_prefixes` | list | `[]` | Permitted durable input/output identifier prefixes. |
| `approved_secret_reference_prefixes` | list | `[]` | Permitted `nomad-var://` runtime secret paths. |
| `workload_templates` | mapping | `{}` | Operator-reviewed batch definitions eligible for offload and benchmarks. |
| `protected_node_names` | list | `["Gato"]` | Nodes avoided when another eligible node fits. |
| `require_image_digest` | bool | `true` | Require `@sha256:` image pins. |
| `max_cpu_mhz`, `max_memory_mb`, `max_disk_mb`, `max_group_count` | int | bounded | Per-job resource and allocation-count ceilings. |
| `max_parallel_jobs` | int | `4` | Global tracked batch concurrency ceiling. |
| `max_retry_attempts` | int | `2` | Maximum explicit batch retry attempts. |
| `max_benchmark_runs` | int | `32` | Warm-up plus measured run ceiling. |
| `max_benchmark_parallelism` | int | `4` | Benchmark fan-out ceiling; cannot exceed `max_parallel_jobs`. |
| `max_wall_time_seconds` | int | `86400` | Batch wall-time ceiling. |
| `plan_ttl_seconds` | int | `900` | Exact-plan validity window. |
| `request_timeout_seconds` | int | `60` | CLI request timeout. |
| `allow_purge` | bool | `false` | First of two required opt-ins for deleting owned job history. |

See [Nomad Workload Orchestration](nomad.md) for credential modes, workload
template format, policy prerequisites, tool workflows, and result protocol.

---

## `providers`

A mapping of provider names to their configuration.  Each key is a logical
name (e.g. `anthropic`, `openai`, `ollama`) and the value is a configuration
block.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | (same as the mapping key) | Canonical provider name.  Must match one of the built-in provider implementations: `anthropic`, `openai`, or `ollama`. |
| `model` | string | (required) | Model identifier to use for inference (e.g. `"claude-opus-5-5"`, `"gpt-5.6-sol"`, `"llama3.2"`). |
| `api_key` | string | `null` | API key for the provider.  **Not recommended** -- use environment variables instead.  When `null`, the provider reads `<PROVIDER_NAME>_API_KEY` from the environment (e.g. `ANTHROPIC_API_KEY`). |
| `base_url` | string | `null` | Override the provider's default API endpoint.  Required for Ollama (`"http://localhost:11434"`).  Also useful for OpenAI-compatible third-party services. |
| `timeout` | int | `30` | Request timeout in seconds. |
| `enabled` | bool | `true` | When `false`, the provider is loaded but treated as unavailable by the registry. |
| `api_keys` | list of strings | `[]` | Multiple API keys for round-robin rotation.  Takes precedence over single `api_key` when non-empty. |
| `fast_model` | string | `""` | Model to use for quick/simple tasks (e.g. `"claude-haiku-4-5"`).  Empty string disables tier routing for fast tasks. |
| `premium_model` | string | `""` | Model to use for complex reasoning tasks (e.g. `"claude-opus-4-6"`).  Empty string disables tier routing for premium tasks. |
| `context_worker_provider` | string | `""` | Provider registry key used by `context_shunt` to analyze stored large outputs. Empty keeps data on the parent provider. A different value explicitly authorizes cross-provider egress. |
| `context_worker_model` | string | `""` | Optional model override for `context_shunt`. Empty uses the worker provider's `fast_model`, then its primary `model`. |

### API Key Resolution Order

For each provider, the API key is resolved in this order:

1. `api_key` field in the YAML config (not recommended).
2. Environment variable `<PROVIDER_NAME>_API_KEY`, where `<PROVIDER_NAME>` is
   the uppercased mapping key (e.g. `ANTHROPIC_API_KEY` for a provider keyed
   as `anthropic`).
3. If neither is set, the provider reports itself as unavailable.

---

## `discord`

Optional Discord bot integration.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for the Discord channel. |
| `accounts` | list | `[]` | List of Discord bot account configurations. |

### Account Configuration

Each entry in `accounts` supports:

| Key | Type | Default | Description |
|---|---|---|---|
| `token_env_var` | string | `"DISCORD_BOT_TOKEN"` | Name of the environment variable holding the bot token.  Tokens are never stored in the config file. |
| `account_id` | string | `null` | Explicit bot user ID.  When omitted, fetched from Discord on startup. |
| `application_id` | string | `""` | Discord application ID, required for slash command registration. |
| `dm_policy` | string | `"disabled"` | How the bot handles direct messages.  One of: `"pairing"`, `"allowlist"`, `"open"`, `"disabled"`. |
| `dm_allowlist` | list of strings | `[]` | User IDs permitted for DM when `dm_policy` is `"allowlist"`. |
| `ack_reaction` | string | `""` | Emoji name used to acknowledge receipt (e.g. `"eyes"`).  Empty string disables. |
| `ignore_bots` | bool | `true` | Ignore messages from other bots. |
| `allow_bots_if_mention_only` | bool | `false` | When `true` and `ignore_bots` is `true`, bot messages that @-mention this bot are not ignored. |
| `guild_policies` | mapping | `{}` | Per-guild access control policies, keyed by guild ID string. |

### Guild Policy

Each entry in `guild_policies` supports:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | When `false`, the bot ignores all messages from this guild. |
| `require_mention` | bool | `false` | When `true`, the bot only responds if explicitly @-mentioned. |
| `allowed_channels` | list of strings | `[]` | Whitelist of channel names the bot responds in.  Empty means all channels. |
| `allowed_roles` | list of strings | `[]` | Whitelist of role names users must hold.  Empty means all roles. |
| `allowed_users` | list of strings | `[]` | Whitelist of user IDs permitted to interact.  Empty means all users. |
| `mode` | string | `"full"` | Feature mode: `"safe_chat_only"`, `"no_tools"`, or `"full"`. |

---

## `heartbeat`

Periodic workspace monitoring during active hours.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for heartbeat monitoring. |
| `interval_seconds` | int | `1800` | Seconds between heartbeat checks. |
| `workspace` | string | `"~/workspace"` | Directory to monitor. |
| `active_hours` | string | `""` | Time window for heartbeat activity (e.g. `"08:00-22:00"`).  Empty means always active. |

---

## `observability`

OpenTelemetry integration for traces and metrics.

| Key | Type | Default | Description |
|---|---|---|---|
| `otel_enabled` | bool | `false` | Enable OpenTelemetry export.  Requires `pip install -e ".[otel]"`. |
| `otel_endpoint` | string | `"http://localhost:4317"` | OTLP collector endpoint. |
| `otel_protocol` | string | `"grpc"` | Transport protocol: `"grpc"` or `"http/protobuf"`. |
| `otel_service_name` | string | `"missy"` | Service name in traces and metrics. |
| `log_file_path` | string | `"~/.missy/missy.log"` | Rotating application log file for Python logs and provider diagnostics. |
| `log_level` | string | `"warning"` | Python logging level for the application. |

---

## `vault`

Encrypted secrets store using ChaCha20-Poly1305.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable the encrypted vault.  When enabled, `vault://KEY_NAME` references in config are resolved from the vault. |
| `vault_dir` | string | `"~/.missy/secrets"` | Directory for vault key and encrypted data files. |

Manage secrets with `missy vault set|get|list|delete`.

---

## `voice`

Voice channel configuration for edge node communication.  Requires `pip install -e ".[voice]"`.

```yaml
voice:
  host: "127.0.0.1"
  port: 8765
  stt:
    engine: "faster-whisper"
    model: "base.en"
  tts:
    engine: "piper"
    voice: "en_US-lessac-medium"
  # Remote binds require both TLS files, unless the explicit
  # allow_insecure_remote escape hatch is set.
  tls_certfile: ""
  tls_keyfile: ""
```

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"127.0.0.1"` | WebSocket server bind address. |
| `port` | int | `8765` | WebSocket server port. |
| `tls_certfile` | string | `""` | TLS certificate for secure remote (`wss://`) binds. |
| `tls_keyfile` | string | `""` | TLS private key; must be set with `tls_certfile`. |
| `allow_insecure_remote` | bool | `false` | Explicit escape hatch for plaintext non-loopback binds. |
| `stt.engine` | string | `"faster-whisper"` | Speech-to-text engine. |
| `stt.model` | string | `"base.en"` | STT model name (faster-whisper model). |
| `tts.engine` | string | `"piper"` | Text-to-speech engine (external binary). |
| `tts.voice` | string | `"en_US-lessac-medium"` | Piper voice model identifier. |

---

## `container`

Optional Docker-based sandbox for tool execution.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable container sandbox.  Requires Docker. |
| `image` | string | `"python:3.12-slim"` | Docker image for sandbox containers. |
| `memory_limit` | string | `"256m"` | Container memory limit. |
| `cpu_quota` | float | `0.5` | CPU quota (fraction of one core). |
| `network_mode` | string | `"none"` | Docker network mode.  `"none"` disables all networking in the sandbox. |

---

## `vision`

On-demand visual capabilities.  Requires `pip install -e ".[vision]"`.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable vision subsystem. |
| `preferred_device` | string | `""` | Preferred camera device path (e.g. `/dev/video0`).  Empty auto-detects. |
| `capture_width` | int | `1920` | Capture resolution width. |
| `capture_height` | int | `1080` | Capture resolution height. |
| `warmup_frames` | int | `5` | Frames to discard on camera warm-up. |
| `max_retries` | int | `3` | Capture retry count on failure. |
| `auto_activate_threshold` | float | `0.80` | Audio intent confidence threshold for automatic vision activation. |
| `scene_memory_max_frames` | int | `20` | Maximum frames stored per scene session. |
| `scene_memory_max_sessions` | int | `5` | Maximum concurrent scene sessions. |

---

## `workspace_path`

| Key | Type | Default |
|---|---|---|
| `workspace_path` | string | `"~/workspace"` |

The agent's working directory.  Tilde expansion is supported.  Created by
`missy init`.

---

## `audit_log_path`

| Key | Type | Default |
|---|---|---|
| `audit_log_path` | string | `"~/.missy/audit.jsonl"` |

Path to the JSONL audit log file.  Tilde expansion is supported.  The parent
directory is created automatically.

---

## `max_spend_usd`

| Key | Type | Default |
|---|---|---|
| `max_spend_usd` | float | `0.0` |

Per-session budget cap in USD.  When set to a positive value, the agent loop
will raise `BudgetExceededError` after any provider call that causes the
accumulated session cost to exceed this limit.  An `agent.budget.exceeded`
audit event is emitted before the error propagates.

Set to `0` (the default) for unlimited spending.

```yaml
max_spend_usd: 5.00    # halt after $5 of API calls per session
```

View the current budget with `missy cost`.

---

## Full Annotated Example

The following YAML shows every configuration option with comments.

```yaml
# ---------------------------------------------------------------------------
# Network policy
# Controls all outbound network access.
# ---------------------------------------------------------------------------
network:
  default_deny: true                      # Block all traffic not explicitly allowed

  allowed_cidrs:                          # Reachable IP ranges (CIDR notation)
    - "10.0.0.0/8"
    - "172.16.0.0/12"
    - "192.168.0.0/16"

  allowed_domains:                        # Domain suffix matching
    - "*.github.com"

  allowed_hosts:                          # Explicit host or host:port pairs
    - "api.anthropic.com"
    - "api.openai.com"
    - "localhost:11434"                   # Local Ollama instance

  provider_allowed_hosts: []              # Extra hosts for provider traffic only
  tool_allowed_hosts: []                  # Extra hosts for tool traffic only
  discord_allowed_hosts:                  # Extra hosts for Discord traffic only
    - "discord.com"
    - "gateway.discord.gg"

# ---------------------------------------------------------------------------
# Filesystem policy
# Restricts which directories the agent may read or write.
# ---------------------------------------------------------------------------
filesystem:
  allowed_read_paths:
    - "~/workspace"
    - "/tmp"

  allowed_write_paths:
    - "~/workspace"

# ---------------------------------------------------------------------------
# Shell policy
# Disabled by default. Enable only specific commands you trust.
# ---------------------------------------------------------------------------
shell:
  enabled: false
  allowed_commands: []                    # e.g. ["git", "python3", "ls"]
  allowed_env_vars: []                    # extra names inherited by shell_exec

# ---------------------------------------------------------------------------
# Plugin policy
# Disabled by default. Enable only trusted plugins by name.
# ---------------------------------------------------------------------------
plugins:
  enabled: false
  allowed_plugins: []                     # e.g. ["my_trusted_plugin"]

# ---------------------------------------------------------------------------
# Scheduling policy
# Controls the APScheduler-backed job scheduler.
# ---------------------------------------------------------------------------
scheduling:
  enabled: true
  max_jobs: 0                             # 0 = unlimited

# ---------------------------------------------------------------------------
# Background memory processing
# Off by default. With no provider, summaries are deterministic and local.
# Set MISSY_DISABLE_SLEEPTIME=1 as an emergency process-level kill switch.
# ---------------------------------------------------------------------------
sleeptime:
  enabled: false
  provider: ""                            # Explicit registry key, e.g. ollama

# ---------------------------------------------------------------------------
# Provider configuration
# API keys must be set as environment variables, not in this file.
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
# ---------------------------------------------------------------------------
providers:
  anthropic:
    name: anthropic
    model: "claude-opus-5-5"
    # api_key: null                       # Reads ANTHROPIC_API_KEY from env
    # base_url: null                      # Uses SDK default
    timeout: 30
    enabled: true
    # Analyze large outputs with a cheaper model while keeping the parent
    # context compact. Empty provider means the same Anthropic endpoint.
    context_worker_provider: ""
    context_worker_model: "claude-haiku-4-5"

  openai:
    name: openai
    model: "gpt-5.6-sol"
    # api_key: null                       # Reads OPENAI_API_KEY from env
    # base_url: null                      # Uses SDK default
    timeout: 30
    enabled: true

  ollama:
    name: ollama
    model: "llama3.2"
    base_url: "http://localhost:11434"    # Required for Ollama
    timeout: 60
    enabled: true

# ---------------------------------------------------------------------------
# Discord integration (optional)
# ---------------------------------------------------------------------------
discord:
  enabled: false
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN    # Env var holding the bot token
      application_id: "1234567890"        # Required for slash commands
      dm_policy: disabled                 # pairing | allowlist | open | disabled
      dm_allowlist: []
      ack_reaction: "eyes"
      ignore_bots: true
      allow_bots_if_mention_only: false
      guild_policies:
        "987654321":                      # Guild ID
          enabled: true
          require_mention: true
          allowed_channels:
            - "general"
            - "bot-commands"
          allowed_roles: []
          allowed_users: []
          mode: full                      # safe_chat_only | no_tools | full

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"

observability:
  log_file_path: "~/.missy/missy.log"
  log_level: "warning"
```
