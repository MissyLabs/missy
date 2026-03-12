# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. Production-grade agent platform with strict security controls, policy enforcement, and full auditability. Python 3.11+.

## Commands

```bash
# Install
pip install -e ".[dev]"
pip install -e ".[voice]"       # adds faster-whisper, numpy, soundfile
pip install -e ".[otel]"        # adds OpenTelemetry SDK + exporters

# Tests
python3 -m pytest tests/ -v
python3 -m pytest tests/unit/test_policy_engine.py -v     # single file
python3 -m pytest tests/ -k "test_name" -v                # single test
pytest tests/ --cov=missy --cov-report=html               # coverage

# Lint / format
ruff check missy/ tests/
ruff format missy/ tests/
```

CLI entry point: `missy` (maps to `missy.cli.main:cli`).

## Architecture

Secure-by-default: all capabilities (shell, plugins, network) are disabled until explicitly enabled in `~/.missy/config.yaml`.

### Data Flow

```
CLI (missy/cli/main.py)
  → missy setup (wizard.py + oauth.py + anthropic_auth.py)
  → load_config() (config/settings.py) + ConfigWatcher (config/hotreload.py)
  → AgentRuntime (agent/runtime.py)
       ├─ InputSanitizer + SecretsDetector + SecretCensor (security/)
       ├─ PolicyEngine (policy/engine.py)
       ├─ CircuitBreaker (agent/circuit_breaker.py)
       ├─ ContextManager (agent/context.py) — token budget with memory/learnings injection
       ├─ ProviderRegistry + ModelRouter (providers/registry.py)
       ├─ RateLimiter (providers/rate_limiter.py)
       ├─ PolicyHTTPClient (gateway/client.py)
       ├─ ToolRegistry (tools/registry.py) + built-in tools
       ├─ McpManager (mcp/manager.py) — MCP server integration
       ├─ ResilientMemoryStore → SQLiteMemoryStore (memory/)
       ├─ DoneCriteria + Learnings + PromptPatchManager (agent/)
       ├─ SubAgentRunner (agent/sub_agent.py)
       ├─ ApprovalGate (agent/approval.py)
       └─ AuditLogger + OtelExporter (observability/)

Channels:
  CLIChannel | DiscordChannel | WebhookChannel | VoiceChannel

VoiceChannel (channels/voice/):
  → VoiceServer (WebSocket, port 8765)
       ├─ DeviceRegistry (devices.json) + PairingManager
       ├─ PresenceStore (occupancy/sensor data)
       ├─ STTEngine → FasterWhisperSTT
       └─ TTSEngine → PiperTTS
```

### Key Subsystems

**Policy Engine (`missy/policy/`)** — Three-layer enforcement facade:
- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, per-category host allowlists (provider, tool, discord)
- `FilesystemPolicyEngine`: Per-path read/write access control
- `ShellPolicyEngine`: Command whitelisting

**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx; single enforcement point for ALL outbound HTTP. Every request checked against network policy before dispatch.

**Providers (`missy/providers/`)** — `BaseProvider` defines the interface (`Message`, `CompletionResponse`, `ToolCall`, `ToolResult`). Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. `ProviderRegistry` handles resolution with fallback. `ProviderConfig` supports API key rotation (`api_keys` list), `fast_model`/`premium_model` tiers.

**Channels (`missy/channels/`)** — Communication interfaces:
- `CLIChannel`: Interactive stdin/stdout
- `DiscordChannel`: Full WebSocket Gateway API with access control (DM allowlist, guild/role policies), slash commands (`/ask`, `/status`, `/model`, `/help`)
- `WebhookChannel`: HTTP webhook ingress
- `VoiceChannel`: WebSocket server (default port 8765) accepting connections from edge nodes (ReSpeaker, Raspberry Pi). Protocol: JSON control frames + binary PCM audio. Device pairing with PBKDF2-hashed tokens. Per-node policy modes: `full`, `safe-chat`, `muted`. STT via faster-whisper, TTS via piper binary.

**Agent Loop Components (`missy/agent/`)**:
- `CircuitBreaker`: Closed/Open/HalfOpen state machine with exponential backoff (threshold=5, base_timeout=60s, max=300s)
- `ContextManager`: Token budget (default 30k) with reserves for system prompt, tool definitions, memory fraction (15%), learnings fraction (5%). Prunes oldest history first.
- `DoneCriteria`: Generates verification prompts injected after each tool-call round
- `Learnings`: Extracts task_type/outcome/lesson from tool-augmented runs, persisted in SQLite
- `PromptPatchManager`: Self-tuning prompt patches with approval workflow (proposed/approved/rejected)
- `SubAgentRunner`: Spawns child agent instances
- `ApprovalGate`: Human-in-the-loop approval for sensitive operations

**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`.

**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions.

**Security (`missy/security/`)**:
- `InputSanitizer`: Detects 13+ prompt injection patterns
- `SecretsDetector`: Detects 9 credential patterns (API keys, JWTs, etc.)
- `SecretCensor`: Redacts secrets from output
- `Vault`: ChaCha20-Poly1305 encrypted key-value store. Key file at `~/.missy/secrets/vault.key`, encrypted data at `~/.missy/secrets/vault.enc`. Supports `vault://KEY_NAME` references in config.

**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days.

**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl`. `OtelExporter` sends traces/metrics to an OTLP endpoint when enabled.

### Default File Locations

| Purpose | Path |
|---|---|
| Config | `~/.missy/config.yaml` |
| Audit log | `~/.missy/audit.jsonl` |
| Memory DB | `~/.missy/memory.db` |
| Scheduler jobs | `~/.missy/jobs.json` |
| MCP config | `~/.missy/mcp.json` |
| Device registry | `~/.missy/devices.json` |
| Vault key | `~/.missy/secrets/vault.key` |
| Vault data | `~/.missy/secrets/vault.enc` |
| Prompt patches | `~/.missy/patches.json` |
| Workspace | `~/workspace` |

### Configuration Schema

```yaml
network:
  default_deny: true
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts: []               # host:port pairs
  provider_allowed_hosts: []      # per-category overrides
  tool_allowed_hosts: []
  discord_allowed_hosts: []

filesystem:
  allowed_write_paths: []
  allowed_read_paths: []

shell:
  enabled: false
  allowed_commands: []

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
    api_key: null                  # or set ANTHROPIC_API_KEY env var
    api_keys: []                   # multiple keys for rotation
    fast_model: ""                 # e.g. claude-haiku-4-5
    premium_model: ""              # e.g. claude-opus-4-6
    enabled: true

scheduling:
  enabled: true
  max_jobs: 0                      # 0 = unlimited
  active_hours: ""                 # e.g. "08:00-22:00"

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
  host: "0.0.0.0"
  port: 8765
  stt:
    engine: "faster-whisper"
    model: "base.en"
  tts:
    engine: "piper"
    voice: "en_US-lessac-medium"

discord:
  # See missy/channels/discord/config.py for full schema

workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
max_spend_usd: 0.0                  # per-session budget cap; 0 = unlimited
```

## CLI Commands

```
missy init                          Create default config at ~/.missy/config.yaml
missy setup                         Interactive setup wizard (API keys, OAuth)
missy ask PROMPT                    Single-turn query (--provider, --session)
missy run                           Interactive REPL session (--provider)
missy providers                     List configured providers and availability
missy skills                        List registered skills
missy plugins                       List plugins and their status
missy doctor                        System health check

missy schedule add                  Add scheduled job (--name, --schedule, --task, --provider)
missy schedule list                 List all scheduled jobs
missy schedule pause JOB_ID         Pause a job
missy schedule resume JOB_ID        Resume a paused job
missy schedule remove JOB_ID        Remove a job

missy audit security                Show recent security events (--limit)
missy audit recent                  Show recent audit events (--limit, --category)

missy gateway start                 Start the gateway server (--host, --port)
missy gateway status                Show gateway status

missy discord status                Show Discord channel configuration
missy discord probe                 Test Discord bot token connectivity
missy discord register-commands     Register slash commands (--guild-id, --global)
missy discord audit                 Show Discord-specific audit events (--limit)

missy vault set KEY VALUE           Store an encrypted secret
missy vault get KEY                 Retrieve a secret
missy vault list                    List stored key names
missy vault delete KEY              Delete a secret

missy sessions cleanup              Delete old conversation history (--older-than, --dry-run)

missy approvals list                List pending approval requests

missy patches list                  List prompt patches
missy patches approve PATCH_ID      Approve a proposed patch
missy patches reject PATCH_ID       Reject a proposed patch

missy mcp list                      List configured MCP servers
missy mcp add NAME                  Connect to an MCP server (--command, --url)
missy mcp remove NAME               Disconnect and remove an MCP server

missy devices list                  List all registered edge nodes
missy devices pair                  Approve a pending pairing request (--node-id)
missy devices unpair NODE_ID        Remove a paired edge node
missy devices status                Show online/offline status of all nodes
missy devices policy NODE_ID        Set node policy mode (--mode full|safe-chat|muted)

missy voice status                  Show voice channel config and STT/TTS status
missy voice test NODE_ID            Test TTS synthesis for an edge node (--text)

missy cost                          Show cost tracking config and budget status (--session)
missy recover                       List incomplete checkpoints from previous sessions (--abandon-all)
```

## Optional Extras

```bash
pip install -e ".[dev]"     # pytest, pytest-asyncio, pytest-cov, black, ruff, mypy, watchdog
pip install -e ".[voice]"   # faster-whisper, numpy, soundfile
pip install -e ".[otel]"    # opentelemetry-sdk, OTLP gRPC + HTTP exporters
```

Piper TTS is a separate binary, not a pip package. Install from https://github.com/rhasspy/piper.

## Test Layout

Tests under `tests/` with subdirectories: `agent/`, `channels/`, `cli/`, `config/`, `core/`, `integration/`, `memory/`, `observability/`, `plugins/`, `policy/`, `providers/`, `scheduler/`, `security/`, `skills/`, `tools/`, `unit/`. 52 test files, 1097 tests, coverage threshold 85% (configured in `pyproject.toml`).
