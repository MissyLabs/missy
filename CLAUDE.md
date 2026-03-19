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
pip install -e ".[vector]"      # adds faiss-cpu for semantic memory search
pip install -e ".[vision]"      # adds opencv-python-headless, numpy

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
  → Config migration (config/migrate.py) — auto-upgrades old configs to preset format
  → load_config() (config/settings.py) + ConfigWatcher (config/hotreload.py)
  → AgentRuntime (agent/runtime.py)
       ├─ InputSanitizer + SecretsDetector + SecretCensor (security/)
       ├─ PromptDriftDetector (security/drift.py) — SHA-256 system prompt tamper detection
       ├─ PolicyEngine (policy/engine.py) + RestPolicy (policy/rest_policy.py)
       ├─ AgentIdentity (security/identity.py) — Ed25519 keypair, signs audit events
       ├─ TrustScorer (security/trust.py) — 0-1000 reliability tracking per tool/provider
       ├─ CircuitBreaker (agent/circuit_breaker.py)
       ├─ AttentionSystem (agent/attention.py) — 5 brain-inspired subsystems
       ├─ ContextManager (agent/context.py) — token budget with memory/learnings injection
       ├─ MemoryConsolidator (agent/consolidation.py) — sleep mode at 80% context
       ├─ MemorySynthesizer (memory/synthesizer.py) — unified relevance-ranked memory block
       ├─ Playbook (agent/playbook.py) — auto-captured successful patterns
       ├─ ProviderRegistry + ModelRouter (providers/registry.py)
       ├─ RateLimiter (providers/rate_limiter.py)
       ├─ PolicyHTTPClient (gateway/client.py) + InteractiveApproval (agent/interactive_approval.py)
       ├─ ToolRegistry (tools/registry.py) + built-in tools
       ├─ McpManager (mcp/manager.py) — MCP server integration + digest pinning
       ├─ SkillDiscovery (skills/discovery.py) — SKILL.md dynamic skill loading
       ├─ ResilientMemoryStore → SQLiteMemoryStore (memory/)
       ├─ VectorMemoryStore (memory/vector_store.py) — optional FAISS semantic search
       ├─ DoneCriteria + Learnings + PromptPatchManager (agent/)
       ├─ ProgressReporter (agent/progress.py) — Null/Audit/CLI implementations
       ├─ SubAgentRunner (agent/sub_agent.py)
       ├─ ApprovalGate (agent/approval.py)
       ├─ PersonaManager (agent/persona.py) — YAML-backed identity/tone/style with backup/rollback
       ├─ BehaviorLayer (agent/behavior.py) — tone analysis, intent classification, response shaping
       ├─ HatchingManager (agent/hatching.py) — 8-step first-run bootstrap (incl. vision check)
       ├─ ContainerSandbox (security/container.py) — optional Docker isolation
       ├─ MessageBus (core/message_bus.py) — async event-driven routing
       ├─ VisionSubsystem (vision/) — camera discovery, capture, analysis, scene memory
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

**Policy Engine (`missy/policy/`)** — Multi-layer enforcement facade:
- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, per-category host allowlists (provider, tool, discord)
- `FilesystemPolicyEngine`: Per-path read/write access control
- `ShellPolicyEngine`: Command whitelisting
- `RestPolicy`: L7 HTTP method + path glob rules per host (e.g. allow GET /repos/**, deny DELETE /**)
- Network presets (`missy/policy/presets.py`): `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs

**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx; single enforcement point for ALL outbound HTTP. Every request checked against network policy + REST policy before dispatch. `InteractiveApproval` TUI prompts operator on denied requests (y/n/a with session memory).

**Providers (`missy/providers/`)** — `BaseProvider` defines the interface (`Message`, `CompletionResponse`, `ToolCall`, `ToolResult`). Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. `ProviderRegistry` handles resolution with fallback. `ProviderConfig` supports API key rotation (`api_keys` list), `fast_model`/`premium_model` tiers.

**Channels (`missy/channels/`)** — Communication interfaces:
- `CLIChannel`: Interactive stdin/stdout
- `DiscordChannel`: Full WebSocket Gateway API with access control (DM allowlist, guild/role policies), slash commands (`/ask`, `/status`, `/model`, `/help`)
- `WebhookChannel`: HTTP webhook ingress
- `VoiceChannel`: WebSocket server (default port 8765) accepting connections from edge nodes (ReSpeaker, Raspberry Pi). Protocol: JSON control frames + binary PCM audio. Device pairing with PBKDF2-hashed tokens. Per-node policy modes: `full`, `safe-chat`, `muted`. STT via faster-whisper, TTS via piper binary.

**Agent Loop Components (`missy/agent/`)**:
- `CircuitBreaker`: Closed/Open/HalfOpen state machine with exponential backoff (threshold=5, base_timeout=60s, max=300s)
- `ContextManager`: Token budget (default 30k) with reserves for system prompt, tool definitions, memory fraction (15%), learnings fraction (5%). Prunes oldest history first.
- `MemoryConsolidator`: Sleep mode — triggers at 80% context capacity, summarizes old turns, extracts key facts, preserves recent 4 messages
- `MemorySynthesizer`: Unified memory block — merges learnings (0.7), playbook (0.6), summaries (0.4) into a single relevance-ranked, deduplicated context block using keyword overlap scoring
- `AttentionSystem`: 5 brain-inspired subsystems — `AlertingAttention` (urgency keywords), `OrientingAttention` (topic extraction), `SustainedAttention` (focus continuity), `SelectiveAttention` (context filtering), `ExecutiveAttention` (tool prioritization)
- `Playbook`: Auto-captures successful tool patterns (task_type + tool_sequence hash), injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals. JSON persistence at `~/.missy/playbook.json`.
- `ProgressReporter`: Protocol with `NullReporter`, `AuditReporter`, `CLIReporter`. Called in tool loop for structured progress events.
- `InteractiveApproval`: Real-time Rich TUI for policy-denied operations (y=allow once, n=deny, a=allow always). Session-scoped memory. Non-TTY auto-denies.
- `DoneCriteria`: Generates verification prompts injected after each tool-call round
- `Learnings`: Extracts task_type/outcome/lesson from tool-augmented runs, persisted in SQLite
- `PromptPatchManager`: Self-tuning prompt patches with approval workflow (proposed/approved/rejected)
- `SubAgentRunner`: Spawns child agent instances
- `ApprovalGate`: Human-in-the-loop approval for sensitive operations

**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`. Digest pinning (`missy mcp pin`) records SHA-256 of tool manifests; mismatches refuse to load.

**Skills (`missy/skills/`)** — `SkillDiscovery` scans directories for SKILL.md files (cross-agent portable skill format with YAML frontmatter). `missy skills scan` lists discovered skills. Fuzzy search by name/description.

**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions.

**Security (`missy/security/`)**:
- `InputSanitizer`: Detects 250+ prompt injection patterns with Unicode normalization, base64 decode, multi-language support
- `SecretsDetector`: Detects 37+ credential patterns (API keys, JWTs, AWS, GCP, etc.)
- `SecretCensor`: Redacts secrets from output with overlap merging
- `Vault`: ChaCha20-Poly1305 encrypted key-value store. Key file at `~/.missy/secrets/vault.key`, encrypted data at `~/.missy/secrets/vault.enc`. Supports `vault://KEY_NAME` references in config.
- `AgentIdentity`: Ed25519 keypair at `~/.missy/identity.pem`. Signs audit events. JWK export.
- `TrustScorer`: 0-1000 reliability tracking per tool/provider/MCP server. Success (+10), failure (-50), violation (-200). Warns below threshold.
- `PromptDriftDetector`: SHA-256 hashes system prompts at start, verifies before each provider call. Emits `security.prompt_drift` audit event on tamper.
- `ContainerSandbox`: Optional Docker-based isolation for tool execution. Per-session containers with `--network=none`, memory/CPU limits. Config: `container: { enabled: true }`.

**Vision (`missy/vision/`)** — On-demand visual capabilities:
- `CameraDiscovery`: USB camera detection via sysfs with vendor/product ID matching, Logitech C922x preferred
- `CameraHandle`: OpenCV-based capture with warm-up, retry, blank-frame detection
- `ImageSource` abstraction: WebcamSource, FileSource, ScreenshotSource, PhotoSource
- `ImagePipeline`: Resize, CLAHE exposure normalization, quality assessment
- `SceneSession`: Task-scoped multi-frame memory for puzzle/painting tasks with change detection
- `AnalysisPromptBuilder`: Domain-specific prompts (puzzle board-state, painting coaching)
- `VisionIntentClassifier`: Audio-triggered vision activation with configurable thresholds
- `VisionDoctor`: Diagnostics (opencv, video group, devices, capture test, health)
- `VisionHealthMonitor`: Per-device capture stats, success rates, quality tracking, recommendations
- Provider-specific formatting: Anthropic/OpenAI/Ollama image message structures
- Agent tools: `vision_capture`, `vision_burst`, `vision_analyze`, `vision_devices`, `vision_scene`
- Voice integration: Auto-captures image when audio intent implies vision need

**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days. Optional `VectorMemoryStore` with FAISS semantic search (`pip install -e ".[vector]"`).

**Message Bus (`missy/core/message_bus.py`)** — Async event-driven routing with fnmatch topic wildcards (e.g. `channel.*`), priority queue, correlation IDs, worker thread. Standard topics in `missy/core/bus_topics.py`. Runtime publishes `AGENT_RUN_START/COMPLETE/ERROR`, `TOOL_REQUEST/RESULT`. Module-level singleton via `init_message_bus()` / `get_message_bus()`.

**Config Migration (`missy/config/migrate.py`)** — Auto-migrates old configs on startup. Detects manual hosts matching presets, replaces with `presets: [...]`, stamps `config_version: 2`. Backs up before modifying. Idempotent.

**Config Plan (`missy/config/plan.py`)** — Automatic backups on config writes (max 5, pruned). `missy config rollback/diff/plan/backups` commands.

**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl`. `OtelExporter` sends traces/metrics to an OTLP endpoint when enabled.

### Default File Locations

| Purpose | Path |
|---|---|
| Config | `~/.missy/config.yaml` |
| Config backups | `~/.missy/config.d/config.yaml.<timestamp>` |
| Audit log | `~/.missy/audit.jsonl` |
| Memory DB | `~/.missy/memory.db` |
| Vector index | `~/.missy/memory.faiss` (optional) |
| Scheduler jobs | `~/.missy/jobs.json` |
| MCP config | `~/.missy/mcp.json` |
| Device registry | `~/.missy/devices.json` |
| Vault key | `~/.missy/secrets/vault.key` |
| Vault data | `~/.missy/secrets/vault.enc` |
| Agent identity | `~/.missy/identity.pem` |
| Prompt patches | `~/.missy/patches.json` |
| Playbook | `~/.missy/playbook.json` |
| Persona | `~/.missy/persona.yaml` |
| Persona backups | `~/.missy/persona.d/persona.yaml.<timestamp>` |
| Persona audit log | `~/.missy/persona_audit.jsonl` |
| Hatching state | `~/.missy/hatching.yaml` |
| Hatching log | `~/.missy/hatching_log.jsonl` |
| Skills directory | `~/.missy/skills/` |
| Vision captures | `~/.missy/captures/` |
| Workspace | `~/workspace` |

### Configuration Schema

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

## CLI Commands

```
missy init                          Create default config at ~/.missy/config.yaml
missy setup                         Interactive setup wizard (API keys, OAuth)
missy setup --no-prompt             Non-interactive setup (--provider, --api-key-env, --model)
missy ask PROMPT                    Single-turn query (--provider, --session)
missy run                           Interactive REPL session (--provider)
missy providers list                List configured providers and availability
missy providers switch NAME         Switch active provider at runtime
missy skills                        List registered skills
missy skills scan                   Scan for SKILL.md files (--path)
missy presets list                  Show built-in network policy presets
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
missy mcp pin NAME                  Pin tool manifest SHA-256 digest for verification

missy devices list                  List all registered edge nodes
missy devices pair                  Approve a pending pairing request (--node-id)
missy devices unpair NODE_ID        Remove a paired edge node
missy devices status                Show online/offline status of all nodes
missy devices policy NODE_ID        Set node policy mode (--mode full|safe-chat|muted)

missy voice status                  Show voice channel config and STT/TTS status
missy voice test NODE_ID            Test TTS synthesis for an edge node (--text)

missy config backups                List config backups
missy config diff                   Diff current config vs latest backup
missy config rollback               Restore config from latest backup
missy config plan                   Show what changed since last backup

missy sandbox status                Show Docker container sandbox config and availability

missy vision devices                Enumerate and diagnose available USB cameras
missy vision capture                Capture frames (--device, --output, --count, --width, --height)
missy vision inspect                Run visual quality assessment (--file, --screenshot, --device)
missy vision review                 LLM-powered visual analysis (--mode general|puzzle|painting|inspection, --file, --context)
missy vision doctor                 Run vision subsystem diagnostics
missy vision health                 Show capture statistics and device health status

missy hatch                         First-run bootstrap wizard (--non-interactive)

missy persona show                  Display current persona configuration
missy persona edit                  Edit persona fields (--name, --tone, --identity)
missy persona reset                 Reset persona to factory defaults
missy persona backups               List available persona backups
missy persona diff                  Show diff between current persona and latest backup
missy persona rollback              Restore persona from latest backup
missy persona log                   Show persona change audit log (--limit)

missy cost                          Show cost tracking config and budget status (--session)
missy recover                       List incomplete checkpoints from previous sessions (--abandon-all)
```

## Optional Extras

```bash
pip install -e ".[dev]"     # pytest, pytest-asyncio, pytest-cov, black, ruff, mypy, watchdog
pip install -e ".[voice]"   # faster-whisper, numpy, soundfile
pip install -e ".[otel]"    # opentelemetry-sdk, OTLP gRPC + HTTP exporters
pip install -e ".[vector]"  # faiss-cpu for semantic memory search
pip install -e ".[vision]"  # opencv-python-headless, numpy for vision subsystem
```

Piper TTS is a separate binary, not a pip package. Install from https://github.com/rhasspy/piper.

## Documentation

Full docs site: **https://missylabs.github.io/** — 60+ pages covering getting started, configuration, security, architecture, CLI reference, channels, providers, extending, edge nodes, and operations. Source at `/home/missy/missylabs.github.io/` (MkDocs Material, deployed via GitHub Actions).

## Test Layout

Tests under `tests/` with subdirectories: `agent/`, `channels/`, `cli/`, `config/`, `core/`, `gateway/`, `integration/`, `mcp/`, `memory/`, `observability/`, `plugins/`, `policy/`, `providers/`, `scheduler/`, `security/`, `skills/`, `tools/`, `unit/`, `vision/`. 350+ test files, 12,000+ tests, coverage threshold 85% (configured in `pyproject.toml`).

## Companion Project: missy-edge

**Repository:** `/home/missy/missy-edge/` ([GitHub](https://github.com/MissyLabs/missy-edge))

`missy-edge` is the standalone Raspberry Pi voice edge node client. It connects to Missy's VoiceChannel server (`missy/channels/voice/server.py`) over WebSocket and provides always-on wake word detection, audio streaming, and TTS playback.

### Relationship to This Repo

- **Server side** (this repo): `missy/channels/voice/server.py` is the authoritative WebSocket protocol implementation. `missy/channels/voice/registry.py` handles device registration, PBKDF2 token hashing, and node management. `missy/channels/voice/edge_client.py` is the original reference client (PipeWire, manual push-to-talk, local testing only).
- **Client side** (missy-edge): Production edge node client with wake word, sounddevice audio, reconnection, LED/mute hardware support, systemd service.

### missy-edge Architecture

```
missy-edge/
├── missy_edge/
│   ├── client.py        # EdgeClient: reconnect loop, auth, concurrent async loops
│   ├── audio.py         # AudioCapture (sounddevice → asyncio.Queue) + AudioPlayback (WAV)
│   ├── wakeword.py      # WakeWordDetector (openWakeWord + ONNX, pre-trigger buffer)
│   ├── protocol.py      # Message builders matching server.py protocol
│   ├── reconnect.py     # ExponentialBackoff (1s→60s) + AuthRateLimiter (5/60s)
│   ├── config.py        # EdgeConfig, YAML load chain, token permission checks
│   ├── noise.py         # RMS noise estimator (reported in heartbeats)
│   ├── led.py           # ReSpeaker USB HID LED ring (6 states)
│   ├── mute.py          # Physical mute button monitor (HID polling thread)
│   └── __main__.py      # CLI: --pair, --manual, --server, --config
├── systemd/missy-edge.service
├── setup_pi.sh          # Pi provisioning script
├── docs/                # protocol.md, deployment.md, development.md
└── tests/               # 56 tests (protocol, reconnect, config, audio, wakeword, client)
```

### Protocol Alignment (Critical)

The spec doc (`docs/edge-node-client.md`) and actual server differ. Both this repo and missy-edge use the **server implementation**:

| Spec Doc | Actual Server (`server.py`) | missy-edge Uses |
|---|---|---|
| `stream_start`/`stream_end` | `audio_start`/`audio_end` | `audio_start`/`audio_end` |
| `tts_audio` + `tts_end` (raw PCM) | `audio_start` + binary WAV + `audio_end` | WAV over `audio_start`/`audio_end` |
| `pair_request` with `node_id` | Server assigns `node_id`, returns in `pair_pending` | Omit `node_id` from request |
| Token via WebSocket `pair_ack` | Token shown in CLI `missy devices pair` output | Out-of-band token provisioning |

### Cross-Repo Development Notes

- Changing the WebSocket protocol in `missy/channels/voice/server.py` requires corresponding updates in `missy-edge/missy_edge/protocol.py` and `missy-edge/missy_edge/client.py`.
- The device pairing flow spans both repos: `missy devices pair` (this repo, CLI) generates the token; the user manually provisions it to `/etc/missy-edge/token` on the Pi.
- Edge node management commands (`missy devices list/status/pair/unpair/policy`) in this repo interact with the `DeviceRegistry` that missy-edge authenticates against.

### missy-edge Commands

```bash
# Install (on Pi)
sudo bash setup_pi.sh

# Or development install
pip install -e ".[dev]"

# Tests
python -m pytest tests/ -v          # 56 tests

# Run
missy-edge                          # Wake word mode (default)
missy-edge --manual                 # Push-to-talk (Enter key)
missy-edge --pair --name N --room R # Pair new device
```

### missy-edge Config Locations

| Purpose | Path |
|---|---|
| System config | `/etc/missy-edge/config.yaml` |
| User config | `~/.config/missy-edge/config.yaml` |
| Auth token | `/etc/missy-edge/token` (chmod 600, required) |
| Node identity | `/etc/missy-edge/node_id` |
| Wake word model | `/opt/missy-edge/models/*.onnx` |
| systemd service | `/etc/systemd/system/missy-edge.service` |
| Venv | `/opt/missy-edge/venv/` |
