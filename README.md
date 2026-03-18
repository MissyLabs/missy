# Missy

**Security-first, self-hosted AI assistant for Linux.**

Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.

**Full documentation: [missylabs.github.io](https://missylabs.github.io)**

---

## Why Missy

Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.

- **No outbound traffic** unless you whitelist the destination (CIDR, domain, host:port, or use presets)
- **No filesystem access** unless you declare read/write paths
- **No shell commands** unless you name each allowed binary
- **No plugins** unless you approve them individually
- **Every action** logged as structured JSONL with full audit trail
- **Every audit event** signed with the agent's Ed25519 identity

This isn't paranoia — it's the only sane default for an AI agent that can execute code, call APIs, and access your files.

---

## Features

### Core Platform
- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
- **API key rotation** — multiple keys per provider, round-robin distribution
- **Model tiers** — `fast_model` for quick tasks, `premium_model` for complex reasoning, auto-routed by ModelRouter
- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
- **Sleep mode** — context consolidation at 80% token capacity: summarizes old turns, extracts key facts, preserves recent context
- **Unified memory** — merges learnings, summaries, and playbook into a single relevance-ranked, deduplicated context block
- **Sub-agents** — spawn child agent instances for parallel work
- **Approval gate** — human-in-the-loop confirmation for sensitive operations
- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
- **Progress reporting** — structured protocol with Null/Audit/CLI reporter implementations
- **Cost tracking** — per-session budget caps with `max_spend_usd`
- **Message bus** — async event-driven routing with topic wildcards, priority queue, and correlation IDs

### Security
- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
- **Input sanitization** — 250+ prompt injection patterns across 10+ languages with Unicode normalization, base64 decode, multi-layer detection
- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
- **Secrets detection** — 37+ credential patterns with automatic response censoring and overlap merging
- **Encrypted vault** — ChaCha20-Poly1305 with atomic key creation, `vault://` config references
- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
- **Config hot-reload safety** — symlink, ownership, and permission checks before reload

### Channels
- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
- **Webhooks** — HTTP ingress with HMAC auth, rate limiting, payload validation
- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth

### Automation & Extensibility
- **Scheduler** — APScheduler with human-friendly syntax (`"daily at 09:00"`, `"every monday at 08:00"`), JSON persistence
- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
- **Vector memory** — optional FAISS-based semantic search alongside SQLite FTS5 (`pip install -e ".[vector]"`)
- **Heartbeat** — periodic workspace monitoring during active hours

### Operations
- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
- **Config migration** — auto-upgrades old configs to preset format on startup, backs up first
- **Config plan/rollback** — `missy config diff`, `missy config rollback`, automatic backups (max 5)
- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`

### Observability
- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
- **OpenTelemetry** — optional traces and metrics via OTLP (gRPC or HTTP)
- **Cost tracking** — per-session spend monitoring with configurable caps

---

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/MissyLabs/missy/master/install.sh | bash
```

This clones to `~/.local/share/missy`, creates a venv, installs, and symlinks `missy` into `~/.local/bin`. Requires Python 3.11+ and git.

## Quick Start

```bash
missy setup
```

The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:

```bash
missy ask "What services are listening on this machine?"
missy run    # interactive session
```

<details>
<summary>Manual install</summary>

```bash
git clone https://github.com/MissyLabs/missy.git
cd missy
pip install -e .
missy setup
```
</details>

<details>
<summary>Non-interactive setup (CI/Docker)</summary>

```bash
missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
```
</details>

### Optional extras

```bash
pip install -e ".[voice]"   # faster-whisper, numpy, soundfile (for voice channel)
pip install -e ".[otel]"    # OpenTelemetry SDK + OTLP exporters
pip install -e ".[vector]"  # FAISS semantic memory search
pip install -e ".[dev]"     # pytest, ruff, mypy, coverage tools
```

---

## Architecture

```
User ─── CLI / Discord / Webhook / Voice
              │
              v
         AgentRuntime
              │
    ┌─────────┼──────────────────────────────┐
    │         │                              │
    v         v                              v
PolicyEngine  ProviderRegistry           ToolRegistry
(network,     (Anthropic, OpenAI,        (built-in tools,
 filesystem,   Ollama + fallback)         skills, plugins,
 shell,                                   MCP servers)
 REST L7)         │                          │
    │             v                          v
    │    ┌────────────────┐     ┌────────────────────┐
    │    │ AttentionSystem │     │ Playbook + SKILL.md│
    │    │ MemorySynth.   │     │ SkillDiscovery     │
    │    │ Consolidator   │     │ VectorMemory       │
    │    └────────────────┘     └────────────────────┘
    │             │                          │
    v             v                          v
PolicyHTTPClient + InteractiveApproval ◄─────┘
    │              (single enforcement point)
    v
 Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
    │
    v
 MessageBus ──► async event routing with topic wildcards
```

Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.

---

## Configuration

Missy uses `~/.missy/config.yaml`. API keys go in environment variables or the encrypted vault — never in the config file. Old configs are auto-migrated on startup.

```yaml
config_version: 2

network:
  default_deny: true
  presets:
    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
    - github                     # auto-expands to api.github.com + github.com
  allowed_hosts:
    - "localhost:11434"          # local Ollama
  rest_policies:                 # L7 HTTP method + path controls
    - host: "api.github.com"
      method: "GET"
      path: "/repos/**"
      action: "allow"

filesystem:
  allowed_read_paths: ["~/workspace", "~/.missy"]
  allowed_write_paths: ["~/workspace", "~/.missy"]

shell:
  enabled: false
  allowed_commands: []           # e.g. ["git", "python3"]

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    fast_model: "claude-haiku-4-5"
    premium_model: "claude-opus-4-6"
    timeout: 30

container:                       # optional Docker sandbox
  enabled: false
  image: "python:3.12-slim"
  network_mode: "none"

voice:
  host: "0.0.0.0"
  port: 8765
  stt: { engine: "faster-whisper", model: "base.en" }
  tts: { engine: "piper", voice: "en_US-lessac-medium" }
```

See the [full configuration reference](https://missylabs.github.io/configuration/reference/) for all options.

---

## CLI Reference

```bash
# Core
missy setup                         # Interactive setup wizard
missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
missy run                           # Interactive REPL (--provider, --mode)
missy providers list                # List providers and availability
missy providers switch NAME         # Hot-swap active provider
missy doctor                        # System health check

# Scheduling
missy schedule add --name NAME --schedule EXPR --task PROMPT
missy schedule list | pause | resume | remove

# Security & audit
missy audit recent                  # Recent events (--limit, --category)
missy audit security                # Policy violations
missy vault set|get|list|delete     # Encrypted secrets

# Config management
missy config backups                # List config backups
missy config diff                   # Diff vs latest backup
missy config rollback               # Restore from backup
missy presets list                  # Show built-in network presets

# Discord
missy discord status | probe | register-commands | audit

# Voice & edge nodes
missy voice status | test NODE_ID
missy devices list | pair | unpair | status | policy

# MCP & skills
missy mcp list | add NAME | remove NAME | pin NAME
missy skills                        # List registered skills
missy skills scan                   # Discover SKILL.md files

# Operations
missy sandbox status                # Docker sandbox availability
missy sessions cleanup              # Prune old conversations
missy cost                          # Budget status
missy recover                       # Resume from checkpoints
```

---

## Voice Edge Nodes

Deploy Raspberry Pi nodes throughout your home. Say the wake word, speak naturally, hear Missy respond.

```
  "Hey Missy, turn off the garage lights"

  [Pi + ReSpeaker]  ──PCM audio──►  [Missy Server]
       edge node                    STT → Agent → TTS
                    ◄──WAV audio──
       speaker plays response       (all on your hardware)
```

The edge node client is a separate project: **[missy-edge](https://github.com/MissyLabs/missy-edge)** — wake word detection, auto-reconnect, LED feedback, hardware mute button, systemd service.

Server-side management:

```bash
missy devices list                  # All registered nodes
missy devices pair --node-id ID     # Approve a new device
missy devices policy ID --mode full|safe-chat|muted
missy devices status                # Online/offline + sensor data
```

---

## Testing

```bash
python3 -m pytest tests/ -v                          # All tests
python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
pytest tests/ --cov=missy --cov-report=html           # Coverage report
ruff check missy/ tests/                              # Lint
ruff format missy/ tests/                             # Format
```

1300+ tests across 70+ test files. 85% coverage threshold.

---

## Documentation

**Full docs: [missylabs.github.io](https://missylabs.github.io)** — 60+ pages with dark mode, search, code tabs, and mermaid diagrams.

| Section | Pages | Covers |
|---------|-------|--------|
| [Getting Started](https://missylabs.github.io/getting-started/) | 5 | Install, quickstart, wizard, first conversation |
| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
| [Security](https://missylabs.github.io/security/) | 11 | Policy engine, gateway, sanitization, secrets, vault, drift, identity, trust, container, threat model |
| [Architecture](https://missylabs.github.io/architecture/) | 10 | Runtime, context, circuit breaker, progress, playbook, sleep mode, synthesizer, attention, message bus |
| [CLI Reference](https://missylabs.github.io/cli/) | 7 | Every command group |
| [Channels](https://missylabs.github.io/channels/) | 7 | CLI, Discord, voice server/protocol/devices |
| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
| [Missy Edge](https://missylabs.github.io/edge/) | 6 | Hardware, Pi setup, pairing, config, wake word |
| [Operations](https://missylabs.github.io/operations/) | 4 | Backup/rollback, observability, troubleshooting |

### In-repo docs

Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.

---

## Project Structure

```
missy/
├── agent/           Runtime, circuit breaker, context, playbook, consolidation,
│                    attention, progress, interactive approval, sub-agents, approvals
├── channels/        CLI, Discord (Gateway + REST), webhooks, voice (WebSocket server)
├── cli/             Click + Rich CLI, setup wizard, OAuth
├── config/          YAML settings, hot-reload, migration, plan/rollback
├── core/            Session management, event bus, message bus, exceptions
├── gateway/         PolicyHTTPClient — single network enforcement point
├── mcp/             MCP server manager, health checks, digest pinning
├── memory/          SQLite FTS5 store, vector memory (FAISS), synthesizer
├── observability/   Audit logger (JSONL), OpenTelemetry exporter
├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
├── scheduler/       APScheduler integration, human schedule parser
├── security/        Input sanitizer, secrets detector, censor, vault, identity,
│                    trust scorer, drift detector, container sandbox
├── skills/          Skill registry + SKILL.md discovery
├── plugins/         Security-gated external plugin loader
└── tools/           Built-in tools + registry
```

---

## License

MIT
