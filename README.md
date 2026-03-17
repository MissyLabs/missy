# Missy

**Security-first, self-hosted AI assistant for Linux.**

Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.

---

## Why Missy

Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.

- **No outbound traffic** unless you whitelist the destination (CIDR, domain, or host:port)
- **No filesystem access** unless you declare read/write paths
- **No shell commands** unless you name each allowed binary
- **No plugins** unless you approve them individually
- **Every action** logged as structured JSONL with full audit trail

This isn't paranoia — it's the only sane default for an AI agent that can execute code, call APIs, and access your files.

---

## Features

### Core Platform
- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback
- **API key rotation** — multiple keys per provider, round-robin distribution
- **Model tiers** — `fast_model` for quick tasks, `premium_model` for complex reasoning
- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
- **Sub-agents** — spawn child agent instances for parallel work
- **Approval gate** — human-in-the-loop confirmation for sensitive operations
- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
- **Cost tracking** — per-session budget caps with `max_spend_usd`

### Security
- **Three-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist)
- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions
- **Input sanitization** — 69+ prompt injection patterns across 10 languages
- **Secrets detection** — 26 credential patterns with automatic response censoring
- **Encrypted vault** — ChaCha20-Poly1305 with atomic key creation, `vault://` config references
- **MCP isolation** — sanitized environment for subprocess servers, timeout + size limits
- **Config hot-reload safety** — symlink, ownership, and permission checks before reload

### Channels
- **CLI** — interactive REPL and single-shot queries with Rich formatting
- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
- **Webhooks** — HTTP ingress with HMAC auth, rate limiting, payload validation
- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth

### Automation
- **Scheduler** — APScheduler with human-friendly syntax (`"daily at 09:00"`, `"every monday at 08:00"`), JSON persistence
- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart on failure
- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
- **Heartbeat** — periodic workspace monitoring during active hours

### Observability
- **Audit logger** — every policy decision, provider call, and tool execution as JSONL
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

### Optional extras

```bash
pip install -e ".[voice]"   # faster-whisper, numpy, soundfile (for voice channel)
pip install -e ".[otel]"    # OpenTelemetry SDK + OTLP exporters
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
    ┌─────────┼─────────────────────────┐
    │         │                         │
    v         v                         v
PolicyEngine  ProviderRegistry      ToolRegistry
    │         │                         │
    │    ┌────┼────┐               Built-in tools
    │    │    │    │               Skills / Plugins
    │    v    v    v               MCP servers
    │  Claude GPT  Ollama              │
    │    │    │    │                    │
    v    v    v    v                    v
PolicyHTTPClient ◄─────────────────────┘
    │              (single enforcement point)
    v
 Network
    │
    v
AuditLogger ──► ~/.missy/audit.jsonl
```

Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.

---

## Configuration

Missy uses `~/.missy/config.yaml`. API keys go in environment variables or the encrypted vault — never in the config file.

```yaml
network:
  default_deny: true
  allowed_hosts:
    - "api.anthropic.com"
    - "localhost:11434"        # local Ollama
  allowed_domains:
    - "*.github.com"
  allowed_cidrs:
    - "10.0.0.0/8"

filesystem:
  allowed_read_paths: ["~/workspace", "~/.missy"]
  allowed_write_paths: ["~/workspace", "~/.missy"]

shell:
  enabled: false
  allowed_commands: []         # e.g. ["git", "python3"]

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    fast_model: "claude-haiku-4-5"
    premium_model: "claude-opus-4-6"
    timeout: 30

voice:
  host: "0.0.0.0"
  port: 8765
  stt: { engine: "faster-whisper", model: "base.en" }
  tts: { engine: "piper", voice: "en_US-lessac-medium" }
```

See [docs/configuration.md](docs/configuration.md) for the complete reference.

---

## CLI Reference

```bash
# Core
missy setup                         # Interactive setup wizard
missy ask PROMPT                    # Single-turn query (--provider, --session)
missy run                           # Interactive REPL (--provider)
missy providers                     # List providers and availability
missy doctor                        # System health check

# Scheduling
missy schedule add --name NAME --schedule EXPR --task PROMPT
missy schedule list | pause | resume | remove

# Security & audit
missy audit recent                  # Recent events (--limit, --category)
missy audit security                # Policy violations
missy vault set|get|list|delete     # Encrypted secrets

# Discord
missy discord status | probe | register-commands | audit

# Voice & edge nodes
missy voice status | test NODE_ID
missy devices list | pair | unpair | status | policy

# MCP servers
missy mcp list | add NAME | remove NAME

# Maintenance
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

1097 tests across 52 test files. 85% coverage threshold.

---

## Documentation

| Guide | Description |
|---|---|
| [Architecture](docs/architecture.md) | System design, data flow, module dependencies, design principles |
| [Configuration](docs/configuration.md) | Complete YAML reference with annotated examples |
| [Security](docs/security.md) | Security policy, hardening guide, vulnerability reporting |
| [Operations](docs/operations.md) | Installation, running, monitoring, backup and recovery |
| [Providers](docs/providers.md) | Provider abstraction, per-provider setup, API key management |
| [Discord](docs/discord.md) | Discord integration, access control, slash commands |
| [Scheduler](docs/scheduler.md) | Job scheduling, human-friendly syntax, persistence |
| [Skills & Plugins](docs/skills-and-plugins.md) | Extension system: tools, skills, plugins |
| [Memory & Persistence](docs/memory-and-persistence.md) | Conversation memory, learnings, storage schema |
| [Testing](docs/testing.md) | Test suite layout, running tests, writing new tests |
| [Troubleshooting](docs/troubleshooting.md) | Common errors and diagnostic procedures |
| [Threat Model](docs/threat-model.md) | Attack vectors and mitigations |
| [Voice Edge Spec](docs/voice-edge-spec.md) | Edge node protocol specification |

### Implementation deep-dives

| Reference | Source |
|---|---|
| [Agent Loop](docs/implementation/agent-loop.md) | `missy/agent/runtime.py` |
| [Policy Engine](docs/implementation/policy-engine.md) | `missy/policy/` |
| [Provider Abstraction](docs/implementation/provider-abstraction.md) | `missy/providers/` |
| [Network Client](docs/implementation/network-client.md) | `missy/gateway/client.py` |
| [Discord Channel](docs/implementation/discord-channel.md) | `missy/channels/discord/` |
| [Audit Events](docs/implementation/audit-events.md) | `missy/observability/` |
| [Persistence Schema](docs/implementation/persistence-schema.md) | `missy/memory/`, `missy/scheduler/` |
| [Scheduler Execution](docs/implementation/scheduler-execution.md) | `missy/scheduler/` |
| [Module Map](docs/implementation/module-map.md) | Full import dependency graph |
| [Manifest Schema](docs/implementation/manifest-schema.md) | Plugin/skill manifests |

---

## Project Structure

```
missy/
├── agent/           Runtime, circuit breaker, context manager, sub-agents, approvals
├── channels/        CLI, Discord (Gateway + REST), webhooks, voice (WebSocket server)
├── cli/             Click + Rich CLI, setup wizard, OAuth
├── config/          YAML settings, hot-reload watcher
├── core/            Session management, event bus, exceptions
├── gateway/         PolicyHTTPClient — single network enforcement point
├── mcp/             MCP server manager, health checks
├── memory/          SQLite FTS5 conversation store, learnings
├── observability/   Audit logger (JSONL), OpenTelemetry exporter
├── policy/          Network, filesystem, shell policy engines + facade
├── providers/       Anthropic, OpenAI, Ollama + registry with fallback
├── scheduler/       APScheduler integration, human schedule parser
├── security/        Input sanitizer, secrets detector, response censor, vault
├── skills/          In-process skill registry
├── plugins/         Security-gated external plugin loader
└── tools/           Built-in tools + registry
```

---

## License

MIT
