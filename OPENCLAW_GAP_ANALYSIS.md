# OpenClaw Gap Analysis

Last updated: 2026-03-12

## Currently Implemented Capabilities

| Category | Capability | Status | Notes |
|---|---|---|---|
| **Runtime** | Self-hosted local operation | ✅ Complete | Linux-native, systemd-ready |
| **Runtime** | Gateway/control-plane | ✅ Complete | PolicyHTTPClient, configurable host/port |
| **Runtime** | Agent loop with tool calling | ✅ Complete | Multi-step with iteration limits |
| **Runtime** | Multi-step planning & verification | ✅ Complete | DoneCriteria, compound task detection |
| **Runtime** | Retry/self-correction on tool failure | ✅ Complete | FailureTracker with strategy rotation |
| **Runtime** | Task checkpointing & recovery | ✅ Complete | SQLite-backed, 3-tier classification |
| **Runtime** | Circuit breaker | ✅ Complete | Closed/Open/HalfOpen with exponential backoff |
| **Runtime** | Sub-agent spawning | ✅ Complete | SubAgentRunner |
| **Runtime** | Proactive task initiation | ✅ Complete | 4 trigger types (schedule, disk, load, file_change) |
| **Runtime** | Cost tracking & budgets | ✅ Complete | Per-model pricing, budget enforcement |
| **CLI** | init/setup | ✅ Complete | Wizard with OAuth (OpenAI PKCE + Anthropic) |
| **CLI** | ask / run (REPL) | ✅ Complete | Single-turn and interactive modes |
| **CLI** | providers list | ✅ Complete | Shows configured providers & availability |
| **CLI** | skills / plugins list | ✅ Complete | Lists registered skills and plugins |
| **CLI** | doctor | ✅ Complete | 10+ health checks |
| **CLI** | schedule add/list/pause/resume/remove | ✅ Complete | APScheduler with cron parsing |
| **CLI** | audit security / recent | ✅ Complete | Structured JSONL audit trail |
| **CLI** | vault set/get/list/delete | ✅ Complete | ChaCha20-Poly1305 encrypted store |
| **CLI** | sessions cleanup | ✅ Complete | TTL-based pruning |
| **CLI** | approvals list | ✅ Complete | Human-in-the-loop approval gate |
| **CLI** | patches list/approve/reject | ✅ Complete | Self-tuning prompt patches |
| **CLI** | mcp list/add/remove | ✅ Complete | MCP server management |
| **CLI** | devices list/pair/unpair/status/policy | ✅ Complete | Voice edge node management |
| **CLI** | voice status/test | ✅ Complete | STT/TTS diagnostics |
| **CLI** | gateway start/status | ✅ Complete | Gateway lifecycle |
| **Providers** | Anthropic | ✅ Complete | Claude models, API key rotation |
| **Providers** | OpenAI | ✅ Complete | GPT-4o/4/3.5, o3/o4 |
| **Providers** | Ollama (local) | ✅ Complete | Zero-cost local models |
| **Providers** | Provider fallback | ✅ Complete | Registry with fallback chain |
| **Providers** | Model tiering | ✅ Complete | fast_model / premium_model per provider |
| **Security** | Default-deny policy | ✅ Complete | Network, filesystem, shell all deny by default |
| **Security** | Input sanitization | ✅ Complete | 13+ prompt injection patterns |
| **Security** | Secret detection & redaction | ✅ Complete | 9 credential patterns, output censoring |
| **Security** | Approval gate | ✅ Complete | Human-in-the-loop for sensitive ops |
| **Security** | Filesystem ACLs | ✅ Complete | Per-path read/write control |
| **Security** | Shell command allowlist | ✅ Complete | Whitelist with deny-all default |
| **Security** | Network allowlists | ✅ Complete | CIDR, domain, per-category host lists |
| **Memory** | SQLite with FTS5 | ✅ Complete | Full-text search, session-scoped |
| **Memory** | Cross-task learnings | ✅ Complete | TaskLearning persistence |
| **Memory** | Resilient store | ✅ Complete | Auto-fallback with repair |
| **Memory** | Context management | ✅ Complete | 7-tier token budget with pruning |
| **Scheduler** | Durable job scheduling | ✅ Complete | JSON persistence, cron + human parsing |
| **Scheduler** | Policy-routed execution | ✅ Complete | Scheduled jobs go through normal policy |
| **Scheduler** | Retry & timezone | ✅ Complete | Configurable retry, timezone-aware |
| **Channels** | CLI channel | ✅ Complete | Interactive stdin/stdout |
| **Channels** | Discord channel | ✅ Complete | WebSocket gateway, REST API |
| **Channels** | Webhook channel | ✅ Complete | HTTP ingress |
| **Channels** | Voice channel | ✅ Complete | WebSocket + STT/TTS + device pairing |
| **Discord** | DM support | ✅ Complete | Allowlist, pairing workflow |
| **Discord** | Guild channel support | ✅ Complete | Channel/role-based policies |
| **Discord** | Slash commands | ✅ Complete | /ask, /status, /model, /help |
| **Discord** | Bot message filtering | ✅ Complete | Ignores bot-authored by default |
| **Discord** | Anti-loop protection | ✅ Complete | Prevents response loops |
| **Discord** | Per-source capability modes | ✅ Complete | full, safe-chat, no-tools |
| **Discord** | Credential message deletion | ✅ Complete | Auto-deletes messages with secrets |
| **Discord** | Register commands CLI | ✅ Complete | Guild-specific and global |
| **Discord** | Status/probe/audit CLI | ✅ Complete | Diagnostics and audit trail |
| **Observability** | Audit logging | ✅ Complete | Structured JSONL |
| **Observability** | OpenTelemetry | ✅ Complete | Traces + metrics via OTLP |
| **Observability** | Config hot-reload | ✅ Complete | Watchdog-based file monitoring |
| **Tools** | Built-in tools | ✅ Complete | file_read/write/delete, shell_exec, web_fetch, calculator, etc. |
| **Tools** | Self-create tool | ✅ Complete | Runtime tool creation |
| **Tools** | MCP integration | ✅ Complete | External tool servers, namespaced |
| **Docs** | SECURITY.md | ✅ Complete | Threat model, policy docs |
| **Docs** | OPERATIONS.md | ✅ Complete | Operator guide |
| **Docs** | ARCHITECTURE.md | ✅ Complete | System design |
| **Docs** | CONFIG_REFERENCE.md | ✅ Complete | Full config schema |
| **Docs** | DISCORD.md | ✅ Complete | Discord setup and usage |
| **Docs** | TESTING.md | ✅ Complete | Test strategy and coverage |
| **Docs** | TROUBLESHOOTING.md | ✅ Complete | Common issues and fixes |

## Missing Capabilities (vs. OpenClaw)

| Priority | Capability | Gap Level | Notes |
|---|---|---|---|
| ~~P1~~ | ~~Docker/container sandbox for tool execution~~ | **Implemented (Session 3)** | DockerSandbox + FallbackSandbox with full security controls |
| ~~P2~~ | ~~Discord thread creation & thread-scoped sessions~~ | **Implemented (Session 3)** | create_thread(), thread-scoped sessions, auto-thread config |
| ~~P2~~ | ~~Doctor command — MCP, memory, watchdog checks~~ | **Implemented (Session 3)** | 15 checks: memory, MCP, watchdog, voice, checkpoints |
| ~~P3~~ | ~~Session friendly names~~ | **Implemented (Session 3)** | sessions table, list/rename CLI, name resolution |
| P3 | Discord multi-account support | **Not implemented** | Single bot token; OpenClaw supports multiple Discord accounts. |
| P3 | Interactive setup for Discord | **Partial** | Setup wizard handles API keys/OAuth but Discord config is YAML-only. |
| P4 | Web UI / dashboard | **Not implemented** | CLI-only operator interface. Intentionally deferred. |
| P4 | Telegram / Slack channels | **Not implemented** | Excluded by user decision. |

## Partially Implemented Capabilities

### Discord Thread Handling
- **What works:** Inbound thread messages are routed correctly, `send_to()` accepts `thread_id`, audit includes thread_id
- **What's missing:** No `create_thread()` method, no thread-scoped session isolation, no auto-thread for long conversations

### Doctor Command
- **What works:** Config, audit log, workspace, secrets dir, network policy, providers, shell, plugins, scheduler, Discord
- **What's missing:** MCP server health ping, memory store connectivity test, watchdog/hot-reload status, voice channel status

### Session Management
- **What works:** SQLite persistence, FTS5 search, history loading, `--session` flag for resume
- **What's missing:** Human-friendly session names, `missy sessions list` command, session metadata (title, created_at)

## Priority Order for Remaining Work

1. ~~**Docker sandbox**~~ — Implemented in Session 3
2. ~~**Discord thread support**~~ — Implemented in Session 3
3. ~~**Doctor enhancements**~~ — Implemented in Session 3
4. ~~**Session friendly names**~~ — Implemented in Session 3
5. **Discord multi-account** — Niche but relevant for multi-guild operators.
6. **Interactive Discord setup** — Streamline onboarding for Discord bot configuration.

## Intentionally Out of Scope

| Capability | Reason |
|---|---|
| Telegram channel | User decision — excluded from scope |
| Slack channel | User decision — excluded from scope |
| Web UI / dashboard | CLI-first philosophy; may be added later |
| Cloud control plane | Self-hosted only by design |
| Code from OpenClaw | Parity is behavioral, not code-level |

## Test Coverage

- **1029 tests passing** (as of session 3)
- Coverage threshold: 85% (configured in pyproject.toml)
- Test areas: agent, channels, cli, config, core, integration, memory, observability, plugins, policy, providers, scheduler, security, skills, tools, unit
