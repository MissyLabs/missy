# OpenClaw Gap Analysis

Last updated: 2026-03-12 (Session 4)

## Currently Implemented Capabilities

| Category | Capability | Status | Notes |
|---|---|---|---|
| **Runtime** | Self-hosted local operation | ✅ Complete | Linux-native, systemd-ready |
| **Runtime** | Gateway/control-plane | ✅ Complete | PolicyHTTPClient, configurable host/port |
| **Runtime** | Agent loop with tool calling | ✅ Complete | Multi-step with iteration limits |
| **Runtime** | Multi-step planning & verification | ✅ Complete | DoneCriteria, compound task detection |
| **Runtime** | Retry/self-correction on tool failure | ✅ Complete | FailureTracker with strategy rotation |
| **Runtime** | Task checkpointing & recovery | ✅ Complete | SQLite-backed, 3-tier classification, recovery scan at startup |
| **Runtime** | Circuit breaker | ✅ Complete | Closed/Open/HalfOpen with exponential backoff |
| **Runtime** | Sub-agent spawning | ✅ Complete | SubAgentRunner |
| **Runtime** | Proactive task initiation | ✅ Complete | 4 trigger types (schedule, disk, load, file_change) |
| **Runtime** | Cost tracking & budgets | ✅ Complete | Per-model pricing, budget enforcement in tool loop, audit event on breach |
| **Runtime** | Docker sandbox | ✅ Complete | ShellExecTool routes through DockerSandbox when enabled |
| **CLI** | init/setup | ✅ Complete | Wizard with OAuth (OpenAI PKCE + Anthropic), Discord setup |
| **CLI** | ask / run (REPL) | ✅ Complete | Single-turn and interactive modes |
| **CLI** | providers list | ✅ Complete | Shows configured providers & availability |
| **CLI** | skills / plugins list | ✅ Complete | Lists registered skills and plugins |
| **CLI** | doctor | ✅ Complete | 15+ health checks |
| **CLI** | schedule add/list/pause/resume/remove | ✅ Complete | APScheduler with cron parsing |
| **CLI** | audit security / recent | ✅ Complete | Structured JSONL audit trail |
| **CLI** | vault set/get/list/delete | ✅ Complete | ChaCha20-Poly1305 encrypted store |
| **CLI** | sessions cleanup/list/rename | ✅ Complete | TTL-based pruning, friendly names |
| **CLI** | approvals list | ✅ Complete | Human-in-the-loop approval gate |
| **CLI** | patches list/approve/reject | ✅ Complete | Self-tuning prompt patches |
| **CLI** | mcp list/add/remove | ✅ Complete | MCP server management |
| **CLI** | devices list/pair/unpair/status/policy | ✅ Complete | Voice edge node management |
| **CLI** | voice status/test | ✅ Complete | STT/TTS diagnostics |
| **CLI** | gateway start/status | ✅ Complete | Gateway lifecycle |
| **CLI** | cost | ✅ Complete | Budget config display, session cost summary |
| **Providers** | Anthropic | ✅ Complete | Claude models, API key rotation |
| **Providers** | OpenAI | ✅ Complete | GPT-4o/4/3.5, o3/o4, OAuth/Codex |
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
| **Security** | Docker sandbox for tools | ✅ Complete | Container isolation with --cap-drop=ALL, read-only root |
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
| **Discord** | Thread management | ✅ Complete | Thread-scoped sessions, auto-thread |
| **Discord** | Register commands CLI | ✅ Complete | Guild-specific and global |
| **Discord** | Status/probe/audit CLI | ✅ Complete | Diagnostics and audit trail |
| **Discord** | Interactive setup wizard | ✅ Complete | Bot token, DM policy, guild policies |
| **Observability** | Audit logging | ✅ Complete | Structured JSONL |
| **Observability** | OpenTelemetry | ✅ Complete | Traces + metrics via OTLP |
| **Observability** | Config hot-reload | ✅ Complete | Watchdog-based file monitoring |
| **Tools** | Built-in tools | ✅ Complete | file_read/write/delete, shell_exec, web_fetch, calculator, browser, etc. |
| **Tools** | Self-create tool | ✅ Complete | Runtime tool creation |
| **Tools** | MCP integration | ✅ Complete | External tool servers, namespaced |
| **Docs** | SECURITY.md | ✅ Complete | Threat model, policy docs |
| **Docs** | OPERATIONS.md | ✅ Complete | Operator guide |
| **Docs** | ARCHITECTURE.md | ✅ Complete | System design |
| **Docs** | CONFIG_REFERENCE.md | ✅ Complete | Full config schema |
| **Docs** | DISCORD.md | ✅ Complete | Discord setup and usage |
| **Docs** | TESTING.md | ✅ Complete | Test strategy and coverage |
| **Docs** | TROUBLESHOOTING.md | ✅ Complete | Common issues and fixes |

## Remaining Gaps (vs. OpenClaw)

| Priority | Capability | Gap Level | Notes |
|---|---|---|---|
| P3 | Discord multi-account support | **Not implemented** | Single bot token; OpenClaw supports multiple Discord accounts. Low demand — most operators run one bot. |
| P4 | Web UI / dashboard | **Not implemented** | CLI-only operator interface. Intentionally deferred — CLI-first philosophy. |
| P4 | Telegram / Slack channels | **Not implemented** | Excluded by user decision. |

## Session 4 Improvements

1. **ShellExecTool sandbox integration** — Commands now routed through Docker sandbox when `sandbox.enabled: true` in config. Previously the sandbox was built but ShellExecTool always used direct subprocess.
2. **Budget enforcement** — `CostTracker.check_budget()` called after every provider response in the agent tool loop. Emits `agent.budget.exceeded` audit event. `max_spend_usd` config field flows from `config.yaml` → `MissyConfig` → `AgentConfig` → `CostTracker`.
3. **Checkpoint recovery scan** — `scan_for_recovery()` called at `AgentRuntime.__init__()`. Incomplete tasks from previous runs are surfaced via `pending_recovery` property. CLI `missy run` displays resumable/restartable tasks at startup.
4. **Cost CLI command** — `missy cost` shows budget configuration and usage hint.
5. **Wizard step numbering fix** — Steps now consistently numbered 1-5.
6. **24 new tests** — ShellExecTool sandbox routing (11), budget enforcement + checkpoint recovery + config parsing (13). Total: 1053.

## Intentionally Out of Scope

| Capability | Reason |
|---|---|
| Telegram channel | User decision — excluded from scope |
| Slack channel | User decision — excluded from scope |
| Web UI / dashboard | CLI-first philosophy; may be added later |
| Cloud control plane | Self-hosted only by design |
| Code from OpenClaw | Parity is behavioral, not code-level |
| Discord multi-account | Low demand; single bot token covers 99% of use cases |

## Test Coverage

- **1053 tests passing** (as of session 4, up from 1029)
- Coverage threshold: 85% (configured in pyproject.toml)
- Test areas: agent, channels, cli, config, core, integration, memory, observability, plugins, policy, providers, scheduler, security, skills, tools, unit
