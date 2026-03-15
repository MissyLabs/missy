# OpenClaw Gap Analysis

Last updated: 2026-03-15 (Session 12)

## Currently Implemented Capabilities

| Category | Capability | Status | Notes |
|---|---|---|---|
| **Runtime** | Self-hosted local operation | ✅ Complete | Linux-native, systemd-ready |
| **Runtime** | Gateway/control-plane | ✅ Complete | PolicyHTTPClient, configurable host/port |
| **Runtime** | Agent loop with tool calling | ✅ Complete | Multi-step with iteration limits |
| **Runtime** | Multi-step planning & verification | ✅ Complete | DoneCriteria, compound task detection |
| **Runtime** | Retry/self-correction on tool failure | ✅ Complete | FailureTracker with strategy rotation |
| **Runtime** | Task checkpointing & recovery | ✅ Complete | SQLite-backed, 3-tier classification, recovery scan at startup, `missy recover` CLI |
| **Runtime** | Circuit breaker | ✅ Complete | Closed/Open/HalfOpen with exponential backoff |
| **Runtime** | Sub-agent spawning | ✅ Complete | SubAgentRunner |
| **Runtime** | Proactive task initiation | ✅ Complete | 4 trigger types (schedule, disk, load, file_change) |
| **Runtime** | Cost tracking & budgets | ✅ Complete | Per-model pricing, budget enforcement, **SQLite persistence**, per-session cost queries |
| **Runtime** | Docker sandbox | ✅ Complete | ShellExecTool routes through DockerSandbox when enabled |
| **Runtime** | Provider rate limiting | ✅ Complete | Token bucket (RPM + TPM), blocking acquire with timeout |
| **Runtime** | Response streaming | ✅ Complete | `run_stream()` yields tokens via provider `stream()` method |
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
| **CLI** | cost | ✅ Complete | Budget config display, **per-session cost breakdown with per-model detail** |
| **CLI** | recover | ✅ Complete | List incomplete checkpoints, abandon stale tasks |
| **Providers** | Anthropic | ✅ Complete | Claude models, API key rotation, **real streaming** |
| **Providers** | OpenAI | ✅ Complete | GPT-4o/4/3.5, o3/o4, OAuth/Codex |
| **Providers** | Ollama (local) | ✅ Complete | Zero-cost local models |
| **Providers** | Provider fallback | ✅ Complete | Registry with fallback chain |
| **Providers** | Model tiering | ✅ Complete | fast_model / premium_model per provider |
| **Providers** | Rate limiting | ✅ Complete | Token bucket with RPM/TPM budgets, blocking acquire |
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
| **Memory** | Cost history | ✅ Complete | Per-session cost records in SQLite `costs` table |
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

## Session 5 Improvements

1. **Provider rate limiting** — New `RateLimiter` class with token bucket algorithm. Enforces requests-per-minute (60 default) and tokens-per-minute (100k default) limits. Blocks with configurable timeout. Wired into AgentRuntime before every `complete()` and `complete_with_tools()` call.
2. **Cost persistence to SQLite** — New `costs` table in `memory.db`. `record_cost()` persists model, tokens, and USD cost per API call. `get_session_costs()` returns per-session breakdown. `get_total_costs()` returns cross-session aggregates. Runtime auto-persists after each provider call.
3. **Fixed `missy cost --session`** — Was crashing with `ImportError: cannot import name 'ResilientMemoryStore' from 'missy.memory.resilient_store'`. Now imports `SQLiteMemoryStore` correctly and displays per-model cost breakdowns.
4. **Response streaming** — `AgentRuntime.run_stream()` method yields text chunks from provider's `stream()` API. Falls back to full `run()` when tools are active (tool calls need complete responses).
5. **`missy recover` command** — Lists incomplete checkpoints with recommended recovery actions. `--abandon-all` flag clears all stale checkpoints. Shows checkpoint age, iteration count, and original prompt.
6. **44 new tests** — Rate limiter (18), cost persistence (11), streaming+integration (6), CLI commands (9). Total: 1097.

## Remaining Gaps (vs. OpenClaw)

| Priority | Capability | Gap Level | Notes |
|---|---|---|---|
| P3 | Discord multi-account support | **Not implemented** | Single bot token; OpenClaw supports multiple Discord accounts. Low demand — most operators run one bot. |
| P4 | Web UI / dashboard | **Not implemented** | CLI-only operator interface. Intentionally deferred — CLI-first philosophy. |
| P4 | Telegram / Slack channels | **Not implemented** | Excluded by user decision. |

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

- **4951 tests passing** (as of session 12)
- **99%+ code coverage** across 140 test files
- Test areas: agent, channels, cli, config, core, integration, memory, observability, plugins, policy, providers, scheduler, security, skills, tools, unit, mcp
- Property-based tests (hypothesis), security fuzz tests, stress tests, end-to-end integration tests
