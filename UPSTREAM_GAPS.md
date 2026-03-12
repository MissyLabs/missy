# Missy — Upstream Gap Analysis

Comparison of Missy against its two upstream inspirations:
- **SkyClaw** — Rust, 15-crate workspace, ~59,100 LOC, autonomous agentic loop
- **OpenClaw** — TypeScript, monorepo, 52+ bundled skills, 23+ channels, production assistant platform

Updated: 2026-03-12.

Gaps are grouped by priority. Features that are platform-specific (iOS/Android companion apps, macOS-only integrations, browser extension relay) are excluded as out-of-scope for a Linux Python implementation.

---

## Critical — Core Agent Functionality

### 1. Multi-step agentic loop (SkyClaw) ✅ Implemented

`AgentRuntime._tool_loop()` in `missy/agent/runtime.py` implements the full iterative loop:
- Calls `provider.complete_with_tools()` via the circuit breaker each iteration.
- Executes tool calls and feeds results back as messages.
- Injects a verification prompt (from `agent/done_criteria.py`) after each tool round.
- Respects `max_iterations` (default 10) with a final fallback completion when the limit is reached.
- Falls back to single-turn mode when no tools are registered or `max_iterations == 1`.

### 2. Tool invocation in the agent loop (both) ✅ Implemented

`ToolRegistry` is called by `AgentRuntime._get_tools()` and `_execute_tool()` on every iteration.

Built-in tools implemented in `missy/tools/builtin/`:
- `file_read`, `file_write`, `file_delete`, `list_files`
- `shell_exec` — policy-checked via `ShellPolicyEngine`
- `web_fetch` — policy-checked via `PolicyHTTPClient`
- `calculator`
- `self_create_tool` — agent-authored persistent tools

### 3. Streaming responses (both) ✅ Implemented

All three providers (`AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`) expose a `stream()` async generator method. `AgentRuntime` and channel implementations consume it for progressive token delivery.

### 4. Function/tool calling protocol (both) ✅ Implemented

`BaseProvider` defines `ToolCall` / `ToolResult` dataclasses and `complete_with_tools()`.
- `AnthropicProvider` and `OpenAIProvider` use native function-calling APIs.
- `OllamaProvider` uses a prompted fallback: schemas injected into system prompt, JSON parsed from model output.
- `AgentRuntime._tool_loop()` processes `finish_reason == "tool_calls"` and loops.

### 5. MCP (Model Context Protocol) integration (SkyClaw) ✅ Implemented

- `missy/mcp/client.py` — `McpClient` connects to stdio subprocess or HTTP MCP servers.
- `missy/mcp/manager.py` — `McpManager` hot-loads servers, auto-namespaces tools (`server__tool`), persists config to `~/.missy/mcp.json`, restarts dead servers via `health_check()`.
- CLI: `missy mcp list/add/remove`.

---

## High — Resilience and Context Management

### 6. Circuit breaker for provider failures (SkyClaw) ✅ Implemented

`missy/agent/circuit_breaker.py` — full Closed → Open → HalfOpen state machine:
- Configurable failure threshold (default 5), base timeout 60 s, max timeout 300 s.
- Exponential backoff: recovery timeout doubles on HalfOpen probe failure.
- Thread-safe via `threading.Lock`.
- Integrated into `AgentRuntime` as `self._circuit_breaker`; wraps every provider call.

### 7. Retry / self-correction on tool failure (SkyClaw) ❌ Remaining

`FailureTracker` per-tool failure counter and strategy-rotation prompt injection are not implemented. Tool failures return an error result and the agent loop continues, but there is no threshold-based strategy-rotation prompt ("analyze why this failed, list 3 alternatives").

### 8. Task checkpointing and startup recovery (SkyClaw) ❌ Remaining

No checkpoint is written after each tool round. Interrupted tasks are lost. No `RecoveryManager` scans for incomplete tasks on startup.

### 9. Context window management (SkyClaw) ✅ Implemented

`missy/agent/context.py` — `ContextManager` with 7-tier token budget:
1. System prompt (always)
2. Tool definitions
3. Task state / DONE criteria
4. Recent N messages (always kept)
5. Memory search results (≤15% of budget)
6. Cross-task learnings (≤5% of budget)
7. Older history (fills remainder, newest-first)

History pruning assigns importance levels to messages, preserves atomic tool+result pairs, and drops oldest low-importance messages first. Called by `AgentRuntime._build_context_messages()`.

### 10. Cost tracking and budget limits (SkyClaw) ❌ Remaining

No per-token cost tracking, no pricing table in `ProviderConfig`, no `max_spend_usd` enforcement. Token usage is not accumulated across runs.

### 11. Resilient memory with failover (SkyClaw) ✅ Implemented

`missy/memory/resilient.py` — `ResilientMemoryStore` wraps the primary backend with an in-memory dict cache. On primary I/O failure all operations fall back to the cache transparently; auto-triggers repair after consecutive failures; syncs cache back to primary on recovery.

### 12. Watchdog / subsystem health monitor (SkyClaw) ✅ Implemented

`missy/agent/watchdog.py` — background task that runs on a configurable interval (default 60 s), monitors provider/memory/channel subsystems, tracks consecutive failures, logs state transitions, and emits health audit events. Integrated into `gateway start`.

---

## High — Missing Built-in Tools

### 13. File operation tools ✅ Implemented

`missy/tools/builtin/file_read.py`, `file_write.py`, `file_delete.py`, `list_files.py` — each checks `FilesystemPolicyEngine` before operating.

### 14. Shell execution tool ✅ Implemented

`missy/tools/builtin/shell_exec.py` — passes commands through `ShellPolicyEngine.check_command()` before executing via `subprocess` with timeout and output cap.

### 15. Web fetch tool ✅ Implemented

`missy/tools/builtin/web_fetch.py` — issues GET via `PolicyHTTPClient` after `NetworkPolicyEngine` approval.

---

## High — Intelligence Features

### 16. DONE criteria engine for compound tasks (SkyClaw) ✅ Implemented

`missy/agent/done_criteria.py` — detects compound tasks, generates verifiable DONE conditions, and provides `make_verification_prompt()` injected by `AgentRuntime._tool_loop()` after each tool round.

### 17. Cross-task learning (SkyClaw) ✅ Implemented

`missy/agent/learnings.py` — `extract_learnings()` runs post-task (called by `AgentRuntime._record_learnings()` after tool-augmented runs). `TaskLearning` objects are persisted via `SQLiteMemoryStore.save_learning()`. `ContextManager` retrieves relevant lessons and injects them at ≤5% of token budget.

### 18. Prompt self-tuning / patch system (SkyClaw) ✅ Implemented

`missy/agent/prompt_patches.py` — `PromptPatchManager` with propose/review/approve/expire lifecycle (ToolUsageHint, ErrorAvoidance, WorkflowPattern, DomainKnowledge, StylePreference). Tracks per-patch success rate; auto-expires underperforming patches.
CLI: `missy patches list/approve/reject`.

### 19. Tiered model routing (SkyClaw) ✅ Implemented

`missy/providers/registry.py` — `ModelRouter` routes to `fast` / `primary` / `premium` tiers based on message length, tool count, complexity keywords (debug/architect/refactor → premium), and history length. User prefix overrides (`!fast`, `!best`) supported.

### 20. Multiple API key rotation (SkyClaw) ✅ Implemented

`missy/providers/registry.py` — providers accept `api_keys: list[str]`; atomic round-robin rotation on 429/401 responses via `rotate_key()`.

---

## Medium — Execution and Security

### 21. Execution approval flow (OpenClaw) ✅ Implemented

`missy/agent/approval.py` — `ApprovalGate` pauses execution on `ApprovalRequiredError`, presents the action to the user/channel, waits for approve/deny, and resumes. Per-agent glob allowlists for pre-approved commands. Discord integration via `approve`/`deny` slash commands.
CLI: `missy approvals list`.

### 22. Vault / encrypted secrets store (SkyClaw) ✅ Implemented

`missy/security/vault.py` — `Vault` class using ChaCha20-Poly1305 (via `cryptography` package). Supports `vault://key-name` URI scheme for config references, resolved by the config loader. Key stored at `~/.missy/secrets/vault.key` (mode 0o600).
CLI: `missy vault set/get/list/delete`.

### 23. Outbound secret censoring (SkyClaw) ✅ Implemented

`missy/security/censor.py` — wraps channel `send()` with redaction of known API key patterns before delivery. All `agent.run()` responses are post-processed through `secrets_detector.redact()` before being printed to console or sent to a channel.

### 24. Docker sandbox for isolated task execution (OpenClaw) ❌ Remaining

No Docker/Podman wrapper for sub-task or scheduled job execution. No `SandboxConfig` in `MissyConfig`. No process isolation for shell or agent execution.

### 25. Credential detection and deletion on inbound messages (SkyClaw) ❌ Remaining

`SecretsDetector` flags credentials in inbound Discord messages but does not delete the message from the channel. No `rest.delete_message()` call on detection.

---

## Medium — Scheduler Enhancements

### 26. Raw cron expression support (OpenClaw) ✅ Implemented

`missy/scheduler/parser.py` — `_RAW_CRON_PATTERN` matches 5- or 6-field cron expressions and passes them to `CronTrigger.from_crontab()` via the `_cron_expression` key in the trigger config.

### 27. Timezone support in scheduler (OpenClaw) ✅ Implemented

`parse_schedule()` accepts an optional `tz: str` IANA timezone parameter. When supplied it is attached as `"timezone"` to cron and date trigger configs and forwarded to APScheduler.
CLI: `missy schedule add --tz IANA_TIMEZONE`.

### 28. Job retry on failure (OpenClaw/SkyClaw) ✅ Implemented

`missy/scheduler/jobs.py` and `missy/scheduler/manager.py` — `ScheduledJob` carries `max_attempts` (default 3) and `backoff_seconds` array (30, 60, 300). `SchedulerManager._run_job()` retries on failure with the configured backoff.

### 29. One-shot future-dated jobs (OpenClaw) ✅ Implemented

`missy/scheduler/parser.py` — `_AT_PATTERN` matches `"at YYYY-MM-DD HH:MM"` / `"at YYYY-MM-DDTHH:MM"` and returns `{"trigger": "date", "run_date": ...}` for APScheduler's `DateTrigger`.

### 30. Active-hours window for heartbeat/scheduler (SkyClaw) ✅ Implemented

`active_hours: str` field in `SchedulingPolicy` (format `"HH:MM-HH:MM"`). `missy/scheduler/jobs.py` and `manager.py` check active hours before executing jobs; jobs outside the window are skipped and rescheduled.

---

## Medium — Memory and Context

### 31. Semantic / full-text memory search (OpenClaw/SkyClaw) ✅ Implemented

`missy/memory/sqlite_store.py` — `SQLiteMemoryStore` with FTS5 virtual table `turns_fts`. `search(query, limit, session_id)` method supports FTS5 prefix, phrase, and boolean operators. `ContextManager` injects relevant memories into each agent call at ≤15% of token budget.

Full vector/semantic search (embedding providers, MMR re-ranking, hybrid search) is not implemented — only keyword FTS is available. This makes this item partial relative to OpenClaw's full implementation.

### 32. Session compaction / history summarization (OpenClaw) ✅ Implemented

`missy/memory/store.py` — `compact_session(session_id)` summarizes old history using the LLM when token count exceeds threshold. Called by `ContextManager` when context pressure is detected.

### 33. Sub-agent / task decomposition (SkyClaw/OpenClaw) ✅ Implemented

`missy/agent/sub_agent.py` — `SubAgentRunner` spawns isolated sub-agents with scoped tool subsets. Hard cap of 10 sub-agents / 3 concurrent, 300 s timeout, no recursive delegation. `spawn_agent(task, tools, timeout)` tool is callable from the primary agent.

---

## Medium — Heartbeat and Proactive Features

### 34. Heartbeat system (SkyClaw/OpenClaw) ✅ Implemented

`missy/agent/heartbeat.py` — background loop in `gateway start`. `HeartbeatConfig` in `MissyConfig` (enabled, interval_seconds, active_hours). Reads `HEARTBEAT.md` on each tick and sends it as a synthetic task to the agent. `HEARTBEAT_OK` suppression file prevents redundant runs.

### 35. Proactive task initiation (SkyClaw) ❌ Remaining

`ProactiveManager` with file-change triggers (`watchdog` package), webhook triggers, and threshold triggers is not implemented. Tasks originate only from user input or the scheduler.

---

## Low — Additional Channels

### 36. Telegram channel ❌ Excluded

Excluded from Missy by user decision. No `TelegramChannel` implementation is planned.

### 37. Slack channel ❌ Excluded

Excluded from Missy by user decision. No `SlackChannel` implementation is planned.

### 38. Webhook channel (OpenClaw) ✅ Implemented

`missy/channels/webhook.py` — `WebhookChannel` provides an HTTP listener for inbound webhooks. Shared secret validation enforced. Policy gating applied to all inbound requests. Activated via `gateway start`.

---

## Low — Developer Experience and Operations

### 39. OpenTelemetry integration (SkyClaw/OpenClaw) ✅ Implemented

`missy/observability/otel.py` — `init_otel(cfg)` initialises `opentelemetry-sdk` and `opentelemetry-exporter-otlp`. `OtelExporter` wraps `AuditLogger`. Config: `observability.otel_enabled` / `otel_endpoint` in `config.yaml`. Disabled by default.

### 40. Config hot-reload (OpenClaw) ✅ Implemented

`missy/config/hotreload.py` — `watchdog`-based file watcher on `config.yaml`. Debounced reload re-applies config to `PolicyEngine`, `ProviderRegistry`, and `SchedulerManager` without restart.

### 41. Multiple API keys per provider (SkyClaw) ✅ Implemented

`ProviderConfig` accepts `api_keys: list[str]`. Round-robin rotation with atomic index advance on 429/401 responses, implemented in `missy/providers/registry.py`.

### 42. Agent-authored persistent custom tools (SkyClaw) ✅ Implemented

`missy/tools/builtin/self_create_tool.py` — `SelfCreateTool` writes bash/Python scripts to `~/.missy/custom-tools/`. `ToolRegistry` dynamically re-registers them on next agent run, making them persistent across restarts.

### 43. Failure alert routing for scheduled jobs (OpenClaw) ✅ Implemented

`missy/scheduler/manager.py` — consecutive failure counter in `ScheduledJob`. After a configurable threshold of consecutive failures, an alert is routed to the configured Discord channel (and emitted as an audit event).

### 44. Session cleanup CLI (OpenClaw) ✅ Implemented

`missy sessions cleanup [--dry-run] [--before DAYS]` — TTL-based purge in `SQLiteMemoryStore.cleanup()`. Default retention: 30 days. `--dry-run` reports what would be deleted without acting.

---

## Additional Capabilities Not in Original 44 Gaps

These were implemented beyond the original gap list:

### Onboarding wizard ✅ Implemented

`missy/cli/wizard.py` — `missy setup` guides the user through workspace selection, provider configuration (Anthropic, OpenAI, Ollama), API key entry with masking and env-var detection, model tier selection, connectivity verification, and atomic `config.yaml` write. Includes vault and OAuth paths.

### OpenAI OAuth PKCE flow ✅ Implemented

`missy/cli/oauth.py` — full PKCE S256 OAuth 2.0 flow against `auth.openai.com`. Local callback server on port 1455 with headless/SSH tunnel fallback. JWT parsing extracts `account_id` and email. Tokens persisted at `~/.missy/secrets/openai-oauth.json` (mode 0o600) with automatic refresh support.

### Anthropic setup-token flow ✅ Implemented

`missy/cli/anthropic_auth.py` — paste flow for Claude Code setup-tokens (`sk-ant-oat01-...`). Mandatory ToS acknowledgement prompt (Anthropic updated ToS on 2026-02-19 to prohibit third-party use of setup-tokens). Token classification (API key vs. setup-token), expiry tracking, vault storage option. `get_current_token()` for runtime resolution (env → token file → vault).

### Voice channel ✅ Implemented

`missy/channels/voice/` — full implementation including:
- WebSocket server for edge node communication
- STT (speech-to-text) and TTS (text-to-speech) pipeline
- `DeviceRegistry` (`channels/voice/registry.py`) — persistent JSON registry of edge nodes with PBKDF2-HMAC-SHA256 token auth, pairing/approval workflow, sensor data (occupancy, noise level), audio log retention, and atomic writes
- Per-device `policy_mode` (`full`, `safe-chat`, `muted`)
- CLI: `missy devices list/pair/unpair/status/policy` and `missy voice status/test`

---

## Out of Scope (excluded from this gap list)

The following OpenClaw features are excluded as they require platform infrastructure not applicable to a local Linux Python CLI:

- iOS/Android companion apps (Swabble, node devices)
- macOS-specific integrations (Apple Notes, Reminders, iMessage native)
- Browser extension relay / Chrome/Chromium management
- TUI (Terminal UI) — desirable but not required
- Canvas host (web UI)
- ClawHub package registry
- ACP (Agent Control Protocol) full implementation
- QR code pairing flow
- mDNS/Bonjour gateway discovery
- Tailscale Funnel integration
- SSH tunnel transport

---

## Summary by Status

| Status | Count | Items |
|---|---|---|
| ✅ Implemented | 35 | #1-6, #9, #11-23, #26-34, #38-44 + wizard, OAuth, Anthropic auth, voice |
| ⚠️ Partial | 1 | #31 (FTS search implemented; vector/semantic search not) |
| ❌ Remaining | 6 | #7, #8, #10, #24, #25, #35 |
| ❌ Excluded | 2 | #36 (Telegram), #37 (Slack) — user decision |

**Original 44 gaps: 35 implemented, 1 partial, 6 remaining, 2 excluded by design.**
