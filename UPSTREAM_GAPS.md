# Missy — Upstream Gap Analysis

Comparison of Missy against its two upstream inspirations:
- **SkyClaw** — Rust, 15-crate workspace, ~59,100 LOC, autonomous agentic loop
- **OpenClaw** — TypeScript, monorepo, 52+ bundled skills, 23+ channels, production assistant platform

Gaps are grouped by priority. Features that are platform-specific (iOS/Android companion apps, macOS-only integrations, browser extension relay) are excluded as out-of-scope for a Linux Python implementation.

---

## Critical — Core Agent Functionality

These gaps make Missy a single-turn chatbot rather than an autonomous agent.

### 1. Multi-step agentic loop (SkyClaw)

**What's missing:** The `AgentRuntime.run()` executes exactly one provider call and returns. Neither iterative tool use nor a think→act→verify loop exists. `max_iterations` is accepted but never used.

SkyClaw's loop:
- Classify message (chat / order / stop) in one combined LLM call
- If "order": immediate acknowledgement → enter tool loop
- Each tool round injects a verification prompt before the next step
- Loop continues until DONE criteria satisfied or iteration limit hit
- Difficulty tiers (Simple → 2 rounds, Standard → 5, Complex → 10)

**Required additions:**
- Iteration control driven by `max_iterations` in `AgentConfig`
- Tool invocation inside the loop (the tool registry exists but is never called by the runtime)
- Verification injection after tool calls
- Classification gate (chat / action / stop) — "stop" should cancel the active task

### 2. Tool invocation in the agent loop (both)

**What's missing:** `ToolRegistry` is fully implemented and policy-enforced, but `AgentRuntime` never calls it. Tools are never executed.

SkyClaw has 13 built-in tools: `shell`, `file_read`, `file_write`, `file_delete`, `web_fetch`, `git`, `browser`, `send_message`, `send_file`, `check_messages`, `memory_manage`, `key_manage`, `self_create_tool`.

**Required built-in tools for Missy (minimum viable set):**
- `file_read` — reads a file (requires filesystem_read policy)
- `file_write` — writes a file (requires filesystem_write policy)
- `shell_exec` — runs a whitelisted command (requires shell policy)
- `web_fetch` — HTTP GET via PolicyHTTPClient (requires network policy)
- `list_files` — lists directory contents

### 3. Streaming responses (both)

**What's missing:** All providers return a single completed string. Both upstreams stream partial tokens to the channel as they arrive, giving immediate feedback.

SkyClaw's `StreamBuffer` flushes on a configurable interval (default 1000ms) and edits messages in place on Telegram.

OpenClaw streams with configurable coalescing and chunking per channel.

**Required:** Async generator interface on `BaseProvider.complete()`, consumed by `AgentRuntime` and forwarded to the active channel's `send()`.

### 4. Function/tool calling protocol (both)

**What's missing:** Missy's providers return plain text. Neither structured tool-call parsing (Anthropic/OpenAI native function calling) nor a fallback prompted tool calling scheme is implemented.

SkyClaw implements both:
- Native function calling for providers that support it
- Prompted tool calling fallback: tool schemas injected into system prompt; model text parsed for `{"response": "...", "tool_call": {...}}` JSON

**Required:** `BaseProvider.complete()` must be able to return both a text response and a list of tool calls. The agent loop processes tool calls before continuing.

### 5. MCP (Model Context Protocol) integration (SkyClaw)

**What's missing:** No MCP support. SkyClaw's `McpManager` hot-loads any stdio or HTTP MCP server, auto-namespaces tools across servers, and lets the agent add/remove servers at runtime via `mcp_manage` and `self_add_mcp` tools.

**Required:** `McpManager` class connecting to stdio subprocess or HTTP MCP servers; tool discovery and injection into `ToolRegistry`; `/mcp list/add/remove` CLI or CLI commands.

---

## High — Resilience and Context Management

### 6. Circuit breaker for provider failures (SkyClaw)

**What's missing:** Provider errors are caught and re-raised. There is no backoff, cooldown, or circuit-breaker pattern.

SkyClaw implements: Closed → Open → HalfOpen state machine, configurable failure threshold (default 5), exponential backoff with deterministic jitter (±25%), doubles recovery timeout on HalfOpen failure, caps at 5 minutes.

**Required:** `CircuitBreaker` wrapper around each provider in the registry, consulted before each call.

### 7. Retry / self-correction on tool failure (SkyClaw)

**What's missing:** If a tool call fails the agent returns an error. SkyClaw's `FailureTracker` counts per-tool failures; after N failures it injects a strategy-rotation prompt: "analyze why this failed, list 3 alternatives, execute the best one."

**Required:** Per-tool failure counter in the agent loop; strategy injection prompt after threshold.

### 8. Task checkpointing and startup recovery (SkyClaw)

**What's missing:** If Missy is interrupted mid-task (crash, kill) the task is lost with no record.

SkyClaw serializes session history to SQLite's `TaskQueue` after every tool round. On startup, `RecoveryManager` scans for incomplete tasks and classifies as Resume (valid checkpoint), Restart (no checkpoint), or Abandon (>24h old).

**Required:** Checkpoint write after each tool round; startup scan in `AgentRuntime.__init__` or `gateway start`.

### 9. Context window management (SkyClaw)

**What's missing:** There is no token counting, no history pruning, and no truncation. Long sessions will silently exceed provider context limits.

SkyClaw's 7-tier token budget:
1. System prompt (always)
2. Tool definitions
3. Task state / DONE criteria
4. Recent N messages (always kept)
5. Memory search results (≤15% of budget)
6. Cross-task learnings (≤5% of budget)
7. Older history (fills remainder, newest-first)

History pruning assigns 5 importance levels to messages, preserves atomic tool+result pairs, and drops oldest low-importance messages first.

OpenClaw implements session compaction (history summarization by LLM) with `postCompactionSections` for selective preservation.

**Required:** Token counting on messages, `ContextManager` applying budget rules before each provider call.

### 10. Cost tracking and budget limits (SkyClaw)

**What's missing:** No per-token cost tracking. SkyClaw has a full pricing table (cost per 1M input/output tokens for each model), tracks spend per task, and enforces `max_spend_usd` limits.

**Required:** Pricing table in `ProviderConfig`, spend accumulator in `AgentRuntime`, enforcement check before each provider call.

### 11. Resilient memory with failover (SkyClaw)

**What's missing:** `MemoryStore` reads/writes a JSON file directly. Any I/O error raises an exception and the store is unavailable.

SkyClaw's `ResilientMemory`: wraps primary backend with in-memory `dict` cache; on primary failure, all operations fall back to cache transparently; auto-triggers repair after N consecutive failures; syncs cache back to primary on recovery.

**Required:** `ResilientMemoryStore` wrapper with in-memory fallback.

### 12. Watchdog / subsystem health monitor (SkyClaw)

**What's missing:** `missy doctor` runs a one-shot check. There is no ongoing background health monitoring.

SkyClaw's `Watchdog` runs on a configurable interval (default 60s), monitors provider/memory/channel subsystems, tracks consecutive failures, logs transitions (info→warn→error), and recommends shutdown if health is critical.

**Required:** Background task in `gateway start` that periodically checks all subsystems and emits health audit events.

---

## High — Missing Built-in Tools

### 13. File operation tools

Neither `file_read` nor `file_write` nor `file_delete` tools exist, only the policy engines that would gate them. The agent cannot read or write files.

**Required:** `FileReadTool`, `FileWriteTool`, `FileDeleteTool`, `ListFilesTool` in `missy/tools/builtin/` — each checks `FilesystemPolicyEngine` before operating.

### 14. Shell execution tool

`ShellPolicy` exists and is fully implemented. No tool actually runs a shell command.

**Required:** `ShellExecTool` that passes the command through `ShellPolicyEngine.check_command()` before executing via `subprocess` with timeout and output cap.

### 15. Web fetch tool

`PolicyHTTPClient` exists. No tool exposes it to the agent.

**Required:** `WebFetchTool` that issues a GET via `create_client()` after `NetworkPolicyEngine` approval.

---

## High — Intelligence Features

### 16. DONE criteria engine for compound tasks (SkyClaw)

**What's missing:** The agent has no mechanism to determine when a multi-step task is complete.

SkyClaw: detects compound tasks (numbered lists, multiple imperatives, "and/then" sequences) heuristically; injects a "define DONE conditions" prompt before execution; tracks verifiable conditions; re-prompts if any criterion unmet; continues loop until all verified.

**Required:** `DoneCriteria` class, compound-task detector, verification injection in the agent loop.

### 17. Cross-task learning (SkyClaw)

**What's missing:** Missy forgets everything between sessions. SkyClaw extracts structured learnings after each task (task_type, approach, outcome, lesson) and injects relevant ones (≤5% of token budget) into future tasks.

**Required:** `extract_learnings()` post-task, `TaskLearning` storage in `MemoryStore`, retrieval and injection in `ContextManager`.

### 18. Prompt self-tuning / patch system (SkyClaw)

**What's missing:** No mechanism to improve the system prompt based on observed outcomes.

SkyClaw proposes prompt patches after tasks (ToolUsageHint, ErrorAvoidance, WorkflowPattern, DomainKnowledge, StylePreference), requires user approval via `/patches` before activation, tracks per-patch success rate, and auto-expires underperforming patches.

**Required:** `PromptPatchManager` with propose/review/approve/expire lifecycle; `/patches` CLI command.

### 19. Tiered model routing (SkyClaw)

**What's missing:** Provider selection is first-configured or CLI-specified. No dynamic routing based on task complexity.

SkyClaw routes to `fast` / `primary` / `premium` tiers based on message length, tool count, keywords (debug/architect/refactor → premium), and history length. Users can override with `!fast` or `!best` prefixes.

**Required:** `ModelRouter` in `AgentRuntime`; tier configuration in `ProviderConfig`; message complexity scoring; user prefix overrides.

### 20. Multiple API key rotation (SkyClaw)

**What's missing:** Each provider takes a single `api_key`. SkyClaw accepts a list of keys and rotates on rate-limit or auth errors.

**Required:** `keys: list[str]` in `ProviderConfig`; atomic round-robin on 429/401 responses.

---

## Medium — Execution and Security

### 21. Execution approval flow (OpenClaw)

**What's missing:** `ApprovalRequiredError` was added but there is no mechanism to pause execution, present the action to the user, wait for approval, and resume.

OpenClaw's approval system: request → 15s grace period → user approves/denies → `consumeAllowOnce()` replay protection. Supports per-agent glob allowlists for pre-approved commands. Approval forwarding to Discord/Telegram with `approve`/`deny` slash commands.

**Required:** `ApprovalGate` class; agent loop yields on `ApprovalRequiredError`; channel receives approval prompt; `missy approvals` CLI commands.

### 22. Vault / encrypted secrets store (SkyClaw)

**What's missing:** `~/.missy/secrets/` directory exists (created by `init`) but has no encryption or structured access. SkyClaw uses ChaCha20-Poly1305 with a `vault://` URI scheme for secret references in config.

**Required:** `VaultManager` with ChaCha20Poly1305 encryption (via `cryptography` package already in deps); `vault://key-name` resolution in config loader; `key_manage` tool.

### 23. Outbound secret censoring (SkyClaw)

**What's missing:** `SecretsDetector` detects secrets in inbound prompts. Nothing scans outbound responses before they are sent to the channel.

SkyClaw wraps every channel's `send()` with `SecretCensorChannel` which redacts known API keys from the text before delivery.

**Required:** Post-process all `agent.run()` responses through `secrets_detector.redact()` before passing to the channel or printing to console.

### 24. Docker sandbox for isolated task execution (OpenClaw)

**What's missing:** SkyClaw has `security.sandbox = "mandatory"`. OpenClaw has full Docker lifecycle management for non-main agent sessions. Missy has no process isolation.

OpenClaw's sandbox config: `image`, `readOnlyRoot`, `capDrop`, `seccompProfile`, `memory`, `cpus`, `pidsLimit`, `network`, `tmpfs`, `binds`.

**Required:** Optional Docker/Podman wrapper for sub-task or scheduled job execution; `SandboxConfig` in `MissyConfig`; `missy sandbox list/explain` commands.

### 25. Credential detection and deletion on inbound messages (SkyClaw)

**What's missing:** Credentials detected in Discord messages are warned about but not deleted from the channel.

SkyClaw deletes the message from the channel after reading it to prevent the key from sitting in chat history.

**Required:** When `SecretsDetector.has_secrets(content)` is true on an inbound Discord message, call `rest.delete_message(channel_id, message_id)` and emit an audit event.

---

## Medium — Scheduler Enhancements

### 26. Raw cron expression support (OpenClaw)

**What's missing:** The parser only handles `"every N units"`, `"daily at HH:MM"`, and `"weekly on DAY at HH:MM"`. OpenClaw accepts full 5- or 6-field cron expressions.

**Required:** Detect and pass raw cron strings directly to APScheduler's `CronTrigger.from_crontab()`.

### 27. Timezone support in scheduler (OpenClaw)

**What's missing:** `parse_schedule()` has no timezone awareness. All cron jobs fire in local time.

**Required:** `--tz IANA_TIMEZONE` option on `schedule add`; stored in `ScheduledJob`; passed to APScheduler trigger.

### 28. Job retry on failure (OpenClaw/SkyClaw)

**What's missing:** If a scheduled job fails (provider error, policy violation) it is silently skipped.

SkyClaw/OpenClaw: configurable `max_attempts` (default 3), `backoff_ms` array (30s, 60s, 300s), `retry_on` error categories.

**Required:** Retry wrapper in `SchedulerManager._run_job()`; per-job `max_attempts` and `backoff_seconds` in `ScheduledJob`.

### 29. One-shot future-dated jobs (OpenClaw)

**What's missing:** All jobs are recurring. OpenClaw supports `--at <ISO datetime>` for one-shot execution.

**Required:** `"at YYYY-MM-DD HH:MM"` format in `parse_schedule()`; APScheduler `DateTrigger`.

### 30. Active-hours window for heartbeat/scheduler (SkyClaw)

**What's missing:** Scheduled jobs fire at any hour. SkyClaw's heartbeat runner respects `active_hours = "08:00-22:00"`.

**Required:** `active_hours: str` in `SchedulingPolicy`; skip-and-reschedule logic in job runner.

---

## Medium — Memory and Context

### 31. Semantic / vector memory search (OpenClaw)

**What's missing:** `MemoryStore.get_session_turns()` returns by session ID only. There is no text search or semantic retrieval.

OpenClaw uses embedding providers (Voyage, Mistral, Gemini, OpenAI, Ollama) with MMR re-ranking, temporal decay scoring, and hybrid search (keyword + vector).

SkyClaw uses SQLite FTS with AND-logic keyword search plus configurable vector/keyword weight blend (default 0.7/0.3).

**Required (minimum viable):** SQLite FTS-backed `MemoryStore` with `search(query, limit)` method; context manager injects relevant memories into each agent call.

### 32. Session compaction / history summarization (OpenClaw)

**What's missing:** Conversation history grows unbounded in `MemoryStore`. OpenClaw compacts old history into a summary using the LLM when context pressure is detected.

**Required:** `compact_session(session_id)` in `MemoryStore`; called by `ContextManager` when token count exceeds threshold.

### 33. Sub-agent / task decomposition (SkyClaw/OpenClaw)

**What's missing:** Missy has no mechanism to spawn isolated sub-agents for parallel or sequential subtask execution.

SkyClaw: `DelegationManager` spawns `SubAgent` instances with scoped tool subsets, hard cap of 10 sub-agents / 3 concurrent, 300s timeout, no recursive delegation.

OpenClaw: `sessions_spawn` tool with `mode=run|session`, depth limit 1–5, up to 20 children.

**Required:** `SubAgentRunner`; `spawn_agent(task, tools, timeout)` tool; result aggregation back to primary agent.

---

## Medium — Heartbeat and Proactive Features

### 34. Heartbeat system (SkyClaw/OpenClaw)

**What's missing:** `missy gateway start` runs but has no periodic self-driven activity. SkyClaw's heartbeat reads `HEARTBEAT.md` every N minutes and sends it as a synthetic task to the agent.

**Required:** Background loop in `gateway start`; `HeartbeatConfig` in `MissyConfig`; `HEARTBEAT.md` checklist processing; `HEARTBEAT_OK` suppression file.

### 35. Proactive task initiation (SkyClaw)

**What's missing:** Tasks only originate from user input or the scheduler. SkyClaw supports file-change triggers (`FileChanged`), cron triggers, webhook triggers, and threshold triggers. All require explicit user opt-in.

**Required:** `ProactiveManager` with `watchdog`-style file watching (via `watchdog` Python package) and webhook receiver (via `aiohttp`); `requires_confirmation` flag gates destructive actions.

---

## Low — Additional Channels

### 36. Telegram channel

SkyClaw has native Telegram with file transfer, streaming in-place edits, image download for vision models, and allowlist-based access control.

OpenClaw adds topic-based agent routing, webhook/polling mode, and custom commands.

**Required:** `TelegramChannel` implementing `BaseChannel`; `python-telegram-bot` or `aiogram` library.

### 37. Slack channel

Both upstreams support Slack. OpenClaw uses Bolt.

**Required:** `SlackChannel` implementing `BaseChannel`; `slack_bolt` library.

### 38. Webhook channel

OpenClaw supports inbound webhooks as a channel, allowing external systems to trigger the agent.

**Required:** `WebhookChannel`; HTTP listener in `gateway start`; shared secret validation; policy gating.

---

## Low — Developer Experience and Operations

### 39. OpenTelemetry integration (SkyClaw/OpenClaw)

**What's missing:** Audit events are JSONL only. Both upstreams export traces and metrics to an OTEL collector.

**Required:** `opentelemetry-sdk` + `opentelemetry-exporter-otlp` in dependencies; `OtelExporter` wrapping `AuditLogger`; `observability.otel_enabled` / `otel_endpoint` in config.

### 40. Config hot-reload (OpenClaw)

**What's missing:** Config is loaded once at startup. OpenClaw supports `hot` reload mode (debounced file watcher re-applies config without restart).

**Required:** `watchdog`-based file watcher on `config.yaml`; reload signals to `PolicyEngine`, `ProviderRegistry`, and `SchedulerManager`.

### 41. Multiple API keys per provider (SkyClaw)

**What's missing:** `ProviderConfig.api_key` is a single optional string. SkyClaw accepts `keys: list[str]` and rotates atomically on 429/401.

**Required:** `api_keys: list[str]` in `ProviderConfig`; round-robin index in provider; rotate on rate-limit.

### 42. Agent-authored persistent custom tools (SkyClaw)

**What's missing:** Tools are only registered at startup. SkyClaw's `self_create_tool` lets the agent write bash/python/node scripts to `~/.skyclaw/custom-tools/` which persist across restarts.

**Required:** `SelfCreateTool` that writes scripts to `~/.missy/custom-tools/`; dynamic re-registration in `ToolRegistry`.

### 43. Failure alert routing for scheduled jobs (OpenClaw)

**What's missing:** Scheduled job failures are audited but not surfaced. OpenClaw triggers a `failureAlert` after N consecutive failures with routing to a configured channel.

**Required:** Consecutive failure counter in `ScheduledJob`; alert routing to Discord when threshold exceeded.

### 44. Session cleanup CLI (OpenClaw)

**What's missing:** Memory store entries accumulate indefinitely. OpenClaw prunes sessions older than 30 days (500-entry cap) with a `sessions cleanup --dry-run` command.

**Required:** `missy sessions cleanup [--dry-run] [--before DAYS]` command; TTL-based purge in `MemoryStore`.

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
- Voice pipeline / TTS (ElevenLabs, Edge TTS)
- mDNS/Bonjour gateway discovery
- Tailscale Funnel integration
- SSH tunnel transport

---

## Summary by Priority

| Priority | # Gaps | Key Items |
|---|---|---|
| Critical | 5 | Multi-step loop, tool invocation, streaming, function calling protocol, MCP |
| High | 15 | Circuit breaker, checkpointing, context management, cost tracking, file/shell/web tools, DONE criteria, cross-task learning, prompt patches, model tiering |
| Medium | 13 | Approval flow, vault, outbound censoring, Docker sandbox, scheduler enhancements (cron syntax, timezone, retry, one-shot), vector search, compaction, sub-agents, heartbeat, proactive triggers |
| Low | 6 | Telegram/Slack/webhook channels, OpenTelemetry, config hot-reload, custom agent-authored tools |
