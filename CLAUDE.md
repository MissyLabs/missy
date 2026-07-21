# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. Production-grade agent platform with strict security controls, policy enforcement, and full auditability. Python 3.11+.

## Commands

```bash
# Install
pip install -e ".[dev]"
pip install -e ".[voice]"         # adds faster-whisper, numpy, soundfile
pip install -e ".[otel]"          # adds OpenTelemetry SDK + exporters
pip install -e ".[vector]"        # adds faiss-cpu for semantic memory search
pip install -e ".[vision]"        # adds opencv-python-headless, numpy
pip install -e ".[desktop]"       # adds playwright (Firefox); run: playwright install firefox
pip install -e ".[discord_voice]" # adds discord.py[voice] + voice recv; requires system ffmpeg

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
       ├─ CostTracker (agent/cost_tracker.py) — per-session spend monitoring + budget caps
       ├─ Checkpoint (agent/checkpoint.py) — WAL-mode SQLite task checkpointing + recovery
       ├─ FailureTracker (agent/failure_tracker.py) — per-tool failure counts + strategy rotation
       ├─ Watchdog (agent/watchdog.py) — background subsystem health monitor
       ├─ ProactiveManager (agent/proactive.py) — file-change/disk/load/schedule triggers
       ├─ CodeEvolutionManager (agent/code_evolution.py) — self-evolving code with approval + git rollback
       ├─ StructuredOutput (agent/structured_output.py) — Pydantic schema enforcement on LLM responses
       ├─ SleeptimeWorker (agent/sleeptime.py) — background memory processing during idle
       ├─ Summarizer (agent/summarizer.py) — DAG-based conversation summarization
       ├─ CondenserPipeline (agent/condensers.py) — 4-stage memory compression
       ├─ CompactionManager (agent/compaction.py) — leaf + condensation pass orchestration
       ├─ ContainerSandbox (security/container.py) — optional Docker isolation
       ├─ LandlockPolicy (security/landlock.py) — Linux Landlock LSM filesystem enforcement
       ├─ SecurityScanner (security/scanner.py) — installation security auditing
       ├─ GraphMemoryStore (memory/graph_store.py) — entity-relationship graph with pattern matching
       ├─ MessageBus (core/message_bus.py) — async event-driven routing
       ├─ ApiServer (api/server.py) — Agent-as-a-Service REST API
       ├─ VisionSubsystem (vision/) — camera discovery, capture, analysis, scene memory
       └─ AuditLogger + OtelExporter (observability/)

Channels:
  CLIChannel | DiscordChannel | WebhookChannel | VoiceChannel | ScreencastChannel

ScreencastChannel (channels/screencast/):
  → Browser-based screen capture with token auth + session management

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
- `ShellPolicyEngine`: Command whitelisting. `ShellPolicy.enabled=True` with an empty `allowed_commands` denies every command (SR-1.8 fail-closed contract) unless `ShellPolicy.unrestricted=True` is explicitly set, in which case any non-empty command is allowed immediately — no allow-list matching (including that empty-list deny), and no subshell/brace-group/malformed-quoting rejection either, since that check exists only to protect the allow-list match from a hidden subcommand and there's no allow-list to protect in this mode (`$(...)`, backticks, and brace groups are all permitted too). Still gated on `enabled: True`, and does not affect any other, independent policy layer (`PolicyEngine.check_shell()`'s SR-1.7 redirect-target-to-filesystem-policy routing still applies — a redirect outside `allowed_write_paths` is denied even in unrestricted mode). The audit event's `policy_rule` is `"unrestricted"` when this path is taken, and launcher-command warnings (`sudo`/`bash`/`python`/etc.) still fire on a best-effort basis (extraction failure is tolerated, not denied).
- `RestPolicy`: L7 HTTP method + path glob rules per host (e.g. allow GET /repos/**, deny DELETE /**)
- Network presets (`missy/policy/presets.py`): `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs

**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx; single enforcement point for ALL outbound HTTP. Every request checked against network policy + REST policy before dispatch. `InteractiveApproval` TUI prompts operator on denied requests (y/n/a with session memory).

**Providers (`missy/providers/`)** — `BaseProvider` defines the interface (`Message`, `CompletionResponse`, `ToolCall`, `ToolResult`). Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`, `CodexProvider`, `AcpxProvider`. `ProviderRegistry.get_available()`/`.rotate_key()`/`.key_for()` are real dispatch-time mechanisms: `AgentRuntime._call_provider_with_fallback()` (used by both `_single_turn()` and `_tool_loop()`) wraps every provider call so a mid-run failure — not just a start-of-run unavailability check — triggers real recovery. Each provider name gets its own `CircuitBreaker` (`missy/agent/circuit_breaker.py`) tracking cooldown/half-open eligibility independently; `missy/providers/health.py`'s `classify_provider_error()` distinguishes auth/rate-limit/timeout/unknown failures from each provider's `ProviderError` message text. An auth failure with `ProviderConfig.api_keys` configured triggers one `rotate_key()` retry on the same provider before falling over; a fallback candidate is chosen by cooldown eligibility (breaker not `OPEN`), tool-capability (prefers a provider overriding `complete_with_tools` when the call requires tools, flagging `tool_compatibility_degraded` in the audit event otherwise), and a pre-flight budget check — never inheriting the failed provider's `model` string, since fallback candidates use their own configured default. Every rotation/fallback/failure is a redacted `agent.provider.*` audit event. `ModelRouter` (fast/primary/premium tier selection by prompt complexity) remains unwired — nothing in `AgentRuntime` calls `score_complexity()`/`select_model()`; `fast_model`/`premium_model` config fields are only consumed directly by `SleeptimeWorker._llm_summarize()`, not via `ModelRouter`. `ProviderConfig.key_rotation_strategy` (`"failover"` default | `"round_robin"`) controls how a 2+-entry `api_keys` list is used: `"failover"` is the original behavior above (one sticky key, `rotate_key()` switches only reactively after an auth failure); `"round_robin"`, currently implemented for `OpenAIProvider` (e.g. two separate OpenAI accounts), proactively balances *every* call across all configured accounts instead. Each account gets its own lazily-built SDK client and its own independent `RateLimiter` (so a second account actually doubles effective throughput rather than sharing one combined budget) — per-call account selection is a thread-local, lock-guarded round-robin index (`OpenAIProvider._select_account()`), never a shared mutable "current key" attribute, so concurrent calls can't race over which account's client/rate-limiter is active. Conversation content itself never passes through this selection (each `complete()`/`complete_with_tools()`/`stream()` call forwards its `messages` argument unchanged regardless of which account handles it), so multi-turn context survives account switching by construction. `ProviderRegistry.rotate_key()` no-ops for a provider reporting `is_multi_account=True`, since such a provider already balances every call internally. `account_index` (never the key itself) is included in `OpenAIProvider`'s `provider_invoke` audit events and its `diagnostics()` output when active. `CodexProvider` (`openai-codex`, OAuth/ChatGPT-backend) supports the OAuth analogue of the same mechanism: `key_rotation_strategy: round_robin` + `ProviderConfig.oauth_accounts` (2+ named OAuth account slugs, not raw keys) round-robins every `complete()`/`complete_with_tools()`/`stream()` call across multiple signed-in ChatGPT accounts, reusing the identical `missy/providers/round_robin.py` helper (`Account.api_key` here holds an account *name*, e.g. `"work"`, rather than a secret — the actual bearer token is loaded lazily per call from that account's own `~/.missy/secrets/openai-oauth-<name>.json`, refreshed independently). `missy providers auth openai-codex --method oauth --account <name>` signs in an additional named account without overwriting any other (the unnamed/default account still uses the original single `openai-oauth.json` file for full backward compatibility); once 2+ accounts are stored, the CLI auto-sets `key_rotation_strategy: round_robin` unless the operator already set it explicitly (e.g. deliberately kept `failover`). `missy providers oauth-accounts` lists every stored account (email/account-id/token-expiry, no secrets) and whether balancing is currently active. Each account also gets its own independent `RateLimiter`, and `complete()`/`stream()`/`complete_with_tools()` all funnel through one shared `_prepare_call()` (account-select + rate-limit-acquire) so the account whose token issues the HTTP request is always the same one its rate limiter gated — `complete()` no longer calls the public `stream()` internally (it shares the private `_iter_deltas()` generator instead) specifically to avoid a second, possibly different, round-robin account being selected mid-call. `account_index`/`account_name` (the name, not a secret here) are included in `CodexProvider`'s `provider_invoke` audit events when active. `RoundRobinAccounts` (shared by both providers) also tracks **per-account health**: plain round-robin is blind to which credential is actually working, so an account whose real upstream quota/auth is exhausted would otherwise keep getting selected on schedule and fail on every one of its turns forever. Both providers report each call's outcome via `RoundRobinAccounts.record_success()`/`record_failure()` (`OpenAIProvider`/`CodexProvider._record_account_outcome()`, called from every exit point of `complete()`/`complete_with_tools()`/`stream()`); after `failure_threshold` consecutive failures (default 5, matching `CircuitBreaker`'s default) an account is skipped by selection for a backoff window (`base_backoff_seconds=60`, doubling on a failed post-cooldown probe up to `max_backoff_seconds=300` — the same closed/open/half-open shape as `CircuitBreaker`, reimplemented locally in `round_robin.py` rather than imported so this provider-layer module has no dependency on the higher-level `agent` package). If every account is currently in backoff, selection fails open (returns whichever recovers soonest) rather than refusing to serve at all. This is a per-account mechanism, independent of and in addition to `AgentRuntime`'s per-*provider* `CircuitBreaker`/fallback in `_call_provider_with_fallback()` above — the two operate at different granularities (one account within a multi-account provider vs. one provider among several configured).

**Channels (`missy/channels/`)** — Communication interfaces:
- `CLIChannel`: Interactive stdin/stdout
- `DiscordChannel`: Full WebSocket Gateway API with access control (DM allowlist, guild/role policies), slash commands (`/ask`, `/status`, `/model`, `/help`)
- `WebhookChannel`: HTTP webhook ingress
- `ScreencastChannel`: Browser-based screen capture with token auth (`ScreencastTokenRegistry`) and session management (`SessionManager`). `!screen stop` (`revoke_session()`) is re-checked on every subsequent WebSocket message in `ScreencastServer._message_loop()`, not just at the initial auth handshake — a revoked session's already-authenticated connection is force-closed (code 1000) as soon as it sends anything else, rather than continuing to stream frames (and have the analyzer keep posting results to Discord) indefinitely until the browser tab is manually closed.
- `VoiceChannel`: WebSocket server (default port 8765) accepting connections from edge nodes (ReSpeaker, Raspberry Pi). Protocol: JSON control frames + binary PCM audio. Device pairing with PBKDF2-hashed tokens. Per-node policy modes: `full`, `safe-chat`, `muted`. `VoiceServer._message_loop()` re-fetches the node's live policy from the registry on every message (not just at the initial auth handshake), so `missy devices policy <id> --mode muted` disconnects an already-connected node on its next message rather than only blocking future connection attempts. `missy devices policy <id> --mode safe-chat` routes that node's agent calls to a dedicated `capability_mode="safe-chat"` `AgentRuntime` (`gateway_start()` constructs one alongside the main/Discord runtimes) via `_build_agent_callback()`'s `policy_mode`-based dispatch — a safe-chat node's request is refused outright (fail closed) rather than silently served with full access if no restricted runtime is configured. STT via faster-whisper, TTS via piper binary.

**Discord inbound attachments (`missy/channels/discord/`)** — `DiscordChannel._handle_message()` classifies every inbound attachment four ways before anything is queued: image (`image_analyze.py`), text-like (`text_attachment.py`, `.md`/`.txt`/`.json`/`.yaml`/`.csv`/`.log` under `MAX_TEXT_ATTACHMENT_BYTES`), zip archive (`zip_attachment.py`, `.zip`/`application/zip`-family under `MAX_ZIP_ATTACHMENT_BYTES`), or denied outright — one denied attachment drops the whole message (all-or-nothing, matching the pre-existing image/text gate). Allowed attachment metadata rides on `ChannelMessage.metadata` as `discord_image_attachments`/`discord_text_attachments`/`discord_zip_attachments`; `attachment_context.py`'s `build_inbound_attachment_context()` (called from `cli/main.py`'s Discord message loop) downloads each one and turns it into prompt content — a local path + `vision_analyze` instruction for images, sanitized inline text for text files, and for zip archives, safe extraction via `zip_extract.py`'s `safe_extract_zip()` followed by a file listing plus inline content for any small text files found inside (bounded by `MAX_INLINE_ZIP_TEXT_FILES`/`MAX_INLINE_ZIP_TEXT_FILE_BYTES`, mirroring the text-attachment splicing pattern). `safe_extract_zip()` guards against the standard untrusted-archive hazards *before* writing anything: the archive's central directory is inspected first for entry count (`MAX_ZIP_ENTRIES`), total uncompressed size (`MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES`), and a compression-ratio zip-bomb heuristic (`MAX_ZIP_COMPRESSION_RATIO`, e.g. classic "42.zip"-style small-download-expands-to-gigabytes archives) — any of these rejects the *whole* archive with nothing written. Per-entry, a resolved-path containment check (zip-slip/path-traversal), a Unix-symlink-bit check, and a password-protected-entry check each skip just that one entry while the rest of the archive still extracts; a per-entry size cap (`MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES`) is enforced against actual streamed bytes (not just the archive's declared size, which a crafted entry could under-report) so a mid-stream-only bomb is caught too. Extracted files land under `~/.missy/captures/discord_inbound_zips/<message>_<index>_<name>/` with `0o700`/`0o600` permissions — like the image-attachment inbound directory, this is not automatically added to `filesystem.allowed_read_paths`, so `file_read`/`shell_exec` on extracted files beyond what's spliced inline requires the operator to explicitly allowlist that directory, matching every other new-capability-class-is-opt-in precedent in this codebase.

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
- `PromptPatchManager`: Self-tuning prompt patches with approval workflow (proposed/approved/rejected). `approve()`/`reject()` only transition a patch out of `PROPOSED` status — a stale/replayed call against an already-`REJECTED`/`EXPIRED`/`APPROVED` patch returns `False` rather than silently overwriting its status, since that would otherwise reinstate a rejected or auto-retired (poor success rate) patch into `get_active_patches()`'s active set with no further human review
- `SubAgentRunner`: Decomposes a compound task into sub-agent calls, wired into production via the `delegate_task` tool. Reuses the calling `AgentRuntime`/session (not a fresh runtime per call) so sub-agent spend aggregates against the same per-session `CostTracker` and policy/capability_mode enforcement is identical to the parent call. Independent subtasks run concurrently via `ThreadPoolExecutor` (capped at `MAX_CONCURRENT`); dependent subtasks wait for their dependency's result. Recursion bounded by `MAX_SUB_AGENT_DEPTH`.
- `ApprovalGate`: Human-in-the-loop approval for sensitive operations
- `CostTracker`: Per-session cost tracking and budget enforcement (`max_spend_usd`). Raises `BudgetExceededError` when cap hit. Each session's tracker is created lazily (`AgentRuntime._make_cost_tracker()`) from `self.config.max_spend_usd` at first use; `gateway_start()`'s config hot-reload callback mutates `_agent.config.max_spend_usd`/`_discord_agent.config.max_spend_usd`/the proactive-trigger runtime's config/`SchedulerManager._default_max_spend_usd` in place on every reload (alongside the `PolicyEngine`/`ProviderRegistry`/`OtelExporter`/`AuditLogger` re-init `_apply_config()` already does) — without this, editing `max_spend_usd` in `config.yaml` while the gateway keeps running had zero effect on any of these long-lived, already-constructed runtimes (not even for brand-new sessions) until a full restart, since none of them are rebuilt on reload the way the other singletons are.
- `Checkpoint`: WAL-mode SQLite checkpointing for task state. Enables `missy recover` to resume incomplete sessions.
- `FailureTracker`: Per-tool consecutive failure counts. Injects strategy-rotation prompts after repeated failures.
- `Watchdog`: Background health monitor for subsystems. Tracks `SubsystemHealth` status and reports degradation.
- `ProactiveManager`: Autonomous task initiation via file-change watchers, disk/load threshold triggers, and schedule-based triggers. Each trigger's `requires_confirmation` defaults to `True` — an unattended trigger gates through a real `ApprovalGate` (`agent/approval.py`) before its agent callback runs, unless a specific trigger opts out. `missy gateway start` constructs and wires a shared `ApprovalGate` into both `ProactiveManager` and the Web API server; `missy approvals list/approve/deny` operate on it via the API's `/api/v1/approvals` endpoints (approval state lives in-process inside the running gateway, so a separate CLI invocation can only reach it over HTTP).
- `CodeEvolutionManager`: Self-evolving code modification engine with approval workflow, git-backed rollback, and `missy evolve` CLI.
- `StructuredOutput`: Pydantic model schema enforcement on LLM responses with automatic retry on validation failure.
- `SleeptimeWorker`: Background daemon thread, constructed and started in `AgentRuntime.__init__` (enabled by default, matching `SleeptimeConfig.enabled=True`). Wakes every `check_interval_seconds` (default 60s); once the agent has been idle for `idle_threshold_seconds` (default 300s, reset via `record_activity()` on every `run()`/`run_stream()`/`resume_checkpoint()` call) it summarizes unsummarised turns and extracts learnings for eligible sessions (inspired by Letta sleeptime computing). `AgentRuntime.shutdown()` stops it cleanly.
- `Summarizer`: DAG-based conversation summarization with escalation tiers for progressive detail reduction.
- `CondenserPipeline`: 4-stage memory compression — observation masking, amortized forgetting, summarizing, windowing.
- `CompactionManager`: Orchestrates leaf passes and condensation passes over conversation history.

**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections and MCP tools are runtime-callable by the agent. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`. Digest pinning (`missy mcp pin`) records SHA-256 of tool manifests; mismatches refuse to load at connect time *and* deny every call at dispatch time (re-verified before each invocation, not just at connect) — `McpManager._check_digest_drift()` is called a *second* time immediately after an `ApprovalGate.request()` wait returns (not only before it), since the gate blocks synchronously for up to its configured timeout (60s default), a window in which a compromised/updated server could mutate its advertised manifest between the pre-approval check and actual dispatch; an operator's approval is honored only against manifest state still current right before the call, not whatever was current when the approval prompt was first shown. `AgentRuntime._sync_mcp_tools()` wraps each connected MCP tool in an `McpToolWrapper(BaseTool)` and registers it into the real `ToolRegistry` every turn, so dispatch goes through the same permission-check/audit reference monitor as any built-in tool. Tools whose MCP annotation sets `requires_approval` (destructive/mutating) block on an `ApprovalGate` (`AgentConfig.mcp_approval_gate`); with no gate configured, such calls fail closed rather than running unconfirmed. `ToolAnnotation.from_mcp_dict()` (`missy/mcp/annotations.py`) parses a *provided* (possibly partial or empty) `annotations` dict per the MCP spec's cautious-by-omission posture: `readOnlyHint` defaults to `False` and `openWorldHint` to `True` when absent, and an omitted `destructiveHint` defaults to `True` unless the tool is read-only — so a real-world server that declares only `{"readOnlyHint": false}` (a common partial annotation, legal per spec) is correctly parsed as mutating/approval-requiring rather than silently trusted safe. This does not extend to a tool with *no* `annotations` key in its manifest at all — `AnnotationRegistry.get_or_default()`'s bare `ToolAnnotation()` fallback (and the dataclass's own field defaults) still treat a fully-unannotated tool as safe by design; whether that broader default should also flip to match the spec's cautious posture is a distinct, larger product-policy question, not addressed here. MCP tool output is scanned for prompt injection and can be blocked or flagged (`McpManager(block_injection=...)`).

**Skills (`missy/skills/`)** — `SkillDiscovery` scans directories for SKILL.md files (cross-agent portable skill format with YAML frontmatter). `missy skills scan` lists discovered skills. Fuzzy search by name/description.

**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions. Each job carries a `capability_mode` (default `"safe-chat"`, read-only tools) that scopes its unattended agent run — a job's tool access does not default to `"full"` the way an interactive session's does; opt a specific job into `"full"` explicitly via `--capability-mode full`. `missy gateway start` constructs a `SchedulerManager`, calls `.start()` (loading persisted jobs and starting its background APScheduler thread), and attaches it to the shared agent runtime as `_scheduler` — this is what makes a job added via the standalone `missy schedule add` CLI (a separate process that only mutates `jobs.json`) actually fire, and is what the Web TUI's scheduler pages/operator controls resolve via `getattr(runtime, "_scheduler", None)`. Gated on `scheduling.enabled` in config; stopped cleanly in the gateway's shutdown path. `gateway_start()` also constructs the `SchedulerManager` with `default_max_spend_usd=getattr(cfg, "max_spend_usd", 0.0)`, and `_run_job()` applies it to every per-job `AgentConfig` (each job run gets a brand-new session/`AgentRuntime`/`CostTracker`) — this closes the same "operator's configured spend cap is silently ignored" gap that also existed in `gateway_start()`'s main/Discord/proactive-trigger `AgentConfig` construction and in `missy api start`, all of which now pass `max_spend_usd` the same way `missy ask`/`missy run`/`missy recover` always did. `SchedulerManager` also accepts `default_tool_policy_kwargs` (the same `_agent_tool_policy_kwargs(cfg)` dict — `tool_policy`/`agent_tool_policy`/`sandbox_tool_policy`/`subagent_tool_policy`/`tool_intelligence`/`agent_id` — every other `AgentConfig` site passes), applied by `_run_job()` to every per-job `AgentConfig`; without it, an operator's global `tools.deny: [...]` config would silently not apply to a `capability_mode="full"` scheduled job, since `AgentRuntime._execute_tool()` enforces the per-turn allowed-tool set as a hard execute-time gate, not just a model-visibility filter. `_remove_pending_retries()` (called by both `pause_job()` and `remove_job()`) cleans up any dangling `f"{job_id}_retry_{n}"` APScheduler entry a prior failed run scheduled independently of the main trigger — `remove_job()` is a more permanent action than pausing and must clean up at least as thoroughly.

**Security (`missy/security/`)**:
- `InputSanitizer`: Detects 250+ prompt injection patterns with Unicode normalization, base64 decode, multi-language support
- `SecretsDetector`: Detects 37+ credential patterns (API keys, JWTs, AWS, GCP, etc.)
- `SecretCensor`: Redacts secrets from output with overlap merging
- `Vault`: ChaCha20-Poly1305 encrypted key-value store. Key file at `~/.missy/secrets/vault.key`, encrypted data at `~/.missy/secrets/vault.enc`. Supports `vault://KEY_NAME` references in config.
- `AgentIdentity`: Ed25519 keypair at `~/.missy/identity.pem`. Signs audit events. JWK export.
- `TrustScorer`: 0-1000 reliability tracking, scored per entity name (tool name today; the class itself is entity-agnostic and the same instance could score providers too). Success (+10) via `record_success()`; ordinary tool failure (-50) via `record_failure()`; a policy-engine denial specifically (-200, harsher) via `record_violation()` — `AgentRuntime._execute_tool()`/`_score_tool_trust()` distinguish the two using `ToolResult.policy_denied` (set by `ToolRegistry.execute()` when `PolicyViolationError` is raised), so a denied tool call no longer scores identically to a tool's own internal failure. Warns via `is_trusted()` below threshold. `_sync_mcp_tools()` wraps every connected MCP tool in a `McpToolWrapper` and registers it into the same `ToolRegistry` built-in tools use, so MCP tool calls dispatch through the identical `registry.execute()` → `_score_tool_trust()` path and DO feed `TrustScorer` in production, scored under their namespaced `server__tool` name exactly like a built-in tool. Only provider calls (`_call_provider_with_fallback`) have their own independent reliability tracking (`CircuitBreaker` per provider name) and do not call into `TrustScorer` — wiring that in is a real but distinct effort, not done here.
- `PromptDriftDetector`: SHA-256 hashes system prompts at start, verifies before each provider call. Emits `security.prompt_drift` audit event on tamper.
- `ContainerSandbox`: Optional Docker-based isolation for tool execution. Per-session containers with `--network=none`, memory/CPU limits. Config: `container: { enabled: true }`.
- `LandlockPolicy`: Linux Landlock LSM filesystem policy enforcement via ctypes syscalls, fully implemented (`LandlockPolicy`, `apply_landlock_from_config()`, `landlock_status()`) but with zero production callers — no CLI command, config flag, or runtime bootstrap path applies it today, so it currently provides no actual protection regardless of kernel support. `SecurityScanner`'s `SEC-094` surfaces this honestly (mirroring `SEC-090`'s treatment of `ContainerSandbox`'s identical gap) when the kernel supports Landlock.
- `SecurityScanner`: Installation security auditor (`missy security scan`). Checks file permissions, config hygiene, exposed secrets, and reports severity-ranked findings. `SEC-002`/`SEC-060` (plaintext-API-key checks) inspect the *raw*, pre-vault-resolution config value (`SecurityScanner._raw_provider_keys`, populated by re-reading the YAML file directly) rather than the already-resolved `ProviderConfig.api_key` `load_config()` produces — otherwise every correctly `vault://`-referenced key would be flagged identically to one typed directly into `config.yaml`.

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

**Video Generation (`missy/tools/builtin/video_generate.py`)** — `VideoGenerateTool` (agent tool name `video_generate`) talks to a separately-running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server's HTTP API (default `127.0.0.1:8199`; see the deployment note below for `MISSY_COMFYUI_HOST`) to render short video clips, optionally with a soundtrack, in four backends selected via a `backend` parameter (audit/design record in `video.md` at the repo root):
- `"wan"` (**default/recommended**) — Wan 2.2 TI2V 5B, 24 fps native. Text-to-video from a `prompt`, or image-to-video when `image_path` is also given. Uses `UNETLoader` + `ModelSamplingSD3(shift=8)` + `Wan22ImageToVideoLatent` (all core nodes), matching ComfyUI's official Wan 2.2 example. Requires `wan2.2_ti2v_5B_fp16.safetensors` (`models/diffusion_models/`), `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (`models/text_encoders/`), `wan2.2_vae.safetensors` (`models/vae/`).
- `"wan14b"` — Wan 2.2 **A14B MoE** text-to-video (higher quality than the 5B; text-to-video only, `image_path` rejected). Two expert UNETs (high-noise handles the first half of sampling, low-noise the second), each wrapped in `ModelSamplingSD3(shift=8)` and chained across a mid-step boundary via two `KSamplerAdvanced` nodes; `EmptyHunyuanLatentVideo` latent; umt5 text encoder; and the **wan 2.1 VAE** (`wan_2.1_vae.safetensors` — the A14B uses 2.1's VAE, not the 5B's 2.2 VAE). Requires `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` + `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` (`models/diffusion_models/`, ~14 GB each, from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`). The experts run sequentially so peak VRAM is ~one expert (tight on 16 GB, workable-but-slower on 8 GB via offload).
- `"svd"` — Stable Video Diffusion image-to-video. Animates a single input image (`image_path`) via `ImageOnlyCheckpointLoader` + `SVD_img2vid_Conditioning` + `VideoLinearCFGGuidance`. Requires `svd_xt.safetensors` in `models/checkpoints/`. 25 frames @ 6 fps natively; auto-interpolated 4x → 24 fps.
- `"animatediff"` — AnimateDiff Evolved text-to-video on SD1.5 with a `FreeU_V2` quality patch and `dpmpp_2m/karras` sampling (the `ComfyUI-AnimateDiff-Evolved` custom node pack). Requires a motion module (e.g. `mm_sd_v15_v2.ckpt`) in `models/animatediff_models/`.

**Audio**: `audio_prompt` generates a soundtrack (music/sfx/ambience) inside the same workflow graph, sized to the final clip duration, and muxes it into the MP4 via `VHS_VideoCombine`'s `audio` input; `audio_path` muxes an existing local audio file instead (mutually exclusive). The `audio_model` parameter selects the backend: **`stable-audio-3`** (default/recommended — Stable Audio 3.0 medium base, open-weight; requires `stable_audio_3_medium_base.safetensors` in `models/checkpoints/` + `t5gemma_b_b_ul2.safetensors` in `models/text_encoders/`, both from `Comfy-Org/stable-audio-3`, and a ComfyUI build whose loaders recognize the SA3 checkpoint architecture) or **`stable-audio-open-1.0`** (legacy fallback for older ComfyUI installs — `stable-audio-open-1.0.safetensors` + `t5_base.safetensors`). Both load the text encoder via `CLIPLoader(type="stable_audio")` (the checkpoint embeds none); SA3 drops the `ConditioningStableAudio` node (clip length comes from `EmptyLatentAudio` alone) and samples with `lcm`/`simple`, while SA1 uses `ConditioningStableAudio` + `dpmpp_3m_sde_gpu`. The preflight names the exact missing SA3 files + download source when they aren't present.

**Quality/robustness**: `interpolate` adds a core-`FrameInterpolate` RIFE pass (`rife_v4.26.safetensors` in `models/frame_interpolation/`; 0 = auto per backend); `upscale=True` adds a 2x RealESRGAN pass (`RealESRGAN_x2plus.pth` in `models/upscale_models/`); `video_format` selects `h264-mp4` (crf 17 default) / `h265-mp4` / `nvenc_h264-mp4` (NVENC GPU encode). Before submitting, the tool preflights `/system_stats` and **refuses a ComfyUI with no CUDA/ROCm/MPS/XPU device** (`allow_cpu=True` to override; the GPU name/VRAM is echoed in the output) and checks every required model file exists via `/models/{folder}`, erroring with the download source when one is missing. Inputs are clamped/snapped to each model's valid ranges (dims to /8 or /32, SVD frames ≤ 25, Wan lengths to 4k+1, etc.). On timeout the job is cancelled server-side (dequeued; `/interrupt` only if the running job is actually ours). The effective `seed` is always echoed back so results can be reproduced/refined.

All backends finish through the `ComfyUI-VideoHelperSuite` custom node pack's `VHS_VideoCombine` node to mux frames (and audio) into a real MP4, which the tool then copies to `~/.missy/videos/` (never overwriting — collisions get a numeric suffix; falls back to an HTTP `/view` download if ComfyUI runs on a different host than Missy). "Improving a video based on feedback" is just calling the tool again with adjusted parameters (prompt wording, the echoed `seed` with more `steps`, `motion_bucket_id`/`augmentation_level` for more/less motion, etc.) — there is no separate revise step. `resolve_network_hosts()`/`resolve_filesystem_targets()` declare the real ComfyUI host and `image_path`/`audio_path`/`save_path` targets so the policy engine enforces against the actual values, not just static declarations.

**Video Editing (`missy/tools/builtin/video_edit.py`)** — `VideoEditTool` (agent tool name `video_edit`) edits existing video files with the external `ffmpeg`/`ffprobe` binaries (direct subprocess, argv lists, sanitized env — never a shell; design record in `video.md` Parts II–III). Operations via an `operation` parameter: `concat` (splices 2+ videos; mixed resolution/fps/audio inputs are normalized first — letterbox scale/pad, common fps, silent `anullsrc` tracks synthesized when some inputs lack audio; optional `transition="crossfade"` chains xfade/acrossfade with probed-duration offsets), `trim` (frame-accurate re-encode, `start` + `end`/`duration`), `text` (drawtext overlay — the text goes through a temp *textfile* with `expansion=none`, so arbitrary content incl. `%`/quotes/colons renders literally and can't inject into the filter graph; 9 position presets or `x`/`y` expressions, whitelist-validated so a crafted expression can't smuggle e.g. a `movie=` source past the filesystem policy; timed captions via `start`/`end`; styled box), `speed` (0.25x–4x, audio pitch preserved via chained `atempo`), `resize` (0 for a dimension derives it from aspect), `extract_frame` (Part III — export one frame as a PNG/JPEG still; `at=-1` default = last frame, derived from probed duration/fps so low-fps clips still land on a real frame; enables vision review of a clip and last-frame → `image_path` scene chaining), and `audio` (Part III — lay an audio file onto a video: `audio_mode="replace"`/`"mix"` (amix, `normalize=0`), `loop` to repeat a short track; the video stream is `-c:v copy`ed bit-exact — never re-encoded — and the track is `apad`ded/`-shortest`ed so output duration always equals the video's; the audio file is probed with `_probe(expect="audio")`). Shares `video_generate`'s quality surface: `video_format` = `h264-mp4`/`h265-mp4`/`nvenc_h264-mp4` (NVENC GPU encode), `crf` 17 default, collision-safe output to `~/.missy/videos/`. The result is ffprobe'd after encoding, so the returned width/height/fps/duration/has_audio describe the actual file. Policy: `ToolPermissions(shell=True, filesystem_read/write=True)`; `resolve_shell_command()` declares `ffmpeg && ffprobe` (both must be allow-listed in ShellPolicy) and `resolve_filesystem_targets()` declares the real `inputs`/`input`/`font_file`/`audio_file`/`save_path` values. Complex edits compose by calling the tool repeatedly (trim each clip → concat → title), same iteration contract as `video_generate`.

**Video Storyboard (`missy/tools/builtin/video_storyboard.py`)** — `VideoStoryboardTool` (agent tool name `video_storyboard`, F16 + Part III) composes `video_generate` × N and `video_edit` into one call: an ordered list of *scenes* (each `{prompt, duration?, caption?, transition?}` plus per-scene generation overrides from a whitelist — `seed`, `negative_prompt`, `image_path`, `audio_prompt`, `steps`, `cfg`, `video_frames` — which win over shared kwargs) is generated clip-by-clip, trimmed, captioned (per-scene `caption` overlays via the `text` op; a caption/trim failure degrades gracefully to the plain clip), spliced honoring per-scene transitions (`scenes[i].transition` = the join *into* scene i; uniform joins collapse to a single concat, mixed joins left-fold pairwise concats), titled, and optionally scored (`audio_path`/`audio_mode`/`audio_loop` mux one continuous soundtrack over the final assembly via the `audio` op — a mux failure is a hard error). `continuity=True` visually chains scenes — the last frame of each finished (post-trim/caption) clip is extracted via `extract_frame` and fed as the next scene's `image_path`, the standard Wan 2.2 multi-scene technique; requires an i2v-capable backend (`wan`/`svd`; `svd` additionally needs scene 0 to supply its own `image_path`), rejected with an actionable error otherwise, and an extraction failure aborts (silently dropping continuity mid-story would defeat it). The output's `scenes` list echoes each scene's `seed`/`prompt`/`path` so any single bad scene can be re-rolled without regenerating the rest. Permissions are the union of its parts (`network`+`shell`+`filesystem`); policy declarations delegate to the underlying tools. Sub-tools are constructor-injectable for tests.

ComfyUI itself is external infrastructure, not part of the Missy codebase — installed at `~/comfyui` (venv at `~/comfyui-venv`), GPU-backed (the local box is an RTX 3070, 8 GB), managed by a system-level systemd unit (`/etc/systemd/system/comfyui.service`, `User=missy`, `Restart=on-failure`, enabled for `multi-user.target` so it survives reboots). Manage via `sudo systemctl {status,restart} comfyui` / `sudo journalctl -u comfyui -f`. Not a Missy-managed subprocess — the tool only ever talks to it over HTTP and has no knowledge of how/where it's actually running.

**Desktop / OBS / VTube Studio Integration** (design record: `desktop_obs_vtube.md`) — Missy's desktop-control surface was already substantial (`missy/tools/builtin/x11_tools.py`: `x11_screenshot`/`x11_click`/`x11_type`/`x11_key`/`x11_window_list`/`x11_read_screen` — the last doing Ollama-vision OCR/scene description; `atspi_tools.py` for accessibility-tree automation; `x11_launch.py`); this pass filled the remaining gaps and added two new WebSocket-based integrations, prioritized ahead of desktop mouse automation per operator request (APIs are safer than blind clicking). `missy/tools/builtin/obs_tools.py` talks directly to OBS Studio's built-in obs-websocket v5 server (`obs_status`, `obs_list_scenes`, `obs_switch_scene`, `obs_set_source_visibility`, `obs_start_recording`, `obs_stop_recording`, `obs_start_streaming_confirmed`, `obs_stop_streaming_confirmed`) — each call opens a fresh connection, computes the SHA-256 challenge/salt auth response per spec, issues one request, closes. `missy/tools/builtin/vtube_tools.py` talks to VTube Studio's public WebSocket API (`vtube_status`, `vtube_load_model`, `vtube_trigger_hotkey`, `vtube_set_parameter`); first-run authorization blocks up to 30s for the operator to click Allow in the VTS app itself, then persists the issued token directly to the encrypted `Vault` (`missy vault`) — never returned in any tool output, same persist-don't-print posture as `missy/cli/oauth.py`'s OAuth flow. `missy/tools/builtin/desktop_tools.py` adds `desktop_status` (detects X11 vs Wayland session type and which of `xdotool`/`wmctrl`/`scrot`/`wtype`/`ydotool`/`grim`/`slurp` are actually installed — GNOME-on-native-Wayland is reported as non-functional rather than silently failing, since Mutter implements neither X11 nor the wlroots protocols `wtype`/`ydotool` need), `desktop_focus_window`, `desktop_mouse_drag`, and `desktop_launch_app` (allowlisted app launch — always an argv-list `subprocess.Popen`, never a shell string, so there's no shell-injection surface regardless of requested arguments). `missy/tools/builtin/audio_route.py` adds `audio_route_tts`/`audio_test_route`, creating an idempotent PipeWire null-sink (`pactl`) that OBS's Audio Input Capture or VTube Studio's own built-in microphone-based lip sync can select as an input — deliberately *not* implemented as Missy computing and streaming synthetic per-frame `InjectParameterDataRequest` amplitude values against TTS playback timing, since VTube Studio's native mic-based lip sync fed by a real PipeWire route is far more reliable. Every destructive/public-facing action (non-allowlisted app launch, non-allowlisted OBS scene switch, and unconditionally `obs_start_streaming_confirmed`/`obs_stop_streaming_confirmed` — no allowlist bypass exists for those two) blocks on a new `missy.agent.approval.get_shared_approval_gate()`/`set_shared_approval_gate()` process-wide accessor, since built-in tools get no constructor injection (`register_builtin_tools()` instantiates every tool class with zero arguments) — `gateway_start()` registers the same `ApprovalGate` it already wires into `ProactiveManager`/the Web API here too, and any tool finding no gate registered denies outright, mirroring `McpManager`'s `requires_approval` handling exactly. New `ObsConfig`/`VtubeConfig`/`DesktopConfig` sections in `missy/config/settings.py` are all disabled by default; `obs.password`/`vtube.auth_token` support `vault://`/`$ENV` references resolved the same way as `ProviderConfig.api_key`. Phase 5 (a fully integrated always-on Discord-voice + VTuber live-performance workflow, tee'ing `discord_voice_say`'s audio to both Discord and the local PipeWire route simultaneously) is designed in `desktop_obs_vtube.md` but not implemented — `discord_voice.py`'s Discord-voice playback and `tts_speak`'s local playback are currently separate code paths. **Follow-up hardening pass** (self-audited against the original request, gaps + fixes in `desktop_obs_vtube.md` §3.5): `_desktop_shared.check_rate_limit()`/`check_window_allowed()` (`missy/tools/builtin/_desktop_shared.py`) add process-wide sliding-window rate limiting (per-tool budgets from each family's new `rate_limit_per_minute` config field, or a fixed 20/min for `audio_route_*`) across every `x11_*`/`desktop_*`/`obs_*`/`vtube_*`/`audio_route_*` tool, and a `desktop.window_allowlist` gate on `x11_click`/`x11_type`/`x11_key`'s `window_name` param + `desktop_focus_window` — enforced only when `desktop.enabled` is `True` (never fail-closed on absent config, to avoid breaking every pre-existing deployment that only ever used the app-level allowlist). `x11_tools._redact_screenshot_secrets()` OCRs (`pytesseract`, optional `[vision]` extra) and blacks out `SecretsDetector`-flagged text in a saved screenshot before it's returned (`x11_screenshot`'s new `redact: bool = True`, and `x11_read_screen`, which additionally censors the vision model's text description via `censor_response()`); degrades to a no-op note when `pytesseract` isn't installed. New tools: `obs_set_source_text` (`SetInputSettings` with literal `text` or `file_path`, for on-stream captions), `vtube_list_models` (`AvailableModelsRequest`, for Live2D model discovery independent of `vtube_load_model`'s by-name lookup), `desktop_mouse_move` (`xdotool mousemove` with no click), and `install_software_confirmed` (argv-only `sudo apt-get install -y <package>`, gated on `desktop.enabled` **and** a new `desktop.allow_software_install` flag defaulting `False`, a package-name regex, and `require_approval()`). `discord_upload.py`'s `execute()` now always calls `require_approval()` before posting (checked after the token-presence check, so a missing-token failure never triggers a pointless approval prompt) — no allowlist bypass, matching the OBS start/stop-streaming posture. `AudioSetVolumeTool` (`tts_speak.py`) now hard-caps an absolute volume at 100% and a relative `+N%`/`-N%` delta at 50 percentage points per call, since PipeWire otherwise lets a relative delta boost a sink arbitrarily far past 100%. **Discord tool-set reversal** (`desktop_obs_vtube.md` §3.6): a prior, unrelated validation pass had deliberately excluded every `x11_*`/`atspi_*` tool (and, by extension, the later-built `desktop_*`/`obs_*`/`vtube_*`/`audio_route_*` tools) from Discord's tool set (`missy/policy/tool_policy_pipeline.py`'s `MISSY_DISCORD_TOOLS`/`"messaging"` profile, used by `capability_mode="discord"`) as an explicit, un-made-unilaterally security-scope call — which directly contradicted this feature's own purpose (Discord-driven desktop/OBS/VTube control) once built. The operator confirmed the intent explicitly; `MISSY_DISCORD_TOOLS` and `DISCORD_SYSTEM_PROMPT` (`missy/agent/runtime.py`) were both updated so these tools are now visible and described (not denied) in Discord's capability mode. `browser_*` remains excluded — narrower, unrelated to this reversal. Every deeper guardrail (rate limiting, allowlists, confirmation gates, redaction) is unchanged by this and still gates individual calls regardless of which capability_mode exposed the schema. **Auto-approve opt-ins** (`desktop_obs_vtube.md` §3.7): once `install_software_confirmed`/`discord_upload_file` were reachable in practice from an unattended Discord session, their unconditional `require_approval()` calls became a real usability problem — no one is reliably available to answer the prompt within its 60s timeout, so requests reliably failed with `Approval timed out after 60.0s`. `DesktopConfig.auto_approve_software_install` and `DiscordConfig.auto_approve_uploads` (both default `False`) let an operator explicitly skip the per-call prompt for each tool independently; neither relaxes any other gate (`allow_software_install`, the package-name format check, rate limiting, `DISCORD_BOT_TOKEN` presence) — this is an explicit trust decision for a specific deployment, not a change to either tool's default "always ask" posture.

**Deployment topology / ComfyUI host selection (current state).** `video_generate` picks its ComfyUI server from `MISSY_COMFYUI_HOST` (a single host or a comma-separated ordered list, each optionally `host:port`) / `MISSY_COMFYUI_PORT`, and **always appends `127.0.0.1:8199` as a final fallback** — the preflight probes each candidate with a short timeout and uses the first reachable, GPU-capable one, so if a configured remote is down generation degrades to the local box. The chosen server is reported in the output dict (`comfyui_host`), with `gpu.fallback_from` listing skipped candidates. An explicit `comfyui_host` *kwarg* is a single target with no fallback (preserves policy-denial semantics); the kwarg is deliberately **not** exposed in the LLM-facing schema (which server to use is infra, not a model choice). This deployment runs a **second GPU box on the LAN at `10.0.0.230` (Windows, RTX 5080, 16 GB)** that hosts its own ComfyUI on `:8199` (installed via `setup-comfyui-5080.ps1`, listening on `0.0.0.0`) — the gateway sets `MISSY_COMFYUI_HOST=10.0.0.230` so video/audio generation runs on the faster 5080, falling back to the local 3070 if it's offline. The 5080 has the Wan 2.2 5B + Stable Audio 3 model set; the Wan 2.2 14B (`wan14b`) experts are additionally present on the local 3070 (they must be downloaded onto the 5080 separately to use `wan14b` there). The `10.0.0.230` box **also** runs **Ollama** on `:11434` (has `qwen3:14b` + `qwen2.5-coder:14b`) — configured as an `ollama` provider in `config.yaml` but currently `enabled: false` (the 14B local models leak tool-call JSON as content / over-call in Missy's full agent loop; `openai-codex`/gpt-5.5 is the active LLM driver). `OllamaProvider` salvages tool calls a model emits as content text (`_salvage_tool_calls_from_content`) but that doesn't make a weak model reliable for the agent loop.

**Over-refusal spiral recovery.** A security-triggering or adversarial-looking Discord message (e.g. a "hack the Gibson" movie-line joke) can prime `openai-codex`/gpt-5.5 into a persistent refusal state that recurs every turn, because the channel's session history keeps re-injecting the offending turn into context (a Discord session is keyed to the channel, so "start a new session" as a message does *not* reset it). Fix: `SQLiteMemoryStore.clear_session_full(session_id)` deletes the session's turns **and** summaries (a plain `clear_session()` leaves summaries, which keep the contamination alive), then restart the gateway to drop in-memory session state. The channel→session id is a deterministic UUID5; find it by grepping `~/.missy/memory.db` `turns.content` for the channel id.

**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days. Optional `VectorMemoryStore` with FAISS semantic search (`pip install -e ".[vector]"`). `GraphMemoryStore` provides SQLite-backed entity-relationship graph memory with rule-based pattern matching for structured knowledge retrieval.

**Message Bus (`missy/core/message_bus.py`)** — Async event-driven routing with fnmatch topic wildcards (e.g. `channel.*`), priority queue, correlation IDs, worker thread. Standard topics in `missy/core/bus_topics.py`. Runtime publishes `AGENT_RUN_START/COMPLETE/ERROR`, `TOOL_REQUEST/RESULT`. Module-level singleton via `init_message_bus()` / `get_message_bus()`.

**Config Migration (`missy/config/migrate.py`)** — Auto-migrates old configs on startup. Detects manual hosts matching presets, replaces with `presets: [...]`, stamps `config_version: 2`. Backs up before modifying. Idempotent.

**Config Plan (`missy/config/plan.py`)** — Automatic backups on config writes (max 5, pruned). `missy config rollback/diff/plan/backups` commands. `list_backups()`/`rollback()`/`_prune_backups()` order backups by filename (the `YYYYMMDD_HHMMSS[_N]` timestamp, which sorts lexicographically in true creation order) rather than `stat().st_mtime` — `shutil.copy2()` preserves the *source* config file's mtime on each backup copy, so two backups of an unchanged source get identical mtimes despite their filenames (and the `_N` disambiguating suffix already added for same-second collisions) correctly encoding which one is actually newer.

**API Server (`missy/api/`)** — Agent-as-a-Service REST API (`missy api start`). Loopback-only binding by default, API key authentication, rate limiting, and automatic secrets censoring on responses. Also serves the **Web TUI** (`missy/api/webui/` package; `missy/api/web_console.py` is a thin compat facade), a **multipage** browser operator console — every page shares the layout/nav/Inspector in `webui/layout.py` and requires the same cookie-session auth + CSRF: `/` dashboard (status tiles, streaming "Ask Missy" run console, safe controls, approvals, Discord pairing), `/memory` full-page memory browser (FTS search or query-less paging via `GET /api/v1/memory/recent`, per-session filter via `GET /api/v1/memory/sessions`, and per-turn pin/delete/**edit** — `PUT /api/v1/memory/turns/{id}` backed by `SQLiteMemoryStore.update_turn_content()`, whose `turns_au` trigger keeps the external-content FTS5 index in sync, since a plain SQL UPDATE silently desyncs it), `/audit` audit trail (filters + clickable facet chips, pagination, JSON export, auto-refresh), `/diagnostics` interactive doctor report (expandable sections, status filter, refresh), `/providers` provider management (redacted config detail via `GET /api/v1/providers/{name}` — key material never leaves the server, only `api_key_configured`/count — plus runtime enable/disable toggles via the `provider.enable`/`provider.disable` operator controls, backed by `ProviderRegistry.set_enabled()`: a disabled provider stays registered but is excluded from `get_available()` and refused by `set_default()`; the current default can't be disabled), `/scheduler` (job list with pause/resume/remove + create form), `/sessions` (transcript viewer, end session). `missy gateway start` launches the Web TUI automatically (disable via `api.enabled: false`); it is also reachable standalone via `missy api start`. Every row is clickable and opens the shared detail Inspector panel (`missy/api/operator_controls.py` backs the safe-control actions).

**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl` by wrapping `event_bus.publish()` so every event is captured regardless of type. `OtelExporter` sends every published audit event as an OTLP span when `otel_enabled: true`, using the same publish-wrapping pattern (its own `subscribe()` re-verifies the wrapper is installed). `detail` fields are redacted (`missy.observability.audit_logger._redact_detail`) before becoming span attributes. Export failures increment `OtelExporter.export_failure_count`/set `.last_export_error` rather than only logging at DEBUG. `init_otel(config)` (`missy/observability/otel.py`) is safe to call more than once per process and tracks the currently active exporter as a module-level singleton (`get_active_exporter()`); `missy/config/hotreload.py`'s `_apply_config()` (the `ConfigWatcher` reload callback) calls it on every config reload, so toggling `observability.otel_enabled`/`otel_endpoint`/`otel_protocol` on a running `missy gateway start` daemon actually takes effect without a restart — previously `init_otel()` was only ever invoked once at process bootstrap, so a hot-reloaded config change to these fields was silently a no-op. Each re-init calls the previous exporter's `unsubscribe()` first, restoring `event_bus.publish` before installing the new wrapper, so repeated reloads never stack multiple layers of export around the same event. `_apply_config()` likewise calls `init_audit_logger(new_config.audit_log_path)` on every reload, so editing `audit_log_path` on a running `missy gateway start` daemon actually takes effect — `init_audit_logger()` reuses and `reconfigure()`s the same, already-subscribed `AuditLogger` instance in place (mutating `log_path`/`identity`, both read fresh on every event) rather than constructing a new one, since a fresh instance's own `_subscribe()` would layer a second wrapper on top without ever actually replacing the old one's write target. Every record also carries a `prev_chain_hash` field (the SHA-256 of the exact preceding line's bytes, set before signing so it's covered by that line's own signature) — per-line signing alone (SR-1.1) detects content tampering but not *reordering*: two validly-signed lines swapped in position, or one deleted, would report every individual line "valid." `verify_audit_log()`'s `AuditLineVerification.chain_ok` catches this (`False` on a chain mismatch, `None` for the first line in a file or a pre-chaining legacy line with no `prev_chain_hash` field at all); `missy audit verify`/`missy doctor`'s audit-signing row both surface a broken chain as a failure distinct from `tampered`. The whole build-sign-write-and-advance-the-chain sequence in `_handle_event()` runs under `AuditLogger._chain_lock` so concurrent publishers can't both read the same "previous hash" and produce two lines chained from the same point; the chain state is seeded from the log file's existing tail on construction/`reconfigure()` so it survives process restarts and log rotation.

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
| Web TUI operator key | `~/.missy/secrets/web_console.key` (auto-generated on first `missy gateway start`) |
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
| Video generation output | `~/.missy/videos/` |
| Checkpoints DB | `~/.missy/checkpoints.db` |
| Graph memory DB | `~/.missy/graph_memory.db` |
| Code evolutions | `~/.missy/evolutions.json` |
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
  unrestricted: false          # true = skip allow-list matching entirely (still needs enabled: true)

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
    api_key: null                  # or set ANTHROPIC_API_KEY env var
    api_keys: []                   # multiple keys for rotation/balancing
    key_rotation_strategy: failover  # "failover" | "round_robin"
    fast_model: ""                 # e.g. claude-haiku-4-5
    premium_model: ""              # e.g. claude-opus-4-6
    enabled: true
  openai:
    name: openai
    model: "gpt-5.5"
    api_keys: ["sk-account-1-...", "sk-account-2-..."]  # two OpenAI accounts
    key_rotation_strategy: round_robin  # balance every call across both
  openai-codex:
    name: openai-codex
    model: "gpt-5.2"
    oauth_accounts: ["work", "personal"]  # 2+ named `missy providers auth openai-codex --method oauth --account <name>` logins
    key_rotation_strategy: round_robin    # balance every call across both signed-in ChatGPT accounts

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

# Web TUI / REST API — auto-started by `missy gateway start` (read from raw
# YAML, not a dataclass). api_key defaults to the persisted key at
# ~/.missy/secrets/web_console.key when unset.
api:
  enabled: true
  host: "127.0.0.1"
  port: 8080
  api_key: ""

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
missy providers switch NAME         Switch active provider (a running `missy gateway start` daemon's live default if reachable via --host/--port, else a local-only, non-persisted fallback)
missy providers auth [NAME]         Refresh OpenAI API-key or OAuth/Codex credentials (--method api-key|oauth, --account NAME for an additional OpenAI OAuth account)
missy providers oauth-accounts      List every stored OpenAI OAuth account (email/account-id/token-expiry, no secrets) and whether round-robin balancing is active
missy skills                        List registered skills
missy skills scan                   Scan for SKILL.md files (--path)
missy presets list                  Show built-in network policy presets
missy plugins                       List plugins and their status
missy doctor                        System health check

missy schedule add                  Add scheduled job (--name, --schedule, --task, --provider, --capability-mode; defaults to safe-chat, not full)
missy schedule list                 List all scheduled jobs
missy schedule pause JOB_ID         Pause a job
missy schedule resume JOB_ID        Resume a paused job
missy schedule remove JOB_ID        Remove a job

missy audit security                Show recent security events (--limit)
missy audit recent                  Show recent audit events (--limit, --category)

missy gateway start                 Start the gateway server (--host, --port); also launches the Web TUI (see api: config)
missy gateway status                Show gateway status

missy discord status                Show Discord channel configuration
missy discord probe                 Test Discord bot token connectivity
missy discord register-commands     Register slash commands (--guild-id, --global)
missy discord audit                 Show Discord-specific audit events (--limit)

missy vault set KEY VALUE           Store an encrypted secret
missy vault get KEY                 Retrieve a secret
missy vault list                    List stored key names
missy vault delete KEY              Delete a secret

missy approvals list                List pending approval requests from a running gateway (--host, --port, --api-key)
missy approvals approve ID          Approve a pending request by ID
missy approvals deny ID             Deny a pending request by ID

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

missy security scan                 Audit installation for security issues (permissions, config, secrets)

missy api start                     Start the Agent-as-a-Service REST API (--host, --port)
missy api status                    Show API server configuration and status

missy evolve list                   List code evolution proposals
missy evolve show EVOLUTION_ID      Show details of a code evolution proposal
missy evolve approve EVOLUTION_ID   Approve a proposed code evolution
missy evolve reject EVOLUTION_ID    Reject a proposed code evolution
missy evolve apply EVOLUTION_ID     Apply an approved code evolution
missy evolve rollback EVOLUTION_ID  Roll back an applied code evolution

missy sessions list                 List active sessions
missy sessions rename SESSION       Rename a session
missy sessions cleanup              Delete old conversation history (--older-than, --dry-run)

missy vision devices                Enumerate and diagnose available USB cameras
missy vision capture                Capture frames (--device, --output, --count, --width, --height)
missy vision inspect                Run visual quality assessment (--file, --screenshot, --device)
missy vision review                 LLM-powered visual analysis (--mode general|puzzle|painting|inspection, --file, --context)
missy vision doctor                 Run vision subsystem diagnostics
missy vision health                 Show capture statistics and device health status
missy vision benchmark              Run vision capture performance benchmarks
missy vision validate               Validate vision pipeline end-to-end
missy vision memory                 Show vision scene memory status and history

missy hatch                         First-run bootstrap wizard (--non-interactive)

missy persona show                  Display current persona configuration
missy persona edit                  Edit persona fields (--name, --tone, --identity)
missy persona reset                 Reset persona to factory defaults
missy persona backups               List available persona backups
missy persona diff                  Show diff between current persona and latest backup
missy persona rollback              Restore persona from latest backup
missy persona log                   Show persona change audit log (--limit)

missy cost                          Show cost tracking config and budget status (--session)
missy recover                       List incomplete checkpoints from previous sessions (--abandon-all, --resume ID, --provider)
```

## Optional Extras

```bash
pip install -e ".[dev]"           # pytest, pytest-asyncio, pytest-cov, black, ruff, mypy, watchdog, hypothesis
pip install -e ".[voice]"         # faster-whisper, numpy, soundfile
pip install -e ".[otel]"          # opentelemetry-sdk, OTLP gRPC + HTTP exporters
pip install -e ".[vector]"        # faiss-cpu for semantic memory search
pip install -e ".[vision]"        # opencv-python-headless, numpy for vision subsystem
pip install -e ".[desktop]"       # playwright (Firefox automation); run: playwright install firefox
pip install -e ".[discord_voice]" # discord.py[voice] + voice recv extension; requires system ffmpeg
```

Piper TTS is a separate binary, not a pip package. Install from https://github.com/rhasspy/piper.
Desktop extra also needs: `sudo apt install python3-pyatspi` for accessibility tools.

## Documentation

Full docs site: **https://missylabs.github.io/** — 80+ pages covering getting started, configuration, security, architecture, CLI reference, channels, providers, extending, edge nodes, operations, and Leyline P2P network. Source at `/home/missy/missylabs.github.io/` (MkDocs Material, deployed via GitHub Actions).

## Test Layout

Tests under `tests/` with subdirectories: `agent/`, `api/`, `channels/`, `cli/`, `config/`, `core/`, `gateway/`, `integration/`, `mcp/`, `memory/`, `observability/`, `plugins/`, `policy/`, `providers/`, `scheduler/`, `security/`, `skills/`, `tools/`, `unit/`, `vision/`. 480+ test files, 20,000+ tests, coverage threshold 90% (configured in `pyproject.toml`).

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
