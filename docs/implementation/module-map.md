# Module Map

Complete map of every module in the `missy/` package. Each entry lists the
module path, its purpose, the key classes or functions it exports, and its
dependencies on other `missy` modules.

---

## missy.core

### missy.core.message_bus

| Field | Value |
|-------|-------|
| **Purpose** | Async event-driven routing with fnmatch topic wildcards, priority queue, correlation IDs, and worker thread. |
| **Key exports** | `MessageBus`, `init_message_bus()`, `get_message_bus()` |
| **Internal deps** | None |

### missy.core.bus_topics

| Field | Value |
|-------|-------|
| **Purpose** | Standard topic constants for the message bus (AGENT_RUN_START/COMPLETE/ERROR, TOOL_REQUEST/RESULT, etc.). |
| **Key exports** | Topic string constants |
| **Internal deps** | None |

### missy.core.events

| Field | Value |
|-------|-------|
| **Purpose** | Audit event bus -- the publish/subscribe backbone for all policy decisions and runtime actions. |
| **Key exports** | `AuditEvent` (dataclass), `EventBus`, `event_bus` (module-level singleton), `EventCategory` (Literal type), `EventResult` (Literal type), `EventCallback` (type alias) |
| **Internal deps** | None |

### missy.core.exceptions

| Field | Value |
|-------|-------|
| **Purpose** | Custom exception hierarchy for the entire framework. |
| **Key exports** | `MissyError`, `PolicyViolationError`, `ConfigurationError`, `ProviderError`, `SchedulerError`, `ApprovalRequiredError` |
| **Internal deps** | None |

### missy.core.session

| Field | Value |
|-------|-------|
| **Purpose** | Thread-local session lifecycle management. |
| **Key exports** | `Session` (dataclass), `SessionMode` (enum: `FULL`, `NO_TOOLS`, `SAFE_CHAT`), `SessionManager` |
| **Internal deps** | None |

---

## missy.config

### missy.config.settings

| Field | Value |
|-------|-------|
| **Purpose** | YAML configuration loading and all policy/config dataclass definitions. |
| **Key exports** | `MissyConfig`, `NetworkPolicy`, `FilesystemPolicy`, `ShellPolicy`, `PluginPolicy`, `ProviderConfig`, `SchedulingPolicy`, `load_config()`, `get_default_config()` |
| **Internal deps** | `missy.core.exceptions` (`ConfigurationError`), `missy.channels.discord.config` (lazy import for `DiscordConfig`) |

### missy.config.migrate

| Field | Value |
|-------|-------|
| **Purpose** | Auto-migrate old configs on startup. Detects manual hosts matching presets, replaces with `presets: [...]`, stamps `config_version: 2`. |
| **Key exports** | `migrate_config()` |
| **Internal deps** | `missy.config.settings`, `missy.policy.presets` |

### missy.config.hotreload

| Field | Value |
|-------|-------|
| **Purpose** | File-watching config hot-reload with symlink, ownership, and permission safety checks. |
| **Key exports** | `ConfigWatcher` |
| **Internal deps** | `missy.config.settings` |

### missy.config.plan

| Field | Value |
|-------|-------|
| **Purpose** | Automatic config backups (max 5), diff, rollback, and plan commands. |
| **Key exports** | `ConfigPlan`, `backup_config()`, `rollback_config()` |
| **Internal deps** | `missy.config.settings` |

---

## missy.policy

### missy.policy.engine

| Field | Value |
|-------|-------|
| **Purpose** | Central policy facade composing network, filesystem, and shell sub-engines behind a single interface. Module-level singleton. |
| **Key exports** | `PolicyEngine`, `init_policy_engine()`, `get_policy_engine()` |
| **Internal deps** | `missy.config.settings` (`MissyConfig`), `missy.policy.network`, `missy.policy.filesystem`, `missy.policy.shell` |

### missy.policy.network

| Field | Value |
|-------|-------|
| **Purpose** | Evaluate outbound network access against CIDR, domain suffix, and explicit host rules. |
| **Key exports** | `NetworkPolicyEngine` |
| **Internal deps** | `missy.config.settings` (`NetworkPolicy`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`PolicyViolationError`) |

### missy.policy.filesystem

| Field | Value |
|-------|-------|
| **Purpose** | Evaluate filesystem read/write access against path allow-lists with symlink resolution. |
| **Key exports** | `FilesystemPolicyEngine` |
| **Internal deps** | `missy.config.settings` (`FilesystemPolicy`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`PolicyViolationError`) |

### missy.policy.shell

| Field | Value |
|-------|-------|
| **Purpose** | Evaluate shell command execution against an allow-list of program basenames. |
| **Key exports** | `ShellPolicyEngine` |
| **Internal deps** | `missy.config.settings` (`ShellPolicy`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`PolicyViolationError`) |

### missy.policy.rest_policy

| Field | Value |
|-------|-------|
| **Purpose** | L7 HTTP method + path glob rules per host (e.g. allow GET /repos/**, deny DELETE /**). |
| **Key exports** | `RestPolicy`, `RestPolicyRule` |
| **Internal deps** | `missy.config.settings` |

### missy.policy.presets

| Field | Value |
|-------|-------|
| **Purpose** | Named network policy presets (anthropic, github, openai, ollama, discord) that auto-expand to hosts/domains/CIDRs. |
| **Key exports** | `PRESETS`, `expand_presets()` |
| **Internal deps** | None |

---

## missy.gateway

### missy.gateway.client

| Field | Value |
|-------|-------|
| **Purpose** | Policy-enforcing HTTP client wrapping `httpx`. All outbound HTTP in the framework must use this client so that network policy is checked before any I/O. |
| **Key exports** | `PolicyHTTPClient`, `create_client()` |
| **Internal deps** | `missy.policy.engine` (`get_policy_engine`), `missy.core.events` (`AuditEvent`, `event_bus`) |

---

## missy.providers

### missy.providers.base

| Field | Value |
|-------|-------|
| **Purpose** | Abstract base class and interchange types for AI provider integrations. |
| **Key exports** | `BaseProvider` (ABC), `Message` (dataclass), `CompletionResponse` (dataclass) |
| **Internal deps** | None |

### missy.providers.anthropic_provider

| Field | Value |
|-------|-------|
| **Purpose** | Concrete provider for the Anthropic Messages API (Claude). |
| **Key exports** | `AnthropicProvider` |
| **Internal deps** | `missy.providers.base` (`BaseProvider`, `Message`, `CompletionResponse`), `missy.config.settings` (`ProviderConfig`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`ProviderError`) |

### missy.providers.openai_provider

| Field | Value |
|-------|-------|
| **Purpose** | Concrete provider for the OpenAI Chat Completions API. |
| **Key exports** | `OpenAIProvider` |
| **Internal deps** | Same pattern as `anthropic_provider`. |

### missy.providers.ollama_provider

| Field | Value |
|-------|-------|
| **Purpose** | Concrete provider for local Ollama inference. |
| **Key exports** | `OllamaProvider` |
| **Internal deps** | Same pattern as `anthropic_provider`. |

### missy.providers.registry

| Field | Value |
|-------|-------|
| **Purpose** | Singleton registry mapping provider name strings to live `BaseProvider` instances. Factory method builds a registry from `MissyConfig`. Supports fallback, hot-swap, and model tier routing (fast/premium). |
| **Key exports** | `ProviderRegistry`, `ModelRouter`, `init_registry()`, `get_registry()` |
| **Internal deps** | `missy.config.settings`, `missy.providers.anthropic_provider`, `missy.providers.openai_provider`, `missy.providers.ollama_provider`, `missy.providers.base`, `missy.providers.rate_limiter` |

### missy.providers.rate_limiter

| Field | Value |
|-------|-------|
| **Purpose** | Token-bucket rate limiter for provider API calls. Per-provider limits with configurable burst. |
| **Key exports** | `RateLimiter` |
| **Internal deps** | None |

---

## missy.agent

### missy.agent.runtime

| Field | Value |
|-------|-------|
| **Purpose** | Top-level agent orchestrator: resolves a provider, builds messages, calls the provider, and returns the model reply. |
| **Key exports** | `AgentRuntime`, `AgentConfig` (dataclass) |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`ProviderError`), `missy.core.session` (`Session`, `SessionManager`), `missy.providers.base` (`Message`, `CompletionResponse`), `missy.providers.registry` (`get_registry`), `missy.tools.registry` (`get_tool_registry`) |

### missy.agent.circuit_breaker

| Field | Value |
|-------|-------|
| **Purpose** | Closed/Open/HalfOpen state machine with exponential backoff (threshold=5, base=60s, max=300s). |
| **Key exports** | `CircuitBreaker`, `CircuitState` |
| **Internal deps** | None |

### missy.agent.context

| Field | Value |
|-------|-------|
| **Purpose** | Token budget management (default 30k) with reserves for system prompt, tool definitions, memory (15%), learnings (5%). |
| **Key exports** | `ContextManager` |
| **Internal deps** | `missy.memory.synthesizer` |

### missy.agent.attention

| Field | Value |
|-------|-------|
| **Purpose** | 5 brain-inspired attention subsystems: Alerting, Orienting, Sustained, Selective, Executive. |
| **Key exports** | `AttentionSystem`, `AlertingAttention`, `OrientingAttention`, `SustainedAttention`, `SelectiveAttention`, `ExecutiveAttention` |
| **Internal deps** | None |

### missy.agent.consolidation

| Field | Value |
|-------|-------|
| **Purpose** | Sleep mode — triggers at 80% context capacity, summarizes old turns, extracts key facts, preserves recent 4 messages. |
| **Key exports** | `MemoryConsolidator` |
| **Internal deps** | `missy.agent.context` |

### missy.agent.playbook

| Field | Value |
|-------|-------|
| **Purpose** | Auto-captures successful tool patterns (task_type + tool_sequence hash). Injects proven approaches. Auto-promotes 3+ successes to skill proposals. |
| **Key exports** | `Playbook` |
| **Internal deps** | None (JSON persistence at `~/.missy/playbook.json`) |

### missy.agent.progress

| Field | Value |
|-------|-------|
| **Purpose** | Structured progress reporting protocol with Null/Audit/CLI implementations. |
| **Key exports** | `ProgressReporter`, `NullReporter`, `AuditReporter`, `CLIReporter` |
| **Internal deps** | `missy.core.events` |

### missy.agent.interactive_approval

| Field | Value |
|-------|-------|
| **Purpose** | Real-time Rich TUI for policy-denied operations (y=allow once, n=deny, a=allow always). Session-scoped memory. |
| **Key exports** | `InteractiveApproval` |
| **Internal deps** | None |

### missy.agent.done_criteria

| Field | Value |
|-------|-------|
| **Purpose** | Generates verification prompts injected after each tool-call round. |
| **Key exports** | `DoneCriteria` |
| **Internal deps** | None |

### missy.agent.learnings

| Field | Value |
|-------|-------|
| **Purpose** | Extracts task_type/outcome/lesson from tool-augmented runs, persisted in SQLite. |
| **Key exports** | `Learnings` |
| **Internal deps** | `missy.memory.sqlite_store` |

### missy.agent.prompt_patches

| Field | Value |
|-------|-------|
| **Purpose** | Self-tuning prompt patches with approval workflow (proposed/approved/rejected). |
| **Key exports** | `PromptPatchManager` |
| **Internal deps** | None (JSON persistence at `~/.missy/patches.json`) |

### missy.agent.sub_agent

| Field | Value |
|-------|-------|
| **Purpose** | Spawns child agent instances for parallel work. |
| **Key exports** | `SubAgentRunner` |
| **Internal deps** | `missy.agent.runtime` |

### missy.agent.approval

| Field | Value |
|-------|-------|
| **Purpose** | Human-in-the-loop approval gate for sensitive operations. |
| **Key exports** | `ApprovalGate` |
| **Internal deps** | `missy.core.events` |

### missy.agent.persona

| Field | Value |
|-------|-------|
| **Purpose** | YAML-backed agent identity/tone/style management with backup, rollback, and audit logging. |
| **Key exports** | `PersonaConfig`, `PersonaManager` |
| **Internal deps** | None (YAML persistence at `~/.missy/persona.yaml`) |

### missy.agent.behavior

| Field | Value |
|-------|-------|
| **Purpose** | Humanistic behavior layer — tone analysis, intent classification, response shaping. |
| **Key exports** | `BehaviorLayer`, `IntentInterpreter`, `ResponseShaper` |
| **Internal deps** | `missy.agent.persona` |

### missy.agent.hatching

| Field | Value |
|-------|-------|
| **Purpose** | 8-step first-run bootstrap wizard with idempotent initialization (incl. vision check). |
| **Key exports** | `HatchingManager` |
| **Internal deps** | Multiple agent subsystems |

### missy.agent.checkpoint

| Field | Value |
|-------|-------|
| **Purpose** | WAL-mode SQLite task checkpointing. Enables `missy recover` to resume incomplete sessions. |
| **Key exports** | `Checkpoint` |
| **Internal deps** | None (SQLite at `~/.missy/checkpoints.db`) |

### missy.agent.cost_tracker

| Field | Value |
|-------|-------|
| **Purpose** | Per-session cost tracking and budget enforcement. |
| **Key exports** | `CostTracker`, `BudgetExceededError` |
| **Internal deps** | `missy.core.events` |

### missy.agent.failure_tracker

| Field | Value |
|-------|-------|
| **Purpose** | Per-tool failure tracking with strategy-rotation prompt injection after consecutive failures. |
| **Key exports** | `FailureTracker` |
| **Internal deps** | None |

### missy.agent.watchdog

| Field | Value |
|-------|-------|
| **Purpose** | Background subsystem health monitor. |
| **Key exports** | `Watchdog`, `SubsystemHealth` |
| **Internal deps** | Multiple agent subsystems |

### missy.agent.proactive

| Field | Value |
|-------|-------|
| **Purpose** | Proactive task initiation via file-change, disk/load threshold, and schedule triggers. |
| **Key exports** | `ProactiveManager` |
| **Internal deps** | `missy.agent.runtime` |

### missy.agent.code_evolution

| Field | Value |
|-------|-------|
| **Purpose** | Self-evolving code modification engine with approval workflow and git-backed rollback. |
| **Key exports** | `CodeEvolutionManager` |
| **Internal deps** | `missy.agent.runtime` |

### missy.agent.structured_output

| Field | Value |
|-------|-------|
| **Purpose** | Pydantic model schema enforcement on LLM responses with automatic retry on validation failure. |
| **Key exports** | `StructuredOutput` |
| **Internal deps** | `missy.providers.base` |

### missy.agent.sleeptime

| Field | Value |
|-------|-------|
| **Purpose** | Background memory processing during idle periods — consolidation, indexing, pruning. |
| **Key exports** | `SleeptimeWorker` |
| **Internal deps** | `missy.memory.sqlite_store`, `missy.memory.vector_store` |

### missy.agent.condensers

| Field | Value |
|-------|-------|
| **Purpose** | 4-stage memory compression pipeline: observation masking, amortized forgetting, summarizing, windowing. |
| **Key exports** | `CondenserPipeline` |
| **Internal deps** | `missy.agent.context` |

### missy.agent.compaction

| Field | Value |
|-------|-------|
| **Purpose** | Orchestrates leaf passes and condensation passes over conversation history. |
| **Key exports** | `CompactionManager` |
| **Internal deps** | `missy.agent.condensers` |

### missy.agent.summarizer

| Field | Value |
|-------|-------|
| **Purpose** | DAG-based conversation summarization with escalation tiers. |
| **Key exports** | `Summarizer` |
| **Internal deps** | `missy.providers.base` |

---

## missy.tools

### missy.tools.base

| Field | Value |
|-------|-------|
| **Purpose** | Abstract base class, permission model, and result type for tools. |
| **Key exports** | `BaseTool` (ABC), `ToolPermissions` (dataclass), `ToolResult` (dataclass) |
| **Internal deps** | None |

### missy.tools.registry

| Field | Value |
|-------|-------|
| **Purpose** | Singleton registry that manages, permission-checks, and executes tools. |
| **Key exports** | `ToolRegistry`, `init_tool_registry()`, `get_tool_registry()` |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`PolicyViolationError`, `ProviderError`), `missy.policy.engine` (`get_policy_engine`), `missy.tools.base` |

### missy.tools.builtin (18+ tools)

| Tool | Purpose |
|------|---------|
| `calculator` | Safe math expression evaluation |
| `file_read` | Policy-checked file reading |
| `file_write` | Policy-checked file writing |
| `file_delete` | Policy-checked file deletion |
| `list_files` | Directory listing |
| `shell_exec` | Policy-checked shell command execution |
| `web_fetch` | HTTP fetching through PolicyHTTPClient |
| `memory_tools` | Memory store read/write/search |
| `vision_tools` | Camera capture, analysis, scene memory |
| `browser_tools` | Playwright-based Firefox automation |
| `x11_tools` | X11 window management |
| `x11_launch` | X11 application launching |
| `atspi_tools` | AT-SPI accessibility toolkit |
| `tts_speak` | Text-to-speech via Piper |
| `discord_upload` | Discord file/image uploads |
| `incus_tools` | LXD/Incus container management |
| `code_evolve` | Self-code modification (with approval) |
| `self_create_tool` | Dynamic tool creation in `~/.missy/custom-tools/` |

---

## missy.skills

### missy.skills.base

| Field | Value |
|-------|-------|
| **Purpose** | Abstract base class, permission model, and result type for skills. |
| **Key exports** | `BaseSkill` (ABC), `SkillPermissions` (dataclass), `SkillResult` (dataclass) |
| **Internal deps** | None |

### missy.skills.registry

| Field | Value |
|-------|-------|
| **Purpose** | Singleton registry that manages and executes skills with audit events. |
| **Key exports** | `SkillRegistry`, `init_skill_registry()`, `get_skill_registry()` |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `event_bus`), `missy.skills.base` |

### missy.skills.builtin.system_info

| Field | Value |
|-------|-------|
| **Purpose** | Built-in skill that reports system information (platform, CPU, memory). |
| **Key exports** | `SystemInfoSkill` |
| **Internal deps** | `missy.skills.base` |

### missy.skills.builtin.workspace_list

| Field | Value |
|-------|-------|
| **Purpose** | Built-in skill that lists files in the configured workspace directory. |
| **Key exports** | `WorkspaceListSkill` |
| **Internal deps** | `missy.skills.base` |

---

## missy.plugins

### missy.plugins.base

| Field | Value |
|-------|-------|
| **Purpose** | Abstract base class and permission manifest for plugins. Plugins are disabled by default and must be explicitly enabled. |
| **Key exports** | `BasePlugin` (ABC), `PluginPermissions` (dataclass) |
| **Internal deps** | None |

### missy.plugins.loader

| Field | Value |
|-------|-------|
| **Purpose** | Plugin gatekeeper: loads, permission-checks, and executes plugins against `PluginPolicy`. |
| **Key exports** | `PluginLoader`, `init_plugin_loader()`, `get_plugin_loader()` |
| **Internal deps** | `missy.config.settings` (`MissyConfig`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`PolicyViolationError`), `missy.plugins.base` |

---

## missy.scheduler

### missy.scheduler.jobs

| Field | Value |
|-------|-------|
| **Purpose** | `ScheduledJob` dataclass with JSON serialisation/deserialisation. |
| **Key exports** | `ScheduledJob` |
| **Internal deps** | None |

### missy.scheduler.parser

| Field | Value |
|-------|-------|
| **Purpose** | Parse human-readable schedule strings (e.g. `"every 5 minutes"`, `"daily at 09:00"`) into APScheduler trigger configurations. |
| **Key exports** | `parse_schedule()` |
| **Internal deps** | None |

### missy.scheduler.manager

| Field | Value |
|-------|-------|
| **Purpose** | Manages scheduled jobs backed by APScheduler `BackgroundScheduler` and a JSON persistence file at `~/.missy/jobs.json`. |
| **Key exports** | `SchedulerManager` |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`SchedulerError`), `missy.scheduler.jobs` (`ScheduledJob`), `missy.scheduler.parser` (`parse_schedule`), `missy.agent.runtime` (lazy import: `AgentRuntime`, `AgentConfig`) |

---

## missy.observability

### missy.observability.audit_logger

| Field | Value |
|-------|-------|
| **Purpose** | Subscribes to every event on `event_bus` and appends a JSONL line to the audit log file. Provides `get_recent_events()` and `get_policy_violations()` read methods. |
| **Key exports** | `AuditLogger`, `init_audit_logger()`, `get_audit_logger()` |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `EventBus`, `event_bus`) |

### missy.observability.otel

| Field | Value |
|-------|-------|
| **Purpose** | OpenTelemetry traces and metrics export via OTLP (gRPC or HTTP). Requires `pip install -e ".[otel]"`. |
| **Key exports** | `OtelExporter` |
| **Internal deps** | `missy.config.settings` |

---

## missy.security

### missy.security.sanitizer

| Field | Value |
|-------|-------|
| **Purpose** | Input sanitisation to detect and log prompt-injection patterns. Truncates oversized payloads. |
| **Key exports** | `InputSanitizer`, `sanitizer` (module-level singleton), `MAX_INPUT_LENGTH` |
| **Internal deps** | None |

### missy.security.secrets

| Field | Value |
|-------|-------|
| **Purpose** | Secrets detection (37+ patterns) and redaction to prevent credential leakage. |
| **Key exports** | `SecretsDetector`, `secrets_detector` (module-level singleton) |
| **Internal deps** | None |

### missy.security.censor

| Field | Value |
|-------|-------|
| **Purpose** | Structured redaction of secrets from output with overlap merging. |
| **Key exports** | `SecretCensor`, `censor_response()` |
| **Internal deps** | `missy.security.secrets` |

### missy.security.vault

| Field | Value |
|-------|-------|
| **Purpose** | ChaCha20-Poly1305 encrypted key-value store. TOCTOU-safe atomic key creation. |
| **Key exports** | `Vault` |
| **Internal deps** | None (files at `~/.missy/secrets/`) |

### missy.security.identity

| Field | Value |
|-------|-------|
| **Purpose** | Ed25519 keypair generation and management. Signs audit events. JWK export. |
| **Key exports** | `AgentIdentity`, `init_identity()`, `get_identity()` |
| **Internal deps** | None (keypair at `~/.missy/identity.pem`) |

### missy.security.drift

| Field | Value |
|-------|-------|
| **Purpose** | SHA-256 system prompt tamper detection. Verifies prompts before each provider call. |
| **Key exports** | `PromptDriftDetector` |
| **Internal deps** | `missy.core.events` |

### missy.security.trust

| Field | Value |
|-------|-------|
| **Purpose** | 0-1000 reliability tracking per tool/provider/MCP server. |
| **Key exports** | `TrustScorer` |
| **Internal deps** | `missy.core.events` |

### missy.security.container

| Field | Value |
|-------|-------|
| **Purpose** | Optional Docker-based isolation for tool execution. Per-session containers with network/memory/CPU limits. |
| **Key exports** | `ContainerSandbox` |
| **Internal deps** | `missy.config.settings` |

### missy.security.landlock

| Field | Value |
|-------|-------|
| **Purpose** | Linux Landlock LSM filesystem policy enforcement via ctypes syscalls. |
| **Key exports** | `LandlockPolicy` |
| **Internal deps** | None |

### missy.security.scanner

| Field | Value |
|-------|-------|
| **Purpose** | Installation security auditor. Checks permissions, config hygiene, exposed secrets. |
| **Key exports** | `SecurityScanner` |
| **Internal deps** | `missy.config.settings`, `missy.security.secrets` |

---

## missy.memory

### missy.memory.store

| Field | Value |
|-------|-------|
| **Purpose** | Base memory store interface and `ConversationTurn` dataclass. |
| **Key exports** | `MemoryStore`, `ConversationTurn` |
| **Internal deps** | None |

### missy.memory.sqlite_store

| Field | Value |
|-------|-------|
| **Purpose** | SQLite-backed memory with FTS5 full-text search at `~/.missy/memory.db`. |
| **Key exports** | `SQLiteMemoryStore` |
| **Internal deps** | `missy.memory.store` |

### missy.memory.resilient

| Field | Value |
|-------|-------|
| **Purpose** | Resilient wrapper around SQLiteMemoryStore with automatic retry and fallback. |
| **Key exports** | `ResilientMemoryStore` |
| **Internal deps** | `missy.memory.sqlite_store` |

### missy.memory.vector_store

| Field | Value |
|-------|-------|
| **Purpose** | Optional FAISS-based semantic search. Requires `pip install -e ".[vector]"`. |
| **Key exports** | `VectorMemoryStore` |
| **Internal deps** | `missy.memory.store` |

### missy.memory.graph_store

| Field | Value |
|-------|-------|
| **Purpose** | SQLite-backed entity-relationship graph memory with rule-based pattern matching. |
| **Key exports** | `GraphMemoryStore` |
| **Internal deps** | None (SQLite at `~/.missy/graph_memory.db`) |

### missy.memory.synthesizer

| Field | Value |
|-------|-------|
| **Purpose** | Unified memory block — merges learnings, playbook, and summaries into a single relevance-ranked, deduplicated context block. |
| **Key exports** | `MemorySynthesizer` |
| **Internal deps** | `missy.memory.sqlite_store`, `missy.agent.playbook`, `missy.agent.learnings` |

---

## missy.mcp

### missy.mcp.manager

| Field | Value |
|-------|-------|
| **Purpose** | MCP server connection manager. Config at `~/.missy/mcp.json`. Auto-restarts dead servers. |
| **Key exports** | `McpManager` |
| **Internal deps** | `missy.mcp.client`, `missy.mcp.digest`, `missy.tools.registry`, `missy.gateway.client` |

### missy.mcp.digest

| Field | Value |
|-------|-------|
| **Purpose** | SHA-256 digest pinning for MCP tool manifests. Mismatches refuse to load. |
| **Key exports** | `pin_digest()`, `verify_digest()` |
| **Internal deps** | None |

---

## missy.api

### missy.api.server

| Field | Value |
|-------|-------|
| **Purpose** | Agent-as-a-Service REST API. Loopback-only binding, API key auth, rate limiting, secrets censoring. |
| **Key exports** | `ApiConfig`, `ApiServer` |
| **Internal deps** | `missy.agent.runtime`, `missy.security.censor` |

---

## missy.vision

| Module | Purpose |
|--------|---------|
| `discovery` | USB camera detection via sysfs with vendor/product ID matching |
| `capture` | OpenCV-based capture with warm-up, retry, blank-frame detection |
| `sources` | ImageSource abstraction: WebcamSource, FileSource, ScreenshotSource, PhotoSource |
| `pipeline` | Resize, CLAHE exposure normalization, quality assessment |
| `scene` | Task-scoped multi-frame memory for puzzle/painting tasks with change detection |
| `prompts` | Domain-specific analysis prompts (puzzle, painting, inspection) |
| `intent` | Audio-triggered vision activation with configurable thresholds |
| `doctor` | Diagnostics (opencv, video group, devices, capture test) |
| `health` | Per-device capture stats, success rates, quality tracking |

---

## missy.channels

### missy.channels.base

| Field | Value |
|-------|-------|
| **Purpose** | Abstract base class and normalised message type for all I/O channels. |
| **Key exports** | `BaseChannel` (ABC), `ChannelMessage` (dataclass) |
| **Internal deps** | None |

### missy.channels.cli_channel

| Field | Value |
|-------|-------|
| **Purpose** | stdin/stdout channel implementation for interactive CLI use. |
| **Key exports** | `CLIChannel` |
| **Internal deps** | `missy.channels.base` |

### missy.channels.discord.config

| Field | Value |
|-------|-------|
| **Purpose** | Configuration dataclasses and YAML parsing for the Discord integration. |
| **Key exports** | `DiscordConfig`, `DiscordAccountConfig`, `DiscordGuildPolicy`, `DiscordDMPolicy` (enum), `parse_discord_config()` |
| **Internal deps** | None |

### missy.channels.discord.channel

| Field | Value |
|-------|-------|
| **Purpose** | Full Discord channel implementation: Gateway connection, access-control pipeline, message routing, and pairing workflow. |
| **Key exports** | `DiscordChannel` |
| **Internal deps** | `missy.channels.base` (`BaseChannel`, `ChannelMessage`), `missy.channels.discord.config`, `missy.channels.discord.gateway` (`DiscordGatewayClient`), `missy.channels.discord.rest` (`DiscordRestClient`), `missy.core.events` (`AuditEvent`, `event_bus`), `missy.gateway.client` (lazy import: `create_client`) |

### missy.channels.discord.gateway

| Field | Value |
|-------|-------|
| **Purpose** | Low-level Discord Gateway WebSocket client (connect, heartbeat, reconnect). |
| **Key exports** | `DiscordGatewayClient` |
| **Internal deps** | `missy.gateway.client` |

### missy.channels.discord.rest

| Field | Value |
|-------|-------|
| **Purpose** | Discord REST API client for sending messages, registering commands, and triggering typing indicators. |
| **Key exports** | `DiscordRestClient` |
| **Internal deps** | `missy.gateway.client` |

### missy.channels.discord.commands

| Field | Value |
|-------|-------|
| **Purpose** | Slash command definitions and dispatch for the Discord integration. |
| **Key exports** | `SLASH_COMMANDS`, `handle_slash_command()` |
| **Internal deps** | `missy.channels.discord.channel` |

### missy.channels.webhook

| Field | Value |
|-------|-------|
| **Purpose** | HTTP webhook ingress with HMAC auth, rate limiting, payload validation, header filtering. |
| **Key exports** | `WebhookChannel` |
| **Internal deps** | `missy.channels.base`, `missy.core.events` |

### missy.channels.voice

| Module | Purpose |
|--------|---------|
| `server` | WebSocket server (default port 8765) for edge node connections |
| `registry` | Device registration with PBKDF2 token hashing |
| `pairing` | Device pairing workflow |
| `presence` | Occupancy and sensor data tracking |
| `stt/` | Speech-to-text engines (faster-whisper) |
| `tts/` | Text-to-speech engines (Piper binary) |
| `edge_client` | Reference client for local testing |

### missy.channels.screencast

| Module | Purpose |
|--------|---------|
| `channel` | Browser-based screen capture channel |
| `auth` | Token-based session authentication (`ScreencastTokenRegistry`) |
| `session_manager` | Session lifecycle, frame metadata, analysis results |

---

## missy.cli

### missy.cli.main

| Field | Value |
|-------|-------|
| **Purpose** | CLI entry point (`missy` command). 30+ subcommands/groups including ask, run, setup, audit, schedule, vault, mcp, devices, voice, vision, persona, evolve, api, security, config, and more. |
| **Key exports** | `main()` |
| **Internal deps** | Nearly all other modules (transitive). |

---

## Dependency Graph (summary)

```
missy.core.events           <-- no internal deps (foundation)
missy.core.exceptions       <-- no internal deps (foundation)
missy.core.session          <-- no internal deps (foundation)
missy.core.message_bus      <-- no internal deps (foundation)
missy.config.settings       <-- core.exceptions, channels.discord.config (lazy)
missy.config.migrate        <-- config.settings, policy.presets
missy.config.hotreload      <-- config.settings
missy.config.plan           <-- config.settings
missy.policy.*              <-- config.settings, core.events, core.exceptions
missy.policy.presets        <-- no internal deps
missy.policy.rest_policy    <-- config.settings
missy.gateway.client        <-- policy.engine, policy.rest_policy, agent.interactive_approval,
                                core.events
missy.providers.base        <-- no internal deps
missy.providers.*_provider  <-- providers.base, config.settings, core.events, core.exceptions
missy.providers.registry    <-- config.settings, providers.base, all concrete providers,
                                providers.rate_limiter
missy.agent.runtime         <-- core.events, core.exceptions, core.session, core.message_bus,
                                providers.base, providers.registry, tools.registry,
                                agent.* (attention, context, circuit_breaker, playbook,
                                consolidation, condensers, done_criteria, learnings,
                                progress, approval, persona, behavior, checkpoint,
                                cost_tracker, failure_tracker, structured_output, sleeptime),
                                memory.synthesizer, security.* (sanitizer, secrets, censor, drift)
missy.tools.*               <-- core.events, core.exceptions, policy.engine, tools.base
missy.skills.*              <-- core.events, skills.base
missy.plugins.*             <-- config.settings, core.events, core.exceptions, plugins.base
missy.scheduler.*           <-- core.events, core.exceptions, scheduler.jobs, scheduler.parser,
                                agent.runtime (lazy)
missy.memory.store          <-- no internal deps
missy.memory.sqlite_store   <-- memory.store
missy.memory.vector_store   <-- memory.store
missy.memory.graph_store    <-- no internal deps
missy.memory.synthesizer    <-- memory.sqlite_store, agent.playbook, agent.learnings
missy.mcp.manager           <-- mcp.client, mcp.digest, tools.registry, gateway.client
missy.api.server            <-- agent.runtime, security.censor
missy.observability.*       <-- core.events
missy.security.*            <-- no internal deps (except scanner: config.settings, secrets)
missy.channels.*            <-- channels.base, core.events, gateway.client, discord.config
missy.channels.voice.*      <-- channels.base, core.events, gateway.client
missy.channels.screencast.* <-- channels.base, core.events
missy.vision.*              <-- providers.base, core.events
```
