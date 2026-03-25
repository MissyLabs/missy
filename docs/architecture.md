# Architecture

This document describes the internal architecture of Missy, a security-first
local agentic assistant platform for Linux.

---

## System Overview

Missy is a **security-first**, **local-first**, **multi-provider** AI agent
platform.  It is designed to run entirely under the operator's control on a
local machine.  Every capability that could cause harm -- outbound network
access, filesystem writes, shell execution, plugin loading -- is disabled by
default and must be explicitly enabled through a YAML configuration file.

Three properties define the system:

1. **Secure-by-default** -- all dangerous capabilities are denied until an
   operator opts in.
2. **Single network enforcement point** -- every outbound HTTP request flows
   through `PolicyHTTPClient` (`missy/gateway/client.py`), which consults the
   policy engine before any bytes leave the machine.
3. **Audit everything** -- every policy decision, provider call, scheduler
   execution, and plugin action is recorded as a structured JSONL event.

---

## Module Layout

```
missy/
  core/            Session management, event bus, message bus, exception hierarchy
  config/          YAML settings, hot-reload, migration, plan/rollback
  policy/          Network, filesystem, shell, REST L7 policy engines + presets
  gateway/         PolicyHTTPClient -- single network enforcement point
  agent/           Runtime, circuit breaker, context, playbook, consolidation,
                   attention, progress, approval, persona, behavior, hatching,
                   checkpoint, cost tracking, sleeptime, condensers, code evolution,
                   structured output, failure tracking, watchdog, proactive triggers
  api/             Agent-as-a-Service REST API server
  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry + rate limiter
  tools/           Tool base class, registry, 18+ built-in tools
  skills/          Skill registry + SKILL.md discovery
  plugins/         Security-gated external plugin loader and base class
  scheduler/       APScheduler integration, human schedule parsing, job persistence
  memory/          SQLite FTS5 store, vector memory (FAISS), graph memory, synthesizer
  observability/   AuditLogger (JSONL), OpenTelemetry exporter
  security/        Input sanitizer, secrets detector, censor, vault, identity,
                   trust scorer, drift detector, container sandbox, Landlock LSM, scanner
  channels/        CLI, Discord (Gateway + REST), webhooks, voice (WebSocket), screencast
  vision/          Camera discovery, capture, analysis, scene memory, health monitoring
  cli/             Click + Rich CLI, setup wizard, OAuth
```

---

## Key Data Flow

The following path describes what happens when a user runs `missy ask "..."`:

```
 1. CLI (click)           Parse command-line arguments, resolve --config path
        |
 2. Config loader         Read YAML, auto-migrate if needed, build MissyConfig
        |                 ConfigWatcher starts hot-reload monitoring
        |
 3. Subsystem init        init_policy_engine(cfg)  -- network, filesystem, shell, REST L7
        |                 init_audit_logger(cfg.audit_log_path) + AgentIdentity (Ed25519)
        |                 init_registry(cfg) -- providers with rate limiter + fallback
        |                 init_message_bus() -- async event routing
        |                 init_tool_registry() -- 18+ built-in tools + MCP servers
        |
 4. Security checks       SecretsDetector scans prompt for credentials
        |                 InputSanitizer truncates and checks for injection (250+ patterns)
        |                 PromptDriftDetector hashes system prompts (SHA-256)
        |
 5. AgentRuntime.run()    Resolve or create a Session
        |                 Resolve provider (with fallback + circuit breaker)
        |                 AttentionSystem extracts urgency/topics/focus
        |                 ContextManager builds message list within token budget
        |                 MemorySynthesizer injects relevant memories + learnings
        |                 Playbook injects proven tool patterns
        |
 6. Provider.complete()   Provider-specific SDK call (Anthropic, OpenAI, Ollama)
        |                 All HTTP through PolicyHTTPClient -> policy + REST check
        |                 CostTracker accumulates spend, enforces budget cap
        |
 7. Tool loop             DoneCriteria verifies completion after each round
        |                 FailureTracker rotates strategy on consecutive failures
        |                 Checkpoint saves state for recovery
        |                 ProgressReporter emits structured updates
        |
 8. Post-processing       Learnings extracted from tool-augmented runs
        |                 MemoryConsolidator triggers at 80% context (sleep mode)
        |                 CondenserPipeline compresses memory (4-stage)
        |                 SecretCensor redacts secrets from output
        |
 9. Audit events          Every step emits AuditEvent -> EventBus -> AuditLogger
        |                 Events signed by AgentIdentity, appended to audit.jsonl
        |                 MessageBus routes to topic subscribers
        |
10. Response              Text returned to channel (CLI/Discord/Voice/API), rendered
```

### ASCII Flow Diagram

```
  User
   |
   v
+--------------------------------------------+
| CLI / Discord / Voice / Webhook / API      |
+---------------------+----------------------+
                      |
               +------+------+
               | Config      |     +------------------+
               | (hot-reload)|     | SecurityScanner  |
               +------+------+     | LandlockPolicy   |
                      |            +------------------+
               +------+------+
               | PolicyEngine|     +------------------+
               | (net/fs/    |     | InputSanitizer   |
               |  shell/REST)|     | SecretsDetector  |
               +------+------+     | PromptDrift      |
                      |            +------------------+
               +------+------+
               | AgentRuntime|
               +------+------+
                      |
    +---------+-------+--------+---------+
    |         |                |         |
+---+---+ +---+--------+ +----+----+ +--+--------+
|Attend.| |Context Mgr | |Provider | |Tool       |
|System | |MemorySynth.| |Registry | |Registry   |
|Playbok| |Condensers  | |RateLimt | |MCP + Skills|
|Sleep  | |Checkpoint  | |Circuit  | |Vision     |
+-------+ +---+--------+ +----+----+ +--+--------+
               |                |         |
          +----+-----+    +----+----+    |
          |Memory    |    |PolicyHTTP|   |
          |SQLite/   |    |Client   |<--+
          |Vector/   |    |(gateway)|
          |Graph     |    +----+----+
          +----------+         |
                          [ Network ]
                               |
                    +----------+---------+
                    | AuditLogger        |
                    | (signed, JSONL)    |
                    | OtelExporter       |
                    | MessageBus         |
                    +--------------------+
```

---

## Key Design Principles

### Secure-by-default

Every policy dataclass defaults to the most restrictive posture:

- `NetworkPolicy.default_deny = True`
- `ShellPolicy.enabled = False`
- `PluginPolicy.enabled = False`
- `FilesystemPolicy` starts with empty path lists

An operator must explicitly add entries to allowlists before any capability is
available.

### Single network enforcement point

All outbound HTTP traffic -- whether initiated by a provider, a tool, a plugin,
or the Discord channel -- passes through `PolicyHTTPClient`
(`missy/gateway/client.py`).  Before any network I/O occurs, the client
extracts the destination hostname from the URL and calls
`get_policy_engine().check_network(host)`.  If the host is not on an allowlist,
a `PolicyViolationError` is raised and no request leaves the machine.

The Anthropic and OpenAI providers use their own SDKs for HTTP, but their API
hosts must still appear in `network.allowed_hosts` for the initial policy check
at the gateway layer.  The Ollama provider routes directly through
`PolicyHTTPClient`.

### Audit everything

The `EventBus` (`missy/core/events.py`) is a process-level singleton.  Every
subsystem publishes `AuditEvent` instances with:

- `timestamp` (UTC, timezone-aware)
- `session_id` / `task_id` (correlation)
- `event_type` (dotted string, e.g. `agent.run.complete`)
- `category` (one of: `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider`, `security`, `agent`, `tool`, `mcp`, `vision`)
- `result` (one of: `allow`, `deny`, `error`)
- `detail` (structured dict)
- `policy_rule` (optional rule name)

The `AuditLogger` (`missy/observability/audit_logger.py`) wraps the bus's
`publish` method to intercept every event and append it as a JSON line to the
audit log file.

---

## Module Dependency Graph

The following shows the primary import dependencies between subsystems.
Arrows point from importer to importee.

```
cli/main
  +-> config/settings + config/migrate + config/hotreload + config/plan
  +-> policy/engine
  +-> observability/audit_logger + observability/otel
  +-> providers/registry
  +-> agent/runtime
  +-> scheduler/manager
  +-> plugins/loader
  +-> security/sanitizer + security/secrets + security/identity + security/drift
  +-> security/landlock + security/scanner + security/vault + security/trust
  +-> channels/cli_channel + channels/discord/* + channels/voice/* + channels/screencast/*
  +-> mcp/manager
  +-> api/server
  +-> vision/*

agent/runtime
  +-> providers/registry + providers/base
  +-> core/session + core/events + core/message_bus
  +-> tools/registry
  +-> agent/attention + agent/context + agent/circuit_breaker
  +-> agent/playbook + agent/consolidation + agent/condensers
  +-> agent/done_criteria + agent/learnings + agent/prompt_patches
  +-> agent/progress + agent/interactive_approval + agent/approval
  +-> agent/sub_agent + agent/persona + agent/behavior + agent/hatching
  +-> agent/checkpoint + agent/cost_tracker + agent/failure_tracker
  +-> agent/structured_output + agent/sleeptime + agent/watchdog
  +-> agent/code_evolution + agent/proactive
  +-> memory/synthesizer + memory/sqlite_store + memory/vector_store + memory/graph_store
  +-> security/sanitizer + security/secrets + security/censor + security/drift

providers/registry
  +-> providers/base
  +-> providers/anthropic_provider + openai_provider + ollama_provider
  +-> providers/rate_limiter
  +-> config/settings

gateway/client
  +-> policy/engine + policy/rest_policy
  +-> agent/interactive_approval
  +-> core/events

policy/engine
  +-> policy/network + policy/filesystem + policy/shell + policy/rest_policy
  +-> policy/presets
  +-> config/settings

mcp/manager
  +-> mcp/client + mcp/digest
  +-> gateway/client
  +-> tools/registry
  +-> core/events

memory/sqlite_store + memory/vector_store + memory/graph_store
  +-> memory/store (base)

scheduler/manager
  +-> scheduler/parser + scheduler/jobs
  +-> agent/runtime (lazy)
  +-> core/events

vision/*
  +-> providers/base (for image formatting)
  +-> core/events
```

### Initialisation Order

The `_load_subsystems()` function in `cli/main.py` enforces a strict
initialisation sequence:

1. `load_config(path)` -- parse YAML into `MissyConfig`
2. `init_policy_engine(cfg)` -- must come first; other subsystems depend on it
3. `init_audit_logger(cfg.audit_log_path)` -- wraps the event bus
4. `init_registry(cfg)` -- constructs provider instances

All three are module-level singletons protected by `threading.Lock`, so they
are safe to initialise from any thread but must be initialised before use.

---

## Subsystem Singletons

Every major subsystem follows the same pattern:

- A module-level `_instance: Optional[T] = None` guarded by a `threading.Lock`.
- An `init_*()` function that constructs and installs the singleton.
- A `get_*()` function that returns the singleton or raises `RuntimeError`.

| Subsystem | init function | get function |
|---|---|---|
| Policy engine | `init_policy_engine(cfg)` | `get_policy_engine()` |
| Provider registry | `init_registry(cfg)` | `get_registry()` |
| Audit logger | `init_audit_logger(path)` | `get_audit_logger()` |
| Plugin loader | `init_plugin_loader(cfg)` | `get_plugin_loader()` |
| Skill registry | `init_skill_registry()` | `get_skill_registry()` |
| Tool registry | `init_tool_registry()` | `get_tool_registry()` |
| Message bus | `init_message_bus()` | `get_message_bus()` |
| Agent identity | `init_identity(path)` | `get_identity()` |

This pattern ensures that subsystems are explicitly initialised (never auto-
created on first access) and that the initialisation order is visible in the
calling code.
