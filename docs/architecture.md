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
  core/            Session management, event bus, exception hierarchy
  config/          YAML configuration loading and policy dataclasses
  policy/          Network, filesystem, and shell policy engines + facade
  gateway/         PolicyHTTPClient -- the single network enforcement point
  agent/           AgentRuntime -- orchestrates provider calls per session
  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry
  tools/           Tool base class and registry (builtin tools)
  skills/          Lightweight in-process callable skills and registry
  plugins/         Security-gated external plugin loader and base class
  scheduler/       APScheduler integration, human schedule parsing, job persistence
  memory/          JSON-based per-session conversation history
  observability/   AuditLogger -- JSONL audit trail writer
  security/        InputSanitizer, SecretsDetector (prompt hygiene)
  channels/        I/O channel abstractions (CLI, Discord)
  cli/             Click + Rich command-line interface
```

---

## Key Data Flow

The following path describes what happens when a user runs `missy ask "..."`:

```
1. CLI (click)           Parse command-line arguments, resolve --config path
       |
2. Config loader         Read YAML, build MissyConfig with policy dataclasses
       |
3. Subsystem init        init_policy_engine(cfg)
       |                 init_audit_logger(cfg.audit_log_path)
       |                 init_registry(cfg)
       |
4. Security checks       SecretsDetector scans prompt for credentials
       |                 InputSanitizer truncates and checks for injection
       |
5. AgentRuntime.run()    Resolve or create a Session
       |                 Resolve provider (with fallback)
       |                 Build message list (system prompt + user input)
       |
6. Provider.complete()   Provider-specific SDK call (Anthropic, OpenAI, Ollama)
       |                 Ollama routes through PolicyHTTPClient -> policy check
       |
7. Audit events          Every step emits AuditEvent -> EventBus -> AuditLogger
       |                 Events appended to ~/.missy/audit.jsonl
       |
8. Response              Plain text returned to CLI, rendered with Rich
```

### ASCII Flow Diagram

```
  User
   |
   v
+-------+     +--------+     +---------------+     +---------------+
|  CLI  | --> | Config | --> | Policy Engine | --> | Agent Runtime |
+-------+     +--------+     +-------+-------+     +-------+-------+
                                     |                     |
                              +------+------+        +-----+------+
                              | Audit       |        | Provider   |
                              | Logger      |        | Registry   |
                              +------+------+        +-----+------+
                                     |                     |
                              +------+------+        +-----+------+
                              | audit.jsonl |        | Anthropic  |
                              +-------------+        | OpenAI     |
                                                     | Ollama     |
                                                     +-----+------+
                                                           |
                                                     +-----+--------+
                                                     | PolicyHTTP   |
                                                     | Client       |
                                                     | (gateway)    |
                                                     +--------------+
                                                           |
                                                      [ Network ]
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
- `category` (one of: `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider`)
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
  +-> config/settings
  +-> policy/engine
  +-> observability/audit_logger
  +-> providers/registry
  +-> agent/runtime
  +-> scheduler/manager
  +-> plugins/loader
  +-> security/sanitizer
  +-> security/secrets
  +-> channels/cli_channel
  +-> channels/discord/*

agent/runtime
  +-> providers/registry
  +-> providers/base
  +-> core/session
  +-> core/events
  +-> tools/registry

providers/registry
  +-> providers/base
  +-> providers/anthropic_provider
  +-> providers/openai_provider
  +-> providers/ollama_provider
  +-> config/settings

providers/ollama_provider
  +-> gateway/client

gateway/client
  +-> policy/engine
  +-> core/events

policy/engine
  +-> policy/network
  +-> policy/filesystem
  +-> policy/shell
  +-> config/settings

scheduler/manager
  +-> scheduler/parser
  +-> scheduler/jobs
  +-> agent/runtime  (lazy import to avoid circular dependency)
  +-> core/events

plugins/loader
  +-> plugins/base
  +-> config/settings
  +-> core/events
  +-> core/exceptions

skills/registry
  +-> skills/base
  +-> core/events

observability/audit_logger
  +-> core/events

memory/store
  (no internal dependencies beyond stdlib)
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

This pattern ensures that subsystems are explicitly initialised (never auto-
created on first access) and that the initialisation order is visible in the
calling code.
