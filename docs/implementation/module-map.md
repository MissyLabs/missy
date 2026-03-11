# Module Map

Complete map of every module in the `missy/` package. Each entry lists the
module path, its purpose, the key classes or functions it exports, and its
dependencies on other `missy` modules.

---

## missy.core

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
| **Purpose** | Singleton registry mapping provider name strings to live `BaseProvider` instances. Factory method builds a registry from `MissyConfig`. |
| **Key exports** | `ProviderRegistry`, `init_registry()`, `get_registry()` |
| **Internal deps** | `missy.config.settings` (`MissyConfig`, `ProviderConfig`), `missy.providers.anthropic_provider`, `missy.providers.openai_provider`, `missy.providers.ollama_provider`, `missy.providers.base` |

---

## missy.agent

### missy.agent.runtime

| Field | Value |
|-------|-------|
| **Purpose** | Top-level agent orchestrator: resolves a provider, builds messages, calls the provider, and returns the model reply. |
| **Key exports** | `AgentRuntime`, `AgentConfig` (dataclass) |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `event_bus`), `missy.core.exceptions` (`ProviderError`), `missy.core.session` (`Session`, `SessionManager`), `missy.providers.base` (`Message`, `CompletionResponse`), `missy.providers.registry` (`get_registry`), `missy.tools.registry` (`get_tool_registry`) |

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

### missy.tools.builtin.calculator

| Field | Value |
|-------|-------|
| **Purpose** | Built-in calculator tool (safe math expression evaluation). |
| **Key exports** | `CalculatorTool` |
| **Internal deps** | `missy.tools.base` |

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

## missy.memory

### missy.memory.store

| Field | Value |
|-------|-------|
| **Purpose** | JSON-based persistence for conversation turns. Provides per-session history retrieval. |
| **Key exports** | `MemoryStore`, `ConversationTurn` (dataclass) |
| **Internal deps** | None |

---

## missy.observability

### missy.observability.audit_logger

| Field | Value |
|-------|-------|
| **Purpose** | Subscribes to every event on `event_bus` and appends a JSONL line to the audit log file. Provides `get_recent_events()` and `get_policy_violations()` read methods. |
| **Key exports** | `AuditLogger`, `init_audit_logger()`, `get_audit_logger()` |
| **Internal deps** | `missy.core.events` (`AuditEvent`, `EventBus`, `event_bus`) |

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
| **Purpose** | Secrets detection and redaction to prevent credential leakage in logs, audit trails, and provider payloads. |
| **Key exports** | `SecretsDetector`, `secrets_detector` (module-level singleton) |
| **Internal deps** | None |

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

---

## missy.cli

### missy.cli.main

| Field | Value |
|-------|-------|
| **Purpose** | CLI entry point (`missy` command). Subcommands include `chat`, `audit recent`, `audit security`, `schedule`, `gateway start`. |
| **Key exports** | `main()` |
| **Internal deps** | Nearly all other modules (transitive). |

---

## Dependency Graph (summary)

```
missy.core.events           <-- no internal deps (foundation)
missy.core.exceptions       <-- no internal deps (foundation)
missy.core.session          <-- no internal deps (foundation)
missy.config.settings       <-- core.exceptions, channels.discord.config (lazy)
missy.policy.*              <-- config.settings, core.events, core.exceptions
missy.gateway.client        <-- policy.engine, core.events
missy.providers.base        <-- no internal deps
missy.providers.*_provider  <-- providers.base, config.settings, core.events, core.exceptions
missy.providers.registry    <-- config.settings, providers.base, all concrete providers
missy.agent.runtime         <-- core.events, core.exceptions, core.session,
                                providers.base, providers.registry, tools.registry
missy.tools.*               <-- core.events, core.exceptions, policy.engine, tools.base
missy.skills.*              <-- core.events, skills.base
missy.plugins.*             <-- config.settings, core.events, core.exceptions, plugins.base
missy.scheduler.*           <-- core.events, core.exceptions, scheduler.jobs, scheduler.parser,
                                agent.runtime (lazy)
missy.memory.store          <-- no internal deps
missy.observability.*       <-- core.events
missy.security.*            <-- no internal deps
missy.channels.*            <-- channels.base, core.events, gateway.client, discord.config
```
