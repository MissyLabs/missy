# Audit Event System

Every policy decision and significant runtime action in Missy is recorded
as an `AuditEvent`. Events flow through an in-process `EventBus`, are
persisted to a JSONL file by `AuditLogger`, and can be queried via the
CLI or programmatically.

**Source files:**

- `missy/core/events.py` -- `AuditEvent`, `EventBus`, `event_bus` singleton
- `missy/observability/audit_logger.py` -- `AuditLogger`, JSONL persistence

---

## AuditEvent Schema

```python
@dataclass
class AuditEvent:
    timestamp: datetime         # UTC, timezone-aware (required)
    session_id: str             # Session that generated the event
    task_id: str                # Task within the session
    event_type: str             # Dotted string (e.g. "network_check")
    category: EventCategory     # "network" | "filesystem" | "shell" | "plugin" | "scheduler" | "provider"
    result: EventResult         # "allow" | "deny" | "error"
    detail: dict[str, Any]      # Structured data specific to the event
    policy_rule: str | None     # The rule that produced the result (e.g. "cidr:10.0.0.0/8")
```

### Construction

The recommended constructor is `AuditEvent.now(...)`, which fills
`timestamp` automatically:

```python
event = AuditEvent.now(
    session_id="abc-123",
    task_id="task-456",
    event_type="network_check",
    category="network",
    result="allow",
    policy_rule="domain:*.github.com",
    detail={"host": "api.github.com"},
)
```

The `timestamp` must be timezone-aware; `__post_init__` raises `ValueError`
if `tzinfo` is `None`.

---

## Complete Event Type Catalogue

### Network Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `network_check` | `network` | `allow` / `deny` | `NetworkPolicyEngine.check_host()` | `host` |
| `network_request` | `network` | `allow` | `PolicyHTTPClient._emit_request_event()` | `method`, `url`, `status_code` |

### Filesystem Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `filesystem_read` | `filesystem` | `allow` / `deny` | `FilesystemPolicyEngine.check_read()` | `path`, `operation` |
| `filesystem_write` | `filesystem` | `allow` / `deny` | `FilesystemPolicyEngine.check_write()` | `path`, `operation` |

### Shell Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `shell_check` | `shell` | `allow` / `deny` | `ShellPolicyEngine.check_command()` | `command` |

### Plugin Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `plugin.load` | `plugin` | `allow` / `deny` / `error` | `PluginLoader.load_plugin()` | `plugin`, `reason`, `manifest`, `error` |
| `plugin.execute` | `plugin` | `allow` / `deny` / `error` | `PluginLoader.execute()` | `plugin`, `reason`, `error` |
| `plugin.execute.start` | `plugin` | `allow` | `PluginLoader.execute()` | `plugin` |

### Tool Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `tool_execute` | `plugin` | `allow` / `deny` / `error` | `ToolRegistry.execute()` | `tool`, `message` |

### Skill Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `skill.execute` | `plugin` | `allow` / `error` | `SkillRegistry.execute()` | `skill`, `message` |

### Provider Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `provider_invoke` | `provider` | `allow` / `error` | `AnthropicProvider.complete()` (and other providers) | `provider`, `model`, `message` |

### Agent Runtime Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `agent.run.start` | `provider` | `allow` | `AgentRuntime.run()` | `user_input_length` |
| `agent.run.complete` | `provider` | `allow` | `AgentRuntime.run()` | `provider`, `model`, `usage` |
| `agent.run.error` | `provider` | `error` | `AgentRuntime.run()` | `error`, `stage`, `provider` |

### Scheduler Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `scheduler.start` | `scheduler` | `allow` | `SchedulerManager.start()` | `job_count` |
| `scheduler.stop` | `scheduler` | `allow` | `SchedulerManager.stop()` | (empty) |
| `scheduler.job.add` | `scheduler` | `allow` / `error` | `SchedulerManager.add_job()` | `job_id`, `name`, `schedule`, `provider`, `error` |
| `scheduler.job.remove` | `scheduler` | `allow` | `SchedulerManager.remove_job()` | `job_id`, `name` |
| `scheduler.job.pause` | `scheduler` | `allow` | `SchedulerManager.pause_job()` | `job_id`, `name` |
| `scheduler.job.resume` | `scheduler` | `allow` | `SchedulerManager.resume_job()` | `job_id`, `name` |
| `scheduler.job.run.start` | `scheduler` | `allow` | `SchedulerManager._run_job()` | `job_id`, `name`, `provider`, `run_count` |
| `scheduler.job.run.complete` | `scheduler` | `allow` | `SchedulerManager._run_job()` | `job_id`, `name`, `run_count` |
| `scheduler.job.run.error` | `scheduler` | `error` | `SchedulerManager._run_job()` | `job_id`, `name`, `error` |

### Discord Channel Events

| event_type | category | result | Emitter | detail keys |
|------------|----------|--------|---------|-------------|
| `discord.channel.message_received` | `network` | `allow` | `DiscordChannel._handle_message()` | `author_id`, `channel_id`, `guild_id` |
| `discord.channel.message_denied` | `network` | `deny` | `DiscordChannel._check_dm_policy()`, `_check_guild_policy()` | `reason`, `author_id`, `guild_id` |
| `discord.channel.bot_filtered` | `network` | `deny` | `DiscordChannel._handle_message()` | `author_id`, `is_bot` |
| `discord.channel.allowlist_denied` | `network` | `deny` | `DiscordChannel._check_dm_policy()`, `_check_guild_policy()` | `reason`, `author_id`, `guild_id`, `channel_name` |
| `discord.channel.require_mention_filtered` | `network` | `deny` | `DiscordChannel._check_guild_policy()` | `reason`, `guild_id`, `author_id` |
| `discord.channel.pairing_wait` | `network` | `allow` | `DiscordChannel._check_pairing()` | `author_id` |
| `discord.channel.attachment_denied` | `network` | `deny` | `DiscordChannel._handle_message()` | `author_id`, `channel_id`, `attachment_count`, `reason` |
| `discord.channel.reply_sent` | `network` | `allow` / `error` | `DiscordChannel.send_to()` | `channel_id`, `reply_to`, `error` |

---

## JSONL Persistence Format

`AuditLogger` appends one JSON object per line to `~/.missy/audit.jsonl`.
Each line contains all `AuditEvent` fields with `timestamp` serialised as
ISO-8601.

### Example Lines

```jsonl
{"timestamp": "2026-03-11T09:00:00.123456+00:00", "session_id": "abc-123", "task_id": "task-456", "event_type": "network_check", "category": "network", "result": "allow", "detail": {"host": "api.anthropic.com"}, "policy_rule": "domain:*.anthropic.com"}
{"timestamp": "2026-03-11T09:00:00.234567+00:00", "session_id": "abc-123", "task_id": "task-456", "event_type": "network_request", "category": "network", "result": "allow", "detail": {"method": "POST", "url": "https://api.anthropic.com/v1/messages", "status_code": 200}, "policy_rule": null}
{"timestamp": "2026-03-11T09:00:00.345678+00:00", "session_id": "abc-123", "task_id": "task-456", "event_type": "provider_invoke", "category": "provider", "result": "allow", "detail": {"provider": "anthropic", "model": "claude-sonnet-4-6", "message": "completion successful"}, "policy_rule": null}
{"timestamp": "2026-03-11T09:00:00.456789+00:00", "session_id": "abc-123", "task_id": "task-456", "event_type": "agent.run.complete", "category": "provider", "result": "allow", "detail": {"provider": "anthropic", "model": "claude-sonnet-4-6", "usage": {"prompt_tokens": 42, "completion_tokens": 128, "total_tokens": 170}}, "policy_rule": null}
{"timestamp": "2026-03-11T09:05:00.000000+00:00", "session_id": "s2", "task_id": "t2", "event_type": "shell_check", "category": "shell", "result": "deny", "detail": {"command": "rm -rf /"}, "policy_rule": null}
{"timestamp": "2026-03-11T09:10:00.000000+00:00", "session_id": "discord", "task_id": "channel", "event_type": "discord.channel.bot_filtered", "category": "network", "result": "deny", "detail": {"author_id": "111222333", "is_bot": true}, "policy_rule": null}
```

---

## How AuditLogger Works

`AuditLogger` subscribes to all events by wrapping
`EventBus.publish()`. At construction time, it replaces
`event_bus.publish` with a patched version that:

1. Calls the original `publish()` (preserving subscriber dispatch)
2. Calls `_handle_event(event)` which serialises the event to JSON and
   appends a line to the log file

This wrapping approach is used because `EventBus` does not support
wildcard subscriptions -- there is no way to subscribe to "all event
types" through the normal `subscribe()` API.

### Read Methods

```python
logger = get_audit_logger()

# Get the last 100 events (chronological order)
events = logger.get_recent_events(limit=100)

# Get up to 100 policy violations (result == "deny")
violations = logger.get_policy_violations(limit=100)
```

`get_recent_events()` reads the entire file, takes the last `limit` non-
empty lines, and parses each as JSON.

`get_policy_violations()` reads the entire file in reverse, collecting up
to `limit` lines where `result == "deny"`, then reverses to return them
in chronological order.

---

## CLI Querying

```bash
# Recent events (default last 100)
missy audit recent

# Security events (policy violations)
missy audit security
```

These commands call `get_audit_logger().get_recent_events()` and
`get_audit_logger().get_policy_violations()` respectively.

---

## EventBus Internals

The `EventBus` (in `missy/core/events.py`) is thread-safe:

- A `threading.Lock` protects the internal subscriber list and event log
- `publish()` appends the event to an in-memory log, then dispatches to
  registered callbacks synchronously in registration order
- Callback exceptions are caught, logged, and do not propagate
- `get_events()` returns a filtered snapshot (AND logic on all provided
  keyword filters)
- `subscribe(event_type, callback)` registers a callback for a specific
  event type string (no wildcards)
- `clear()` resets all state (for tests)

The module-level `event_bus` singleton is the process-wide instance used
by all components.
