# Persistence Schemas

Missy persists data in four files. This document defines the complete
schema for each, with annotated examples and migration guidance.

---

## 1. config.yaml

**Location:** User-specified (typically `~/.missy/config.yaml` or
`missy.yaml` in the project root)

**Parser:** `missy.config.settings.load_config(path)`

### Full Annotated Example

```yaml
# ---------------------------------------------------------------
# Network policy
# ---------------------------------------------------------------
network:
  # When true (default), all outbound requests are blocked unless
  # the destination is explicitly listed below.
  default_deny: true

  # CIDR blocks that are always reachable (e.g. private ranges).
  allowed_cidrs:
    - "10.0.0.0/8"
    - "172.16.0.0/12"
    - "127.0.0.1/32"

  # Domain names and wildcard patterns.
  # "*.github.com" matches api.github.com, github.com, etc.
  allowed_domains:
    - "*.anthropic.com"
    - "*.openai.com"
    - "*.github.com"

  # Explicit host or host:port entries.
  allowed_hosts:
    - "api.anthropic.com:443"

  # Per-category host overrides (merged with allowed_domains/hosts
  # at the caller's discretion).
  provider_allowed_hosts:
    - "localhost:11434"        # Ollama
  tool_allowed_hosts: []
  discord_allowed_hosts:
    - "discord.com"
    - "gateway.discord.gg"

# ---------------------------------------------------------------
# Filesystem policy
# ---------------------------------------------------------------
filesystem:
  # Absolute paths the agent may read from.
  allowed_read_paths:
    - "/home/user/workspace"
    - "/home/user/.missy"

  # Absolute paths the agent may write to.
  allowed_write_paths:
    - "/home/user/workspace/output"
    - "/home/user/.missy"

# ---------------------------------------------------------------
# Shell policy
# ---------------------------------------------------------------
shell:
  # Master switch: false = no shell commands ever (default).
  enabled: false

  # Basename allow-list (only effective when enabled: true).
  allowed_commands:
    - "git"
    - "ls"
    - "cat"
    - "grep"

# ---------------------------------------------------------------
# Plugin policy
# ---------------------------------------------------------------
plugins:
  # Master switch: false = no plugins may be loaded (default).
  enabled: false

  # Explicit plugin names permitted when enabled: true.
  allowed_plugins:
    - "weather"

# ---------------------------------------------------------------
# Provider configurations
# ---------------------------------------------------------------
providers:
  anthropic:
    name: anthropic
    model: claude-sonnet-4-6
    # api_key: null  -- falls back to ANTHROPIC_API_KEY env var
    timeout: 30
    enabled: true

  openai:
    name: openai
    model: gpt-4o
    # api_key: null  -- falls back to OPENAI_API_KEY env var
    timeout: 30
    enabled: false

  ollama:
    name: ollama
    model: llama3
    base_url: "http://localhost:11434"
    timeout: 60
    enabled: false

# ---------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------
scheduling:
  enabled: true
  max_jobs: 0   # 0 = unlimited

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
workspace_path: "/home/user/workspace"
audit_log_path: "~/.missy/audit.jsonl"

# ---------------------------------------------------------------
# Discord integration (optional)
# ---------------------------------------------------------------
discord:
  enabled: false
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      application_id: "1234567890123456789"
      account_id: null     # auto-detected from READY event
      dm_policy: pairing   # pairing | allowlist | open | disabled
      dm_allowlist: []
      ack_reaction: ""
      ignore_bots: true
      allow_bots_if_mention_only: false
      guild_policies:
        "987654321098765432":
          enabled: true
          require_mention: true
          allowed_channels:
            - "general"
            - "bot-commands"
          allowed_roles: []
          allowed_users: []
          mode: full         # full | no_tools | safe_chat_only
```

### Schema Rules

| Section | Required | Default |
|---------|----------|---------|
| `network` | No | `default_deny: true`, empty allow-lists |
| `filesystem` | No | Empty allow-lists |
| `shell` | No | `enabled: false`, empty allow-list |
| `plugins` | No | `enabled: false`, empty allow-list |
| `providers` | No | Empty dict (no providers) |
| `scheduling` | No | `enabled: true`, `max_jobs: 0` |
| `workspace_path` | No | `"."` |
| `audit_log_path` | No | `"~/.missy/audit.log"` |
| `discord` | No | `None` |

Each provider entry must contain a `model` field; omitting it raises
`ConfigurationError`. The `api_key` field falls back to the
`{PROVIDER_KEY}_API_KEY` environment variable at load time.

---

## 2. audit.jsonl

**Location:** `~/.missy/audit.jsonl` (configurable via `audit_log_path`)

**Writer:** `missy.observability.audit_logger.AuditLogger`

### JSON Schema

```json
{
  "type": "object",
  "required": ["timestamp", "session_id", "task_id", "event_type", "category", "result"],
  "properties": {
    "timestamp":   {"type": "string", "format": "date-time"},
    "session_id":  {"type": "string"},
    "task_id":     {"type": "string"},
    "event_type":  {"type": "string"},
    "category":    {"type": "string", "enum": ["network", "filesystem", "shell", "plugin", "scheduler", "provider"]},
    "result":      {"type": "string", "enum": ["allow", "deny", "error"]},
    "detail":      {"type": "object"},
    "policy_rule": {"type": ["string", "null"]}
  }
}
```

### Example Line

```json
{"timestamp": "2026-03-11T14:22:01.456789+00:00", "session_id": "a1b2c3d4", "task_id": "e5f6g7h8", "event_type": "network_check", "category": "network", "result": "allow", "detail": {"host": "api.anthropic.com"}, "policy_rule": "domain:*.anthropic.com"}
```

---

## 3. jobs.json

**Location:** `~/.missy/jobs.json`

**Writer:** `missy.scheduler.manager.SchedulerManager`

### JSON Schema

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "required": ["id", "name", "schedule", "task"],
    "properties": {
      "id":          {"type": "string", "format": "uuid"},
      "name":        {"type": "string"},
      "description": {"type": "string"},
      "schedule":    {"type": "string"},
      "task":        {"type": "string"},
      "provider":    {"type": "string", "default": "anthropic"},
      "enabled":     {"type": "boolean", "default": true},
      "created_at":  {"type": ["string", "null"], "format": "date-time"},
      "last_run":    {"type": ["string", "null"], "format": "date-time"},
      "next_run":    {"type": ["string", "null"], "format": "date-time"},
      "run_count":   {"type": "integer", "default": 0},
      "last_result": {"type": ["string", "null"]}
    }
  }
}
```

### Example

```json
[
  {
    "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "name": "Morning summary",
    "description": "Generate a daily agenda summary",
    "schedule": "daily at 09:00",
    "task": "Summarise today's agenda from my calendar.",
    "provider": "anthropic",
    "enabled": true,
    "created_at": "2026-01-15T08:30:00+00:00",
    "last_run": "2026-03-10T09:00:02+00:00",
    "next_run": "2026-03-11T09:00:00",
    "run_count": 54,
    "last_result": "Here is your agenda for today..."
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Weekly report",
    "description": "",
    "schedule": "weekly on friday 17:00",
    "task": "Generate a summary of this week's git commits.",
    "provider": "anthropic",
    "enabled": false,
    "created_at": "2026-02-01T10:00:00+00:00",
    "last_run": "2026-03-07T17:00:01+00:00",
    "next_run": null,
    "run_count": 5,
    "last_result": "## Weekly Report\n\n..."
  }
]
```

---

## 4. memory.json

**Location:** `~/.missy/memory.json`

**Writer:** `missy.memory.store.MemoryStore`

### JSON Schema

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "required": ["id", "session_id", "role", "content"],
    "properties": {
      "id":         {"type": "string", "format": "uuid"},
      "session_id": {"type": "string"},
      "timestamp":  {"type": ["string", "null"], "format": "date-time"},
      "role":       {"type": "string", "enum": ["user", "assistant", "system"]},
      "content":    {"type": "string"},
      "provider":   {"type": "string", "default": ""}
    }
  }
}
```

### Example

```json
[
  {
    "id": "c0ffee00-1234-5678-9abc-def012345678",
    "session_id": "sess-abc-123",
    "timestamp": "2026-03-11T14:00:00+00:00",
    "role": "user",
    "content": "What is the weather in San Francisco?",
    "provider": ""
  },
  {
    "id": "deadbeef-0000-1111-2222-333344445555",
    "session_id": "sess-abc-123",
    "timestamp": "2026-03-11T14:00:01.234567+00:00",
    "role": "assistant",
    "content": "I don't have access to real-time weather data, but ...",
    "provider": "anthropic"
  }
]
```

---

## Migration Strategy

All four persistence files are JSON-based and follow these evolution rules:

### Safe Changes (backward-compatible)

- **Add a new field with a default.** `from_dict()` and `to_dict()` in
  `ScheduledJob` and `ConversationTurn` use `.get(key, default)` so new
  fields are automatically populated with defaults when loading old data.
- **Add a new event type.** The JSONL audit log is append-only; new event
  types appear alongside old ones with no conflict.
- **Add a new config section.** `load_config()` uses `.get(key) or {}`
  for all sections, so missing sections produce safe defaults.

### Unsafe Changes (require migration)

- **Rename a field.** Old data would be loaded with the old name into a
  default value, losing the original data. Instead: add the new field,
  populate it from the old field in `from_dict()`, and keep reading the
  old field for backward compatibility.
- **Remove a field.** Old data would contain the extra field, which is
  harmless (Python `dict.get()` ignores extras). But code that relied on
  the field would break. Instead: deprecate by ignoring the field in
  `from_dict()`.
- **Change a field type.** Add a type coercion step in `from_dict()` to
  handle both old and new formats.

### Recommended Process

1. Write a migration function in a new module (e.g.
   `missy/migrations/v002_add_priority.py`)
2. Load the old file, transform each record, write the new file
3. Bump a `schema_version` field (not currently present -- add it as the
   first migration)
4. Have `_load_jobs()` / `_load()` check `schema_version` and apply
   pending migrations on startup

### Config Evolution

`config.yaml` follows the same principle: new sections and fields always
have defaults in the parser functions (`_parse_network`, etc.), so old
config files work without modification. Removing or renaming a config key
requires a deprecation period where both old and new keys are accepted.
