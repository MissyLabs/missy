# Memory and Persistence

Missy persists several types of data to disk: conversation history, scheduled
jobs, and audit events.  This document covers the storage format, schema, and
migration guidance for each.

---

## File Locations

| File | Default Path | Purpose |
|---|---|---|
| `config.yaml` | `~/.missy/config.yaml` | Configuration (YAML) |
| `memory.json` | `~/.missy/memory.json` | Conversation history (JSON) |
| `jobs.json` | `~/.missy/jobs.json` | Scheduled job definitions (JSON) |
| `audit.jsonl` | `~/.missy/audit.jsonl` | Audit event log (JSONL) |

All paths support tilde expansion.  The `audit_log_path` is configurable in
the YAML file.  The memory and jobs paths are currently hardcoded as defaults
but accept a constructor argument.

The `~/.missy/` directory and its contents are created by `missy init`.

---

## Conversation Memory

### Overview

The `MemoryStore` (`missy/memory/store.py`) persists conversation turns to a
JSON file at `~/.missy/memory.json`.  All turns are kept in memory as a list
and flushed to disk on every write operation.

### ConversationTurn Schema

Each turn is serialised by `ConversationTurn.to_dict()`:

```json
{
  "id": "uuid-string",
  "session_id": "uuid-string",
  "timestamp": "2026-03-11T09:00:00.123456+00:00",
  "role": "user",
  "content": "What is the weather today?",
  "provider": ""
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique turn identifier (UUID, generated at creation) |
| `session_id` | string | Session this turn belongs to |
| `timestamp` | ISO-8601 string | UTC timestamp of the turn |
| `role` | string | Speaker role: `"user"` or `"assistant"` |
| `content` | string | The message text |
| `provider` | string | AI provider name (empty for user turns) |

### File Format

The file contains a JSON array of turn objects:

```json
[
  {"id": "...", "session_id": "...", "timestamp": "...", "role": "user", ...},
  {"id": "...", "session_id": "...", "timestamp": "...", "role": "assistant", ...}
]
```

### API

```python
from missy.memory.store import MemoryStore

store = MemoryStore()                                    # Default path
store = MemoryStore(store_path="~/.missy/memory.json")   # Explicit path

# Write
turn = store.add_turn(
    session_id="session-abc",
    role="user",
    content="What is the weather today?",
    provider="",
)

# Read
session_turns = store.get_session_turns("session-abc", limit=50)
recent_turns = store.get_recent_turns(limit=10)

# Clear a session
store.clear_session("session-abc")
```

### Persistence Behaviour

- `add_turn()` appends a turn and immediately saves to disk.
- `clear_session()` removes all turns for the given session and saves.
- On construction, the store loads all existing turns from disk.
- Malformed records are skipped with a warning during load.
- If the file does not exist, the store starts empty.
- The parent directory is created automatically on save.
- Write errors are logged but not re-raised, so callers are not disrupted.

---

## Scheduled Jobs Persistence

### Overview

The `SchedulerManager` (`missy/scheduler/manager.py`) persists job definitions
to `~/.missy/jobs.json`.

### ScheduledJob Schema

Each job is serialised by `ScheduledJob.to_dict()` (`missy/scheduler/jobs.py`):

```json
{
  "id": "uuid-string",
  "name": "Daily digest",
  "description": "",
  "schedule": "daily at 09:00",
  "task": "Summarise the top 5 news stories.",
  "provider": "anthropic",
  "enabled": true,
  "created_at": "2026-03-11T09:00:00+00:00",
  "last_run": "2026-03-11T09:00:05+00:00",
  "next_run": "2026-03-12T09:00:00+00:00",
  "run_count": 1,
  "last_result": "Here are today's top stories..."
}
```

See [SCHEDULER.md](SCHEDULER.md) for a detailed field description.

### File Format

The file contains a JSON array of job objects:

```json
[
  {"id": "...", "name": "Daily digest", ...},
  {"id": "...", "name": "Weekly report", ...}
]
```

### Persistence Behaviour

- Jobs are saved after every mutation (add, remove, pause, resume, run
  completion).
- On `start()`, all persisted jobs are loaded and re-registered with
  APScheduler.
- Malformed records are skipped with a warning.
- If the file does not exist or contains `[]`, the scheduler starts with no
  jobs.
- Write errors are logged but not re-raised to avoid interrupting the
  scheduler background thread.

### Recovery

If `jobs.json` becomes corrupted:

```bash
# Reset to empty
echo "[]" > ~/.missy/jobs.json

# Re-add jobs
missy schedule add --name "My job" --schedule "daily at 09:00" --task "..."
```

---

## Audit Log

### Overview

The `AuditLogger` (`missy/observability/audit_logger.py`) writes structured
audit events to a JSONL (newline-delimited JSON) file at
`~/.missy/audit.jsonl`.

### Audit Event Schema

Each line in the file is a self-contained JSON object:

```json
{
  "timestamp": "2026-03-11T09:00:00.123456+00:00",
  "session_id": "uuid-string",
  "task_id": "uuid-string",
  "event_type": "agent.run.complete",
  "category": "provider",
  "result": "allow",
  "detail": {
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
  },
  "policy_rule": null
}
```

| Field | Type | Values | Description |
|---|---|---|---|
| `timestamp` | ISO-8601 string | -- | UTC timestamp |
| `session_id` | string | -- | Session that generated the event |
| `task_id` | string | -- | Task within the session |
| `event_type` | string | Dotted name | Type of event (see table below) |
| `category` | string | `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider` | Broad category |
| `result` | string | `allow`, `deny`, `error` | Outcome of the action |
| `detail` | object | -- | Structured data specific to the event type |
| `policy_rule` | string or null | -- | Policy rule that produced the result |

### Event Types

| Event Type | Category | Description |
|---|---|---|
| `agent.run.start` | `provider` | Agent run begins |
| `agent.run.complete` | `provider` | Agent run completes successfully |
| `agent.run.error` | `provider` | Agent run fails |
| `provider_invoke` | `provider` | Provider completion call |
| `network_request` | `network` | HTTP request made via PolicyHTTPClient |
| `skill.execute` | `plugin` | Skill execution |
| `plugin.load` | `plugin` | Plugin load attempt |
| `plugin.execute` | `plugin` | Plugin execution |
| `plugin.execute.start` | `plugin` | Plugin execution begins |
| `scheduler.start` | `scheduler` | Scheduler starts |
| `scheduler.stop` | `scheduler` | Scheduler stops |
| `scheduler.job.add` | `scheduler` | Job created |
| `scheduler.job.remove` | `scheduler` | Job removed |
| `scheduler.job.pause` | `scheduler` | Job paused |
| `scheduler.job.resume` | `scheduler` | Job resumed |
| `scheduler.job.run.start` | `scheduler` | Scheduled job execution begins |
| `scheduler.job.run.complete` | `scheduler` | Scheduled job execution succeeds |
| `scheduler.job.run.error` | `scheduler` | Scheduled job execution fails |

### Viewing the Audit Log

**Via CLI**:

```bash
# Recent events (all categories)
missy audit recent

# Filter by category
missy audit recent --limit 20 --category network

# Policy violations only
missy audit security
missy audit security --limit 100
```

**Via jq**:

```bash
# All policy denials
jq 'select(.result == "deny")' ~/.missy/audit.jsonl

# Count events by category
jq -r '.category' ~/.missy/audit.jsonl | sort | uniq -c | sort -rn
```

### Log Rotation

The audit log grows without bound.  Use `logrotate` or a cron job to manage
its size:

```
# /etc/logrotate.d/missy
/home/USER/.missy/audit.jsonl {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

`copytruncate` is used because Missy appends to the file directly (no
reopen signal is needed).

---

## Schema Versioning

Missy does not currently embed a schema version number in its persistence
files.  The schemas are defined implicitly by the `to_dict()` / `from_dict()`
methods on `ConversationTurn` and `ScheduledJob`.

### Forward Compatibility

The `from_dict()` methods on both classes use `.get()` with defaults for all
fields.  This means:

- New fields added in future versions are optional and default safely.
- Old persistence files without new fields will load without error.

### Migration Guidance

If the schema changes in a future release:

1. **Adding a field**: No migration needed.  The `from_dict()` method provides
   a default value for missing fields.
2. **Renaming a field**: The new `from_dict()` should check for both the old
   and new field names.
3. **Removing a field**: The old field is silently ignored during deserialisation.
4. **Changing a field type**: A migration script should be provided to convert
   existing data.

For breaking changes, the recommended approach is:

```bash
# Back up existing data
cp ~/.missy/memory.json ~/.missy/memory.json.backup
cp ~/.missy/jobs.json ~/.missy/jobs.json.backup

# Upgrade Missy
pip install --upgrade missy

# If migration is needed, run:
# missy migrate  (hypothetical future command)
```

Currently, no migrations have been needed.  The schema has been stable since
the initial release.
