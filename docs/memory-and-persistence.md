# Memory and Persistence

Missy persists several types of data to disk: conversation history and
derived memory, scheduled jobs, and audit events.  This document covers the
storage format, schema, and subsystem responsibilities for each.

---

## File Locations

| File | Default Path | Purpose |
|---|---|---|
| `config.yaml` | `~/.missy/config.yaml` | Configuration (YAML) |
| `memory.db` | `~/.missy/memory.db` | Conversation turns, learnings, sessions, costs, summaries, entity graph (SQLite) |
| `memory.faiss` | `~/.missy/memory.faiss` | Optional FAISS semantic vector index (requires `.[vector]`) |
| `jobs.json` | `~/.missy/jobs.json` | Scheduled job definitions (JSON) |
| `audit.jsonl` | `~/.missy/audit.jsonl` | Audit event log (JSONL) |

All paths support tilde expansion.  The `audit_log_path` is configurable in
the YAML file.  `memory.db` and `memory.faiss` paths are hardcoded defaults
but accept a constructor argument on their respective store classes.

The `~/.missy/` directory and its contents are created by `missy init`.

---

## Conversation Memory

### Overview

`SQLiteMemoryStore` (`missy/memory/sqlite_store.py`) is the primary
conversation memory backend.  It persists turns, learnings, session
metadata, per-call cost records, a DAG of conversation summaries, and
overflow "large content" records (oversized tool results that don't fit
inline) to a single SQLite database at `~/.missy/memory.db`.  The database
runs in WAL mode with a thread-local connection per accessing thread.

Conversation turns are indexed for full-text search via a SQLite **FTS5**
virtual table (`turns_fts`), kept in sync with the `turns` table through
`AFTER INSERT`/`AFTER DELETE` triggers.  A parallel `summaries_fts` table
provides the same full-text search over the summary DAG.

### ConversationTurn Schema

Each turn is represented by the `ConversationTurn` dataclass
(`missy/memory/sqlite_store.py`):

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique turn identifier (UUID, generated at creation) |
| `session_id` | string | Session this turn belongs to |
| `timestamp` | ISO-8601 string | UTC timestamp of the turn |
| `role` | string | Speaker role: `"user"`, `"assistant"`, or `"tool"` |
| `content` | string | The message text |
| `provider` | string | AI provider name (empty for user turns) |
| `metadata` | dict | Arbitrary extra data (e.g. `{"pinned": true}`) |

### Store API

```python
from missy.memory.sqlite_store import SQLiteMemoryStore, ConversationTurn

store = SQLiteMemoryStore()                              # ~/.missy/memory.db
store = SQLiteMemoryStore(db_path="~/.missy/memory.db")   # Explicit path

# Write
turn = ConversationTurn.new(session_id="session-abc", role="user", content="Hello")
store.add_turn(turn)

# Read
session_turns = store.get_session_turns("session-abc", limit=50)
recent_turns = store.get_recent_turns(limit=10)

# Full-text search (FTS5)
results = store.search("docker networking", limit=10, session_id="session-abc")

# Pin a turn so cleanup() never deletes it
store.set_turn_pinned(turn.id, pinned=True)

# Clear or delete
store.clear_session("session-abc")
store.delete_turn(turn.id)

# Retention
deleted_count = store.cleanup(older_than_days=30)
```

Sessions, per-call cost records, and the summary DAG have their own
tables and accessor methods (`register_session`, `record_cost`,
`get_total_costs`, `add_summary`, `get_summaries`, `search_summaries`, etc.)
— see the module docstrings for the full API. `missy sessions list` /
`rename` / `cleanup` and `missy cost` are thin CLI wrappers over these
methods.

### Resilient Wrapper

`ResilientMemoryStore` (`missy/memory/resilient.py`) wraps a primary store
(normally `SQLiteMemoryStore`) and is what `AgentRuntime` actually talks to.
Behaviour:

- Every write goes to an in-memory cache **and** the primary store.
- If the primary raises, the operation transparently falls back to the
  cache (reads) or is retried on the next write (writes accumulate in the
  cache regardless of primary health).
- After 3 consecutive failures (configurable via `max_failures`) the store
  is marked unhealthy; `is_healthy` reflects this.
- On the next successful call, cached turns are replayed back into the
  primary (`_sync_cache_to_primary`).

This means a transient SQLite lock or disk error degrades gracefully to
in-memory operation rather than crashing the agent loop.

### Retention and Pinning

`cleanup(older_than_days=30)` deletes turns older than the threshold.
Turns with `metadata.pinned = true` (set via `set_turn_pinned`) are
excluded from cleanup regardless of age, allowing an operator to preserve
specific memories past the normal retention window.

---

## Vector Memory (Optional)

`VectorMemoryStore` (`missy/memory/vector_store.py`) provides approximate
semantic search over free-text entries using a FAISS `IndexFlatL2` index
and a dependency-free hashing TF-IDF-style vectorizer
(`SimpleVectorizer`, no sklearn or embedding model required).

- Requires `pip install -e ".[vector]"` (installs `faiss-cpu`). When
  `faiss` is not installed, every method becomes a no-op that logs a
  debug message and returns empty results — the rest of Missy runs
  unaffected.
- Index and metadata persist to `~/.missy/memory.faiss` and
  `~/.missy/memory.faiss.meta` (written with `0o600` permissions).

```python
from missy.memory.vector_store import VectorMemoryStore

store = VectorMemoryStore()
store.add("The deployment failed due to a missing env var", {"category": "solution"})
results = store.search("environment variable error", top_k=3)
store.save()   # persist index + metadata to disk
store.load()   # restore from disk
```

Vision scene memory (`missy/vision/vision_memory.py`) is one current
consumer of `VectorMemoryStore` for semantic recall over captured scene
descriptions.

---

## Graph Memory (Entity-Relationship)

`GraphMemoryStore` (`missy/memory/graph_store.py`) extracts entities
(tools, files, URLs, projects, persons) and typed relationships
(`uses`, `creates`, `modifies`, `depends_on`, `related_to`, `owns`,
`triggers`) from conversation turns using rule-based regex pattern
matching — no NLP or ML dependency is required.

It shares the **same** `~/.missy/memory.db` file as `SQLiteMemoryStore`,
adding `entities` and `relationships` tables to that database rather than
maintaining a separate file.

```python
from missy.memory.graph_store import GraphMemoryStore

store = GraphMemoryStore()
entities, rels = store.ingest_turn(
    "I used file_write to create ~/workspace/notes.txt", role="user", session_id="sess-1"
)
store.get_entity_summary("file_write")
store.get_context_subgraph("config file")   # formatted block for prompt injection
```

Key behaviour:

- `add_entity` / `add_relationship` upsert by `(name, entity_type)` and
  `(source_id, target_id, relation_type)` respectively — repeated mentions
  increment `mention_count`; repeated relationship observations nudge
  `weight` upward (capped at 1.0).
- `get_neighbors` performs a BFS traversal up to a configurable depth.
- `find_related` / `get_context_subgraph` resolve free-text queries into a
  relevant subgraph, formatted as `- src (type) --[relation]--> tgt (type)`
  lines suitable for direct injection into a system prompt.
- `prune(min_mentions, older_than_days)` deletes stale, rarely-mentioned
  entities and their relationships; `merge_entities` collapses duplicate
  entities discovered later (e.g. two spellings of the same file path).

---

## Memory Synthesizer

`MemorySynthesizer` (`missy/memory/synthesizer.py`) is the unification
layer that merges fragments from every memory subsystem — conversation
history, learnings, playbook entries, summaries, and graph context — into
a single deduplicated, relevance-ranked context block for injection into
the system prompt. Used by `AgentRuntime` (`missy/agent/runtime.py`).

Base relevance tiers per source (before query-specific boosting):

| Source | Base relevance |
|---|---|
| `learnings` | 0.70 |
| `graph` | 0.65 |
| `playbook` | 0.60 |
| `conversation` | 0.50 (default) |
| `summaries` | 0.40 |

```python
from missy.memory.synthesizer import MemorySynthesizer

synth = MemorySynthesizer(max_tokens=4500)
synth.add_fragments("conversation", ["discussed Docker setup"])
synth.add_fragments("learnings", ["always check ports first"], base_relevance=0.7)
synth.add_fragments("graph", ["file_read --[modifies]--> config.yaml"], base_relevance=0.65)
block = synth.synthesize("How do I fix Docker networking?")
```

`synthesize(query)`:

1. Scores each fragment: `base_relevance * 0.5 + keyword_overlap * 0.5`
   against the current query.
2. Deduplicates near-identical fragments (word-overlap ratio ≥ 0.8; the
   higher-relevance fragment is kept).
3. Sorts by relevance descending.
4. Greedily fills the `max_tokens` budget (4-chars-per-token estimate),
   dropping lowest-relevance fragments first.
5. Formats each surviving fragment as `[source] content`.

---

## Sleep Mode (Context Consolidation)

`MemoryConsolidator` (`missy/agent/consolidation.py`) triggers when the
active context window fills up, summarizing older turns to free token
budget without losing information outright.

- `should_consolidate(current_tokens)` returns `True` once
  `current_tokens / max_tokens >= threshold_pct` (default threshold
  **0.8**, default `max_tokens` **30,000**).
- `consolidate(messages, system_prompt)` always preserves the most recent
  4 messages (`_RECENT_KEEP`) intact and delegates compression of the
  remainder to a `PipelineCondenser` (`missy/agent/condensers.py`, built
  lazily via `create_default_pipeline`). The pipeline runs a 4-stage
  compression process — observation masking, amortized forgetting,
  summarizing, and windowing — described in `missy/agent/condensers.py`
  and orchestrated across passes by `CompactionManager`
  (`missy/agent/compaction.py`).
- `extract_key_facts(messages)` is a lighter-weight heuristic fallback: it
  keeps lines containing fact keywords (`result:`, `decided:`, `error:`,
  etc.), full `tool`-role message content (truncated to 200 chars), and
  short user messages (≤120 chars) that likely carry instructions —
  used when the pipeline drops old messages without producing an explicit
  summary block.
- Summaries produced during consolidation are persisted to the
  `summaries` table in `memory.db` via `SQLiteMemoryStore.add_summary`,
  forming a DAG: depth-0 summaries compress raw turns, depth-1+ summaries
  condense groups of same-depth summaries (see `SummaryRecord` in
  `missy/memory/sqlite_store.py`).

---

## Scheduled Jobs Persistence

### Overview

The `SchedulerManager` (`missy/scheduler/manager.py`) persists job
definitions to `~/.missy/jobs.json`.

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

See [scheduler.md](scheduler.md) for a detailed field description.

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
  completion), written atomically (`tempfile.mkstemp()` + `os.replace()`)
  with `0o600` permissions.
- On `start()`, all persisted jobs are loaded and re-registered with
  APScheduler.
- Malformed records are skipped with a warning.
- If the file does not exist or contains `[]`, the scheduler starts with no
  jobs.

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
    "model": "claude-sonnet-4-6",
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
| `mcp.digest_mismatch` | `plugin` | MCP tool manifest digest verification failed |
| `security.prompt_drift` | `network` | System prompt hash mismatch detected mid-session |

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

`memory.db` tables are created with `CREATE TABLE IF NOT EXISTS`, so
adding a new table or column set in a future release is additive and
does not require a migration step for existing databases; the `to_dict()`
/ `from_dict()` methods on `ConversationTurn` and `SummaryRecord` use
`.get()` with defaults so old rows missing newer JSON metadata keys still
load. `jobs.json` follows the same forward-compatible pattern via
`ScheduledJob.from_dict()`.

Config schema versioning is separate and handled by
`missy/config/migrate.py` (`config_version` field in `config.yaml`,
auto-migrated on startup) — see [configuration.md](configuration.md).

### Migration Guidance

If a breaking schema change is ever needed:

1. **Adding a field/table**: No migration needed — additive changes are
   backward compatible by construction.
2. **Renaming a field**: The updated `from_dict()`/row mapping should
   check for both the old and new names.
3. **Removing a field**: The old value is silently ignored on
   deserialisation.
4. **Changing a field's type or semantics**: A one-off migration script
   should be provided; back up first:

```bash
# Back up existing data
cp ~/.missy/memory.db ~/.missy/memory.db.backup
cp ~/.missy/jobs.json ~/.missy/jobs.json.backup

# Upgrade Missy
pip install --upgrade missy
```

No destructive memory-schema migrations have been required to date.
