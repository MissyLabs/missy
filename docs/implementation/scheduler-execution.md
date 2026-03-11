# Scheduler Internals

The scheduler subsystem allows Missy to execute agent tasks on a recurring
basis. It is built on top of APScheduler's `BackgroundScheduler` and adds
a JSON-based persistence layer so that jobs survive process restarts.

**Source files:**

- `missy/scheduler/manager.py` -- `SchedulerManager`
- `missy/scheduler/jobs.py` -- `ScheduledJob` dataclass
- `missy/scheduler/parser.py` -- human-readable schedule string parser

---

## SchedulerManager

`SchedulerManager` wraps an APScheduler `BackgroundScheduler` and owns a
dictionary of `ScheduledJob` objects keyed by job ID. All mutations
(add, remove, pause, resume) persist to a JSON file after the in-memory
state is updated.

### Construction

```python
mgr = SchedulerManager(jobs_file="~/.missy/jobs.json")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jobs_file` | `str` | `"~/.missy/jobs.json"` | Path to the JSON persistence file. Tilde expansion is applied. |

### Lifecycle

```python
mgr.start()   # Load persisted jobs, register enabled ones, start scheduler
mgr.stop()    # Graceful shutdown (waits for in-progress jobs to finish)
```

`start()`:
1. Calls `_load_jobs()` to read `jobs.json` into `_jobs`
2. Iterates all jobs; for each where `job.enabled` is `True`, calls
   `_schedule_job(job)` to register it with APScheduler
3. Calls `self._scheduler.start()`
4. Emits a `scheduler.start` audit event

`stop()`:
1. Calls `self._scheduler.shutdown(wait=True)` -- in-progress jobs finish
2. Emits a `scheduler.stop` audit event

---

## ScheduledJob Dataclass

```python
@dataclass
class ScheduledJob:
    id: str                        # UUID string (auto-generated)
    name: str                      # Human-readable name
    description: str               # Optional description
    schedule: str                  # e.g. "every 5 minutes", "daily at 09:00"
    task: str                      # Prompt text sent to the agent
    provider: str = "anthropic"    # Provider name for this job
    enabled: bool = True           # False = paused
    created_at: datetime           # UTC creation timestamp
    last_run: Optional[datetime]   # UTC timestamp of last execution
    next_run: Optional[datetime]   # UTC timestamp of next scheduled execution
    run_count: int = 0             # Total executions
    last_result: Optional[str]     # Agent response from last run
```

### Serialisation

`to_dict()` produces a JSON-compatible dictionary with datetime fields as
ISO-8601 strings. `from_dict(data)` deserialises back, using
`datetime.fromisoformat()` for timestamp parsing. Missing fields fall back
to sensible defaults.

---

## Job Persistence: jobs.json

Jobs are stored as a JSON array of objects in `~/.missy/jobs.json`. The
file is rewritten (not appended) on every mutation.

### Example

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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
  }
]
```

### Save Logic (`_save_jobs`)

1. Creates the parent directory if needed
2. Serialises all jobs via `[job.to_dict() for job in self._jobs.values()]`
3. Writes with `json.dumps(payload, indent=2, default=str)`
4. Errors are logged but not re-raised (to avoid interrupting the
   scheduler thread)

### Load Logic (`_load_jobs`)

1. If the file does not exist, returns silently (empty job list)
2. Reads and parses the JSON
3. Validates the top-level value is a list
4. Iterates records; skips non-dict entries and malformed records with
   a warning
5. Constructs `ScheduledJob.from_dict(record)` for each valid record

---

## Schedule Parsing: `parse_schedule()`

The parser converts human-readable schedule strings into APScheduler
trigger configurations. It is defined in `missy/scheduler/parser.py`.

### Supported Formats

| Input | Trigger Type | APScheduler Config |
|-------|-------------|-------------------|
| `"every 5 minutes"` | interval | `{"trigger": "interval", "minutes": 5}` |
| `"every 2 hours"` | interval | `{"trigger": "interval", "hours": 2}` |
| `"every 30 seconds"` | interval | `{"trigger": "interval", "seconds": 30}` |
| `"daily at 09:00"` | cron | `{"trigger": "cron", "hour": 9, "minute": 0}` |
| `"weekly on Monday at 09:00"` | cron | `{"trigger": "cron", "day_of_week": "mon", "hour": 9, "minute": 0}` |
| `"weekly on friday 14:30"` | cron | `{"trigger": "cron", "day_of_week": "fri", "hour": 14, "minute": 30}` |

The parser uses three compiled regular expressions tested in order. The
first match wins. Unrecognised strings raise `ValueError` with a message
listing all supported formats.

### Day-of-Week Mapping

Full day names (case-insensitive) are mapped to APScheduler abbreviations:

```
monday -> mon, tuesday -> tue, wednesday -> wed, thursday -> thu,
friday -> fri, saturday -> sat, sunday -> sun
```

---

## How a Scheduled Job Fires

When APScheduler triggers a job, it calls `SchedulerManager._run_job(job_id)`
on the background thread. The execution flow is:

```
APScheduler trigger
    |
    v
_run_job(job_id)
    |
    +-- Look up job in self._jobs
    |   (if gone, log warning and return)
    |
    +-- Generate fresh session_id and task_id (UUIDs)
    |
    +-- Emit scheduler.job.run.start audit event
    |
    +-- Lazy import AgentRuntime and AgentConfig
    |   (avoids circular imports at module load time)
    |
    +-- agent = AgentRuntime(AgentConfig(provider=job.provider))
    |
    +-- result_text = agent.run(job.task, session_id=session_id)
    |       |
    |       +-- (full agent loop: provider resolution, policy checks,
    |       |    completion, audit events -- see agent-loop.md)
    |       |
    |       v
    |   returned response text
    |
    +-- Update job state:
    |     job.last_run = now
    |     job.run_count += 1
    |     job.last_result = result_text
    |     job.next_run = APScheduler's next_run_time
    |
    +-- _save_jobs()  (persist to JSON)
    |
    +-- Emit scheduler.job.run.complete audit event
```

### Error Handling

If `agent.run()` raises any exception:

1. The exception is logged with `logger.exception()`
2. `job.last_run` is updated to now
3. `job.last_result` is set to `"ERROR: {exc}"`
4. Jobs are persisted
5. A `scheduler.job.run.error` audit event is emitted
6. The method returns (does not re-raise -- APScheduler continues)

---

## Job State Transitions

```
                  add_job()
                     |
                     v
              +-------------+
              |   enabled    |<------- resume_job()
              |  (running    |
              |   on timer)  |
              +------+------+
                     |
              pause_job()
                     |
                     v
              +-------------+
              |   paused     |
              | (enabled=    |
              |  False)      |
              +------+------+
                     |
              remove_job()
                     |
                     v
              +-------------+
              |  (deleted)   |
              +-------------+
```

### add_job(name, schedule, task, provider)

1. Validates the schedule string via `parse_schedule()` (raises
   `ValueError` on failure)
2. Creates a `ScheduledJob` instance
3. Registers it with APScheduler via `_schedule_job()`
4. On APScheduler failure, rolls back the in-memory entry
5. Persists to JSON
6. Emits `scheduler.job.add` audit event
7. Returns the new job

### remove_job(job_id)

1. Looks up the job (raises `KeyError` if not found)
2. Removes from APScheduler if present
3. Removes from in-memory dict
4. Persists to JSON
5. Emits `scheduler.job.remove` audit event

### pause_job(job_id)

1. Looks up the job (raises `KeyError`)
2. Calls `self._scheduler.pause_job(job_id)`
3. Sets `job.enabled = False`
4. Persists to JSON
5. Emits `scheduler.job.pause` audit event

### resume_job(job_id)

1. Looks up the job (raises `KeyError`)
2. Calls `self._scheduler.resume_job(job_id)`
3. Sets `job.enabled = True`
4. Persists to JSON
5. Emits `scheduler.job.resume` audit event

---

## APScheduler Registration: `_schedule_job(job)`

```python
trigger_config = parse_schedule(job.schedule)
trigger = trigger_config.pop("trigger")  # e.g. "interval" or "cron"

self._scheduler.add_job(
    func=self._run_job,
    trigger=trigger,
    kwargs={"job_id": job.id},
    id=job.id,
    name=job.name,
    replace_existing=True,
    **trigger_config,    # e.g. minutes=5, or hour=9, minute=0
)
```

`replace_existing=True` ensures that reloading a persisted job does not
raise a conflict error.

---

## Audit Events

| Event Type | Result | When |
|------------|--------|------|
| `scheduler.start` | `allow` | Scheduler started successfully |
| `scheduler.stop` | `allow` | Scheduler shut down |
| `scheduler.job.add` | `allow` / `error` | Job added or schedule parse failed |
| `scheduler.job.remove` | `allow` | Job removed |
| `scheduler.job.pause` | `allow` | Job paused |
| `scheduler.job.resume` | `allow` | Job resumed |
| `scheduler.job.run.start` | `allow` | Job execution begins |
| `scheduler.job.run.complete` | `allow` | Job execution succeeded |
| `scheduler.job.run.error` | `error` | Job execution failed |

All scheduler events use `category="scheduler"`.
