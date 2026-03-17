# Scheduler

Missy includes a built-in job scheduler for recurring agent tasks.  This
document covers the scheduling subsystem architecture, supported schedule
formats, job persistence, CLI commands, and policy enforcement.

---

## Overview

The scheduler is backed by [APScheduler](https://apscheduler.readthedocs.io/)
(`BackgroundScheduler`) and managed by `SchedulerManager`
(`missy/scheduler/manager.py`).  It provides:

- Human-readable schedule expressions (parsed by `missy/scheduler/parser.py`)
- JSON-based job persistence (`~/.missy/jobs.json`)
- Full integration with the agent pipeline (jobs run through `AgentRuntime`)
- Audit events for every scheduler operation
- Policy enforcement -- scheduled jobs cannot bypass network, filesystem, or
  shell policies

---

## Supported Schedule Formats

Schedule expressions are parsed by `parse_schedule()` in
`missy/scheduler/parser.py`.  Three formats are supported:

### Interval Triggers

Run a job at a fixed interval.

| Expression | APScheduler Trigger | Result |
|---|---|---|
| `"every 30 seconds"` | `interval`, `seconds=30` | Every 30 seconds |
| `"every 5 minutes"` | `interval`, `minutes=5` | Every 5 minutes |
| `"every 2 hours"` | `interval`, `hours=2` | Every 2 hours |

The pattern is: `every <N> <unit>` where `<unit>` is `seconds`, `minutes`, or
`hours` (singular or plural).

### Daily Triggers

Run a job once per day at a specific time.

| Expression | APScheduler Trigger | Result |
|---|---|---|
| `"daily at 09:00"` | `cron`, `hour=9`, `minute=0` | Every day at 09:00 |
| `"daily at 14:30"` | `cron`, `hour=14`, `minute=30` | Every day at 14:30 |

The pattern is: `daily at HH:MM` (24-hour format).

### Weekly Triggers

Run a job once per week on a specific day at a specific time.

| Expression | APScheduler Trigger | Result |
|---|---|---|
| `"weekly on Monday at 09:00"` | `cron`, `day_of_week=mon`, `hour=9`, `minute=0` | Every Monday at 09:00 |
| `"weekly on friday 14:30"` | `cron`, `day_of_week=fri`, `hour=14`, `minute=30` | Every Friday at 14:30 |

The pattern is: `weekly on <day> [at] HH:MM`.  The `at` keyword is optional.
Day names are case-insensitive and must be the full English name (Monday,
Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday).

### Invalid Expressions

Any expression that does not match one of the above patterns raises a
`ValueError` with a message listing the supported formats.

---

## Job Persistence

### Storage File

Jobs are persisted to `~/.missy/jobs.json` as a JSON array.  This file is
created by `missy init` with an initial value of `[]`.

### Schema

Each job is serialised by `ScheduledJob.to_dict()` (`missy/scheduler/jobs.py`):

```json
{
  "id": "uuid-string",
  "name": "Daily digest",
  "description": "",
  "schedule": "daily at 09:00",
  "task": "Summarise the top 5 news stories for today.",
  "provider": "anthropic",
  "enabled": true,
  "created_at": "2026-03-11T09:00:00+00:00",
  "last_run": "2026-03-11T09:00:05+00:00",
  "next_run": "2026-03-12T09:00:00+00:00",
  "run_count": 1,
  "last_result": "Here are today's top stories..."
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | UUID, generated at creation |
| `name` | string | Human-readable job name |
| `description` | string | Optional description |
| `schedule` | string | The human-readable schedule expression |
| `task` | string | Prompt text sent to the agent on each firing |
| `provider` | string | AI provider name (default: `"anthropic"`) |
| `enabled` | bool | Whether the job is active |
| `created_at` | ISO-8601 string | UTC creation timestamp |
| `last_run` | ISO-8601 string or null | UTC timestamp of the most recent run |
| `next_run` | ISO-8601 string or null | UTC timestamp of the next scheduled run |
| `run_count` | int | Total number of executions |
| `last_result` | string or null | The agent's response from the most recent run |

### Persistence Behaviour

- Jobs are saved to disk after every mutation (add, remove, pause, resume,
  and after each execution completes).
- On startup, `_load_jobs()` reads the JSON file and reconstructs
  `ScheduledJob` instances.  Malformed records are skipped with a warning.
- If the file does not exist, the scheduler starts with an empty job list.
- If the file contains invalid JSON, an error is logged and the scheduler
  starts empty.
- The parent directory is created automatically if it does not exist.

**Important**: Do not edit `jobs.json` while Missy is running.  Stop any
running process first.

---

## CLI Commands

### `missy schedule add`

Create a new scheduled job.

```bash
missy schedule add \
    --name "Daily digest" \
    --schedule "daily at 09:00" \
    --task "Summarise the top 5 news stories for today." \
    --provider anthropic
```

| Option | Required | Default | Description |
|---|---|---|---|
| `--name` | Yes | -- | Human-readable job name |
| `--schedule` | Yes | -- | Schedule expression (see formats above) |
| `--task` | Yes | -- | Prompt text for each execution |
| `--provider` | No | `anthropic` | AI provider to use |

On success, prints the job ID, name, schedule, and provider.

### `missy schedule list`

List all scheduled jobs as a table.

```bash
missy schedule list
```

Output columns: ID (truncated), Name, Schedule, Provider, Enabled, Runs,
Last Run, Next Run.

### `missy schedule pause <job-id>`

Pause a job so it no longer fires.

```bash
missy schedule pause <job-id>
```

The job remains in the persistence file with `enabled: false`.  Use the full
UUID job ID (not the truncated ID shown in the list table).

### `missy schedule resume <job-id>`

Resume a previously paused job.

```bash
missy schedule resume <job-id>
```

Sets `enabled: true` and re-registers the job with APScheduler.

### `missy schedule remove <job-id>`

Permanently remove a job.

```bash
missy schedule remove <job-id>
```

A confirmation prompt is displayed before removal.  The job is removed from
both APScheduler and the persistence file.

---

## How Scheduled Jobs Execute

When a job fires, APScheduler calls `SchedulerManager._run_job(job_id)` on a
background thread.  The execution flow is:

1. Look up the `ScheduledJob` by ID.
2. Generate a fresh `session_id` and `task_id` (both UUIDs).
3. Emit a `scheduler.job.run.start` audit event.
4. Construct an `AgentRuntime` with `AgentConfig(provider=job.provider)`.
5. Call `agent.run(job.task, session_id=session_id)`.
6. Store the result in `job.last_result`, increment `job.run_count`, update
   `job.last_run`.
7. Save jobs to disk.
8. Emit a `scheduler.job.run.complete` audit event (or `scheduler.job.run.error`
   on failure).

Because the job runs through the standard `AgentRuntime`, it is subject to
all the same subsystem initialisation and policy enforcement as an interactive
`missy ask` command.

---

## Policy Enforcement for Scheduled Jobs

Scheduled jobs **cannot bypass** any security policy.  They execute through
the same `AgentRuntime` -> `Provider` -> `PolicyHTTPClient` pipeline as
interactive commands.  Specifically:

- **Network**: All outbound requests are checked against
  `network.allowed_hosts`, `allowed_domains`, and `allowed_cidrs`.
- **Filesystem**: Any file access by tools or skills is checked against
  `filesystem.allowed_read_paths` and `allowed_write_paths`.
- **Shell**: Any shell execution is checked against `shell.enabled` and
  `shell.allowed_commands`.

If a scheduled job triggers a policy violation, the violation is logged as an
audit event and the job's `last_result` is set to `"ERROR: ..."`.

---

## Job Lifecycle

```
  Created (add_job)
     |
     v
  Enabled -----> Running (on trigger)
     |               |
     |               v
     |           Completed (back to Enabled, awaiting next trigger)
     |
     v
  Paused (pause_job)
     |
     v
  Resumed (resume_job) --> Enabled
     |
     v
  Removed (remove_job) --> Deleted from persistence
```

| State | `enabled` | In APScheduler | Description |
|---|---|---|---|
| Created / Enabled | `true` | Registered | Waiting for next trigger time |
| Running | `true` | Registered | Currently executing via AgentRuntime |
| Paused | `false` | Paused | Will not fire until resumed |
| Removed | -- | Removed | Deleted from persistence and scheduler |

---

## Audit Events

The scheduler emits audit events with `category: "scheduler"` for every
operation:

| Event Type | Result | When |
|---|---|---|
| `scheduler.start` | `allow` | Scheduler starts |
| `scheduler.stop` | `allow` | Scheduler stops |
| `scheduler.job.add` | `allow` | Job created successfully |
| `scheduler.job.add` | `error` | Invalid schedule expression |
| `scheduler.job.remove` | `allow` | Job removed |
| `scheduler.job.pause` | `allow` | Job paused |
| `scheduler.job.resume` | `allow` | Job resumed |
| `scheduler.job.run.start` | `allow` | Job execution begins |
| `scheduler.job.run.complete` | `allow` | Job execution succeeds |
| `scheduler.job.run.error` | `error` | Job execution fails |

Review scheduler events with:

```bash
missy audit recent --category scheduler
```
