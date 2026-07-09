# Missy Operations Guide

This guide covers installation, configuration, day-to-day operation,
monitoring, and maintenance of a Missy deployment.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Configuration Guide](#2-configuration-guide)
3. [Running Missy](#3-running-missy)
4. [Monitoring and Audit Logs](#4-monitoring-and-audit-logs)
5. [Scheduler Management](#5-scheduler-management)
6. [Provider Configuration](#6-provider-configuration)
7. [Backup and Recovery](#7-backup-and-recovery)

---

## 1. Installation and Setup

### Prerequisites

- Python 3.11 or later.
- `pip` (or `pipx` for isolated installs).
- API keys for the AI providers you intend to use.

### Install from source

```bash
git clone https://github.com/MissyLabs/missy.git
cd missy
pip install -e ".[dev]"   # editable install with dev extras
```

### Install as a package

```bash
pip install missy
# or, for isolated installs:
pipx install missy
```

### Verify installation

```bash
missy --help
```

### Initialise the workspace

```bash
missy init
```

This command:

1. Creates `~/.missy/` if it does not exist.
2. Writes a default `~/.missy/config.yaml` (secure-by-default posture).
3. Creates empty `~/.missy/audit.jsonl` and `~/.missy/jobs.json`.
4. Creates `~/workspace/` as the default agent working directory.

If `~/.missy/config.yaml` already exists it is left unchanged.

---

## 2. Configuration Guide

The configuration file is YAML and is loaded by `load_config()` from
`missy.config.settings`.  The default path is `~/.missy/config.yaml`.
Override it with the `--config` flag or the `MISSY_CONFIG` environment
variable.

### Full configuration reference

```yaml
# -----------------------------------------------------------------------
# Network policy
# Blocks all outbound traffic unless destination is explicitly listed.
# -----------------------------------------------------------------------
network:
  default_deny: true                  # true = block-by-default (recommended)
  allowed_hosts:                      # host or host:port strings
    - "api.anthropic.com"
    - "api.openai.com"
    - "localhost:11434"               # local Ollama instance
  allowed_domains:                    # domain suffix matching
    - "*.github.com"
  allowed_cidrs:                      # CIDR blocks (RFC 1918 for private networks)
    - "10.0.0.0/8"
    - "172.16.0.0/12"
    - "192.168.0.0/16"

# -----------------------------------------------------------------------
# Filesystem policy
# Restricts which directories the agent may read from or write to.
# -----------------------------------------------------------------------
filesystem:
  allowed_read_paths:
    - "~/workspace"
    - "~/.missy"
    - "/tmp"
  allowed_write_paths:
    - "~/workspace"
    - "~/.missy"

# -----------------------------------------------------------------------
# Shell policy
# Disabled by default. Enable only if required.
# -----------------------------------------------------------------------
shell:
  enabled: false
  allowed_commands: []                # e.g. ["git", "python3", "ls"]

# -----------------------------------------------------------------------
# Plugin policy
# Disabled by default. Enable only trusted plugins.
# -----------------------------------------------------------------------
plugins:
  enabled: false
  allowed_plugins: []                 # e.g. ["my_plugin_name"]

# -----------------------------------------------------------------------
# Provider configuration
# API keys must be set as environment variables, not in this file.
# -----------------------------------------------------------------------
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
  openai:
    name: openai
    model: "auto"
    timeout: 30
  ollama:
    name: ollama
    base_url: "http://localhost:11434"
    model: "llama3.2"
    timeout: 60

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
```

### Environment variables

| Variable | Purpose |
|---|---|
| `MISSY_CONFIG` | Override the config file path. |
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider. |
| `OPENAI_API_KEY` | API key for the OpenAI provider. |

---

## 3. Running Missy

### Single-shot query

```bash
missy ask "What is the capital of France?"
missy ask --provider ollama "Explain quantum entanglement in simple terms."
missy ask --session my-session-id "Continue our earlier conversation."
```

### Interactive session

```bash
missy run
missy run --provider openai
```

Type `quit`, `exit`, or `q` to end the session.  Press Ctrl-D (EOF) or
Ctrl-C to exit immediately.

### Using a custom config

```bash
missy --config /path/to/my-config.yaml ask "Hello"
MISSY_CONFIG=/path/to/my-config.yaml missy run
```

### Debug mode

```bash
missy --debug ask "Hello"
```

`--debug` enables Python `DEBUG`-level logging, which includes policy check
details, provider selection, and audit event publishing. Python application
logs are also written to the configured rotating log file, defaulting to
`~/.missy/missy.log`.

Find or inspect the application log:

```bash
missy logs path
missy logs tail --limit 120
```

### Web TUI / operator console

```bash
missy api start                   # loopback-only by default
missy api start --host 127.0.0.1 --port 8080
missy api status
```

`missy api start` serves both the JSON REST API (`/api/v1/*`) and the
browser-based operator console (`/`) from the same `ApiServer`. Set
`MISSY_API_KEY` (or `ApiConfig.api_key`) before starting — every request is
rejected with `401` otherwise. Open `http://127.0.0.1:8080/` and sign in with
the operator key to reach the console; the browser session is a signed,
`HttpOnly`, `SameSite=Strict` cookie, and every unsafe (`POST`/`PUT`/`PATCH`/
`DELETE`) request from the browser requires the console's per-session CSRF
token (already wired into the console's own JavaScript).

The console's **Ask Missy** panel is the primary "ask the bot and watch a run
stream" workflow: submitting a message calls `POST /api/v1/runs`, which starts
the run on a background thread and returns immediately with a `run_id`. The
browser then opens `GET /api/v1/runs/{run_id}/events` as a Server-Sent Events
stream and renders tool calls and the final response live, without blocking
the HTTP connection that started the run. Reconnecting after a run has
already finished (or a `GET /api/v1/runs/{run_id}` poll) returns the run's
terminal state immediately instead of hanging. Only one run may be in flight
per session at a time — a second `POST /api/v1/runs` against a busy session
returns `409` and is recorded as a `web.run` audit denial.

Both the run console and the dashboard's providers/tools/sessions/audit
panels are read via the same JSON API, so any script can drive the agent the
same way:

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/runs \
  -H "X-API-Key: $MISSY_API_KEY" -H 'Content-Type: application/json' \
  -d '{"message": "list the files in ~/workspace"}'
# => {"status": "ok", "data": {"run_id": "...", "session_id": "...", "status": "pending", ...}}

curl -N -H "X-API-Key: $MISSY_API_KEY" \
  http://127.0.0.1:8080/api/v1/runs/<run_id>/events   # text/event-stream
```

Once a run completes, both the poll response (`GET /api/v1/runs/{id}`) and the
terminal SSE `run.complete` event carry `resolved_provider`/`provider`,
`tools_used`, and `cost` (the same summary already recorded on the
`agent.run.complete` audit event), so the console's run log renders which
provider actually served the request, which tools it called, and what it cost
without a second request.

### Tool intelligence operations

Missy records completed turns in `RequestTracker` so repeated requests and
manual workflows can be reviewed:

```bash
missy tools requests stats --min-count 3
```

Automatic tool-candidate synthesis is off by default. When
`tool_intelligence.candidate_generation.enabled: true` is configured, the
runtime periodically scans frequent patterns and stores proposed candidates in
`~/.missy/tool_candidates.db`. Generated candidates are not executable by
default. They must move through the audited lifecycle
`proposed -> experimental -> benchmarked -> approved -> enabled`; invalid
shortcuts such as `proposed -> enabled` are rejected by `CandidateStore` and
emit `tool.candidate.transition_denied`.

Review and lifecycle commands:

```bash
missy tools candidates list
missy tools candidates show <candidate_id>
missy tools benchmark run <tool_name>
missy tools candidates import-benchmarks <candidate_id>
missy tools candidates approve <candidate_id>
missy tools candidates enable <candidate_id>
missy tools candidates deny <candidate_id> --reason "unsafe permissions"
```

`import-benchmarks` reads aggregate results from `~/.missy/benchmark_results.db`,
stores provider summaries back on the candidate, and sets benchmark-derived
provider flags using conservative thresholds for sample count, composite score,
safety, and schema adherence. It can move `proposed` or `experimental`
candidates to `benchmarked`, but it never approves or enables a tool.

Provider-specific tool availability is controlled separately from execution
policy. `ToolProviderGate` can hide a tool from a weak provider based on
benchmark data, while explicit operator overrides remain auditable:

```bash
missy tools providers status calculator --provider ollama
missy tools providers disable shell_exec ollama
missy tools providers clear shell_exec ollama
missy tools providers recommend calculator
```

The console's **Scheduled Jobs** panel covers the full job lifecycle:
listing (`GET /api/v1/scheduler/jobs`), creation via a guarded form (`POST
/api/v1/scheduler/jobs` — `name`, `schedule`, `task` required; `provider`,
`description`, `active_hours`, `timezone` optional), and pause/resume/remove
through the existing safe-controls confirmation flow (`scheduler.pause_job` /
`scheduler.resume_job` / `scheduler.remove_job`, each requiring a
`confirm: "<action>-job:<id>"` body field). `DELETE
/api/v1/scheduler/jobs/{id}` is a thin alias for the `scheduler.remove_job`
control so scripts can use a conventional REST verb while still going through
the same confirmation and audit path as the console button.

The **Memory Browser** panel searches conversation history
(`GET /api/v1/memory/search?q=...&session_id=...`, same endpoint the
dashboard already used) and adds per-turn retention controls:
`POST /api/v1/memory/turns/{id}/pin` (body `{"pinned": true|false}`) marks a
turn to survive `missy sessions cleanup` / the memory store's age-based
`cleanup()`, and `DELETE /api/v1/memory/turns/{id}` permanently removes a
single turn (and its full-text index entry). Both search results and session
history responses now include each turn's `id` and `pinned` state so the UI
can render pin/unpin and delete actions inline.

---

## 4. Monitoring and Logs

### Application log location

The application log captures Python warnings/errors, provider diagnostics, and
debug logs when enabled. The default path is `~/.missy/missy.log`; override it
with `observability.log_file_path`.

### Audit log location

The audit log is a newline-delimited JSON (JSONL) file, by default at
`~/.missy/audit.jsonl`.  Each line is a self-contained JSON object.

### Audit event structure

```json
{
  "timestamp": "2026-03-11T09:00:00.123456+00:00",
  "session_id": "uuid-...",
  "task_id": "uuid-...",
  "event_type": "agent.run.complete",
  "category": "provider",
  "result": "allow",
  "detail": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
  "policy_rule": null
}
```

`result` is one of `allow`, `deny`, or `error`.

### Viewing recent events via CLI

```bash
# Show last 50 events (all categories)
missy audit recent

# Show last 20 network events
missy audit recent --limit 20 --category network

# Show policy violations only
missy audit security
missy audit security --limit 100
```

### Parsing the log with jq

```bash
# Show all policy denials
jq 'select(.result == "deny")' ~/.missy/audit.jsonl

# Show agent runs in the last hour
jq 'select(.event_type | startswith("agent.run"))' ~/.missy/audit.jsonl

# Count events by category
jq -r '.category' ~/.missy/audit.jsonl | sort | uniq -c | sort -rn
```

### Log rotation

The audit log grows without bound.  Rotate it with `logrotate` or a cron
job:

```bash
# Rotate daily, keep 30 days, compress
cat > /etc/logrotate.d/missy <<'EOF'
/home/USER/.missy/audit.jsonl {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

---

## 5. Scheduler Management

### Add a job

```bash
missy schedule add \
    --name "Daily digest" \
    --schedule "daily at 09:00" \
    --task "Summarise the top 5 news stories for today." \
    --provider anthropic
```

Supported schedule formats (parsed by `missy.scheduler.parser`):

| Expression | Meaning |
|---|---|
| `every 5 minutes` | Every 5 minutes |
| `every hour` | Every 60 minutes |
| `every 2 hours` | Every 2 hours |
| `every day` / `daily` | Every 24 hours |
| `daily at 09:00` | Every day at 09:00 local time |
| `every monday at 08:00` | Weekly on Monday at 08:00 |

### List jobs

```bash
missy schedule list
```

Output columns: ID, Name, Schedule, Provider, Enabled, Runs, Last Run, Next Run.

### Pause a job

```bash
missy schedule pause <job-id>
```

Use the full job ID from `missy schedule list` (the first 8 chars plus `…`
are shown in the table; use the full ID).

### Resume a job

```bash
missy schedule resume <job-id>
```

### Remove a job

```bash
missy schedule remove <job-id>
```

You will be prompted to confirm before removal.

### Job persistence

Jobs are stored in `~/.missy/jobs.json`.  Do not edit this file while
Missy is running.  To inspect or manually fix jobs, stop any running Missy
process first.

---

## 6. Provider Configuration

### Anthropic

```yaml
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### OpenAI

```yaml
providers:
  openai:
    name: openai
    model: "gpt-4o"
    timeout: 30
```

```bash
export OPENAI_API_KEY="sk-..."
```

### Ollama (local, no API key required)

```yaml
providers:
  ollama:
    name: ollama
    base_url: "http://localhost:11434"
    model: "llama3.2"
    timeout: 60
```

Remember to add `localhost:11434` to `network.allowed_hosts`.

### Listing provider status

```bash
missy providers
```

Shows each configured provider, its model, base URL, timeout, and whether it
is currently available (API key present and reachable).

### Provider fallback

When the requested provider is unavailable (no API key, network error, etc.)
the agent runtime automatically falls back to the first available provider.
A warning is logged.  To disable fallback behaviour, configure only the
provider you intend to use.

---

## 7. Backup and Recovery

### Files to back up

| File | Contents | Criticality |
|---|---|---|
| `~/.missy/config.yaml` | Policy and provider settings | High |
| `~/.missy/jobs.json` | Scheduled job definitions | Medium |
| `~/.missy/audit.jsonl` | Audit event log | Medium (compliance) |

**Do not back up API keys.**  Store them in a secrets manager (e.g.
HashiCorp Vault, AWS Secrets Manager, macOS Keychain) and inject them at
runtime via environment variables.

### Backup procedure

```bash
# Create a timestamped archive of the Missy directory
tar -czf "missy-backup-$(date +%Y%m%d).tar.gz" ~/.missy/
```

### Restoring from backup

```bash
# Stop any running Missy processes first
tar -xzf missy-backup-20260311.tar.gz -C ~/

# Verify the restored config loads cleanly
missy --config ~/.missy/config.yaml providers
```

### Recovering a corrupted jobs file

If `~/.missy/jobs.json` is corrupted (invalid JSON), Missy will log an error
and continue with an empty job list.  Restore from backup, or reset with:

```bash
echo "[]" > ~/.missy/jobs.json
```

Then re-add your jobs with `missy schedule add`.

### Recovering a corrupted config file

```bash
# Back up the broken file
cp ~/.missy/config.yaml ~/.missy/config.yaml.broken

# Re-initialise (will not overwrite an existing file)
# So rename first:
mv ~/.missy/config.yaml ~/.missy/config.yaml.broken
missy init
```

Then re-apply your custom configuration on top of the new default.
