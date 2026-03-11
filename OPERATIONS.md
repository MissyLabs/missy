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
git clone https://github.com/your-org/missy.git
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
    model: "claude-3-5-sonnet-20241022"
    timeout: 30
  openai:
    name: openai
    model: "gpt-4o"
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
details, provider selection, and audit event publishing.

---

## 4. Monitoring and Audit Logs

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
  "detail": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
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
    model: "claude-3-5-sonnet-20241022"
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
