# Hatching Log Format

The hatching log at `~/.missy/hatching_log.jsonl` records every step of the bootstrapping process.

## Entry Format

Each line is a JSON object:

```json
{
  "timestamp": "2025-01-15T10:30:00.123456",
  "step": "validate_environment",
  "status": "success",
  "message": "Environment validation passed",
  "details": {
    "python_version": "3.12.0",
    "missy_dir": "/home/user/.missy"
  }
}
```

## Fields

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | ISO 8601 timestamp |
| `step` | string | Step identifier (e.g. `validate_environment`, `generate_persona`) |
| `status` | string | `success`, `warning`, `error`, or `info` |
| `message` | string | Human-readable description |
| `details` | object | Optional structured metadata |

## Step Names

- `validate_environment` — Python version, directory creation, permissions
- `initialize_config` — Config file creation or detection
- `verify_providers` — API key detection across providers
- `initialize_security` — Vault directory, agent identity
- `generate_persona` — Default persona creation
- `seed_memory` — Memory database initialization
- `finalize` — Completion marker
- `reset` — Hatching state cleared

## Viewing the Log

```bash
# View raw log
cat ~/.missy/hatching_log.jsonl

# Pretty-print with jq
cat ~/.missy/hatching_log.jsonl | jq .

# Filter by status
cat ~/.missy/hatching_log.jsonl | jq 'select(.status == "error")'
```
