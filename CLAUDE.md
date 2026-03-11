# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. It is a production-grade local AI agent platform with strict security controls, policy enforcement, and full auditability. The project is implemented in Python 3.11+.

## Commands

```bash
# Install (editable with dev dependencies)
pip install -e ".[dev]"

# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/unit/test_policy_engine.py -v

# Run a single test by name
python3 -m pytest tests/ -k "test_name" -v

# Coverage report
pytest tests/ --cov=missy --cov-report=html

# Lint
ruff check missy/ tests/

# Format
ruff format missy/ tests/
# or
black missy/ tests/
```

The CLI entry point after installation: `missy` (maps to `missy.cli.main:cli`).

## Architecture

The system follows a **secure-by-default** design: all capabilities (shell, plugins, network) are disabled until explicitly enabled in config (`~/.missy/config.yaml`).

### Data Flow

```
CLI (missy/cli/main.py)
  → load_config() (missy/config/settings.py)
  → AgentRuntime (missy/agent/runtime.py)
       ├─ InputSanitizer + SecretsDetector (security/)
       ├─ PolicyEngine (policy/engine.py) — checks before any action
       ├─ ProviderRegistry (providers/registry.py) — Anthropic/OpenAI/Ollama
       ├─ PolicyHTTPClient (gateway/client.py) — wraps ALL outbound HTTP
       ├─ MemoryStore (memory/store.py) — conversation history (JSON)
       └─ AuditLogger (observability/audit_logger.py) — JSONL audit trail
```

### Key Subsystems

**Policy Engine (`missy/policy/`)** — Three-layer enforcement:
- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, explicit host allowlists
- `FilesystemPolicyEngine`: Per-path read/write access control
- `ShellPolicyEngine`: Command whitelisting
- `PolicyEngine`: Facade unifying all three

**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx and is the single enforcement point for ALL outbound HTTP. Every request is checked against network policy before being made.

**Providers (`missy/providers/`)** — Abstraction over AI backends:
- `BaseProvider` defines the interface (`Message`, `CompletionResponse`)
- Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`
- `ProviderRegistry` handles resolution with fallback logic

**Channels (`missy/channels/`)** — Communication interfaces:
- `CLIChannel`: Interactive stdin/stdout
- `DiscordChannel` (`missy/channels/discord/`): Full WebSocket Gateway API implementation with access control (DM allowlist, guild/role policies) and slash commands (`/ask`, `/status`, `/model`, `/help`)

**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. The parser converts human-friendly schedules ("every 5 minutes") to cron expressions.

**Security (`missy/security/`)** — Input-level checks run before the agent acts:
- `InputSanitizer`: Detects 13+ prompt injection patterns
- `SecretsDetector`: Detects 9 credential patterns (API keys, JWTs, etc.)

**Observability (`missy/observability/audit_logger.py`)** — Structured JSONL events at `~/.missy/audit.jsonl` covering: network allow/deny, filesystem allow/deny, shell allow/deny, plugin allow/deny, scheduler execution, provider invocation, agent run.

### Default File Locations

| Purpose | Path |
|---|---|
| Config | `~/.missy/config.yaml` |
| Audit log | `~/.missy/audit.jsonl` |
| Scheduler jobs | `~/.missy/jobs.json` |
| Workspace | `~/workspace` |

### Configuration Schema

```yaml
network:
  default_deny: true
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts: []        # host:port pairs

filesystem:
  allowed_write_paths: []
  allowed_read_paths: []

shell:
  enabled: false
  allowed_commands: []

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-3-5-sonnet-20241022"
    timeout: 30

workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
```

## Test Layout

Tests are organized under `tests/` with subdirectories mirroring the source (`unit/`, `integration/`, `policy/`, `channels/`, `provider/`, `config/`, `cli/`). There are 814 tests with ~86% coverage.
