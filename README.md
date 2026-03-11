# Missy

A security-first, self-hosted local agentic assistant for Linux.

## Overview

Missy is a production-grade local assistant platform that enforces strict network security, filesystem sandboxing, and full auditability. It supports multiple AI providers (OpenAI, Anthropic, Ollama), a rich tool/skills/plugin system, and a built-in scheduler.

## Key Features

- **Default-deny network egress** — all outbound traffic gated by CIDR allowlists and wildcard domain rules
- **Filesystem sandboxing** — writes restricted to configured workspace paths
- **Shell execution policy** — shell commands require explicit allowlisting
- **Audit trail** — every privileged action logged as structured JSONL events
- **Multi-provider** — OpenAI, Anthropic, Ollama with fallback and timeout logic
- **Scheduler** — cron-style recurring jobs with human-friendly syntax
- **Plugin system** — plugins declare permissions; disabled by default
- **Threat model** — documented coverage of prompt injection, SSRF, data exfiltration, and more

## Architecture

```
missy/
  core/          - session, events, exceptions
  config/        - settings, YAML loading
  policy/        - network, filesystem, shell engines + facade
  gateway/       - PolicyHTTPClient (central network policy enforcement)
  providers/     - base, anthropic, openai, ollama, registry
  tools/         - base, registry, builtin tools
  skills/        - base, registry
  plugins/       - base, loader (security-gated)
  scheduler/     - APScheduler, human schedule parsing, job persistence
  memory/        - JSON-based conversation history
  observability/ - AuditLogger, JSONL audit trail
  security/      - InputSanitizer, SecretsDetector
  channels/      - base, cli_channel
  agent/         - AgentRuntime
  cli/           - click + rich CLI
```

## Installation

```bash
pip install -e .
missy init
```

## CLI Commands

```bash
missy init                      # Initialize workspace and config
missy ask "What is 2+2?"        # One-shot query
missy run                       # Interactive session

missy schedule add "every 5 minutes" "summarize logs"
missy schedule list
missy schedule pause <job-id>
missy schedule resume <job-id>
missy schedule remove <job-id>

missy audit security            # Review security audit log
missy providers list            # List configured providers
missy skills list               # List available skills
missy plugins list              # List installed plugins
```

## Configuration

Missy uses `~/.missy/config.yaml` by default. Example secure configuration:

```yaml
# examples/config.secure.yaml
network:
  default_deny: true
  allowed_cidrs:
    - 10.0.0.0/8
  allowed_domains:
    - "*.github.com"
    - api.openai.com
    - api.anthropic.com

filesystem:
  workspace_paths:
    - ~/missy-workspace

shell:
  enabled: false

plugins:
  enabled: false
```

See `examples/` for full example configurations.

## Security Model

- All outbound HTTP requests go through `PolicyHTTPClient` which enforces allowlists
- CIDR-based IP allowlists and wildcard domain matching
- Structured audit events for: network allow/deny, shell allow/deny, filesystem allow/deny, plugin allow/deny, scheduler execution, provider invocation
- See `SECURITY.md` for full security policy and `docs/THREAT_MODEL.md` for threat analysis

## Documentation

- [SECURITY.md](SECURITY.md) — security policy and hardening guide
- [OPERATIONS.md](OPERATIONS.md) — deployment and operations guide
- [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) — threat model

## Testing

```bash
python3 -m pytest tests/ -v
```

740 tests, 86% coverage.

## License

MIT
