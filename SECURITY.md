# Security Policy

## Supported Security Features

Missy is built on a **secure-by-default** philosophy.  Every capability that
could cause harm is disabled until an operator explicitly enables it in the
configuration file.

### Network Access Control

All outbound network traffic is blocked by default (`network.default_deny:
true`).  Allowed destinations must be explicitly listed via:

- `network.allowed_hosts` — specific hostnames or `host:port` pairs.
- `network.allowed_domains` — domain names (suffix matching supported).
- `network.allowed_cidrs` — CIDR blocks for IP-range allowances.

### Filesystem Access Control

Read and write access to the local filesystem is restricted to explicitly
declared directory trees:

- `filesystem.allowed_read_paths` — directories the agent may read.
- `filesystem.allowed_write_paths` — directories the agent may write.

Attempts to access paths outside these trees raise `PolicyViolationError`.

### Shell Execution Control

Shell command execution is **disabled by default** (`shell.enabled: false`).
When enabled, only commands named in `shell.allowed_commands` may run.  An
empty `allowed_commands` list blocks all commands even when the shell is
nominally enabled.

### Plugin Control

The plugin system is **disabled by default** (`plugins.enabled: false`).
When enabled, only plugins whose names appear in `plugins.allowed_plugins`
may be loaded.

### Input Sanitization

User input is sanitized before reaching the AI provider:

- Truncated to 10 000 characters to prevent oversized-payload attacks.
- Scanned for 13+ prompt-injection patterns; violations are logged as
  warnings.

### Secrets Detection

The `SecretsDetector` scans text for 9 credential patterns (API keys,
private keys, tokens, passwords, JWTs, etc.) and can redact them before
text is stored or transmitted.  The CLI warns users when potential secrets
are detected in a prompt.

### Audit Logging

Every significant operation (agent runs, policy checks, scheduler
executions, plugin loads) produces a structured JSONL audit event.  Events
are appended to the configured `audit_log_path` and can be reviewed with
`missy audit recent` and `missy audit security`.

---

## Default Security Posture

A freshly initialised Missy installation has the following defaults:

| Capability | Default | Opt-in mechanism |
|---|---|---|
| Outbound network | **Denied** | `network.allowed_hosts/domains/cidrs` |
| Filesystem read | **Denied** | `filesystem.allowed_read_paths` |
| Filesystem write | **Denied** | `filesystem.allowed_write_paths` |
| Shell execution | **Disabled** | `shell.enabled: true` + `allowed_commands` |
| Plugins | **Disabled** | `plugins.enabled: true` + `allowed_plugins` |
| AI providers | **None configured** | `providers.<name>` block in config |

API keys are never stored in the configuration file.  Providers read them
from environment variables (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

---

## Security Configuration Guide

### Minimal provider-only setup (recommended starting point)

```yaml
network:
  default_deny: true
  allowed_hosts:
    - "api.anthropic.com"

filesystem:
  allowed_write_paths: ["~/.missy"]
  allowed_read_paths: ["~/.missy"]

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

### Enabling filesystem access for a workspace

```yaml
filesystem:
  allowed_read_paths:
    - "~/workspace"
    - "/tmp"
  allowed_write_paths:
    - "~/workspace/output"
    - "~/.missy"
```

### Enabling a specific shell command

```yaml
shell:
  enabled: true
  allowed_commands:
    - "git"
    - "python3"
```

### Enabling a trusted plugin

```yaml
plugins:
  enabled: true
  allowed_plugins:
    - "my_trusted_plugin"
```

---

## Environment Variables for Secrets

Never put API keys in the config file.  Set them as environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Missy reads `<PROVIDER_NAME>_API_KEY` automatically (e.g. `ANTHROPIC_API_KEY`
for a provider named `anthropic`).

---

## Reporting Vulnerabilities

If you discover a security vulnerability in Missy, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

### How to report

1. Email the maintainers directly (see `pyproject.toml` for contact
   information) with the subject line `[SECURITY] Missy vulnerability`.
2. Include a description of the vulnerability, steps to reproduce, and any
   proof-of-concept code or configuration.
3. Indicate whether you would like to be credited in the security advisory.

### What to expect

- Acknowledgement within 48 hours.
- An initial assessment within 5 business days.
- A coordinated disclosure timeline agreed with the reporter before any
  public announcement.

### Scope

In scope:

- Policy bypass (executing denied network/filesystem/shell operations).
- Prompt injection that successfully overrides the system prompt.
- Secrets leakage via the audit log or CLI output.
- Authentication / authorisation issues in any HTTP-facing component.

Out of scope:

- Vulnerabilities in third-party AI providers (Anthropic, OpenAI, Ollama).
- Model-level jailbreaks or hallucinations.
- Issues in underlying Python dependencies not exploitable via Missy's
  public API.

---

## Security Changelog

| Version | Change |
|---|---|
| 0.1.0 | Initial release with secure-by-default policy engine, input sanitization, secrets detection, and JSONL audit logging. |
