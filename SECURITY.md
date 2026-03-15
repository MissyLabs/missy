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

Compound commands (using `&&`, `||`, `;`, `|`, `&`) are split and each
sub-command is individually checked against the allowlist.  The following
shell constructs are unconditionally blocked to prevent hidden command
execution:

- Process substitution: `<(...)`, `>(...)`, `<<(...)`
- Command substitution: `$(...)`, backticks
- Here-strings: `<<<`
- Heredocs: `<<EOF`
- Brace groups: `{ cmd; }`, `{;cmd;}` — scanned anywhere in the command,
  not just at the start, to prevent bypass via `echo; { dangerous; }`

### Plugin Control

The plugin system is **disabled by default** (`plugins.enabled: false`).
When enabled, only plugins whose names appear in `plugins.allowed_plugins`
may be loaded.

### Input Sanitization

User input is sanitized before reaching the AI provider:

- Truncated to 10 000 characters to prevent oversized-payload attacks.
- Scanned for 69 prompt-injection patterns covering system/role delimiters,
  jailbreak attempts, multi-language injection (English, Spanish, French,
  German, Italian, Portuguese, Russian, Japanese, Korean), tool/function abuse,
  prompt leaking/exfiltration, model-specific tokens (Llama 2/3, GPT,
  Claude, FIM), base64-encoded payloads, hidden instruction vectors,
  trigger-based injection, conditional overrides, memory poisoning, future
  response control, and role confusion attacks; violations are logged as
  warnings.

### Secrets Detection & Response Censoring

The `SecretsDetector` scans text for 26 credential patterns (API keys
including `sk-proj-...`, private keys, tokens, passwords, JWTs, AWS
credentials, GitHub/GitLab/npm/PyPI/Slack/Discord/SendGrid tokens,
Azure AccountKey, Twilio SK, Mailgun key,
database connection strings, etc.) and can redact them before text is
stored or transmitted.  The CLI warns users when potential secrets are
detected in a prompt.

The agent runtime applies `censor_response()` to all final output before
it reaches any channel, providing a last-resort defense against secret
leakage through AI-generated responses.  Audit event detail messages are
also redacted through `censor_response()` to prevent secrets from leaking
into audit logs.

### Tool Output Injection Scanning

Tool results (shell output, file contents, MCP server responses, web
fetches) are scanned for prompt injection patterns before being fed back
to the AI model.  When suspicious patterns are detected, a security
warning label is prepended to the tool output so the model treats it as
untrusted data rather than instructions.

### Gateway Security

All outbound HTTP requests are routed through `PolicyHTTPClient`, which:

- Restricts URL schemes to `http://` and `https://` only (blocks `file://`,
  `ftp://`, `data://` SSRF vectors).
- Disables redirect following (`follow_redirects=False`) to prevent
  redirect-based SSRF.
- Uses a kwargs allowlist — only safe parameters (`headers`, `params`,
  `data`, `json`, `content`, `cookies`, `timeout`, `files`) pass through.
  Dangerous kwargs like `verify=False`, `base_url`, `transport`, and `auth`
  are silently stripped.
- Enforces connection pool limits (20 connections, 10 keepalive, 30s expiry)
  to prevent resource exhaustion.
- Performs DNS rebinding protection: all resolved IPs are checked, and if
  any address is private/reserved without explicit CIDR allowance, the
  entire request is denied.

### Webhook Security

The webhook channel enforces multiple layers of protection:

- **Content-Type validation**: Only `application/json` requests are accepted;
  all others receive 415 (Unsupported Media Type), preventing CSRF via form
  submissions.
- **Content-Length validation**: Parsed as a non-negative integer; invalid
  values receive 400 (Bad Request).
- **Payload size limits**: Payloads larger than 1 MB receive 413.
- **Per-IP rate limiting**: 60 requests per 60-second sliding window; excess
  receives 429 (Too Many Requests) with `Retry-After` header.
- **Queue bounds**: Maximum 1000 pending messages; excess receives 503.
- **HMAC authentication**: Optional SHA-256 signature verification via
  `X-Missy-Signature` header.
- **Header filtering**: Only safe headers (Content-Type, User-Agent,
  X-Request-Id, X-Missy-Signature) are stored in message metadata;
  Authorization, Cookie, and forwarding headers are stripped.

### MCP Server Isolation

MCP server subprocesses receive a sanitized environment containing only
safe variables (PATH, HOME, LANG, etc.), preventing API keys and other
secrets from leaking to potentially untrusted MCP servers.  Server names
are validated to prevent namespace collision attacks.  RPC reads have a
30-second timeout and 1 MB size limit to prevent denial-of-service.
Response IDs are validated against request IDs to detect response
confusion attacks.  MCP configuration files are written with restrictive
permissions (0o600) and verified for ownership and writability before
loading.

### Vault Security

Secrets are stored using ChaCha20-Poly1305 authenticated encryption.
Key file creation uses `O_CREAT|O_EXCL` for TOCTOU-safe atomic creation.
Vault writes use a temp-file-then-rename pattern with `fsync` to prevent
data loss on process interruption.  Symlinks and hard links (st_nlink > 1)
in key file paths are rejected.

### Custom Tool Content Validation

The `self_create_tool` tool allows the agent to create custom scripts in
`~/.missy/custom-tools/`.  Before writing any script, the content is
scanned for 15+ dangerous patterns including network access (`curl`,
`wget`, `nc`, `/dev/tcp/`), code execution (`eval`, `exec`, `os.system`,
`subprocess`), and privilege escalation (`chmod +s`, `setuid`).  Scripts
matching any pattern are rejected with a descriptive error.  Tool names
are validated to alphanumeric/underscore/hyphen characters only.

### Device Registry Safety

The voice device registry (`~/.missy/devices.json`) verifies file
ownership and permissions before loading.  Files not owned by the current
user, or that are group- or world-writable, are rejected to prevent
a compromised file from injecting pre-paired device entries.

### Scheduler Job File Safety

Scheduler job state (`~/.missy/jobs.json`) is written atomically using
`tempfile.mkstemp()` + `os.replace()` with restrictive 0o600 permissions.
This prevents partial writes on crash and unauthorized read access to
task descriptions that may contain sensitive prompt content.

### Config Hot-Reload Safety

Before reloading configuration, the watcher verifies the config file is
not a symlink, is owned by the current user, and is not group- or
world-writable.  This prevents an attacker from injecting a malicious
config via a symlink or permission escalation.

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

## Secrets Management

API keys can be stored in three ways (from most to least secure):

1. **Encrypted vault** — `missy vault set ANTHROPIC_API_KEY sk-ant-...`
   Reference in config as `vault://ANTHROPIC_API_KEY`.
2. **Environment variables** — `export ANTHROPIC_API_KEY="sk-ant-..."`
   Reference in config as `$ANTHROPIC_API_KEY`.
3. **Never in plaintext config** — API keys should never appear directly
   in `config.yaml`.

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
| 0.2.0 | Added tool output injection scanning, response censoring, webhook rate limiting, MCP server isolation, process substitution blocking, vault atomic writes, config hotreload safety, encrypted vault with ChaCha20-Poly1305. |
| 0.3.0 | Extended sanitizer to 40+ patterns (tool abuse, prompt leaking, Japanese, Anthropic delimiters); added Azure/Twilio/Mailgun secret detection; gateway async aput/ahead methods; web_fetch header sanitization as class constant. |
