# AUDIT_SECURITY

- Timestamp: 2026-03-15 (updated session 12)
- Auditor: Automated build analysis

## Security Architecture Summary

Missy implements defense-in-depth with 6 security layers:

1. **Input Sanitization** — 13+ prompt injection pattern detectors
2. **Secrets Detection** — 15 credential patterns (API keys, JWTs, AWS, sk-proj-, etc.)
3. **Output Censoring** — `censor_response()` applied in agent runtime before output delivery
4. **Tool Output Injection Scanning** — Tool results scanned for prompt injection, warning labels prepended
5. **Policy Enforcement** — 3-layer default-deny (network, filesystem, shell) with process substitution blocking
6. **Encrypted Vault** — ChaCha20-Poly1305 key-value store
7. **Docker Sandbox** — Optional container isolation for shell commands
8. **MCP Server Isolation** — Sanitized environment variables, name validation, response timeouts/size limits

## Threat Model Coverage

| Threat | Defense | Test Coverage |
|---|---|---|
| Prompt injection (user input) | InputSanitizer (13 patterns) | 30+ tests |
| Prompt injection (tool output) | Tool output scanning + warning labels | 19+ tests |
| Plugin abuse | Plugin allowlist + disabled by default | 30+ tests |
| Data exfiltration | Default-deny network + output censoring | 120+ policy tests |
| SSRF | PolicyHTTPClient blocks unauthorized outbound | 50+ gateway tests |
| Scheduler abuse | Active hours, max_jobs, policy enforcement | 100+ scheduler tests |
| Secrets leakage | SecretsDetector + SecretCensor on all output | 55+ security tests |
| Tool abuse | Tool registry policy checks + approval gate | 280+ tool tests |
| Channel impersonation | Discord access control (DM/guild/role) | 440+ channel tests |

## Policy Engine Tests

### Network Policy

- Default deny blocks all traffic when no allowlists configured
- CIDR allowlist correctly matches IP ranges (10.0.0.0/8, 192.168.0.0/16)
- Domain suffix matching (*.github.com matches api.github.com)
- Exact hostname matching
- Per-category host lists (provider, tool, discord)
- PolicyHTTPClient enforces policy before every HTTP dispatch

### Filesystem Policy

- Write blocked outside allowed_write_paths
- Read blocked outside allowed_read_paths
- Path traversal attacks (../) properly normalized and blocked
- Symlink resolution before policy check

### Shell Policy

- Disabled by default (shell.enabled: false)
- Only whitelisted commands allowed when enabled
- Empty allowed_commands blocks all commands
- Compound command detection (&&, ||, ;, |)
- Process substitution blocked (<(...), >(...), <<(...))
- Command substitution blocked ($(...), backticks)
- Docker sandbox optional isolation

## Secrets Detection Patterns

| Pattern | Description |
|---|---|
| API keys | Generic API key patterns (sk-*, key-*) |
| AWS credentials | AKIA... access key IDs |
| JWT tokens | eyJ... base64-encoded tokens |
| GitHub tokens | ghp_*, gho_*, ghs_* patterns |
| Private keys | BEGIN RSA/DSA/EC PRIVATE KEY |
| Connection strings | postgresql://, mysql://, mongodb:// |
| Bearer tokens | Bearer ... in authorization headers |
| Slack tokens | xoxb-*, xoxp-* patterns |
| Discord tokens | Bot token patterns |

## Vault Security

- ChaCha20-Poly1305 authenticated encryption
- Key file stored at ~/.missy/secrets/vault.key (32 bytes)
- Encrypted data at ~/.missy/secrets/vault.enc
- vault:// references resolved at config load time
- Invalid key length rejected
- Corrupted data detected via authentication tag

## Audit Logging

All privileged actions are logged with:
- Timestamp (ISO 8601)
- Task ID
- Session ID
- Policy rule applied
- Result (allow/deny)
- Category (network, shell, filesystem, plugin, scheduler, provider)

Audit events stored as structured JSONL at ~/.missy/audit.jsonl.

## Security Test Summary

| Category | Tests |
|---|---|
| Input sanitizer | 30+ |
| Secrets detector | 20+ |
| Secret censor | 10+ |
| Vault | 20+ |
| Docker sandbox | 15+ |
| Network policy | 50+ |
| Filesystem policy | 30+ |
| Shell policy | 30+ |
| Audit logger | 40+ |
| Discord access control | 35+ |
| MCP server safety | 19+ |
| Tool output injection | 19+ |
| Response censoring | 10+ |
| **Total security-related** | **330+** |
