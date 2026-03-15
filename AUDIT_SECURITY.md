# AUDIT_SECURITY

- Timestamp: 2026-03-15 (updated session 16)
- Auditor: Automated build analysis + security audit agent

## Security Architecture Summary

Missy implements defense-in-depth with 12 security layers:

1. **Input Sanitization** — 26+ prompt injection pattern detectors (including Llama 3, multilingual, data URI, unclosed HTML)
2. **Secrets Detection** — 23 credential patterns (API keys, JWTs, AWS, GitLab, npm, PyPI, SendGrid, DB connection strings)
3. **Output Censoring** — `censor_response()` applied in agent runtime and audit events
4. **Tool Output Injection Scanning** — Tool results scanned for prompt injection, warning labels prepended
5. **Policy Enforcement** — 3-layer default-deny (network, filesystem, shell) with:
   - Process substitution blocking (`<()`, `>()`, `<<()`)
   - Bare `&` splitting for compound commands
   - Here-string rejection (`<<<`)
   - Brace group rejection (`{ cmd; }`)
   - File tool path enforcement via kwargs
6. **Encrypted Vault** — ChaCha20-Poly1305 key-value store with atomic writes, symlink rejection, hard-link check
7. **Docker Sandbox** — Optional container isolation for shell commands
8. **MCP Server Isolation** — Sanitized environment, name validation, response timeouts/size limits, config file permission checks
9. **Gateway SSRF Prevention** — URL scheme restriction (http/https only), redirect following disabled, kwargs allowlist
10. **Shell Launcher Warnings** — Policy engine warns when command-launching programs (env, bash, sudo) are whitelisted
11. **Webhook Hardening** — Content-Type validation, Content-Length safety, HMAC signatures, rate limiting, metadata header filtering
12. **DNS Rebinding Protection** — All resolved IPs checked before access; mixed public/private records denied

## Threat Model Coverage

| Threat | Defense | Test Coverage |
|---|---|---|
| Prompt injection (user input) | InputSanitizer (26+ patterns) | 40+ tests |
| Prompt injection (tool output) | Tool output scanning + warning labels | 22+ tests |
| Plugin abuse | Plugin allowlist + disabled by default | 30+ tests |
| Data exfiltration | Default-deny network + output censoring + audit redaction | 120+ policy tests |
| SSRF | PolicyHTTPClient: scheme restriction, no redirects, kwargs allowlist, DNS rebinding check | 90+ gateway tests |
| Scheduler abuse | Active hours, max_jobs, policy enforcement, input validation | 100+ scheduler tests |
| Secrets leakage | SecretsDetector (23 patterns) + SecretCensor on output + audit redaction | 85+ security tests |
| Tool abuse | Tool registry policy checks + file path enforcement + approval gate | 290+ tool tests |
| Channel impersonation | Discord access control (DM/guild/role), webhook HMAC, Content-Type validation | 460+ channel tests |
| Shell bypass | Here-strings, brace groups, process substitution, heredocs all blocked | 40+ shell tests |
| Webhook CSRF | Content-Type: application/json required, header filtering | 30+ webhook tests |

## Policy Engine Tests

### Network Policy

- Default deny blocks all traffic when no allowlists configured
- CIDR allowlist correctly matches IP ranges (10.0.0.0/8, 192.168.0.0/16)
- Domain suffix matching (*.github.com matches api.github.com)
- Exact hostname matching
- Per-category host lists (provider, tool, discord)
- PolicyHTTPClient enforces policy before every HTTP dispatch
- DNS rebinding protection: mixed public/private record sets denied
- Unparseable IP addresses from getaddrinfo safely skipped

### Filesystem Policy

- Write blocked outside allowed_write_paths
- Read blocked outside allowed_read_paths
- Path traversal attacks (../) properly normalized and blocked
- Symlink resolution before policy check
- File tool kwargs path enforcement (H2 fix)

### Shell Policy

- Disabled by default (shell.enabled: false)
- Only whitelisted commands allowed when enabled
- Empty allowed_commands blocks all commands
- Compound command detection (&&, ||, ;, |, &)
- Process substitution blocked (<(...), >(...), <<(...))
- Command substitution blocked ($(...), backticks)
- Here-string injection blocked (<<<)
- Brace group execution blocked ({ cmd; })
- Docker sandbox optional isolation

## Gateway Security

- **Kwargs Allowlist**: Only safe parameters pass through (headers, params, data, json, content, cookies, timeout, files, extensions). Dangerous kwargs like `verify`, `base_url`, `transport`, `auth`, `event_hooks` are stripped.
- **Scheme Restriction**: Only http:// and https:// allowed
- **Redirect Prevention**: `follow_redirects=False` enforced on all clients
- **Connection Pool Limits**: max_connections=20, max_keepalive_connections=10, keepalive_expiry=30s

## Webhook Security

- **Content-Type Validation**: Only `application/json` accepted (prevents CSRF via form submissions)
- **Content-Length Safety**: Validated as non-negative integer; non-integer/negative values rejected with 400
- **HMAC Signatures**: Optional SHA-256 HMAC verification via X-Missy-Signature
- **Rate Limiting**: Per-IP rate limiting with configurable window
- **Payload Size**: Enforced maximum payload size
- **Queue Overflow**: Returns 503 when message queue is full
- **Header Filtering**: Only safe headers stored in metadata (Content-Type, User-Agent, X-Request-Id, X-Missy-Signature); Authorization, Cookie, X-Forwarded-* stripped

## Secrets Detection Patterns

| Pattern | Description |
|---|---|
| API keys | Generic API key patterns (sk-*, key-*) |
| AWS credentials | AKIA... access key IDs |
| JWT tokens | eyJ... base64-encoded tokens |
| GitHub tokens | ghp_*, gho_*, ghs_* patterns |
| GitLab tokens | glpat-* patterns |
| npm tokens | npm_* patterns |
| PyPI tokens | pypi-* patterns |
| SendGrid keys | SG.* patterns |
| Private keys | BEGIN RSA/DSA/EC PRIVATE KEY |
| Connection strings | postgresql://, mysql://, mongodb:// |
| Bearer tokens | Bearer ... in authorization headers |
| Slack tokens | xoxb-*, xoxp-* patterns |
| Discord tokens | Bot token patterns |
| OpenAI keys | sk-proj-* extended format |

## Vault Security

- ChaCha20-Poly1305 authenticated encryption
- Key file stored at ~/.missy/secrets/vault.key (32 bytes)
- Encrypted data at ~/.missy/secrets/vault.enc
- vault:// references resolved at config load time
- Invalid key length rejected
- Corrupted data detected via authentication tag
- Atomic file creation with O_CREAT|O_EXCL (TOCTOU-safe)
- Symlink rejection on key and data files

## Audit Logging

All privileged actions are logged with:
- Timestamp (ISO 8601)
- Task ID
- Session ID
- Policy rule applied
- Result (allow/deny)
- Category (network, shell, filesystem, plugin, scheduler, provider)
- Detail messages redacted via censor_response() to prevent secret leakage

Audit events stored as structured JSONL at ~/.missy/audit.jsonl.

## Security Test Summary

| Category | Tests |
|---|---|
| Input sanitizer | 40+ |
| Secrets detector | 25+ |
| Secret censor | 10+ |
| Vault | 20+ |
| Docker sandbox | 15+ |
| Network policy | 55+ |
| Filesystem policy | 35+ |
| Shell policy | 40+ |
| Audit logger | 40+ |
| Discord access control | 35+ |
| MCP server safety | 54+ |
| Tool output injection | 22+ |
| Response censoring | 10+ |
| Gateway SSRF | 30+ |
| Webhook hardening | 30+ |
| **Total security-related** | **560+** |

## Session 16 Security Fixes

| Finding | Severity | Fix |
|---|---|---|
| Shell heredoc bypass | HIGH | Added `<<` to _SUBSHELL_MARKERS to block heredoc injection |
| Shell brace group bypass | HIGH | Brace group scanning now checks entire command, not just start |
| FTS5 query injection | MEDIUM | Query input wrapped in double-quotes; OperationalError caught |
| Format string injection | MEDIUM | Proactive triggers use string.Template instead of str.format() |
| Scheduler jobs file perms | MEDIUM | Atomic write with tempfile+rename, restrictive 0o600 permissions |
| MCP response confusion | MEDIUM | Response ID validated against request ID |
| Self-create tool content | HIGH | Script content scanned for dangerous patterns before writing |
| Code evolve silent exit | LOW | SystemExit now logged as warning instead of silently suppressed |
| Webhook log secret leak | LOW | censor_response() applied to log output |
| Device registry perms | LOW | File ownership and group/world-writable checks on load |
| Audio log file perms | LOW | Directory created with 0o700, files with 0o600 |

## Session 15 Security Fixes

| Finding | Severity | Fix |
|---|---|---|
| H2: File tools bypass filesystem policy | HIGH | Registry now checks actual path from kwargs against check_read/check_write |
| H3: Gateway kwargs blocklist | HIGH | Replaced with explicit allowlist of safe kwargs |
| H1: Shell heredoc/brace bypass | HIGH | Added <<<, { }, {; to rejection lists |
| H4: Webhook header leakage | HIGH | Metadata filtered to safe allowlist (Content-Type, User-Agent, X-Request-Id) |
| M5: Webhook Content-Type | MEDIUM | Requires application/json to prevent CSRF |
| M6: Webhook Content-Length | MEDIUM | Validated as non-negative integer |
| L3: Audit event secret leakage | LOW | Detail messages run through censor_response() |
