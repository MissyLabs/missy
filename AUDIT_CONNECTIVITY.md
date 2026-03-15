# AUDIT_CONNECTIVITY

- Timestamp: 2026-03-15 (updated session 21)
- Auditor: Automated build analysis

## Network Architecture

Missy enforces **default-deny** outbound networking. All HTTP traffic passes through `PolicyHTTPClient` (`missy/gateway/client.py`) which checks every request against the network policy engine before dispatch.

### Security Layers

1. **URL scheme restriction** — Only `http://` and `https://` schemes are allowed; `file://`, `ftp://`, `data://` etc. are blocked at the gateway level
2. **DNS rebinding protection** — ALL resolved IPs are checked; if ANY resolved address is private/reserved and not in `allowed_cidrs`, the entire request is denied
3. **Redirect bypass prevention** — `follow_redirects` is explicitly set to `False`; redirects to blocked hosts are not followed
4. **Response size limits** — Default 50 MB max response body; checked via Content-Length header AND actual body length for chunked responses
5. **Connection pool limits** — max_connections=20, max_keepalive_connections=10, keepalive_expiry=30s to prevent resource exhaustion
6. **kwargs sanitization** — Only safe httpx kwargs are forwarded (allowlist: headers, params, data, json, content, cookies, timeout, files, extensions); blocks verify, base_url, transport, auth, event_hooks

## Allowed Connectivity Patterns

### Provider Endpoints (per-category: provider_allowed_hosts)

| Provider | Required Hosts |
|---|---|
| Anthropic | api.anthropic.com |
| OpenAI | api.openai.com |
| Ollama | localhost:11434 (configurable) |
| Codex | localhost:configurable |

### Tool Endpoints (per-category: tool_allowed_hosts)

| Tool | Required Hosts |
|---|---|
| WebFetchTool | Operator-configured domains |
| BrowserTools | Operator-configured domains |

### Discord Endpoints (per-category: discord_allowed_hosts)

| Service | Required Hosts |
|---|---|
| Discord API | discord.com, gateway.discord.gg |
| Discord CDN | cdn.discordapp.com |

### Infrastructure Endpoints

| Service | Required Hosts |
|---|---|
| OpenTelemetry | Operator-configured OTLP endpoint (default: localhost:4317) |
| MCP servers | Operator-configured per-server |

## Secure Configuration Example

```yaml
network:
  default_deny: true
  allowed_cidrs:
    - "10.0.0.0/8"          # Internal network
  allowed_domains:
    - "*.github.com"         # GitHub API access
  allowed_hosts: []          # No general host access
  provider_allowed_hosts:
    - "api.anthropic.com"
    - "api.openai.com"
  tool_allowed_hosts: []     # No tool network access by default
  discord_allowed_hosts:
    - "discord.com"
    - "gateway.discord.gg"
    - "cdn.discordapp.com"
```

## Inbound Traffic Controls

| Channel | Control | Details |
|---|---|---|
| Webhook | Rate limiting | 60 req/min per IP, 1 MB max payload, 1000 max queue |
| Webhook | HMAC authentication | Optional SHA-256 signature validation |
| Webhook | Content-Type validation | Requires application/json (CSRF prevention) |
| Webhook | Content-Length validation | Must be non-negative integer |
| Webhook | Header filtering | Only safe headers stored (strips Authorization, Cookie, X-Forwarded-*) |
| Webhook | Reverse proxy support | Optional X-Forwarded-For parsing for correct rate limiting |
| Webhook | Rate tracker cleanup | Evicts stale IPs when tracked count exceeds 10K |
| Voice | Device pairing | PBKDF2-hashed tokens, per-device policy modes (full/safe-chat/muted) |
| Voice | Input validation | sample_rate clamped [8000,48000], channels clamped [1,2] |
| Discord | Access control | DM allowlist, guild/role policies |

## MCP Server Isolation

MCP server subprocesses receive a sanitized environment with only safe
variables (PATH, HOME, LANG, etc.).  Server names must not contain `__`
to prevent namespace collision.  RPC reads have a 30s timeout and 1 MB
size limit.  Config file ownership and permissions are verified before loading.
Server names are validated against `^[a-zA-Z0-9_\-]+$`.  Response IDs are
checked against request IDs.  Tool outputs can optionally be scanned for
injection patterns (`block_injection=True` mode).

## Shell Execution Controls

Shell commands pass through `ShellPolicyEngine` which blocks:
- Pipes (`|`), semicolons (`;`), `&&`, `||` compound operators
- Command substitution: backticks (`` ` ``), `$()`, `<()`, `>()`
- Heredocs (`<<`, `<<<`), brace groups (`{ }`)
- Background execution (`&`)
- Subshells: `(`, `)`, `<(`, `>(`, `<<(`
- Environment is sanitized to safe-only variables (PATH, HOME, LANG, TERM, etc.)
- Subprocess timeout: 30s default

## Filesystem Access Controls

- `FilesystemPolicyEngine` enforces per-path read/write access control
- Symlinks are resolved before policy check (TOCTOU fix)
- Path argument (`path` kwarg) validated against filesystem policy at execution time
- Browser session IDs validated against directory traversal (regex validation)

## Policy Enforcement Points

1. **PolicyHTTPClient._check_url()** — Every outbound HTTP request checked against network policy
2. **PolicyHTTPClient._check_response_size()** — Response body size limits enforced
3. **PolicyHTTPClient._sanitize_kwargs()** — Unsafe httpx kwargs stripped
4. **ToolRegistry._check_permissions()** — Tool path args checked against filesystem policy
5. **ShellExecTool.execute()** — Commands checked against shell policy
6. **FileReadTool/FileWriteTool** — Paths checked against filesystem policy
7. **SchedulerManager._run_job()** — Jobs run through policy enforcement
8. **WebhookChannel._check_rate_limit()** — Per-IP rate limiting on inbound
9. **McpClient._rpc()** — Timeout and size limits on MCP server responses
10. **McpManager.connect_all()** — Config file ownership/permissions verified
11. **AgentRuntime._run_loop()** — Tool output scanned for injection patterns

## Network Policy Test Coverage

| Test Category | Tests |
|---|---|
| CIDR allowlist matching | 20+ |
| Domain suffix matching | 15+ |
| Exact hostname matching | 15+ |
| Per-category host lists | 10+ |
| Default deny enforcement | 10+ |
| PolicyHTTPClient integration | 30+ |
| DNS rebinding protection | 5+ |
| URL scheme restriction | 5+ |
| Response size limits | 10+ |
| Connection pool limits | 5+ |
| Shell injection vectors | 25+ |
| Filesystem policy enforcement | 10+ |
| **Total** | **160+** |

## Connectivity Verification

All outbound connections are auditable via:
- `missy audit security` — Shows network allow/deny events
- `~/.missy/audit.jsonl` — Structured JSONL with every network decision
- OpenTelemetry traces — When OTEL is enabled, traces include HTTP spans
- Gateway thread safety — Verified with concurrent test harness

## Secret Protection in Transit

- `SecretsDetector` (40 patterns) scans all text for credentials
- `SecretCensor` redacts detected secrets from agent output and audit logs
- Overlapping redaction spans are merged to prevent partial leakage
- Shell subprocess environment sanitized to prevent API key leakage
- Piper TTS subprocess environment also sanitized
- Webhook log output runs through `censor_response()`
