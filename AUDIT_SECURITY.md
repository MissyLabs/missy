# Missy Security Audit

## Security Model

Missy enforces **DEFAULT DENY** across all subsystems:

### Network Security
- All outbound HTTP routed through `PolicyHTTPClient`
- CIDR allowlists, domain suffix matching, exact hostname matching
- Per-category host allowlists (provider, tool, discord)
- L7 REST policies: HTTP method + path glob rules per host
- Network presets auto-expand to correct hosts/domains/CIDRs

### Filesystem Security
- Per-path read/write access control via `FilesystemPolicyEngine`
- Default: only `~/workspace` and `~/.missy` are writable

### Shell Security
- Shell execution disabled by default
- Command whitelisting via `ShellPolicyEngine`

### Input Security
- `InputSanitizer`: 250+ prompt injection patterns, Unicode normalization, base64 decode
- `SecretsDetector`: 37+ credential patterns (API keys, JWTs, AWS, GCP)
- `SecretCensor`: Redacts secrets from output with overlap merging

### Cryptographic Security
- `Vault`: ChaCha20-Poly1305 encrypted key-value store
- `AgentIdentity`: Ed25519 keypair, signs audit events
- `PromptDriftDetector`: SHA-256 system prompt tamper detection

### Trust Model
- `TrustScorer`: 0-1000 reliability tracking per tool/provider/MCP server
- Success: +10, failure: -50, violation: -200
- Warns below threshold

## Threat Defenses

| Threat | Defense |
|--------|---------|
| Prompt injection | InputSanitizer with 250+ patterns, Unicode normalization |
| Plugin abuse | Plugin permission declarations, policy enforcement |
| Data exfiltration | Default-deny network policy, PolicyHTTPClient |
| SSRF | All HTTP through gateway with policy check |
| Scheduler abuse | Max jobs limit, active hours constraint |
| Secrets leakage | SecretsDetector + SecretCensor on all output |
| Tool abuse | TrustScorer tracks reliability, CircuitBreaker stops cascading failures |
| Channel impersonation | DM allowlist, guild/role policies for Discord |
| Unsafe visual activation | Vision is on-demand only, ≥0.80 confidence threshold |
| Confused audio trigger | Ambiguous intents require explicit user confirmation |
| Prompt drift/tampering | SHA-256 hash verification before each provider call |
| Unauthorized camera access | OS-level permissions, video group membership required |

## Audit Infrastructure

- `AuditLogger`: Structured JSONL at `~/.missy/audit.jsonl`
- Every privileged action logged with timestamp, task ID, session ID, policy rule, result
- Vision operations logged with device, source type, trigger reason, success/failure
- `OtelExporter`: Optional OpenTelemetry traces and metrics
- `AgentIdentity`: Ed25519 signature on audit events for tamper evidence

## Vision Security

- Images processed in-memory only (not persisted unless explicitly requested)
- Vision activation requires user command or high-confidence intent (≥0.80)
- Ambiguous intents (0.50–0.79) require explicit user confirmation
- All activation decisions logged for audit
- Image data sent to LLM API subject to network policy
- No always-on surveillance mode
- Scene memory is in-process only, cleared on session close
