# Security Audit Report

## Date: 2026-03-18

## Threat Model

### 1. Prompt Injection
- **Defense**: `InputSanitizer` with 250+ patterns, Unicode normalization, base64 decode
- **Detection**: Multi-language support, nested encoding detection
- **Response**: Logged to audit trail, user warned, input sanitized

### 2. Plugin Abuse
- **Defense**: Plugin allowlist in config, disabled by default
- **Detection**: Permission declarations required
- **Response**: Blocked at policy engine

### 3. Data Exfiltration
- **Defense**: Default-deny network policy, CIDR/domain/host allowlists
- **Detection**: All outbound HTTP goes through PolicyHTTPClient
- **Response**: Blocked and audit logged

### 4. SSRF
- **Defense**: PolicyHTTPClient validates all URLs against network policy
- **Detection**: REST policy (L7) controls method + path per host
- **Response**: Denied with audit event

### 5. Scheduler Abuse
- **Defense**: Max jobs limit, active hours restriction
- **Detection**: All scheduled tasks audit logged
- **Response**: Jobs blocked outside active hours

### 6. Secrets Leakage
- **Defense**: `SecretsDetector` (37+ patterns), `SecretCensor` for output
- **Detection**: Input scanning, output redaction with overlap merging
- **Response**: Redacted before display, warned in audit log

### 7. Tool Abuse
- **Defense**: Capability modes (full/safe-chat/no-tools), approval gate
- **Detection**: Trust scorer tracks reliability per tool
- **Response**: Human-in-the-loop approval for sensitive operations

### 8. Channel Impersonation
- **Defense**: Discord access control (DM allowlist, guild/role policies)
- **Detection**: Voice channel device pairing with PBKDF2 tokens
- **Response**: Unauthorized requests rejected

## Security Subsystems

| Subsystem | File | Status |
|---|---|---|
| Input Sanitizer | `missy/security/sanitizer.py` | Active |
| Secrets Detector | `missy/security/secrets.py` | Active |
| Secret Censor | `missy/security/censor.py` | Active |
| Vault | `missy/security/vault.py` | Active |
| Agent Identity | `missy/security/identity.py` | Active |
| Trust Scorer | `missy/security/trust.py` | Active |
| Prompt Drift | `missy/security/drift.py` | Active |
| Container Sandbox | `missy/security/container.py` | Active |
| Policy Engine | `missy/policy/engine.py` | Active |
| Network Policy | `missy/policy/network.py` | Active |
| Filesystem Policy | `missy/policy/filesystem.py` | Active |
| REST Policy | `missy/policy/rest_policy.py` | Active |
| Gateway Client | `missy/gateway/client.py` | Active |

## Persona Security Boundaries

The persona system enforces:
- Never execute destructive operations without confirmation
- Never expose secrets or credentials
- Always respect policy engine decisions
- Flag security concerns proactively

These boundaries are injected into every LLM system prompt.
