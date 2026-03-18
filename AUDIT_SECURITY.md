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

## File Permission Hardening (Session 3)

All directory creation under `~/.missy/` now uses `mode=0o700` to prevent
other users on shared systems from reading sensitive configuration, audit logs,
memory databases, or identity keys.

Sensitive files (persona, hatching state, vault) are created with `mode=0o600`
using `os.open()` with restrictive flags (`O_CREAT | O_TRUNC` or `O_CREAT | O_APPEND`).

### Files hardened

| File | mkdir mode | File mode |
|---|---|---|
| `hatching.py` — state file | 0o700 | 0o600 (os.open) |
| `hatching.py` — log file | 0o700 | 0o600 (os.open + O_APPEND) |
| `hatching.py` — env validate | 0o700 | — |
| `hatching.py` — config init | 0o700 | — |
| `hatching.py` — secrets dir | 0o700 | — |
| `persona.py` — persona file | 0o700 | 0o600 (chmod) |
| `persona.py` — audit log | 0o700 | 0o600 (os.open + O_APPEND) |
| `persona.py` — backup dir | 0o700 | — |
| `cli/main.py` — init | 0o700 | — |
| `cli/wizard.py` — config | 0o700 | — |
| `config/plan.py` — backup | 0o700 | — |
| `security/identity.py` — key | 0o700 | 0o600 (existing) |
| `security/vault.py` — vault | 0o700 | 0o600 (existing) |
| `memory/sqlite_store.py` | 0o700 | — |
| `memory/store.py` | 0o700 | — |
| `memory/vector_store.py` | 0o700 | — |
| `mcp/manager.py` | 0o700 | 0o600 (existing) |
| `scheduler/manager.py` | 0o700 | 0o600 (existing) |
| `agent/playbook.py` | 0o700 | — |
| `agent/prompt_patches.py` | 0o700 | — |
| `agent/code_evolution.py` | 0o700 | — |
| `channels/voice/registry.py` | 0o700 | — |
| `agent/checkpoint.py` | 0o700 | — |
| `channels/screencast/analyzer.py` | 0o700 | — |
| `channels/discord/image_analyze.py` | 0o700 | 0o600 (os.open, 2 locations) |
| `memory/vector_store.py` | 0o700 | 0o600 (os.open) |
| `tools/builtin/tts_speak.py` | — | 0o600 (os.open) |
