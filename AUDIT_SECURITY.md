# AUDIT_SECURITY

- Timestamp: 2026-07-08 09:59:08 EDT

## Expected common security and operations docs

- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security Posture Updates

- Web TUI authentication remains explicit: API key login creates hardened
  browser sessions and API-key/bearer auth still protects `/api/v1/*`.
- Unsafe browser-authenticated API requests still require CSRF tokens.
- Diagnostics remain authenticated and redacted before API/browser delivery.
- New diagnostics check gateway policy binding, response-size caps, default-deny
  network posture, scoped provider/tool allowlists, REST method/path policy
  coverage, Discord prerequisites, and scheduling policy posture.
- Discord diagnostics report token presence only; token values are not returned.
- Remediation hints are redacted with the same redaction path as summaries.

## Residual Risks

- Safe operator controls are still pending and must be default-deny,
  policy-gated, CSRF-protected, confirmation guarded, and audited.
- Run/session viewer is still pending and must defend against prompt injection,
  XSS in streamed output, secret leakage, and unsafe replay/resume behavior.
- Browser visual/responsive regression coverage is still pending.
