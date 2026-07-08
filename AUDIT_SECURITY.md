# AUDIT_SECURITY

- Timestamp: 2026-07-08 15:03:20 EDT

## Expected common security and operations docs

- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md
- present: docs/providers.md

## Security Notes

- Provider diagnostics are local-only and do not make live OpenAI calls.
- OpenAI diagnostics report API key source (`config`, `environment`, or
  `missing`) but never the key value.
- Custom OpenAI-compatible endpoints are summarized by host only, avoiding
  leakage from userinfo, query strings, or embedded tokens.
- API diagnostics and `missy doctor` consume provider diagnostics through the
  provider-neutral hook and continue to apply existing redaction.
- Network posture diagnostics inspect configured allowlists without DNS
  resolution or audit-event side effects.

## Remaining Security Work

- Add audit events for retry, rate-limit cooldown, usage/cost recording,
  fallback, and provider-side validation denials.
- Add optional explicit live diagnostic probes with clear operator intent before
  credential/model-list checks.
