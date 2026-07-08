# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 15:03:20 EDT

Expected connectivity posture:

- default-deny network where practical
- exact OpenAI provider endpoints
- explicit provider endpoint checks in diagnostics
- explicit local Web TUI bind address and origin policy
- exact benchmark and non-OpenAI provider endpoints where enabled
- no unreviewed broad outbound access

This session added local OpenAI endpoint diagnostics:

- Native OpenAI defaults to `api.openai.com`.
- Custom OpenAI-compatible `base_url` values are reduced to host-only posture.
- Diagnostics check `allowed_hosts`, `provider_allowed_hosts`, and
  `allowed_domains` locally without DNS or live API calls.
