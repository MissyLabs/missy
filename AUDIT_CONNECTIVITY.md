# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 14:14:39 EDT

Expected connectivity posture:

- default-deny network where practical
- exact OpenAI provider endpoints
- native OpenAI Responses completions and streams both use the same policy-aware
  SDK client construction
- OpenAI-compatible `base_url` providers remain on Chat Completions fallback
  and should be explicitly allowed by host
- explicit local Web TUI bind address and origin policy
- exact benchmark and non-OpenAI provider endpoints where enabled
- no unreviewed broad outbound access
