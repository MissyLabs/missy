# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 14:34:10 EDT

Expected connectivity posture:

- default-deny network where practical
- exact OpenAI provider endpoints
- explicit local Web TUI bind address and origin policy
- exact benchmark and non-OpenAI provider endpoints where enabled
- no unreviewed broad outbound access

Session connectivity notes:

- No new outbound hosts or broad network permissions were added.
- OpenAI-native structured output reuses existing SDK client construction and
  policy-aware HTTP wiring.
- Compatibility `base_url` behavior remains routed through Chat Completions and
  does not introduce new endpoint assumptions outside the configured provider.
