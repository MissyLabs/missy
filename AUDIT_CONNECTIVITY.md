# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 09:31:21

Expected connectivity posture:
- default-deny network where practical
- exact provider endpoints
- explicit local Web TUI bind address and origin policy
- exact benchmark and provider endpoints
- no unreviewed broad outbound access

Current diagnostics coverage:
- `/api/v1/diagnostics` reports whether the Web API is bound to loopback and
  summarizes network allowlist counts without exposing secrets.
- Future diagnostics should add explicit gateway/network probes and explain
  provider, tool, and Discord endpoint reachability.
