# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 09:59:08 EDT

Expected connectivity posture:

- default-deny network where practical
- exact provider endpoints
- explicit local Web TUI bind address and origin policy
- exact benchmark and provider endpoints
- no unreviewed broad outbound access

This session:

- Added diagnostics summaries for provider-scoped, tool-scoped, and
  Discord-scoped network allowlists.
- Added diagnostics visibility for REST method/path policy rule count.
- Added gateway diagnostics that confirm `PolicyHTTPClient` availability and
  policy-engine binding without performing outbound probes.
- Added Discord readiness checks for `discord.com` and `gateway.discord.gg`
  allowlist prerequisites.

Remaining:

- Add optional live reachability probes only when policy allows them and only
  with short timeouts and redacted failures.
