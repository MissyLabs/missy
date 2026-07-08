# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08
- Branch: overhaul/tools-20260708-020326
- Base: origin/master

## Expected Connectivity Posture

- Default-deny network where practical.
- Exact provider, benchmark, Discord Gateway, Discord REST, and CDN endpoints should remain policy-visible.
- No unreviewed broad outbound access should be introduced by tool intelligence, provider schema adaptation, benchmarks, or Discord diagnostics.

## Session Connectivity Notes

- Provider tool schema normalization is local-only and does not add network access.
- `missy tools benchmark run` executes through the local tool registry and labels provider results; it does not bypass existing provider or tool policy.
- Discord REST remains routed through `PolicyHTTPClient`.
- Discord Gateway lifecycle diagnostics record heartbeat ACK, reconnect request, invalid-session, resume-sent, and session-resumed audit signals without exposing tokens or Gateway resume URLs.
