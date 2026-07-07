# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-07 19:48 EDT

Expected connectivity posture:
- default-deny network where practical
- exact provider endpoints
- exact Discord gateway and REST endpoints
- no unreviewed broad outbound access

Session connectivity notes:
- Discord REST remains routed through `PolicyHTTPClient`.
- Discord Gateway WSS lifecycle now records heartbeat ACK, reconnect request, invalid-session, resume-sent, and session-resumed audit signals.
- `missy discord diagnostics` summarizes recent lifecycle signals from the audit log without exposing bot tokens or Gateway resume URLs.
