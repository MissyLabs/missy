# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-07 18:49:30 EDT

Expected connectivity posture:

- default-deny network where practical
- exact provider endpoints
- exact Discord REST endpoint: `discord.com`
- exact Discord Gateway endpoint: `gateway.discord.gg`
- no unreviewed broad outbound access

Current Discord connectivity work:

- `missy discord diagnostics` now checks whether `discord.com` and `gateway.discord.gg` are present in allowed domains, allowed hosts, or Discord-specific allowed hosts.
- `missy discord probe` remains the live REST token/API connectivity check.
- Gateway heartbeat/reconnect/resume live status still needs a service-visible diagnostics surface.
