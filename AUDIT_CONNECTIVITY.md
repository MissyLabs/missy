# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-07 19:08:32 EDT

Expected connectivity posture:

- default-deny network where practical
- exact provider endpoints
- exact Discord REST and Gateway endpoints
- exact Discord CDN hosts for attachment downloads
- no unreviewed broad outbound access

Session update:

- Discord image attachment metadata must reference `https://cdn.discordapp.com` or `https://media.discordapp.net` before routing or download.
- `DiscordRestClient.download_attachment()` still enforces Discord CDN hosts at download time through the policy-backed HTTP client.
