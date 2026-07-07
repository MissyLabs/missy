# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-07 18:30:57 EDT

Expected connectivity posture:
- default-deny network where practical
- exact provider endpoints
- exact Discord gateway and REST endpoints
- no unreviewed broad outbound access

Current Discord connectivity state:
- Discord text REST still routes through `DiscordRestClient` and `PolicyHTTPClient`.
- Discord Gateway remains a raw WebSocket client for Discord Gateway lifecycle traffic.
- Discord voice is optional and uses `discord.py` for voice transport after the channel lazy-starts `DiscordVoiceManager`.
- Agent-callable voice actions require a live account/guild scoped binding and declare Discord network permissions at the tool layer.
- Voice binding lookup denies missing, wrong-guild, and ambiguous multi-account scopes by returning no binding.

Known connectivity gaps:
- Library-level Discord voice sockets are not individually mediated by `PolicyHTTPClient`; policy enforcement is currently at tool execution/startup boundaries.
- Discord diagnostics should report whether REST, Gateway, slash commands, text routing, and voice transports are separately reachable and policy-allowed.
- Accepted Discord image attachments need stronger URL-fetch policy and metadata validation before analysis.
