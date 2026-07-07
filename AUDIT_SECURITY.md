# AUDIT_SECURITY

- Timestamp: 2026-07-07 18:30:57 EDT

## Security Posture

- Privileged Discord voice actions remain behind normal tool registry permission checks and audit.
- `discord_voice_*` tools declare Discord network permissions for `discord.com` and `gateway.discord.gg`.
- Voice dispatch now requires a matching account/guild scoped binding; missing, wrong-guild, or ambiguous multi-account lookups fail closed.
- Channel startup failure clears the failed voice scope and emits `discord.voice.start_failed`.
- Successful voice startup emits `discord.voice.binding_registered` with account, guild, and text channel details.
- Shutdown stops the voice manager and clears scoped bindings owned by the stopped manager/account.

## Security Risks Still Open

- Accepted Discord image attachments still need stricter size, MIME, filename, dimension, and URL-fetch policy checks.
- Discord diagnostics need a concise operator view of policy gates and recent denials.
- Library-level Discord voice sockets remain outside `PolicyHTTPClient`; this is currently mitigated at startup/tool boundaries rather than per-socket enforcement.
