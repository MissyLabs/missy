# TEST_EDGE_CASES

- Timestamp: 2026-07-07 18:49:30 EDT

Current edge-case focus:

- Discord diagnostics when no accounts are configured
- Discord diagnostics with token presence, application ID, DM policy, require-mention routing, and bot-loop filtering
- Network policy readiness for `discord.com` and `gateway.discord.gg`
- Discord capability-mode visibility for `discord_voice_*`
- Runtime account/guild voice binding diagnostics with ready/listen/speak flags
- Recent Discord audit event rendering without leaking secrets
- Discord gateway reconnect and shutdown
- Discord REST retry, rate-limit, and failure handling
- DM pairing and unknown sender denial
- guild/channel allowlist denial
- require-mention filtering
- bot-authored message and loop prevention
- attachment/media safety
- slash command registration diagnostics
- policy-gated tool execution from Discord
- future overhaul compatibility for provider/tool delegation
