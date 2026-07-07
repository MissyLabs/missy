# Test Edge Cases

Last updated: 2026-07-07

## Covered This Session

- Discord voice tools return clear errors when no active voice binding exists.
- Discord voice tools return clear errors when the voice manager exists but is not ready.
- `discord_voice_join` validates `guild_id`, supports channel name, channel ID, and user-follow joins, and reports available listen/speak capabilities.
- `discord_voice_leave` handles connected and not-connected states.
- `discord_voice_say` validates required guild/text inputs.
- `discord_voice_status` reports connection state, current channel, listening state, capabilities, and available channels.
- Built-in tool registration includes all four `discord_voice_*` tools.
- Voice tools explicitly declare Discord network permissions.
- Discord channel clears stale voice binding when voice startup fails.
- Discord channel stops the voice manager and clears binding on shutdown.
- Optional STT tests now cover missing `faster_whisper`, `torch`, and `ctranslate2` branches even when those packages are installed in the host environment.

## Still Needed

- Multi-account and multi-guild voice binding collision tests.
- Policy visibility tests for `discord_voice_*` tools under `full`, `discord`, `safe-chat`, and `no-tools` capability modes.
- Discord diagnostics tests for binding readiness and lifecycle audit event display.
- Attachment/media edge tests for Discord image handling: file size, MIME mismatch, oversized dimensions, remote URL policy denial, and malicious filename handling.
