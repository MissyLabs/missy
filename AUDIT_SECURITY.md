# Security Audit Notes

Last updated: 2026-07-07

## Session Notes

- Discord voice tools now declare network permissions for `discord.com` and `gateway.discord.gg`; normal `ToolRegistry.execute()` calls therefore pass through existing network policy checks before dispatch.
- The Discord voice binding is registered only after `DiscordVoiceManager.start()` succeeds and is cleared on startup failure and channel shutdown.
- `DiscordChannel.stop()` now stops the optional voice manager before disconnecting the raw Gateway client, reducing stale voice-client and stale-binding risk.
- New Discord voice lifecycle audit events:
  - `discord.voice.binding_registered`
  - `discord.voice.start_failed`
- Tool invocation audit still uses the existing `tool_execute` event path.
- Optional-dependency tests were hardened without changing runtime dependency behavior.

## Residual Risks

- `voice_binding.py` is process-wide. It is acceptable for the current single active Discord voice manager model, but it is not sufficient for concurrent multi-account or cross-guild voice routing. Next work should key bindings by account/guild and require Discord context to select one.
- `DiscordVoiceManager` uses `discord.py` directly for voice transport. This is not routed through `PolicyHTTPClient`; the practical policy gate is currently at tool execution and channel startup, not each library-level voice socket operation.
- Voice STT/TTS content should be treated as untrusted user input. Existing voice conversation paths still need deeper prompt-injection and transcript redaction tests.

## Follow-up Security Work

1. Replace the process-wide voice binding with a scoped binding registry.
2. Add diagnostics/audit display for voice binding state and recent voice lifecycle failures.
3. Add explicit policy profile tests for `discord_voice_*` visibility and denial behavior.
4. Continue Discord media hardening: attachment size, MIME, URL, filename, and image-dimension checks.
