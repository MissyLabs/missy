# Build Status

Last updated: 2026-07-07

## Completed Work

- Continued the Discord integration overhaul without replacing existing channel, gateway, REST, policy, or voice modules.
- Added a process-local Discord voice binding layer so built-in tools can safely dispatch onto the active Discord voice manager's asyncio loop.
- Added built-in Discord voice tools:
  - `discord_voice_join`
  - `discord_voice_leave`
  - `discord_voice_say`
  - `discord_voice_status`
- Registered the Discord voice tools in the built-in tool registry so the agent runtime can actually expose them through normal tool selection.
- Declared Discord network permissions on the voice tools for `discord.com` and `gateway.discord.gg`, keeping tool execution behind the registry policy checks.
- Tightened Discord channel lifecycle:
  - voice binding is registered only after `DiscordVoiceManager.start()` succeeds;
  - stale voice binding is cleared on startup failure;
  - `DiscordChannel.stop()` stops the voice manager and clears the binding before disconnecting the raw Gateway client.
- Added audit events for voice tool lifecycle:
  - `discord.voice.binding_registered`
  - `discord.voice.start_failed`
- Hardened optional-dependency tests so missing `faster_whisper`, `torch`, and `ctranslate2` paths are simulated even when those packages are installed in the host environment.
- Installed OpenCV test dependency in the runtime environment after the full suite exposed missing/incompatible `cv2` state:
  - apt package: `python3-opencv`
  - user-site wheel: `opencv-python-headless 5.0.0.93`, required because user-site NumPy is `2.4.3`.

## Current Architecture State

- Discord text traffic continues to use Missy's raw Gateway client plus `DiscordRestClient` over `PolicyHTTPClient`.
- Discord access control still enforces own-message filtering, bot-loop prevention, DM policy, guild policy, allowlists, require-mention behavior, credential detection, and attachment gating before messages enter the agent queue.
- Discord voice remains an optional surface backed by `DiscordVoiceManager` and lazy-started from recognized voice commands.
- Agent-callable Discord voice actions now flow through:
  1. runtime tool selection,
  2. `ToolRegistry` permission checks and audit,
  3. `discord_voice_*` built-in tools,
  4. `voice_binding.py`,
  5. active `DiscordVoiceManager` coroutine dispatch.
- The binding is process-local and lifecycle-scoped. Multi-account voice routing still needs a stronger account/guild scoped binding model.

## Remaining Tasks

- Replace the process-wide single voice binding with an account-aware or guild-aware binding registry before relying on concurrent multi-account Discord voice.
- Add explicit policy/profile defaults for when `discord_voice_*` tools should be visible in Discord capability mode.
- Extend structured Discord diagnostics to report voice binding state, manager readiness, listen/speak availability, and configured policy gates.
- Continue hardening Discord attachment/media flows with size limits, MIME validation, URL fetch policy, and audit details for accepted image attachments.
- Continue gateway lifecycle improvements: reconnect backoff tuning, shutdown ordering tests, heartbeat diagnostics, and slash command registration observability.
- Expand operator docs for Discord voice prerequisites, system ffmpeg, optional STT/TTS availability, and safe troubleshooting commands.

## Blockers

- No code blocker remains for this session.
- The first full-suite run failed because the environment lacked a compatible OpenCV import path. This was corrected in the runtime environment and the full suite now passes.

## Next Actions

1. Add an account/guild scoped `DiscordVoiceBindingRegistry` and migrate tools to select a binding from Discord context.
2. Add Discord diagnostics output for voice binding readiness and lifecycle audit events.
3. Add policy visibility tests proving `discord_voice_*` tools are available only in intended Discord/tool profiles.
