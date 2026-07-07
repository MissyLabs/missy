# Build Status

Last updated: 2026-07-07

## Completed Work

- Continued the Discord integration overhaul without replacing existing channel, gateway, REST, policy, voice, or tool-registry modules.
- Replaced the process-wide Discord voice tool binding with a thread-safe scoped registry in `missy/channels/discord/voice_binding.py`.
- Discord voice bindings are now keyed by account and guild. Ambiguous lookup, such as multiple bot accounts bound to the same guild without an `account_id`, fails closed.
- Updated `discord_voice_join`, `discord_voice_leave`, `discord_voice_say`, and `discord_voice_status` to resolve bindings from `guild_id` plus optional `account_id`.
- Updated `DiscordChannel` lifecycle so voice startup registers account/guild scoped bindings, existing managers publish new guild scopes as voice commands arrive, startup failure clears the failed scope, and shutdown clears bindings owned by the stopped manager/account.
- Added `account_id` to voice binding lifecycle audit details.
- Added tests for wrong-guild denial, ambiguous multi-account denial, account-scoped selection, new-guild registration for an existing manager, and scoped cleanup.
- Updated Discord user and implementation docs for scoped voice bindings and fail-closed lookup behavior.

## Current Architecture State

- Discord text traffic continues to use Missy's raw Gateway client plus `DiscordRestClient` over `PolicyHTTPClient`.
- Discord access control still enforces own-message filtering, bot-loop prevention, DM policy, guild policy, allowlists, require-mention behavior, credential detection, and attachment gating before messages enter the agent queue.
- Discord voice remains optional and lazy-started from recognized voice commands.
- Agent-callable Discord voice actions now flow through:
  1. runtime tool selection,
  2. `ToolRegistry` permission checks and audit,
  3. `discord_voice_*` built-in tools,
  4. account/guild scoped `voice_binding.py`,
  5. active `DiscordVoiceManager` coroutine dispatch.
- Voice binding lookup is no longer a single process-wide global. It is still process-local and should next gain richer diagnostics and operator-facing status output.

## Remaining Tasks

- Extend structured Discord diagnostics to report voice binding scopes, manager readiness, listen/speak availability, configured policy gates, and recent lifecycle failures.
- Add explicit policy/profile tests proving `discord_voice_*` tools are visible only in intended Discord/tool capability modes.
- Continue hardening Discord attachment/media flows with size limits, MIME validation, URL fetch policy, image-dimension checks, filename hygiene, and audit details for accepted image attachments.
- Continue gateway lifecycle improvements: reconnect backoff tuning, shutdown ordering tests, heartbeat diagnostics, resume state visibility, and slash command registration observability.
- Expand operator docs for Discord voice prerequisites, system ffmpeg, optional STT/TTS availability, and safe troubleshooting commands.

## Blockers

- No code blocker remains for this session.

## Next Actions

1. Add Discord diagnostics output for scoped voice binding readiness and lifecycle audit events.
2. Add policy visibility tests for `discord_voice_*` under Discord-focused runtime profiles.
3. Harden accepted Discord image attachment metadata validation and audit details.
