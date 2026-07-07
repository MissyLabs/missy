# Build Status

Last updated: 2026-07-07

## Completed Work

- Continued the Discord integration overhaul without replacing the existing Gateway, REST, policy, channel, voice, or CLI subsystems.
- Added `missy discord diagnostics`, an operator-facing status surface for account configuration, token presence, application IDs, DM policy, guild routing, bot-loop filtering, require-mention counts, network policy readiness, Discord-mode tool visibility, runtime voice bindings, and recent Discord audit events.
- Extended `voice_binding.py` diagnostics snapshots to include manager readiness plus listen/speak capability flags for each account/guild scoped binding.
- Fixed Discord capability-mode tool visibility so `discord_voice_join`, `discord_voice_leave`, `discord_voice_say`, and `discord_voice_status` are exposed in Discord/messaging mode while desktop/X11/browser controls remain excluded.
- Added focused CLI and policy tests for the diagnostics command and Discord voice tool visibility.
- Updated Discord user and implementation docs for the diagnostics command and runtime voice binding status behavior.

## Current Architecture State

- Discord text traffic continues to use Missy's raw Gateway client plus `DiscordRestClient` over `PolicyHTTPClient`.
- Discord access control still enforces own-message filtering, bot-loop prevention, DM policy, guild policy, allowlists, require-mention behavior, credential detection, and attachment gating before messages enter the agent queue.
- Discord voice remains optional and lazy-started from recognized voice commands.
- Agent-callable Discord voice actions flow through runtime tool selection, `ToolRegistry` permission/audit checks, `discord_voice_*` built-in tools, account/guild scoped `voice_binding.py`, and the active `DiscordVoiceManager` coroutine dispatch.
- `missy discord diagnostics` now provides a static-plus-process-local readiness view. Runtime voice bindings are only visible to the process that owns the registry; separate CLI processes still show static config/policy state and recent audit history.

## Remaining Tasks

- Continue hardening Discord attachment/media flows with stronger accepted-image metadata validation: size limits, MIME validation, image-dimension checks, filename hygiene, URL fetch policy, and audit details.
- Continue gateway lifecycle improvements: reconnect backoff tuning, shutdown ordering tests, heartbeat diagnostics, resume state visibility, and slash command registration observability.
- Expand diagnostics with live Gateway heartbeat/resume state when a service process exposes it to the CLI.
- Expand operator docs for Discord voice prerequisites, system ffmpeg, optional STT/TTS availability, and safe troubleshooting commands.
- Keep OpenClaw-style diagnostics patterns generic enough for later primary focuses such as provider routing, scheduler execution, and tool delegation.

## Blockers

- No code blocker remains for this session.

## Next Actions

1. Harden accepted Discord image attachment metadata validation and audit details.
2. Add gateway lifecycle diagnostics for heartbeat health, reconnect/resume state, and slash command registration failures.
3. Add safe operator troubleshooting examples for Discord voice dependencies and policy-denied Discord hosts.
