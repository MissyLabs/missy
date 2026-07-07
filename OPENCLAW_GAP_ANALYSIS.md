# OpenClaw Gap Analysis

Last updated: 2026-07-07

## Current Focus: Discord Integration

Missy now has a stronger Discord voice control path: voice tools are registered, policy-declared, lifecycle-bound to the running Discord channel, audited on startup success/failure, and resolved through account/guild scoped bindings instead of a single process-wide binding.

## Remaining Discord Gaps

- Discord voice diagnostics do not yet expose scoped binding state, manager readiness, listen/speak capability, or recent voice lifecycle failures.
- Tool visibility policy does not yet have explicit tests for `discord_voice_*` under Discord-focused capability modes.
- Discord media safety remains partial: accepted image attachments need stronger size, MIME, image-dimension, filename, and URL-fetch policy coverage.
- Gateway lifecycle diagnostics still need clearer operator-facing status for reconnects, heartbeat health, resume state, and slash command registration failures.

## OpenClaw-Style Operator Ergonomics Gaps

- Operators need one command that summarizes Discord REST, Gateway, slash command, text routing, scoped voice binding, and policy readiness.
- Audit output should make allow/deny causes easy to scan without requiring raw event inspection.
- Recovery guidance should be attached to common failures: missing token, missing intents, policy-denied Discord host, voice dependency missing, ffmpeg missing, STT/TTS unavailable, wrong-guild voice scope, and ambiguous multi-account scope.

## Cross-Subsystem Gaps To Preserve For Future Focus

- Scoped binding/registry patterns should remain generic enough for future provider routing, tool delegation, gateway, scheduler, and media subsystems.
- Policy pipelines should keep separating visibility, approval, and execution enforcement.
- Local-first secure runtime still needs consistent diagnostics across Discord, scheduler, plugins, providers, filesystem, shell, and network actions.

## Recommended Next Slice

Implement a Discord diagnostics command or status surface that reports REST, Gateway, slash command registration, text routing, scoped voice bindings, and policy readiness with actionable recovery hints.
