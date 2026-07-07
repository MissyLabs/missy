# OpenClaw Gap Analysis

Last updated: 2026-07-07

## Current Focus: Discord Integration

Missy now has a stronger Discord operator path: voice tools are policy-visible in Discord mode, scoped voice bindings expose readiness/capability snapshots, and `missy discord diagnostics` summarizes configuration, routing, network policy, tool visibility, runtime voice bindings, and recent Discord audit events.

## Remaining Discord Gaps

- Discord media safety remains partial: accepted image attachments need stronger size, MIME, image-dimension, filename, and URL-fetch policy coverage.
- Gateway lifecycle diagnostics still need clearer operator-facing status for reconnects, heartbeat health, resume state, and slash command registration failures.
- Runtime voice binding diagnostics are process-local; a separate service status channel is needed for out-of-process CLI visibility.

## OpenClaw-Style Operator Ergonomics Gaps

- Audit output should make allow/deny causes easy to scan without requiring raw event inspection across subsystems.
- Recovery guidance should be attached to common failures: missing token, missing intents, policy-denied Discord host, voice dependency missing, ffmpeg missing, STT/TTS unavailable, wrong-guild voice scope, and ambiguous multi-account scope.
- Diagnostics patterns should become reusable for scheduler, provider routing, plugin/tool policy, filesystem, shell, and network actions.

## Cross-Subsystem Gaps To Preserve For Future Focus

- Scoped binding/registry patterns should remain generic enough for future provider routing, tool delegation, gateway, scheduler, and media subsystems.
- Policy pipelines should keep separating visibility, approval, and execution enforcement.
- Local-first secure runtime still needs consistent diagnostics across Discord, scheduler, plugins, providers, filesystem, shell, and network actions.

## Recommended Next Slice

Harden accepted Discord image attachment handling with explicit metadata validation, URL fetch policy checks, and accepted/denied attachment audit detail.
