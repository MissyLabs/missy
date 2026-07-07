# OpenClaw Gap Analysis

Last updated: 2026-07-07 19:48 EDT

## Current Focus: Discord Integration

Missy now has stronger Discord operator and safety paths: diagnostics summarize configuration and runtime readiness, voice tools are policy-visible in Discord mode, scoped voice bindings expose readiness/capability snapshots, image attachment metadata is validated before agent routing or download, and Gateway lifecycle signals are visible through redacted snapshots plus audit-backed CLI summaries.

## Remaining Discord Gaps

- Live Gateway snapshots are process-local; a separate service status channel is needed for exact out-of-process CLI visibility.
- Byte-level image verification remains partial: metadata is bounded, but file signature and true decoded dimensions are not yet verified without an image dependency.
- Runtime voice binding diagnostics are process-local; a separate service status channel is needed for out-of-process CLI visibility.

## OpenClaw-Style Operator Ergonomics Gaps

- Audit output should make allow/deny causes easy to scan without requiring raw event inspection across subsystems.
- Recovery guidance should be attached to common failures: missing token, missing intents, policy-denied Discord host, voice dependency missing, ffmpeg missing, STT/TTS unavailable, wrong-guild voice scope, ambiguous multi-account scope, slash registration failure, invalid Gateway session, and stale heartbeat ACKs.
- Diagnostics patterns should become reusable for scheduler, provider routing, plugin/tool policy, filesystem, shell, and network actions.

## Cross-Subsystem Gaps To Preserve For Future Focus

- Scoped binding/registry patterns should remain generic enough for future provider routing, tool delegation, gateway, scheduler, and media subsystems.
- Policy pipelines should keep separating visibility, approval, and execution enforcement.
- Local-first secure runtime still needs consistent diagnostics across Discord, scheduler, plugins, providers, filesystem, shell, and network actions.

## Recommended Next Slice

Add a service status surface for live out-of-process Discord Gateway snapshots.
