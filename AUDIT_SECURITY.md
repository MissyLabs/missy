# AUDIT_SECURITY

- Timestamp: 2026-07-08
- Branch: overhaul/tools-20260708-020326

## Expected Security Posture

- Privileged tool creation and execution remain default-deny and policy-gated.
- Generated or proposed tool candidates must stay disabled until reviewed and approved.
- Benchmark results must never enable provider/tool access by themselves without an explicit approval path.
- Provider schema adapters must preserve tool schemas without expanding permissions.
- Discord message, image, voice, Gateway, and REST handling must continue to enforce access control, attachment gating, and policy-routed network access.
- Diagnostics and audit output must avoid exposing secrets, bot tokens, API keys, Gateway resume URLs, or sensitive request payloads.

## Tool Intelligence Notes

- Runtime request tracking records completed turns through `RequestTracker` on a best-effort basis and fails closed to normal runtime behavior if tracking is unavailable.
- OpenClaw A3 mutation fingerprinting detects repeated failing tool calls with identical arguments and injects a sticky `lastToolError` strategy prompt without reclassifying tool policy decisions.
- `missy tools benchmark run` builds suites from registered tool metadata and executes through the registry, preserving registry execution controls.
- Provider schema conversion now routes through `normalize_for_provider()` with existing inline fallbacks.

## Discord Notes From Master

- Discord text traffic still uses Missy's raw Gateway client plus `DiscordRestClient` over `PolicyHTTPClient`.
- Access control continues to enforce own-message filtering, bot-loop prevention, DM policy, guild policy, allowlists, require-mention behavior, credential detection, and attachment gating before messages enter the agent queue.
- Accepted image metadata is normalized into `discord_image_attachments`; downloads revalidate metadata and restrict REST download hosts to Discord CDN domains.
- Gateway lifecycle state is observable through redacted diagnostics and structured Discord lifecycle audit events.
- Discord voice remains optional and lazy-started from recognized voice commands with scoped runtime bindings for `discord_voice_*` tools.

## Follow-Up Security Work

- Add provider-specific enablement gates from benchmark scores without automatic activation.
- Add fallback routing only after provider/tool approval semantics are explicit.
- Add byte-level Discord image validation when an image dependency is available or tied to the existing vision extra.
- Keep diagnostics patterns reusable across Discord, scheduler, provider routing, plugin/tool policy, filesystem, shell, and network actions.
