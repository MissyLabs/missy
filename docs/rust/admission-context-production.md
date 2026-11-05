# Immutable production admission context

Every native runtime constructed by the CLI, including the unified gateway,
scheduled turns, Discord, voice and reasoning-only children, uses the same
`ScopedPromptResolver` adapter over the operator governance registry. Resolution
is exact tenant/user/channel/session, with no implicit global persona fallback.

New turns select the current approved system/persona revision once. The selected
text, digest, exact Scope and configuration binding are committed in the same
Store transaction as admission. A queued turn therefore cannot pick up a newer
persona merely because execution capacity became available later. Resume uses
the original selected prompt. New revisions affect subsequent new requests.

Optional relevant memory uses the same admission pinning. It is a separate
bounded untrusted user-data message, never a system/persona patch. Source updates
or new maintenance summaries do not silently alter an existing turn. Existing
request IDs return their original durable outcome and never readmit work.

`crates/missy-runtime/src/tests/prompts.rs` exercises shared-runtime scope
separation, deduplication, queued admission, restart and continuation with changed
revisions. `crates/missy-cli/tests/prompt_production_e2e.rs` starts the real native
gateway against a loopback fake provider, approves a persona after readiness,
pauses an exact tool call for approval, approves a new persona, verifies resume
still uses the old one, denies cross-session propagation and verifies the next
new turn sees the new revision. No real account, provider or process cutover is
part of these fixtures.
