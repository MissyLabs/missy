## Native Rust rewrite

24 native Rust crates and the `missy-rs` executable. Not Python behind a Rust
launcher. Existing Python code, tests, installer and live state remain untouched.
Draft intentionally: a broad implementation is not permission to replace the bot.

### Continuation, sessions and memory

- Payload-bound exact scoped admission, cross-process leases/fencing, SQLite WAL
  sessions, versioned checksummed checkpoints and durable cancellation.
- Tool intent/RUNNING/settlement and successful-result rehydration after crashes.
  Unknown outcomes halt generation rather than retry with fresh tool IDs.
- Persisted token/spend/step/time/continuation controls and immutable lineage.
- Restart-safe exact human approvals resume original calls without regeneration.
- Scope-bound memory, graph/citation retrieval and no global fallback. Personal
  memory currently remains session-scoped, not automatic cross-session recall.
- Compared against Odin's development implementation. Reproduced two stale-policy
  bugs before repairing runtime fingerprint revalidation; both probes now pass.
  Full evidence and limitations: `docs/rust/odin-comparison.md`.

### Native operator surfaces

- OpenAI/Anthropic/Ollama and explicitly approved chat-only CLI provider.
- Filesystem/argv/DNS-pinned HTTP/WS policy, owned child cleanup and actual
  child-only Landlock confinement. Pinned stdio plus Streamable HTTP/SSE MCP.
- Authenticated loopback API/browser console and durable Discord intake/outbox.
- Voice edge pairing/auth/modes, separate NoTools SafeChat, trusted speech commands.
- Nomad/Incus/ComfyUI/OBS/VTube adapters with explicit mutation approvals.
- X11/AT-SPI/isolated CDP capability layer, all desktop tools approval-gated.
- Real FFmpeg editing and image transforms; discovery/file vision, not recording.
- Explicit local neural embeddings, actually exercised with pinned CPU MiniLM.
- Signed scope-bound mesh/replay/receipts, LWW memory/quorum and budgeted delegation.
- Human/cron/IANA/DST durable scheduling and restart-idempotent literal workflows.
- Dedicated HMAC webhook, authenticated diagnostics/Prometheus, bounded event bus,
  policy-bound OTLP and signed export APIs. Not an automatic telemetry deployment.
- Scoped persona/proposal review, trust, encrypted vault and offline copy-only
  migration with explicit scope mapping, source verification and atomic destination.

### Local validation on Rust 1.99.1

- **350 passing native test entries**, 47 result groups, no failures. Includes
  subprocess fixture entrypoints, not 350 independent end-to-end scenarios.
- Full-workspace formatting and Clippy `-D warnings` pass.
- Release executable built. Chromium against release: **15 assertions pass**.
- Strict dependency audit passes, 354 dependencies and 1,409 loaded advisories.
  Removed unmaintained D-Bus dependencies by migrating to zbus 6, no suppressions.
- Actual executable/tool/SQLite/restart/dedup/isolation/approval/migration/workflow
  acceptance, actual child Landlock, actual FFmpeg, isolated Xvfb/private D-Bus,
  persistent SSE, signed mesh-to-runtime and real local neural embedding checks.
- CI is verified separately against the exact head; old green runs are not proof
  of the expanded tree. See `docs/rust/validation.md` and individual crate READMEs.

### Deliberate limits and rollout gates

No production replacement/restart/live-data migration, real language-provider
inference, Discord login, hardware recording or active-desktop input. Dedicated
authorized Missy credentials must be provisioned for live canaries, not borrowed
from another bot. No live federation or infrastructure mutation acceptance.

Remaining gaps include Discord voice, live camera/screencast, music composition,
some specialty integrations, cross-session personal recall, and source
auto-evolution. Codex completion remains blocked without verified no-tools mode.
Export libraries are not an instrumented external telemetry deployment. Spend
checks are consumed-usage controls, not prepaid financial guarantees. Landlock
TCP restrictions are not all-protocol network isolation. Missing old effect
receipts cannot be manufactured by migration. See `docs/rust/parity.md` before
considering a drop-in replacement.
