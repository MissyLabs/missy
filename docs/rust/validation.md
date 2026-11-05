# Native Rust validation evidence

Date: 2026-11-05. Local Linux x86_64 checkout, starting from Python Missy
`d152c54e04f5daf1381c84527643a958fcb7d252`. Rust 1.96.1 / Cargo 1.96.1.
This report concerns the new Rust workspace, not the production Python bot.

## Results observed locally

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --all-targets --locked` | 127 passing native test entries, 0 failed |
| `cargo build --release -p missy-cli --locked` | Pass, native Linux executable |
| `scripts/test-rust-console.py --binary target/release/missy-rs` | 15 Chromium/HTTP acceptance assertions passed |
| `cargo audit --deny warnings` | Pass against 1,409 loaded RustSec advisories and 293 locked dependencies |
| Legacy Python source/tests/installer/config schema diff | Unchanged |

The native total includes two subprocess-fixture entrypoints which intentionally
return without doing work in the ordinary runner and are driven separately by
their parent crash/lock tests. It is not 127 independent end-to-end scenarios.
The optional browser harness is test tooling, not a runtime Python dependency.

### Native test breakdown

| Area | Passing test entries |
|---|---:|
| Automation DAG/skills + scheduler | 12 |
| CLI config + executable end-to-end | 4 |
| Shared protocol/config | 4 |
| HTTP gateway + Discord local wire protocol | 7 |
| Filesystem/network/transport policy | 19 |
| OpenAI/Anthropic/Ollama HTTP fixtures | 18 |
| Agent runtime/context/continuation | 15 |
| Vault/signature/concurrency | 8 |
| SQLite lock/persistence/crash/recovery/cancellation | 27 |
| Builtin tools/MCP/owned child cleanup | 13 |
| **Total** | **127** |

## What was exercised, not merely compiled

- Actual native executable -> real local OpenAI-format HTTP endpoint -> multi-tool
  file write and private memory -> SQLite checkpoint/history -> process exit ->
  new executable process -> deduplicated request -> memory retrieval -> second
  user isolated from the first. Duplicate payload mismatch performs no inference.
- Actual headless Chromium -> authenticated native loopback API -> provider
  fixture -> asynchronously admitted turn -> read-only polling to completion ->
  persisted history. Literal model HTML remains inert. A foreign session cannot
  fetch the turn by ID, and browser reload forgets the bearer token.
- Real local policy-pinned WebSocket server exercises Discord HELLO, Identify,
  READY, message dispatch, heartbeat ACK, reconnect and Resume. Production TLS
  and policy restrictions remain enforced; no real Discord credential was used.
- Abrupt child exit leaves WAL/checkpoint/tool intent durable. Recovery refuses
  uncertain operations and blocks provider generation, including fresh tool IDs.
- Successful journal result missing from an older checkpoint is rehydrated with
  zero additional tool dispatch. Missing journal, changed arguments or PREPARED
  uncertainty blocks continuation instead of guessing external reality.
- Cross-process admission, stale leases, live-worker preservation, payload-bound
  request IDs, scoped memory/history/direct IDs, safe FTS and migration failures.
- Cancellation cannot be overwritten by late success. A cancelled turn cannot
  prepare/start a new effect, but an already-running effect can record its
  truthful receipt before terminal cancellation.
- Cancellation/drop/timeout/output-overflow process-group cleanup, separated
  stderr EOF, exact literal argv, stripped child environment, MCP malformed
  JSON-RPC, changed manifests, strict arguments and bounded output.
- Distinct vault handles serialize with flock; shared-handle concurrent writes
  serialize with an internal mutex. AEAD corruption and missing keys fail closed.

## Defects found and fixed during acceptance

- Replaced the initial YAML backend after strict audit identified unmaintained
  `unsafe-libyaml`; maintained `yaml_serde` is used through a module alias. The
  advisory was removed, not suppressed.
- Fixed resumption ordering so checkpoint/history reload occurs after acquiring
  session ownership, not against potentially stale pre-claim state.
- Fixed never-started admitted requests stranded by capacity/session contention.
- Added journal-to-transcript repair for the successful-settlement crash window.
- Persisted unknown billing for failed/interrupted provider attempts and refused
  tool dispatch under an enforced finite budget with unknown prior charges.
- Preserved effect receipts during compaction and refused to erase an oversized
  latest batch merely to fit context.
- Added monotonic live deadlines, persistent continuation budgets, heartbeat
  cleanup ownership, complete serialized tool-result bounds and cancellation
  ordering checks.
- Browser acceptance exposed the original 202-admission UI leaving an empty
  result forever. Added bounded status polling rather than replaying the request.
- Added same-instance vault mutation serialization and secret-buffer cleanup.

## Limits and unperformed acceptance

- No production replacement, service restart, live state migration, active
  desktop interaction or hardware capture.
- No real-provider inference, real Discord bot login, GPU benchmark or live
  operator credential used. Mock protocol coverage is not live acceptance.
- No claim that every historical Python integration is ported. See
  [parity.md](parity.md) before deciding this can replace a particular deployment.
- Spend checks operate on consumed/known usage, not an externally guaranteed
  prepaid cap. Same-user commands/MCP are not kernel sandboxes.
- SQLite hash-chain audit has no external anchor. Export signature verification
  only authenticates a bundle against an independently pinned key.
- GitHub workflow results must be checked separately from these local results.
  The workflow includes native tests/lint/build/audit and Chromium acceptance.
