# Native Rust validation evidence

Date: 2026-11-05. Local Linux x86_64 checkout, starting from Python Missy
`d152c54e04f5daf1381c84527643a958fcb7d252`. Initial core validation used Rust
1.96.1; the final expanded workspace was validated with Rust/Cargo 1.99.1.
This report concerns the new Rust workspace, not the production Python bot.

## Final expanded acceptance, 17:45 UTC

The finished source tree contains **35 native crates**. All implementation
owners were frozen before this run. The exact implementation commit and hosted
workflow result are recorded in the final PR handoff; earlier results below
are retained as dated evidence, not substituted for this expanded-tree run.

| Final check | Actual local result |
|---|---|
| `cargo +1.99.1 fmt --all -- --check` | Pass |
| `cargo +1.99.1 clippy --workspace --all-targets --locked -- -D warnings` | Pass, all 35 crates |
| `cargo +1.99.1 test --workspace --all-targets --locked` | **521 passed, 0 failed, 3 explicitly ignored, 69 result groups** |
| Native opt-in: `cargo test -p missy-dave -p missy-discord-voice --locked -- --include-ignored` | **22 passed**, including all 3 ignored tests; 19 overlap the ordinary suite |
| `cargo +1.99.1 build --release -p missy-cli --locked` | Pass, native x86_64 Linux ELF |
| `python3 scripts/test-rust-console.py --binary target/release/missy-rs` | **15/15 Chromium assertions passed** against final release binary |
| `cargo +1.99.1 audit --deny warnings` | Pass, **388 locked dependencies, 1409 advisories**, no suppressions |
| Legacy Python source/tests/installer | No diff from baseline `d152c54e04f5daf1381c84527643a958fcb7d252` |

The three ordinary ignored cases explicitly require an operator-supplied pinned
official native libdave library and upstream external-sender helper. They were
actually enabled and passed separately, not silently skipped. There are **524
distinct passing test entries** across the ordinary plus opt-in runs, not 543
independent tests. Some entries are subprocess fixture helpers, not independent
end-to-end scenarios. Upstream native suites previously passed 62 C++ + 9 C API
tests; Cargo audit does not audit those C/C++ dependencies.

### Executable and reproducible local evidence

* Binary: `target/release/missy-rs`, ELF x86_64 GNU/Linux PIE, approximately 31 MiB.
* SHA-256: `7168957c5f6cb8ba6875af162ca3415933e7aaa6624a206870d1e6486094751e`.
* Dynamic dependencies observed by `ldd`: `libopus.so.0`, libc, libm, libgcc_s and
  the ELF loader. This is not a static or universally portable Linux binary.
* Optional official DAVE library: explicit trusted source/version/hash with native
  provenance and linked dependencies documented in `crates/missy-dave/README.md`.
* Logs retained locally under `/tmp/missy-rust-final-{workspace-tests,clippy,audit,
  native-release,dave-voice,browser-acceptance}.log`; temporary logs are not durable
  CI artifacts. GitHub workflow uploads the executable and native library bundle.

### Newly exercised quality and regression repairs

* Actual executable personal recall across processes/sessions, foreign-user and
  tenant denial; principal-local FTS ranking repairs a reproduced cross-principal
  ranking side channel. No global fallback or auto-injected personal instructions.
* Executable history search/UTF-8 receipt expansion, exact four-component scope,
  checkpoint/raw source coverage, persistent immutable receipt reopening, old
  source changes and forged/corrupt/foreign receipt denial. Acceptance exposed
  current-turn query checkpoints polluting recall; current turn/checkpoint sources
  are now excluded and full CLI acceptance passes.
* Exact approval then checksum-bound file deletion; duplicate delivery after
  recreating the file does not delete it. Integration startup rejects omitted
  newly registered sensitive tools rather than silently bypassing approval.
* Actual vision provider image bytes, metadata stripping, approval before
  disclosure and durable independently priced reservations across restart.
* Real native voice loopback WS/UDP/Opus/MLS/media roundtrip using official DAVE;
  mock STT/agent/TTS, exact consent and real NoTools binding selection. Malformed,
  replayed, wrong-sender, plaintext/downgraded media cause zero STT; downgrade and
  cancellation tear down owned sessions. No live Discord claim.
* Model native-plugin proposal -> exact-hash operator review/enable -> new-runtime
  execution under actual native sandbox, with environment/files/network/argv/hash
  drift/timeout/drop cleanup denial tests. Proposal cannot activate itself.
* Explicit provider probe invokes no tools and durable dedup does not replay
  inference; uncertain billing stops subsequent probe dispatch.
* Reviewed source apply/rollback/recovery, agent DAG reservations, telemetry flush,
  capture on synthetic/isolated devices, audio/video composition, specialty and
  expanded infrastructure/Discord adapters have native fixture acceptance.

No production bot was replaced/restarted; no live credentials/state migration,
active desktop, microphone, speaker or GPU workload was used. These are the
remaining canary/quality gates, not claims of complete legacy parity.

### Hosted runner correction

The first expanded-head hosted run passed format/lint/audit and the complete
native DAVE job (including upstream suites and Rust voice roundtrip), but failed
the ACP fixture because the runner has no `/usr/bin/node`. The fixture is a
test-only JavaScript ACP server; the Rust runtime does not require Node unless
that optional CLI adapter is explicitly configured. Added distribution `nodejs`
to isolated CI prerequisites rather than skipping the six ACP tests or weakening
executable validation. Final-head hosted verification is checked after that fix.

## Results observed locally

The following section is the **earlier 24-crate/350-entry snapshot**, retained
for history. The expanded acceptance above supersedes its aggregate totals.

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --all-targets --locked` | Final expanded workspace: 350 passing native test entries across 47 result groups, 0 failed |
| `cargo build --release -p missy-cli --locked` | Pass, native Linux executable |
| `scripts/test-rust-console.py --binary target/release/missy-rs` | 15 Chromium/HTTP acceptance assertions passed |
| `cargo audit --deny warnings` | Pass against 1,409 loaded RustSec advisories and 354 locked dependencies; no advisory suppressions |
| Legacy Python source/tests/installer/config schema diff | Unchanged |

The native total includes subprocess-fixture entrypoints which intentionally
return without doing work in the ordinary runner and are driven separately by
their parent crash/lock tests. It is not 350 independent end-to-end scenarios.
The optional browser harness is test tooling, not a runtime Python dependency.

### Initial core test breakdown (248 entries, before the final expansion)

| Area | Passing test entries |
|---|---:|
| Automation DAG/skills + scheduler | 12 |
| CLI config, voice cancellation + executable end-to-end | 8 |
| Shared protocol/config | 4 |
| HTTP gateway + durable Discord transport | 21 |
| Exact approvals, trust and prompt revisions | 11 |
| Native infrastructure/media HTTP/WS integrations | 19 |
| Offline legacy migration | 11 |
| Chat-only CLI provider | 12 |
| Scoped graph/hybrid citation retrieval | 17 |
| Filesystem/network/transport policy | 19 |
| OpenAI/Anthropic/Ollama HTTP fixtures | 18 |
| Agent runtime/context/continuation/approval | 26 |
| Native Landlock and container argument construction | 6 |
| Vault/signature/concurrency | 8 |
| SQLite lock/persistence/crash/recovery/cancellation | 27 |
| Builtin tools/MCP/owned child cleanup/fingerprints | 14 |
| Voice edge protocol/registry/native speech fixtures | 15 |
| **Total** | **248** |

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
- Actual native executable pauses an exact file-write batch for human approval,
  exits, accepts an operator decision in a separate process, resumes the original
  call without another model generation, writes once and consumes approval once.
- Actual migration command plans then atomically publishes a separate usable
  tree from a fixture database. Source bytes stay identical, old system text is
  inert scoped memory, foreign principals cannot retrieve it, and old checkpoints
  are never runnable.
- Citation ingestion/query/delete through actual CLI preserves exact UTF-8 byte
  spans and refuses cross-user recall. Local graph traversal and document updates
  are transactionally scoped and bounded.
- Dedicated native subprocesses were actually restricted with Landlock ABI8.
  Allowed/forbidden filesystem paths, truncate/symlink denial, TCP bind/connect,
  descriptor/environment isolation and cancellation/abort reaping were exercised.
  The parent process remained unrestricted. Container tests inspect argv only.
- Voice tests use real loopback WebSockets and fixture speech executables. Mute,
  revoke, disconnect and dropped callback cancel work, including detached runtime
  execution. No real microphone, speaker or model was exercised.
- Discord SQLite intake/outbox tests cover sequence backpressure, restart windows,
  uncertain sends reconciled by nonce without blind repost, scoped interactions,
  persistent throttling and bounded retries. A live read-only preflight was
  attempted in an isolated empty environment and correctly stopped at missing
  `MISSY_DISCORD_TOKEN`; no network login or send occurred.

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
- The first hosted Rust build passed native lint/tests/release but its browser
  dependency installation failed because `ubuntu-latest` had migrated to 26.04,
  unsupported by the tested Playwright version. Pinned the Rust workflow to
  supported Ubuntu 24.04 rather than weakening browser acceptance.
- Added same-instance vault mutation serialization and secret-buffer cleanup.
- Fixed dropped voice callback allowing a detached runtime to continue after
  transport authority was revoked; an armed drop guard now cancels the turn.
- Fixed MCP environment-value-only changes escaping continuation fingerprints.
  Full environment content is hashed without exposing its values.
- Fixed migrated Anthropic endpoints missing `/v1`, and state-only imports
  missing a usable secure default-deny native configuration.

## Final expansion and focused correction evidence

The workspace now contains 24 native crates. The final locked all-target suite
passes 350 entries, including the retained 248-entry core and new regression,
native extension and executable acceptance entries. Formatting and full-workspace
Clippy with warnings denied pass on the same final dependency set.

- **Actual stale-policy bug reproduced and fixed:** changing a shared tool
  executor's policy while paused or during preflight previously allowed dispatch
  under a cached binding. Two regression probes failed before the narrow repair
  and passed afterward. Runtime rechecks existing configuration identity at
  generation, resume and dispatch boundaries. The Odin source comparison and
  residual atomicity limits are recorded in `odin-comparison.md`.
- **Scheduler defects repaired:** fractional one-shot and interval timestamps are
  rounded upward rather than fired early or repeated at the same second. Schema
  migration rechecks the version under a cross-process IMMEDIATE transaction.
  Actual worker tests cover one-shot exhaustion, daily/cron DST rescheduling,
  migration of v1 jobs and concurrent independent-process migration.
- **Executable workflow acceptance:** a real new process creates a human-language
  one-shot, another executes it once, and later processes show it exhausted.
  A literal two-node workflow completes through provider HTTP, then redelivery
  returns the original turns without further inference. This exposed a Clap
  argument-ID collision between a positional extension `config` and the global
  `--config`; positional fields were renamed, not worked around in the test.
  A final default-deny smoke check also exposed calendar preview unnecessarily
  constructing a provider/tool runtime. Pure preview now returns before any
  database/provider/tool initialization, with an executable no-provider/no-state
  regression test.
- **HTTP MCP:** real loopback Streamable HTTP and never-EOF SSE, persistent legacy
  SSE correlation, same-origin credential routing, initialized-session manifest
  checks, changed pins and uncertain outcomes. No live MCP server was used.
- **Desktop:** native allowlisted X11 enumeration/screenshots on a fresh
  authenticated isolated Xvfb, a private D-Bus AT-SPI mock and fixed-script CDP
  protocol fixtures. No active desktop, production browser or actual input.
- **Media:** seven actual tiny FFmpeg operations with FFprobe metadata and full
  decode checks, native image edits, policy/symlink denial, playlist refusal,
  child Landlock and process timeout/cancellation. No hardware or playback.
- **Neural retrieval:** a separate CPU-only llama.cpp process loaded a public
  SHA256-verified 45.9 MB MiniLM model. Three token-disjoint semantic/lexical-decoy
  cases and persistent retrieval passed. The exact-quotation weakness is retained
  in `crates/missy-embedding/ACCEPTANCE.md`, not hidden. Dedicated process reaped.
  HTTP-config model fingerprints remain operator assertions, not remote attestation.
- **Mesh:** signed scope/recipient grants, durable replay/receipts, revocation,
  memory/quorum primitives and signed HTTP intake to an ordinary mock-provider
  Runtime. Stable delegated task IDs deduplicate; an approval-required effect
  does not execute. No production peer federation was exercised.
- **Operations:** real Runtime admission through dedicated HMAC webhook auth,
  replay persistence, unknown-crash refusal, body/scope spoof denial, separate
  diagnostics authorization, bounded event bus, Prometheus and policy-bound OTLP
  payload tests. Usage is explicitly a recent sample, not fabricated lifetime totals.
- **Dependency audit defect repaired:** strict audit rejected three unmaintained
  transitive D-Bus crates. Migrated native AT-SPI code to maintained zbus 6.2.1;
  reran all desktop tests and the entire workspace. No advisories were ignored.

## Limits and unperformed acceptance

- No production replacement, service restart, live state migration, active
  desktop interaction or hardware capture.
- No real language-provider inference, real Discord bot login, GPU benchmark or live
  operator credential used. Mock protocol coverage is not live acceptance.
  The separately tested local embedding model is the explicit exception above.
- No claim that every historical Python integration is ported. See
  [parity.md](parity.md) before deciding this can replace a particular deployment.
- Spend checks operate on consumed/known usage, not an externally guaranteed
  prepaid cap. Exact argv policy alone is not a sandbox; optional Landlock is
  child-only and its network control covers TCP, not every communication family.
- SQLite hash-chain audit has no external anchor. Export signature verification
  only authenticates a bundle against an independently pinned key.
- GitHub workflow results must be checked separately from these local results.
  The workflow includes native tests/lint/build/audit and Chromium acceptance.
  One final hosted run timed out installing OS test packages from the runner's
  Azure Ubuntu mirror before any Rust validation ran. The dependency step now
  uses official signed Ubuntu archives, bounded acquisition, no recommended
  extras and an explicit six-minute step timeout. It still installs all test
  dependencies; no tests were skipped to hide the infrastructure failure.
