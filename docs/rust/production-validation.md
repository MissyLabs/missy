# Production conversion acceptance, 2026-11-05

This report supersedes the earlier 17:45 UTC expanded-rewrite checkpoint in
`validation.md`. All implementation owners were frozen before the integrated
acceptance run. Changes remain on `rewrite/rust-runtime`, PR171. No account
connection, live-state import, real hardware capture, service deployment,
process replacement or release tag was authorized or performed.

## Scope and actual evidence

The source inventory contains **37 native workspace crates**. The exact106
legacy normally registered builtin names and order were checked against the
Python registration AST in `production-replacement-matrix.md`. This is a
behavior-disposition inventory, not106 independently exercised live services.
Deliberate typed APIs and stricter isolation are documented rather than hidden
behind a claim of byte-identical Python configuration or unrestricted execution.

Initial integrated toolchain: `rustc 1.96.1`, `cargo 1.96.1`. The exact final
source was then revalidated on **Rust/Cargo1.99.1**, matching hosted stable:
740 ordinary entries, strict all-target Clippy, release build,15 Chromium
assertions and all25 native crypto/voice entries passed again. Locked Linux
x86_64 build. Current stable introduced an array-chunk lint; both affected loops
were updated without suppressing the lint or weakening strict CI.

| Check | Actual result |
|---|---|
| Frozen workspace `cargo test --workspace --all-targets --locked --offline` with screencast browser interpreter enabled | **740 passed, zero failed, four ignored entries, 94 result groups** |
| Formatting | `cargo fmt --all -- --check` passed |
| Strict workspace lint | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` passed |
| Optimized native executable | `cargo build --release -p missy-cli --locked --offline` passed; ELF x86_64 Linux |
| Final release browser console | Actual headless Chromium: **15/15 assertions passed** against `target/release/missy-rs` |
| Isolated default configuration | Final executable init/config-check/doctor passed with disposable HOME, zero allowed origins and zero configured providers; no inference |
| Dependency advisory audit | `cargo audit --deny warnings` passed;419 locked dependencies,1409 loaded advisories, no advisory suppressions |
| Native DAVE/voice opt-in | **25 passed** including all three real-native ignored cases;22 overlap the ordinary suite |
| Installer and release slots | `bash tests/installer_offline.sh` passed actual disposable installer invocation, branch resolution, stage/activate/upgrade/refusals/failed-build preservation/checksum and rollback |

The fourth ordinary ignored entry is an OAuth child-process helper. Its parent
fixture invokes it with its private environment and exercises actual cross-process
vault contention/refresh/restart; it is not a missing live account test. The three
native DAVE cases are separately enabled using the pinned official library and
upstream external-sender helper. Thus743 distinct ordinary+opt-in passing entries
are represented, not765 independent end-to-end scenarios. Some entries are
fixture helpers. Browser/UI assertions and the shell installer fixture are
additional evidence, not additional Rust test entries.

## Audited defects fixed before final acceptance

* Historical context now retains the newest complete message/tool-result groups,
  with source hashes and cumulative omission metadata under the actual escaped
  JSON budget. It no longer prefix-clips away the newest exchanges.
* Approved scoped persona/system revisions now reach the shared production
  runtime, including HTTP, Discord, voice and scheduler admissions. Atomic
  admission metadata pins prompt and optional scoped memory through queueing,
  crash/restart and continuation. A running daemon fixture proves a new persona
  does not mutate an already-paused turn or cross into another session.
* Expected trigger approval pauses no longer poison the worker and stop the
  whole daemon. Actual Store status is reconciled without replay; paused/inflight
  receipts and explicit operator recovery preserve uncertainty.
* Maintenance summaries and observed outcomes have an opt-in bounded exact-scope
  relevance consumer. They remain sanitized untrusted DATA, never prompt patches.
  General memory ranking no longer uses foreign-corpus BM25 statistics.
* The native installer resolves the selected remote conversion branch instead of
  silently assuming a local branch/default master. Build failure preserves the
  previously active native release, and foreign commands are not overwritten.
* Screencast binding/restart invalidation completes before unified readiness.
  The real capture page is exercised by headless Chromium using synthetic canvas
  frames, never the active desktop or a capture permission bypass.
* Hosted acceptance exposed a real owner-lock/status-probe startup race. A
  read-only status probe can briefly own the same lock while checking for stale
  status; startup now tolerates bounded transient contention without unlinking,
  stealing or weakening a live owner's lock. A deterministic held-lock/release
  regression was added to the actual restart fixture. The exact corrected source
  passed all740 native entries, strict Clippy, release and Chromium acceptance.

## Representative real integration fixtures

* Actual native daemon startup, private owner lock, authenticated HTTP and
  operations listeners, shared scheduler, readiness failure, configuration drift,
  bounded SIGTERM drain and restart/no duplicate work.
* Model tool admission through exact durable approvals, original-call resume,
  private immutable evidence, unsafe JSON/input projection and guard retry caps.
* Durable Nomad workload benchmark/offload/finite scheduling, actual mock HTTP
  optimistic plan/submit/log/artifact/reconcile/cancel, owned service CAS and
  rolling restart with full replacement/overlap capacity charging.
* Actual pinned FFmpeg transformations/full bounded external MP4 decoding and
  generated storyboards; real pinned Tesseract/Poppler document adapters,
  isolated Xvfb desktop tests and blank-screen refusal before analysis.
* Real native Opus/MLS/AEAD media, loopback voice WebSocket+UDP signalling,
  consent scope, replay/downgrade denial, cancellation and mocked speech backends.
* Actual provider HTTP protocol/SSE/NDJSON/usage fixtures, weighted routing,
  rate/concurrency/circuit bounds and locked private OAuth refresh fixtures.
* Offline migration planning, inventory, sealed copy-only application, paused
  schedules/pending persona/scoped graph conversion and nondestructive rollback.
* Operator security scan, session inventory, config plans/snapshots, reviewed
  Rust source plugins, skill scans and MCP manifest-bound administration.

## Evidence boundaries

No Robby-host access is needed to implement the conversion. None was used.
Real account credentials, actual Discord/provider endpoints, GPU model quality,
physical audio/camera devices and target infrastructure acceptance were not
tested. Those are later operator deployment acceptance, not work performed here.
The branch does not claim old Python policies, encrypted vaults or in-flight
effects can be automatically trusted or replayed under the new format.

Local detailed logs use `/tmp/missy-production-{frozen-workspace-tests,
frozen-clippy,frozen-release,frozen-native-voice,frozen-browser,audit,installer}.log`.
These temporary logs are not durable CI artifacts. The branch workflow repeats
format/lint/native tests/release build/audit/native crypto and browser acceptance;
the final PR handoff identifies the exact pushed commit and hosted run.

Final executable SHA-256:
`91f7d332d3502696de7ad006cdb6da9865475a3ffa11f7697bbaa0f930bcb5a1`.
Observed dynamic dependencies: `libopus.so.0`, libc, libm, libgcc_s and the ELF
loader. This is not a static or universal Linux binary; official DAVE remains a
separately operator-configured pinned native library.
