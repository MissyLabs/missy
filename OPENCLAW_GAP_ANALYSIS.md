# OpenClaw Gap Analysis

Last updated: 2026-07-08

## Current Focus

The branch now contains both the tool-intelligence overhaul work and the Discord integration diagnostics work from `origin/master`.

## Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | tested | Runtime support remains in place. |
| A2 | Layered tool policy pipeline | hardened | Policy surfaces include current `master` updates. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated identical failing tool calls are fingerprinted and surfaced to the model. |
| A4 | Compaction retry coordination | not_started | Manager-level retry coordination remains future work. |
| A5 | Auth profile cooldown + fallback | not_started | Must preserve user-pinned profile behavior. |
| A6 | Per-provider tool schema normalization | live | Provider schema methods now delegate to `normalize_for_provider()` with fallbacks. |
| A7 | Block-reply chunking with flush points | not_started | Channel delivery remains future work. |
| A8 | Per-channel identity cascade | not_started | |
| A9 | Before/after hook system | not_started | |
| A10 | Sub-agent depth + child caps | not_started | |
| A11 | Raw-stream JSONL diagnostics | not_started | |
| A12 | Transcript dual-repair | not_started | |
| A13 | Context-window guard | not_started | |

## Tool Intelligence Gaps

| Feature | Status | Path |
|---|---|---|
| Request tracking + pattern detection | wired | `missy/tools/intelligence/request_tracker.py`, `missy/agent/runtime.py` |
| Tool candidate lifecycle store | implemented | `missy/tools/intelligence/candidate_store.py` |
| Candidate generation from patterns | implemented | `missy/tools/intelligence/candidate_generator.py` |
| Per-provider schema normalization | live | `missy/providers/schema_adapter.py`, provider implementations |
| Benchmark scoring and store | implemented | `missy/tools/benchmark/` |
| Benchmark runner | implemented | `missy/tools/benchmark/runner.py` |
| CLI `missy tools benchmark run` | implemented | `missy/cli/main.py` |
| Provider-specific enablement from benchmarks | not_started | Populate and enforce `ToolCandidate.provider_enabled`. |
| Fallback routing based on benchmark scores | not_started | Router should consult provider enablement/score thresholds. |

## Discord Gaps From Master

- Live Gateway snapshots are process-local; a separate service status channel is needed for exact out-of-process CLI visibility.
- Byte-level image verification remains partial without an image dependency or vision-extra integration.
- Runtime voice binding diagnostics are process-local.
- Operator troubleshooting guidance should keep expanding around missing tokens, intents, policy-denied hosts, voice dependencies, ffmpeg, STT/TTS, slash registration, invalid Gateway sessions, and stale heartbeat ACKs.

## Recommended Next Slice

Add provider-specific benchmark enablement for tools, then add benchmark-aware provider fallback routing.
