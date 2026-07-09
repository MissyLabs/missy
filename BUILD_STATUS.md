# Build Status

Last updated: 2026-07-09 14:12 EDT

## Current State

Primary focus is the **tool usage and tool intelligence overhaul**. Missy now
has request-pattern tracking, conservative candidate generation, candidate
lifecycle storage, benchmark harnesses, benchmark-to-candidate reconciliation,
benchmark-backed provider gates, runtime request-tracker wiring, and CLI
diagnostics for candidates, benchmarks, provider decisions, and overrides.

This session completed the next lifecycle gap: persisted benchmark results can
now be imported into matching `ToolCandidate` review records without manual
glue. The import updates provider benchmark summaries, records conservative
provider enablement flags, and can move proposed/experimental candidates to
`benchmarked`; approval and runtime enablement remain separate audited actions.

## Completed Work This Session

- `missy/tools/intelligence/benchmark_reconciler.py`
  - Added `CandidateBenchmarkReconciler`.
  - Converts `BenchmarkStore.provider_summary()` aggregates into candidate
    `BenchmarkSummary` entries.
  - Computes provider flags from sample count, composite score, safety score,
    and schema-adherence thresholds.
  - Emits `tool.candidate.benchmarks_reconciled` audit events.
- `missy/tools/benchmark/benchmark_store.py`
  - Provider summaries now include mean schema score and mean tool-call quality
    so candidate review can inspect schema/tool-call behavior.
- `missy/cli/main.py`
  - Added `missy tools candidates import-benchmarks <candidate_id>`.
  - Supports benchmark tool-name override and threshold options.
  - Fixed candidate enable pre-check to require `approved`, matching the
    store-level lifecycle gate.
- `missy/tools/intelligence/__init__.py`
  - Exported reconciliation types.
- Tests
  - Added reconciler unit coverage for successful import, tool-name override,
    missing benchmark data, insufficient samples, and missing candidates.
  - Added CLI coverage for benchmark import success and missing-data errors.
- Docs
  - Updated operations docs with the benchmark import command.
  - Updated module map with the reconciler boundary.

## Current Architecture State

- Request tracking lives in `missy.tools.intelligence.request_tracker` and is
  wired into `AgentRuntime._track_request()` after completed turns.
- Candidate synthesis is opt-in via `tool_intelligence.candidate_generation`
  and produces proposed candidates only.
- `CandidateStore` remains the lifecycle authority for all candidate callers.
- `CandidateBenchmarkReconciler` is the evidence bridge from raw benchmark
  results into candidate review metadata. It does not approve or enable tools.
- Benchmark results live in `missy.tools.benchmark.benchmark_store` and feed
  both provider gates and candidate reconciliation.
- Provider gates remain opt-in via
  `tool_intelligence.provider_gating.enabled`; they remove weak tool/provider
  pairings from runtime tool exposure without bypassing normal tool execution
  policy.
- Explicit provider enable/disable overrides are persisted by
  `ProviderGateStore` and audited.

## Tests

- `python3 -m pytest tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/tools/test_benchmark.py tests/tools/test_provider_gate.py tests/cli/test_tool_provider_cli.py -q`
  - 112 passed.
- `python3 -m ruff check missy/ tests/`
  - passed.
- `python3 -m ruff format --check missy/ tests/`
  - 743 files already formatted.
- `python3 -m pytest -q`
  - 20643 passed, 13 skipped in 426.31s.

## Remaining Work

1. Add candidate lifecycle controls to the Web/API operator surface with the
   same `CandidateStore` transition rules and typed confirmations for
   destructive states.
2. Connect enabled approved candidates to a controlled runtime loading path,
   still behind policy, schema validation, provenance checks, tests, and
   rollback.
3. Add provider-family fallback diagnostics to runtime output when a tool is
   gated off for the active provider and a better provider is known.
4. Add richer schema-compatibility reporting per provider/tool family for
   candidate review.
5. Add lifecycle commands/API for `experimental` and `deprecated` transitions
   where operator review needs those intermediate states.

## Blockers

- None for the next slice. Benchmark reconciliation is additive and uses the
  existing candidate store and benchmark store formats.

## Next Actions

Expose candidate lifecycle and benchmark-import controls through the Web/API
operator surface, using the same store-level lifecycle rules and explicit
confirmations for destructive transitions.
