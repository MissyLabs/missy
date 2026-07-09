# Build Status

Last updated: 2026-07-09 15:25 EDT

## Current State

Primary focus remains the **tool usage and tool intelligence overhaul**. Missy
now has request-pattern tracking, conservative candidate generation, candidate
lifecycle storage, benchmark harnesses, benchmark-to-candidate reconciliation,
provider-specific gates, CLI diagnostics, Web/API review controls, and an
opt-in controlled runtime loader for enabled candidates.

This session completed the first runtime-loading slice. Enabled candidates can
now carry explicit implementation metadata and, when
`tool_intelligence.candidate_runtime.enabled: true`, `AgentRuntime` invokes
`CandidateRuntimeLoader` before tool exposure. The loader currently supports
only a narrow `delegated_tool` implementation that wraps an already-registered
tool. It validates lifecycle state, provenance, schema, permissions,
provider-enable flags, implementation type, target registration, and name
conflicts before registration. Incomplete candidates are skipped fail-closed
and audited.

## Completed Work This Session

- `missy/tools/intelligence/candidate_loader.py`
  - Added `CandidateRuntimeLoader`, `CandidateDelegatedTool`,
    `CandidateLoadReport`, and structured skip issues.
  - Loads only `enabled` candidates for the active provider.
  - Requires valid provenance, object schema, safe names, known permission keys,
    `provider_enabled[provider] is True`, and explicit implementation metadata.
  - Supports only `{"type": "delegated_tool", "tool": "<registered_tool>"}`.
  - Emits `tool.candidate.loaded` and `tool.candidate.load_skipped` audit events.
- `missy/tools/intelligence/candidate_store.py`
  - Added persisted `implementation` metadata to `ToolCandidate`.
  - Added SQLite migration for existing candidate databases.
- `missy/agent/runtime.py`
  - Added opt-in candidate runtime loading before tool-policy and provider-gate
    filtering.
  - Loader runs once per runtime instance and degrades to registered tools only
    on loader errors.
- `missy/config/settings.py`
  - Added `tool_intelligence.candidate_runtime.enabled`, default `false`.
- `missy/vision/capture.py`
  - Hardened deadline-aware retry sleeps/checks against exhausted mocked
    `time.monotonic()` sequences, fixing a late full-suite flake.
- Tests
  - Added loader tests for successful delegated loading, missing
    implementation, provider-disabled candidates, name conflicts, and schema
    rejection.
  - Added runtime opt-in wiring tests, config parsing tests, and candidate-store
    implementation persistence coverage.
- Docs
  - Updated configuration, operations, and module-map docs for candidate runtime
    loading and implementation metadata.

## Current Architecture State

- Request tracking lives in `missy.tools.intelligence.request_tracker` and is
  wired into `AgentRuntime._track_request()` after completed turns.
- Candidate synthesis is opt-in via `tool_intelligence.candidate_generation`
  and produces proposed candidates only.
- `CandidateStore` remains the lifecycle authority for CLI, Web/API controls,
  runtime automation, and future loaders.
- `CandidateRuntimeLoader` is the runtime boundary for enabled candidate
  exposure. It does not execute generated code or infer behavior from
  descriptions.
- Runtime loading remains opt-in via
  `tool_intelligence.candidate_runtime.enabled` and runs before configured tool
  policy and provider benchmark gates.
- `CandidateBenchmarkReconciler` remains the evidence bridge from raw benchmark
  results into candidate review metadata. It does not approve, enable, or load
  tools.
- Benchmark results live in `missy.tools.benchmark.benchmark_store` and feed
  both provider gates and candidate reconciliation.
- Provider gates remain opt-in via
  `tool_intelligence.provider_gating.enabled`; they remove weak tool/provider
  pairings from runtime tool exposure without bypassing normal tool execution
  policy.

## Tests

- `python3 -m pytest tests/vision/test_frame_eviction_hardening.py::TestCaptureDeadlineAwareSleep tests/tools/test_candidate_loader.py tests/tools/test_candidate_store.py tests/agent/test_tool_intelligence_wiring.py tests/config/test_settings.py -q`
  - 122 passed.
- `python3 -m ruff check missy/ tests/`
  - passed.
- `python3 -m ruff format --check missy/ tests/`
  - 745 files already formatted.
- `python3 -m pytest -q -o faulthandler_timeout=120`
  - 20656 passed, 13 skipped in 420.71s.

## Remaining Work

1. Add review/API/CLI surfaces for setting or revising candidate
   implementation metadata rather than relying on direct store writes.
2. Add additional implementation adapters beyond `delegated_tool` only after
   they have policy gates, provenance checks, tests, sandboxing, and rollback.
3. Add provider-family fallback diagnostics to runtime output when a tool is
   gated off for the active provider and a better provider is known.
4. Add richer schema-compatibility reporting per provider/tool family for
   candidate review.
5. Add lifecycle commands/API for `experimental` and `deprecated` transitions
   where operator review needs those intermediate states.

## Blockers

- None for the next slice.
- A `.stop` marker is present in the worktree, so the external loop controller
  may stop before another session.

## Next Actions

Add safe operator controls and CLI/API support for candidate implementation
metadata, starting with `delegated_tool` bindings, so approved candidates can
be reviewed end to end without manual database edits.
