# Build Status

Last updated: 2026-07-09 14:47 EDT

## Current State

Primary focus is the **tool usage and tool intelligence overhaul**. Missy now
has request-pattern tracking, conservative candidate generation, candidate
lifecycle storage, benchmark harnesses, benchmark-to-candidate reconciliation,
benchmark-backed provider gates, runtime request-tracker wiring, CLI
diagnostics, and Web/API operator controls for candidate review actions.

This session completed the next operator-surface gap: candidate list/show and
lifecycle controls are exposed through the REST API and the existing safe
controls model. Web/API candidate actions reuse `CandidateStore` and
`CandidateBenchmarkReconciler`, require typed confirmations, audit allow/deny
results, and do not bypass lifecycle rules.

## Completed Work This Session

- `missy/api/operator_controls.py`
  - Added `tool_candidate.import_benchmarks`, `tool_candidate.approve`,
    `tool_candidate.enable`, and `tool_candidate.deny` controls.
  - Candidate controls require typed confirmations and emit structured
    `web.control` audit details.
  - Denial is destructive and requires an explicit operator reason.
  - Invalid lifecycle shortcuts still fail through `CandidateStore.transition`.
- `missy/api/server.py`
  - Added injectable `candidate_store` and `benchmark_store` dependencies.
  - Added `GET /api/v1/tool-candidates` and
    `GET /api/v1/tool-candidates/{id}` review endpoints.
  - Routed candidate controls through the shared operator-control executor.
- `missy/cli/main.py`
  - API startup now attaches the normal candidate and benchmark stores when
    available, allowing the browser console to surface candidate controls from
    `GET /api/v1/controls`.
- Tests
  - Added API coverage for candidate list/show, control target visibility,
    import -> approve -> enable, deny confirmation/reason requirements, and
    skipped lifecycle-gate rejection.
- Docs
  - Updated operations docs with candidate REST routes and control IDs.
  - Updated the module map for API and operator-control boundaries.

## Current Architecture State

- Request tracking lives in `missy.tools.intelligence.request_tracker` and is
  wired into `AgentRuntime._track_request()` after completed turns.
- Candidate synthesis is opt-in via `tool_intelligence.candidate_generation`
  and produces proposed candidates only.
- `CandidateStore` remains the lifecycle authority for CLI, Web/API controls,
  runtime automation, and future loaders.
- `CandidateBenchmarkReconciler` remains the evidence bridge from raw benchmark
  results into candidate review metadata. It does not approve or enable tools.
- Web/API candidate controls are a thin policy-shaped layer over the same
  store/reconciler logic, with explicit confirmations and `web.control` audit
  events.
- Benchmark results live in `missy.tools.benchmark.benchmark_store` and feed
  both provider gates and candidate reconciliation.
- Provider gates remain opt-in via
  `tool_intelligence.provider_gating.enabled`; they remove weak tool/provider
  pairings from runtime tool exposure without bypassing normal tool execution
  policy.
- Explicit provider enable/disable overrides are persisted by
  `ProviderGateStore` and audited.

## Tests

- `python3 -m pytest tests/api/test_server.py::TestOperatorControls tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/cli/test_tool_provider_cli.py -q`
  - 73 passed.
- `python3 -m ruff check missy/ tests/`
  - passed.
- `python3 -m ruff format --check missy/ tests/`
  - 743 files already formatted.
- `python3 -m pytest -q -o faulthandler_timeout=120`
  - 20648 passed, 13 skipped in 423.79s.

## Remaining Work

1. Connect enabled approved candidates to a controlled runtime loading path,
   still behind policy, schema validation, provenance checks, tests, and
   rollback.
2. Add provider-family fallback diagnostics to runtime output when a tool is
   gated off for the active provider and a better provider is known.
3. Add richer schema-compatibility reporting per provider/tool family for
   candidate review.
4. Add lifecycle commands/API for `experimental` and `deprecated` transitions
   where operator review needs those intermediate states.
5. Expand the browser console from generic candidate control rows to a richer
   candidate review panel if operators need benchmark summaries inline.

## Blockers

- None for the next slice. Candidate Web/API controls are additive and share
  existing lifecycle and benchmark storage.

## Next Actions

Build the controlled runtime loading path for enabled candidates with schema,
permissions, provenance, tests, provider enablement, policy checks, and
rollback guardrails before any generated candidate can become executable.
