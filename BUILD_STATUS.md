# Build Status

Last updated: 2026-07-09 13:52 EDT

## Current State

Primary focus is the **tool usage and tool intelligence overhaul**. Missy now
has a real tool-intelligence backbone: request-pattern tracking, conservative
candidate generation, candidate lifecycle storage, direct/LLM benchmark
harnesses, benchmark-backed provider gates, runtime request-tracker wiring,
and CLI diagnostics for candidates, benchmarks, provider decisions, and
overrides.

This session tightened the highest-risk gap in that path: tool-candidate
lifecycle transitions were documented as gated, but `CandidateStore.transition`
accepted arbitrary jumps. Store-level enforcement now prevents callers from
skipping review/benchmark gates or resurrecting disabled candidates.

## Completed Work This Session

- `missy/tools/intelligence/candidate_store.py`
  - Added `is_valid_transition()` and a single transition matrix for
    candidate lifecycle rules.
  - Enforced lifecycle transitions inside `CandidateStore.transition()`.
  - Invalid transitions now keep the stored state unchanged, raise
    `ValueError`, and emit `tool.candidate.transition_denied` with
    `result="deny"`.
  - Allowed path: `proposed -> experimental -> benchmarked -> approved ->
    enabled`.
  - Rollback path: approved/enabled candidates may move to `deprecated` or
    `disabled`; deprecated candidates may be restored to enabled or disabled.
  - Disabled candidates are terminal and cannot be re-enabled in place.
- `missy/tools/intelligence/__init__.py`
  - Exported `is_valid_transition()` for CLI/API diagnostics and future
    control-plane code.
- `missy/cli/main.py`
  - Candidate approve/enable/deny commands now catch lifecycle `ValueError`
    and report a clean CLI error instead of a traceback.
  - Approval help now describes approving a benchmarked candidate, matching
    the enforced gate.
- `tests/tools/test_candidate_store.py`
  - Updated lifecycle tests to follow benchmark-before-approval.
  - Added coverage for rejected direct enable, rejected pre-benchmark
    approval, disabled-candidate resurrection denial, deprecated rollback,
    no-op transitions, and exported transition-rule helper behavior.
- Docs
  - `docs/operations.md` now documents tool-intelligence operations,
    request-pattern review, candidate lifecycle commands, and provider-gate
    commands.
  - `docs/implementation/module-map.md` now includes
    `missy.tools.intelligence` and `missy.tools.benchmark`.

## Current Architecture State

- Request tracking lives in `missy.tools.intelligence.request_tracker` and is
  wired into `AgentRuntime._track_request()` after completed turns.
- Candidate synthesis is opt-in via `tool_intelligence.candidate_generation`
  and produces proposed candidates only.
- `CandidateStore` is now the lifecycle authority for all candidate callers,
  not just the CLI. Future Web/API controls should call the same store methods
  rather than duplicating transition checks.
- Benchmark results live in `missy.tools.benchmark.benchmark_store` and feed
  `ToolProviderGate` decisions.
- Provider gates remain opt-in via
  `tool_intelligence.provider_gating.enabled`; they remove weak tool/provider
  pairings from runtime tool exposure without bypassing normal tool execution
  policy.
- Explicit provider enable/disable overrides are persisted by
  `ProviderGateStore` and audited.

## Tests

- `python3 -m pytest tests/tools/test_candidate_store.py tests/tools/test_candidate_generator.py tests/tools/test_request_tracker.py tests/tools/test_provider_gate.py tests/agent/test_tool_intelligence_wiring.py tests/agent/test_request_tracker_wiring.py tests/cli/test_tool_provider_cli.py tests/cli/test_benchmark_run_cmd.py -q`
  - 157 passed.
- `python3 -m pytest -q`
  - 20636 passed, 13 skipped in 441.06s.
- `python3 -m ruff check missy/ tests/`
  - passed.
- `python3 -m ruff format --check missy/ tests/`
  - passed, 741 files already formatted.

## Remaining Work

1. Add a CLI/API command to import benchmark summaries into matching
   `ToolCandidate` records so candidates can move to `benchmarked` from real
   benchmark store data without manual glue.
2. Add candidate lifecycle controls to the Web/API operator surface with the
   same `CandidateStore` transition rules and typed confirmations for
   destructive states.
3. Connect enabled approved candidates to a controlled runtime loading path,
   still behind policy, schema validation, provenance checks, and rollback.
4. Add provider-family fallback diagnostics to runtime output when a tool is
   gated off for the active provider and a better provider is known.
5. Add schema-compatibility reporting per provider/tool family for candidate
   review.

## Blockers

- None for the next slice. The lifecycle gate is store-level and additive;
  remaining work can build on it without changing existing benchmark or
  provider-gate persistence formats.

## Next Actions

Implement benchmark-to-candidate reconciliation: read stored benchmark
summaries for a tool/candidate, update `CandidateStore.update_benchmark()`,
compute provider enablement flags, and expose it through
`missy tools candidates benchmark-status` or a similarly direct command.
