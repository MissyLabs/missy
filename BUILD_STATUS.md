# Build Status

Last updated: 2026-04-27

## Current State

- Loop session 3 continued from the initialized OpenClaw/humanize artifacts.
- Implemented the first A1 slice: `missy/agent/subscription.py` provides an `AgentSubscription` state machine for streaming message/tool/compaction events.
- Lightly wired A1 into `AgentRuntime.run_stream()` so single-turn streaming uses the same tag-stripping and monotonic buffering path.
- Added targeted tests for subscription reconciliation, split tag stripping, code-span awareness, reasoning streaming, reply directives, block flush points, compaction retry state, and runtime streaming integration.
- Implemented A2 `missy/policy/tool_policy_pipeline.py` with profile layers, standard profile→provider→global→agent→group→sandbox→subagent layer construction, group expansion, glob matching, `-tool` deny syntax, fail-warning unknown allowlists, and per-step trace records.
- Wired `AgentRuntime._get_tools()` through the A2 pipeline while preserving existing `full`, `safe-chat`, `discord`, and `no-tools` capability-mode behavior.
- Hardened A2 with config-backed policy surfaces:
  - `tools.profile/allow/deny/alsoAllow/byProvider/byModel/groups`
  - `agents.<id>.tools.*`
  - `agents.<id>.subagents.tools.*`
  - `sandbox.tools.*`
- Routed parsed YAML policy objects through CLI-created `AgentConfig` instances for ask/run/gateway/API paths.
- Documented tool visibility policy in `docs/configuration.md`.

## Verification

- `pytest tests/agent/test_subscription.py tests/agent/test_runtime_streaming.py -q` passed: 18 tests.
- `pytest -q` passed: 20077 passed, 14 skipped in 369.30s.
- `ruff check .` passed.
- `ruff format --check .` passed.
- Session 2 focused verification passed:
  - `pytest tests/policy/test_tool_policy_pipeline.py tests/agent/test_runtime_streaming.py tests/agent/test_coverage_gaps.py::TestRuntimeCapabilityMode -q`: 19 passed.
  - `pytest tests/agent/test_coverage_gaps.py::TestRuntimeCapabilityMode tests/tools/test_registry_policy_edges.py -q`: 58 passed.
  - `ruff check missy/policy/tool_policy_pipeline.py missy/agent/runtime.py tests/policy/test_tool_policy_pipeline.py tests/agent/test_runtime_streaming.py`: passed.
  - `ruff format --check missy/policy/tool_policy_pipeline.py missy/agent/runtime.py tests/policy/test_tool_policy_pipeline.py tests/agent/test_runtime_streaming.py`: passed.
  - `ruff check .`: passed.
  - `ruff format --check .`: passed, 702 files already formatted.
- Session 3 focused verification passed:
  - `pytest tests/policy/test_tool_policy_pipeline.py tests/config/test_settings.py tests/agent/test_runtime_config_edges.py tests/agent/test_runtime_streaming.py tests/tools/test_registry_policy_edges.py -q`: 222 passed.
  - `pytest -q`: 20085 passed, 14 skipped in 365.02s.
  - `ruff check .`: passed.
  - `ruff format --check .`: passed, 702 files already formatted.

## Repository Notes

- Pre-existing untracked files at session start: `.embedded_prompt.txt`, `build_log.txt`.
- No new third-party dependencies were added.

## Remaining Work

- A1 still needs deeper runtime wiring for the tool-call loop and channel block replies.
- A2 is wired and config-backed for provider/global/agent/sandbox/subagent policies; future hardening can add channel/group policy sources and richer audit display.
- A3 through A13 remain to be implemented after A1/A2 hardening.
- H_A through H_I are tracked but not yet implemented.
