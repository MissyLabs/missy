# Test Results

Last updated: 2026-04-27

## Latest Runs

| Command | Result | Notes |
| --- | --- | --- |
| `pytest tests/policy/test_tool_policy_pipeline.py tests/agent/test_runtime_streaming.py tests/agent/test_coverage_gaps.py::TestRuntimeCapabilityMode -q` | pass | 19 passed in 5.06s. |
| `pytest tests/agent/test_coverage_gaps.py::TestRuntimeCapabilityMode tests/tools/test_registry_policy_edges.py -q` | pass | 58 passed in 1.57s. |
| `pytest -q` | pass | 20077 passed, 14 skipped in 369.30s. |
| `ruff check .` | pass | Full-repo lint passed. |
| `ruff format --check .` | pass | 702 files already formatted. |
| `pytest tests/policy/test_tool_policy_pipeline.py tests/config/test_settings.py tests/agent/test_runtime_config_edges.py tests/agent/test_runtime_streaming.py tests/tools/test_registry_policy_edges.py -q` | pass | 222 passed in 10.40s. |
| `pytest -q` | pass | 20085 passed, 14 skipped in 365.02s. |
| `ruff check .` | pass | Full-repo lint passed. |
| `ruff format --check .` | pass | 702 files already formatted. |

## Not Yet Run This Session

- No known verification gaps for this session.
