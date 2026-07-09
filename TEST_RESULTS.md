# TEST_RESULTS

- Timestamp: 2026-07-09 13:52:21 EDT

## Focused Tool-Intelligence Tests

- Command:

```text
python3 -m pytest tests/tools/test_candidate_store.py tests/tools/test_candidate_generator.py tests/tools/test_request_tracker.py tests/tools/test_provider_gate.py tests/agent/test_tool_intelligence_wiring.py tests/agent/test_request_tracker_wiring.py tests/cli/test_tool_provider_cli.py tests/cli/test_benchmark_run_cmd.py -q
```

- Result:

```text
157 passed in 8.34s
```

## Full Test Suite

- Command:

```text
python3 -m pytest -q
```

- Result:

```text
20636 passed, 13 skipped in 441.06s (0:07:21)
```

## Lint And Format

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check missy/ tests/
741 files already formatted
```
