# TEST_RESULTS

- Timestamp: 2026-07-09 14:12:59 EDT

## Focused Tool-Intelligence Suite

Command:

```bash
python3 -m pytest tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/tools/test_benchmark.py tests/tools/test_provider_gate.py tests/cli/test_tool_provider_cli.py -q
```

Result:

```text
112 passed in 7.80s
```

## Lint And Formatting

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check missy/ tests/
743 files already formatted
```

## Full Suite

Command:

```bash
python3 -m pytest -q
```

Result:

```text
20643 passed, 13 skipped in 426.31s (0:07:06)
```
