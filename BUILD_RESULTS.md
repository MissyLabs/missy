# BUILD_RESULTS

- Timestamp: 2026-07-09 14:12:59 EDT
- Branch: overhaul/tools-20260709-174109
- Primary focus: complete tool usage and tool intelligence overhaul

## Python

Python 3.12.3

## Build Summary

Implemented benchmark-to-candidate reconciliation for tool intelligence.
Persisted benchmark aggregates can now be imported into candidate review
records with provider summaries and benchmark-derived provider flags.

## Changed Files

- `missy/tools/intelligence/benchmark_reconciler.py`
- `missy/tools/intelligence/__init__.py`
- `missy/tools/benchmark/benchmark_store.py`
- `missy/cli/main.py`
- `tests/tools/test_benchmark_reconciler.py`
- `tests/cli/test_tool_provider_cli.py`
- `docs/operations.md`
- `docs/implementation/module-map.md`
- Required loop artifacts

## Verification

```text
python3 -m pytest tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/tools/test_benchmark.py tests/tools/test_provider_gate.py tests/cli/test_tool_provider_cli.py -q
112 passed in 7.80s
```

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check missy/ tests/
743 files already formatted
```

```text
python3 -m pytest -q
20643 passed, 13 skipped in 426.31s (0:07:06)
```
