# TEST_RESULTS

- Timestamp: 2026-07-08 14:34:10 EDT
- Command: `python3 -m pytest -q`

```text
20487 passed, 6 skipped, 3 warnings in 397.50s (0:06:37)
```

## Focused Checks

```text
python3 -m pytest tests/agent/test_structured_output.py tests/providers/test_openai_provider.py -q
101 passed in 1.30s
```

```text
python3 -m pytest tests/providers -q
843 passed in 22.61s
```

```text
python3 -m pytest tests/agent -q
4109 passed, 4 skipped in 44.75s
```

## Lint/Format

```text
python3 -m ruff format --check .
731 files already formatted
```

```text
python3 -m ruff check .
All checks passed!
```

## Warnings

- Deprecation warnings from SWIG import types remain pre-existing:
  `SwigPyPacked`, `SwigPyObject`, and `swigvarlink` missing `__module__`.
