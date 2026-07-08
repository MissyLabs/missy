# TEST_RESULTS

- Timestamp: 2026-07-08 09:19:07 EDT
- Branch: overhaul/web-tui-20260708-122250
- Primary focus: complete web TUI and operator console overhaul

## Focused API Suite

```text
python3 -m pytest tests/api/test_server.py -q
75 passed in 8.36s
```

## Lint And Format

```text
python3 -m ruff check .
All checks passed!
```

```text
python3 -m ruff format --check .
728 files already formatted
```

## Full Suite

```text
python3 -m pytest -q
20453 passed, 13 skipped in 382.78s (0:06:22)
```
