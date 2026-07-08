# BUILD_RESULTS

- Timestamp: 2026-07-08 10:38:28 EDT
- Branch: overhaul/web-tui-20260708-122250
- Primary focus: complete web TUI and operator console overhaul

## Python

Python 3.12.3

## Session Build Summary

- Extracted main Web TUI console rendering from `missy/api/server.py`.
- Added `render_console()` and `console_script()` to `missy/api/web_console.py`.
- Preserved API routing, browser auth, session handling, CSRF protections,
  controls behavior, diagnostics loading, and audit browser behavior.
- Added renderer contract coverage in `tests/api/test_server.py`.

## Verification

```text
python3 -m ruff format --check .
731 files already formatted
```

```text
python3 -m ruff check .
All checks passed!
```

```text
python3 -m pytest tests/api/test_server.py -q
85 passed in 10.19s
```

```text
python3 -m pytest -q
20463 passed, 13 skipped in 381.13s (0:06:21)
```
