# TEST_RESULTS

- Timestamp: 2026-07-09 15:25:53 EDT

## Focused Tests

```text
python3 -m pytest tests/vision/test_frame_eviction_hardening.py::TestCaptureDeadlineAwareSleep tests/tools/test_candidate_loader.py tests/tools/test_candidate_store.py tests/agent/test_tool_intelligence_wiring.py tests/config/test_settings.py -q
122 passed in 5.03s
```

## Lint And Format

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check missy/ tests/
745 files already formatted
```

## Full Suite

```text
python3 -m pytest -q -o faulthandler_timeout=120
20656 passed, 13 skipped in 420.71s (0:07:00)
```
