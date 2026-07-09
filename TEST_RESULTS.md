# TEST_RESULTS

- Timestamp: 2026-07-09 14:47 EDT
- Focused command:
  `python3 -m pytest tests/api/test_server.py::TestOperatorControls tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/cli/test_tool_provider_cli.py -q`
- Focused result: 73 passed.
- Lint command: `python3 -m ruff check missy/ tests/`
- Lint result: All checks passed.
- Format command: `python3 -m ruff format --check missy/ tests/`
- Format result: 743 files already formatted.
- Full command: `python3 -m pytest -q -o faulthandler_timeout=120`
- Full result: 20648 passed, 13 skipped in 423.79s.

Notes:
- An initial plain `python3 -m pytest -q` run was interrupted after a long
  quiet period without traceback output. The protected rerun with
  `faulthandler_timeout=120` completed successfully.
