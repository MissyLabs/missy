# TEST_RESULTS

- Timestamp: 2026-07-08 15:03:20 EDT
- Command: pytest -q

```text
20489 passed, 6 skipped, 3 warnings in 392.42s (0:06:32)
```

Warnings:

```text
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Focused checks:

```text
python3 -m pytest tests/providers/test_openai_provider.py tests/api/test_server.py::TestDiagnostics::test_diagnostics_reports_redacted_operator_posture tests/cli/test_cli_commands.py::TestDoctor::test_doctor_shows_provider_not_available -q
39 passed in 1.91s
```

```text
python3 -m pytest tests/providers -q
845 passed in 23.46s
```

```text
python3 -m pytest tests/api/test_server.py tests/cli/test_cli_commands.py::TestDoctor tests/cli/test_cli_commands.py::TestDoctorBranches -q
97 passed in 17.12s
```

Lint/format:

```text
python3 -m ruff format --check .
731 files already formatted
```

```text
python3 -m ruff check .
All checks passed!
```
