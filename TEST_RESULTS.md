# TEST_RESULTS

- Timestamp: 2026-07-07 18:49:30 EDT
- Command: `pytest -q`

```text
20260 passed, 13 skipped in 364.17s (0:06:04)
```

Additional checks:

```text
pytest tests/cli/test_cli_commands.py::TestDiscordDiagnostics tests/policy/test_tool_policy_pipeline.py tests/tools/test_discord_voice_tools.py -q
34 passed in 0.36s

pytest tests/cli/test_cli_commands.py::TestDiscordStatus tests/cli/test_cli_commands.py::TestDiscordAudit tests/cli/test_cli_commands.py::TestDiscordDiagnostics tests/cli/test_cli_main_extended.py::TestDiscordProbeBranches tests/cli/test_cli_main_extended.py::TestDiscordRegisterCommandsBranches tests/channels/test_discord_channel_gap_coverage.py tests/channels/test_discord_voice_extended.py tests/tools/test_discord_voice_tools.py tests/policy/test_tool_policy_pipeline.py -q
144 passed in 1.17s

pytest tests/agent/test_runtime_coverage_gaps.py::TestGetToolsDiscordMode tests/agent/test_runtime_config_edges.py -q
95 passed in 1.47s

ruff check .
All checks passed!

ruff format --check .
708 files already formatted
```
