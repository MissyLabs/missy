# TEST_RESULTS

- Timestamp: 2026-07-08
- Branch: overhaul/tools-20260708-020326
- Context: rebase conflict resolution

## Verification Before Rebase

```text
python3 -m pytest tests/agent/test_mutation_fingerprint.py tests/agent/test_request_tracker_wiring.py tests/cli/test_benchmark_run_cmd.py tests/providers/test_schema_adapter_wiring.py -q
37 passed in 1.12s
```

```text
python3 -m pytest tests/channels/test_discord_protocol_deep.py tests/channels/test_discord_channel_coverage.py tests/channels/test_discord_channel_gap_coverage.py tests/channels/test_discord_image_analyze.py tests/cli/test_cli_commands.py::TestDiscordDiagnostics tests/tools/test_discord_voice_tools.py tests/tools/test_builtin_init_coverage.py -q
281 passed in 8.60s
```

```text
python3 -m pytest tests/policy/test_tool_policy_pipeline.py tests/security/test_discord_attachment_codeevolution_filewrite_security.py tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTLoad tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTAutoDevice -q
43 passed in 0.86s
```

```text
python3 -m ruff check .
All checks passed!
```

```text
python3 -m ruff format --check .
726 files already formatted
```

## Tool Branch Full Suite Before Merge

```text
python3 -m pytest -q
2 failed, 20404 passed, 13 skipped in 363.06s (0:06:03)
```

Known failures:

- `tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTLoad::test_load_raises_import_error_when_faster_whisper_missing`
- `tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTAutoDevice::test_auto_device_falls_back_to_cpu_when_torch_missing`

Notes:

- The failures were optional voice/STT environment detection tests.
- The focused post-merge STT slice passed after `master` was merged.

## Master Verification Before Merge

```text
pytest tests/channels/test_discord_protocol_deep.py tests/channels/test_discord_extended.py tests/channels/test_discord_channel_coverage.py tests/cli/test_cli_commands.py::TestDiscordDiagnostics tests/unit/test_hardening_piper_discord.py::TestDiscordGatewayOpcodes -q
288 passed
```

```text
pytest tests/security/test_identity_drift_edges.py tests/security/test_injection_multimodal_token_patterns.py tests/security/test_landlock.py tests/security/test_property_based.py tests/security/test_property_based_fuzz.py -q
212 passed
```

```text
timeout 1200 pytest -q
20270 passed, 13 skipped
```
