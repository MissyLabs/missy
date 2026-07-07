# TEST_RESULTS

- Timestamp: 2026-07-07 19:48 EDT
- Primary focus: complete Discord integration overhaul

## Commands

```text
pytest tests/channels/test_discord_protocol_deep.py tests/channels/test_discord_extended.py tests/channels/test_discord_channel_coverage.py tests/cli/test_cli_commands.py::TestDiscordDiagnostics tests/unit/test_hardening_piper_discord.py::TestDiscordGatewayOpcodes -q
288 passed in 15.36s
```

```text
pytest tests/security/test_identity_drift_edges.py tests/security/test_injection_multimodal_token_patterns.py tests/security/test_landlock.py tests/security/test_property_based.py tests/security/test_property_based_fuzz.py -q
212 passed in 15.47s
```

```text
timeout 1200 pytest -q
20270 passed, 13 skipped in 377.44s (0:06:17)
```

```text
ruff check .
All checks passed.
```

```text
ruff format --check .
708 files already formatted.
```

## Notes

- An earlier full-suite run found four `tests/unit/test_hardening_piper_discord.py::TestDiscordGatewayOpcodes` failures because those low-level tests construct `DiscordGatewayClient` with `__new__`, bypassing `__init__`. The fix added `_ensure_diagnostic_state()` guards and the targeted tests now pass.
- A subsequent unbounded full-suite rerun was killed after it stayed quiet while still active. The bounded full-suite rerun completed successfully.
