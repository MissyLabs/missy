# Test Results

## Date: 2026-03-18

## Summary

| Metric | Value |
|---|---|
| Total tests | 7388 |
| Passed | 7388 |
| Failed | 0 |
| Skipped | 14 |
| Duration | ~154s |

## Test Distribution

| Directory | Description |
|---|---|
| tests/agent/ | Agent runtime, behavior, persona, hatching, attention, etc. |
| tests/channels/ | CLI, Discord, webhook, voice channels |
| tests/cli/ | CLI commands, wizard, auth flows |
| tests/config/ | Config loading, migration, hot-reload |
| tests/core/ | Events, sessions, message bus |
| tests/integration/ | Cross-module integration tests |
| tests/memory/ | SQLite store, resilient wrapper, vector store |
| tests/observability/ | Audit logger, OpenTelemetry |
| tests/plugins/ | Plugin loader and registry |
| tests/policy/ | Network, filesystem, shell, REST policies |
| tests/providers/ | Anthropic, OpenAI, Ollama providers |
| tests/scheduler/ | Job scheduling, parser |
| tests/security/ | Sanitizer, secrets, censor, vault, identity, trust |
| tests/skills/ | Skill discovery, built-in skills |
| tests/tools/ | Built-in tools (file, shell, web, calculator, etc.) |
| tests/unit/ | Focused unit tests, hardening tests |

## New Tests Added (This Session)

| File | Tests | Coverage |
|---|---|---|
| tests/agent/test_persona.py | 60 | PersonaConfig, PersonaManager, YAML round-trip, backup/rollback/diff |
| tests/agent/test_behavior.py | 129 | BehaviorLayer, IntentInterpreter, ResponseShaper, edge cases |
| tests/agent/test_hatching.py | 51 | HatchingManager, HatchingLog, state machine, edge cases |
| tests/cli/test_cli_hatch_persona.py | 33 | CLI: hatch, persona show/edit/reset/backups/diff/rollback |
| **Total new** | **273** | |

## Tests Fixed (This Session)

| Test | Issue | Fix |
|---|---|---|
| test_run_passes_system_prompt | Persona now appended to system prompt | Changed to `startswith` check |
| test_iteration_limit_fallback_* | Missing runtime attributes | Added new attributes to mock |
| test_aput/ahead_calls_check_url | _check_url now takes method arg | Updated expected call args |
| test_allowed_hosts_listed | Hosts collapsed into presets | Accept either form |
| test_wizard_discord_configuration | Discord hosts → preset | Accept either form |
| test_list (incus snapshots) | API uses query not info | Accept both |
| test_setup_calls_run_wizard | Missing Click option destination | Added "setup_api_key_env" dest |
