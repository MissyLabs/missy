# Test Results

## Date: 2026-03-18

## Summary

| Metric | Value |
|---|---|
| Total tests | 7830 |
| Passed | 7830 |
| Failed | 0 |
| Skipped | 17 |
| Duration | ~163s |

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

## New Tests Added (Sessions 1-3)

| File | Tests | Coverage |
|---|---|---|
| tests/agent/test_persona.py | 60 | PersonaConfig, PersonaManager, YAML round-trip, backup/rollback/diff |
| tests/agent/test_behavior.py | 129 | BehaviorLayer, IntentInterpreter, ResponseShaper, edge cases |
| tests/agent/test_hatching.py | 51 | HatchingManager, HatchingLog, state machine, edge cases |
| tests/cli/test_cli_hatch_persona.py | 33 | CLI: hatch, persona show/edit/reset/backups/diff/rollback |
| **Total new (sessions 1-3)** | **273** | |

## New Tests Added (Session 4 — Coverage Expansion)

| File | Tests | Coverage |
|---|---|---|
| tests/agent/test_heartbeat.py | 29 | HeartbeatRunner: fire, active hours, suppression, threading |
| tests/agent/test_watchdog.py | 19 | Watchdog: health checks, recovery, audit events, thread lifecycle |
| tests/cli/test_anthropic_auth.py | 24 | Token classification, storage, expiry, runtime resolution |
| tests/providers/test_anthropic_provider.py | 22 | Init, complete, tools, error handling, setup-token rejection |
| tests/providers/test_openai_provider.py | 19 | Init, complete, tools, streaming, error handling |
| tests/providers/test_ollama_provider.py | 18 | Init, complete, tools, streaming, error handling |
| tests/gateway/test_gateway_error_paths.py | 13 | URL validation, REST policy, graceful degradation |
| tests/channels/voice/test_pairing.py | 14 | PairingManager: initiate, approve, reject, unpair lifecycle |
| tests/mcp/test_mcp_manager.py (additions) | 9 | Injection blocking, unsafe names, permissions, digest mismatch |
| tests/agent/test_runtime_behavior_integration.py | 14 | Subsystem creation, graceful degradation, factory methods |
| **Total new (session 4)** | **181** | |

## New Tests Added (Session 5 — Hypothesis + Voice + Discord + Concurrent)

| File | Tests | Coverage |
|---|---|---|
| tests/security/test_sanitizer_hypothesis.py | 31 | Property-based: never-crash, truncation, zero-width, injection, base64, false positives |
| tests/security/test_secrets_hypothesis.py | 10 classes | Property-based: never-crash, known secrets, redaction, ordering, passwords |
| tests/channels/discord/test_voice_commands.py | 20 | !join/!leave/!say parsing, guards, errors, case sensitivity |
| tests/channels/discord/test_image_commands.py | 22 | is_image_attachment, find_latest_image, !analyze, !screenshot |
| tests/channels/voice/test_voice_server.py | 17 | Auth, pair request, heartbeat, audio pipeline, STT/TTS failure |
| tests/channels/voice/test_stt_whisper.py | 11 | Lifecycle, device resolution, transcription, abstract base |
| tests/channels/voice/test_tts_piper.py | 15 | PCM-to-WAV, env sanitization, lifecycle, model resolution, synthesis |
| tests/memory/test_memory_concurrent.py | 10 | Thread-safe writes, read/write, search, compaction, SQLite concurrent |
| tests/channels/discord/test_voice_utils.py | 22 | _clean_for_speech markdown stripping, _resample_pcm audio resampling |
| tests/core/test_exceptions.py | 15 | Exception hierarchy, PolicyViolationError, ApprovalRequiredError, bus topics |
| **Total new (session 5)** | **190** | |

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
