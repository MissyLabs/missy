# Missy Test Results

## Summary

- **Total tests**: 13,722
- **Passed**: 13,722
- **Failed**: 0
- **Skipped**: 14
- **Duration**: ~5 minutes

## Vision Tests

- **Total**: 808
- **Passed**: 808
- **Failed**: 0

### By Module (tests/vision/)

| Module | Tests | Status |
|--------|-------|--------|
| test_discovery.py | 18 | ALL PASS |
| test_discovery_edge_cases.py | 54 | ALL PASS |
| test_discovery_hardening.py | 19 | ALL PASS |
| test_capture.py | 13 | ALL PASS |
| test_capture_extended.py | 27 | ALL PASS |
| test_adaptive_blank.py | 18 | ALL PASS |
| test_sources.py | 27 | ALL PASS |
| test_sources_extended.py | 24 | ALL PASS |
| test_source_validation.py | 17 | ALL PASS |
| test_source_security.py | 15 | ALL PASS |
| test_pipeline.py | 6 | ALL PASS |
| test_pipeline_extended.py | 30 | ALL PASS |
| test_pipeline_edge_cases.py | 18 | ALL PASS |
| test_scene_memory.py | 25 | ALL PASS |
| test_scene_memory_extended.py | 19 | ALL PASS |
| test_scene_memory_stress.py | 21 | ALL PASS |
| test_perceptual_hash.py | 14 | ALL PASS |
| test_intent.py | 25 | ALL PASS |
| test_intent_extended.py | 44 | ALL PASS |
| test_analysis.py | 20 | ALL PASS |
| test_analysis_extended.py | 25 | ALL PASS |
| test_doctor.py | 16 | ALL PASS |
| test_vision_tools.py | 23 | ALL PASS |
| test_edge_cases.py | 30 | ALL PASS |
| test_provider_format.py | 12 | ALL PASS |
| test_audit.py | 7 | ALL PASS |
| test_audit_extended.py | 19 | ALL PASS |
| test_integration.py | 12 | ALL PASS |
| test_resilient_capture.py | 9 | ALL PASS |
| test_resilient_extended.py | 15 | ALL PASS |
| test_resilient_edge_cases.py | 13 | ALL PASS |
| test_hardening.py | 32 | ALL PASS |
| test_burst_and_diff.py | 14 | ALL PASS |
| test_security.py | 13 | ALL PASS |
| test_timeout_and_backoff.py | 10 | ALL PASS |
| test_failure_classification.py | 33 | ALL PASS |
| test_health_monitor.py | 51 | ALL PASS |
| test_health_persistence.py | 22 | ALL PASS |

### CLI Vision Tests

| Module | Tests | Status |
|--------|-------|--------|
| tests/cli/test_vision_cli.py | 14 | ALL PASS |

### Voice-Vision Integration

| Module | Tests | Status |
|--------|-------|--------|
| tests/channels/voice/test_voice_vision_integration.py | 11 | ALL PASS |

## Non-Vision Test Breakdown

| Directory | Tests | Status |
|-----------|-------|--------|
| tests/agent/ | 2,160 | ALL PASS |
| tests/unit/ | 1,825 | ALL PASS |
| tests/channels/ | 1,690 | ALL PASS |
| tests/security/ | 1,444 | ALL PASS |
| tests/tools/ | 1,199 | ALL PASS |
| tests/cli/ | 986 | ALL PASS |
| tests/providers/ | 683 | ALL PASS |
| tests/vision/ | 808 | ALL PASS |
| tests/integration/ | 531 | ALL PASS |
| tests/memory/ | 415 | ALL PASS |
| tests/policy/ | 375 | ALL PASS |
| tests/scheduler/ | 316 | ALL PASS |
| tests/core/ | 276 | ALL PASS |
| tests/config/ | 275 | ALL PASS |
| tests/gateway/ | 270 | ALL PASS |
| tests/skills/ | 183 | ALL PASS |
| tests/mcp/ | 133 | ALL PASS |
| tests/plugins/ | 115 | ALL PASS |
| tests/observability/ | 114 | ALL PASS |

## Session 6 New Tests (125 total)

| File | New Tests | Focus |
|------|-----------|-------|
| test_health_persistence.py | 22 | SQLite save/load, round-trip, merge |
| test_perceptual_hash.py | 14 | aHash, hamming distance, boundaries |
| test_adaptive_blank.py | 18 | Adaptive threshold, integration |
| test_discovery_hardening.py | 19 | Rediscovery, validation, resilient capture |
| test_resilient_edge_cases.py | 13 | Concurrency, permission, backoff |
| test_pipeline_edge_cases.py | 18 | 1x1, BGRA, quality, denoise, sharpen |
| test_scene_memory_stress.py | 21 | Eviction, concurrent, phash boundaries |
| test_otel.py | 12 | OtelExporter mock tracer, factory |
| test_plugin_hardening.py | 19 | Policy, audit events, singleton |

## Lint Status

- **ruff check**: All checks passed (0 errors)
- **ruff format**: Clean
