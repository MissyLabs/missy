# Missy Test Results

## Summary

- **Total tests**: 14,236
- **Passed**: 14,236
- **Failed**: 0
- **Skipped**: 14
- **Duration**: ~5 minutes

## Vision Tests

- **Total**: 1,269
- **Passed**: 1,269
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
| test_multi_camera.py | 79 | ALL PASS |
| test_benchmark.py | 94 | ALL PASS |
| test_memory_usage.py | 73 | ALL PASS |
| test_config_validator.py | 114 | ALL PASS |
| test_vision_memory.py | 67 | ALL PASS |
| test_dedup_and_timeout.py | 30 | ALL PASS |

### CLI Vision Tests (tests/cli/)

| Module | Tests | Status |
|--------|-------|--------|
| test_vision_cli.py | 14 | ALL PASS |
| test_vision_cli_extended.py | 22 | ALL PASS |

### Integration Tests

| Module | Tests | Status |
|--------|-------|--------|
| test_voice_vision_integration.py | 11 | ALL PASS |

## Non-Vision Test Categories

| Category | Approximate Tests |
|----------|------------------|
| Agent Runtime | 800+ |
| Policy Engine | 200+ |
| Security | 300+ |
| Providers | 200+ |
| Channels | 400+ |
| Memory | 150+ |
| Scheduler | 100+ |
| Tools | 200+ |
| Config | 150+ |
| MCP | 100+ |
| Observability | 50+ |
| Plugins | 50+ |

## Session History

| Session | Total Tests | New Tests | Key Changes |
|---------|-------------|-----------|-------------|
| 6 | 13,757 | 160 | Health persistence, perceptual hash, adaptive blank, rediscovery |
| 7 | 14,236 | 479 | Multi-camera, benchmark, memory usage, config validator, vision memory, dedup, timeout |
