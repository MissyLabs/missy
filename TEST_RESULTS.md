# Missy Test Results

## Summary

- **Total tests**: 12,234
- **Passed**: 12,234
- **Failed**: 0
- **Skipped**: 14
- **Duration**: ~4.5 minutes

## Vision Tests

- **Total**: 289
- **Passed**: 289
- **Failed**: 0

### By Module

| Module | Tests | Status |
|--------|-------|--------|
| test_discovery.py | 18 | ALL PASS |
| test_capture.py | 13 | ALL PASS |
| test_sources.py | 27 | ALL PASS |
| test_pipeline.py | 6 | ALL PASS |
| test_scene_memory.py | 25 | ALL PASS |
| test_intent.py | 25 | ALL PASS |
| test_analysis.py | 20 | ALL PASS |
| test_doctor.py | 16 | ALL PASS |
| test_vision_tools.py | 23 | ALL PASS |
| test_edge_cases.py | 30 | ALL PASS |
| test_provider_format.py | 12 | ALL PASS |
| test_audit.py | 7 | ALL PASS |
| test_integration.py | 12 | ALL PASS |
| test_resilient_capture.py | 9 | ALL PASS |
| test_hardening.py | 32 | ALL PASS |
| test_burst_and_diff.py | 14 | ALL PASS |

## Previously-Failing Tests (Fixed in Session 2)

All 9 previously-failing tests were fixed:

| Test | Fix |
|------|-----|
| test_acpx_provider (6 tests) | Fixed command index assumptions (cmd[-1] for prompt) |
| test_hardening_s23 | Used spec= to properly mock provider without complete_with_tools |
| test_remaining_gaps (whisper) | Set sys.modules["numpy"] = None to block import |
| test_session10_coverage (ollama) | Patched missy.core.events.event_bus (correct import path) |

## Test Infrastructure

- Framework: pytest 8.4.1
- Python: 3.12.3
- Plugins: asyncio, cov, hypothesis
- Config: `pyproject.toml` (asyncio_mode=auto)
