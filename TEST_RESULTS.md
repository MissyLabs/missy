# Missy Test Results

## Summary

- **Total tests**: 12,095
- **Passed**: 12,086
- **Failed**: 9 (pre-existing, non-vision)
- **Skipped**: 14
- **Duration**: ~4.5 minutes

## Vision Tests

- **Total**: 150
- **Passed**: 150
- **Failed**: 0

### By Module

| Module | Tests | Status |
|--------|-------|--------|
| test_discovery.py | 16 | ALL PASS |
| test_capture.py | 10 | ALL PASS |
| test_sources.py | 18 | ALL PASS |
| test_pipeline.py | 6 | ALL PASS |
| test_scene_memory.py | 19 | ALL PASS |
| test_intent.py | 18 | ALL PASS |
| test_analysis.py | 15 | ALL PASS |
| test_doctor.py | 10 | ALL PASS |

## Pre-existing Failures (Not Vision-Related)

| Test | Issue |
|------|-------|
| test_acpx_provider (6 tests) | ACPX provider module attribute issues |
| test_hardening_s23 | Agent tool loop edge case |
| test_remaining_gaps (whisper) | NumPy reimport warning in test |
| test_session10_coverage (ollama) | Module attribute patching issue |

## Test Infrastructure

- Framework: pytest 8.4.1
- Python: 3.12.3
- Plugins: asyncio, cov, hypothesis
- Config: `pyproject.toml` (asyncio_mode=auto)
