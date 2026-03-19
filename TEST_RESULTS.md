# Missy Test Results

## Summary

- **Total tests**: 17,193
- **Passed**: 17,193
- **Failed**: 0
- **Skipped**: 14
- **Duration**: ~5 minutes 47 seconds
- **Last updated**: Session 14 (2026-03-19)

## Session 14 New Tests (456 tests, 8 files)

| Test File | Count | Coverage Area |
|-----------|-------|---------------|
| test_session14_summarizer_proactive.py | 40 | Summarizer tiers, proactive cooldown/templates/approval |
| test_session14_resilient_multi.py | 39 | Reconnect edge cases, multi-camera properties |
| test_session14_container.py | 48 | Docker sandbox lifecycle, security flags |
| test_session14_vision_modules.py | 72 | Shutdown, vision memory, config validator, benchmark |
| test_session14_cost_failure.py | 52 | Pricing, budget enforcement, failure strategy |
| test_session14_orientation_format.py | 42 | EXIF parsing, aspect ratio, provider formats |
| test_session14_analysis_intent_audit.py | 66 | Analysis prompts, intent classification, audit |
| test_session14_compaction_context.py | 20 | Chunk splitting, compaction threshold, fresh tail |
| test_session14_secrets_sanitizer.py | 37 | Secret patterns, redaction, injection detection |
| test_session14_tools_integration.py | 40 | Vision tools, scene memory, pipeline integration |

## Session 13 New Tests (921 tests, 14 files)

| Test File | Count | Coverage Area |
|-----------|-------|---------------|
| test_session13_resilient.py | 41 | Jitter backoff, thread safety, failure types |
| test_session13_health_monitor.py | 54 | Auto-save recovery, transactions, SQLite |
| test_session13_scene_memory.py | 58 | Collisions, eviction, phash, concurrency |
| test_session13_consolidation_approval.py | 81 | Threshold boundaries, concurrent approvals |
| test_session13_multi_camera.py | 43 | Closed handle, health monitor args, failures |
| test_session13_discovery_capture.py | 72 | Symlink cycles, sysfs, adaptive blank, cv2 |
| test_session13_message_bus.py | 44 | Worker lifecycle, fnmatch, sequence counter |
| test_session13_watchdog_ratelimiter.py | 64 | Log levels, bucket deduction, concurrency |
| test_session13_provider_audit.py | 72 | Provider format, audit fields, privacy |
| test_session13_hotreload_plan.py | 43 | File safety, backup fidelity, diff edges |
| test_session13_vault_trust.py | 53 | Key rotation, corrupt data, trust boundaries |
| test_session13_persona_behavior.py | 87 | Defaults, intent categories, vision guidelines |
| test_session13_hatching_checkpoint.py | 59 | First-run detection, checkpoint concurrency |
| test_session13_scheduler_memory.py | 68 | Retry boundary, FTS search, concurrent writes |
| test_session13_policy_gateway.py | 82 | CIDR IPv6, domain matching, DNS rebinding |
| test_session13_mcp_skills_plugins.py | 66 | Digest pinning, frontmatter, plugin manifest |
| test_session13_circuitbreaker_attention.py | 79 | Threshold boundaries, attention pipeline |

## Non-Vision Test Categories

| Category | Approximate Tests |
|----------|------------------|
| Agent Runtime | 1,100+ |
| Policy Engine | 300+ |
| Security | 400+ |
| Providers | 250+ |
| Channels | 450+ |
| Memory | 200+ |
| Scheduler | 150+ |
| Tools | 200+ |
| Config | 200+ |
| MCP | 170+ |
| Observability | 50+ |
| Plugins | 70+ |
| Vision | 2,000+ |

## Session History

| Session | Total Tests | New Tests | Key Changes |
|---------|-------------|-----------|-------------|
| 6 | 13,757 | 160 | Health persistence, perceptual hash, adaptive blank |
| 7 | 14,388 | 631 | Multi-camera, benchmark, memory usage, config validator |
| 8 | 14,711 | 323 | Orientation, EXIF parsing, doctor diagnostics |
| 9 | 14,895 | 168 | Thread safety, security hardening, context sanitization |
| 10 | 15,033 | 138 | Memory cleanup, thread safety fixes, timeout hardening |
| 11 | 15,296 | 235 | Code quality fixes, prompt injection tests, secrets detection |
| 12 | 15,527 | 164 | Property-based tests, cross-module integration, security |
| 13 | 16,737 | 1,210 | 6 code fixes, deep cross-codebase hardening |
