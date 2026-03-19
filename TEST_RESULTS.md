# Missy Test Results

## Summary

- **Total tests**: ~19,271
- **Passed**: ~19,271
- **Failed**: 0
- **Skipped**: 14
- **Duration**: ~5 minutes 52 seconds
- **Last updated**: Session 15 (2026-03-19)

## Session 15 New Tests (2,319 tests, 17 files)

| Test File | Count | Coverage Area |
|-----------|-------|---------------|
| test_session15_circuit_breaker.py | 69 | Thread safety, TOCTOU, escalating backoff, stress tests |
| test_session15_filesystem.py | 92 | Symlink traversal, unicode, concurrent access, audit events |
| test_session15_voice_server.py | 97 | Constants, lifecycle, flood protection, sample rate clamping |
| test_session15_persona.py | 112 | Atomic save, backup/rollback/diff, audit JSONL, prune |
| test_session15_json_error_paths.py | 72 | Malformed JSON recovery across 5 modules |
| test_session15_runtime.py | 92 | Heredoc rewriting, capability modes, bus publish |
| test_session15_vault.py | 67 | Encryption internals, symlink rejection, atomic writes |
| test_session15_behavior.py | 215 | Tone/intent/urgency detection, response shaping, vision modes |
| test_session15_trust.py | 57 | Score bounds, weight edge cases, thread safety |
| test_session15_context.py | 108 | Token budget, history pruning, fresh tail, memory injection |
| test_session15_synthesizer.py | 84 | Relevance scoring, deduplication, truncation |
| test_session15_hatching.py | 106 | State lifecycle, step execution, resume/retry |
| test_session15_network.py | 142 | CIDR, DNS rebinding, domain wildcards, per-category hosts |
| test_session15_attention.py | 152 | 5 attention subsystems, focus continuity, keyword sets |
| test_session15_playbook.py | 83 | Pattern hashing, record/increment, thread safety |
| test_session15_consolidation.py | 92 | Threshold boundaries, fact extraction, keyword detection |
| test_session15_migrate.py | 63 | needs_migration edge cases, preset detection, atomic write |
| test_session15_manager.py | 127 | MCP name validation, permission checks, digest pinning |
| test_session15_learnings_done.py | 150 | Task type extraction, outcome scanning, compound tasks |
| test_session15_drift_identity.py | 57 | SHA-256 vectors, Ed25519 signing, JWK export |

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
| Agent Runtime | 2,000+ |
| Policy Engine | 550+ |
| Security | 600+ |
| Providers | 250+ |
| Channels | 550+ |
| Memory | 300+ |
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
| 14 | 17,234 | 497 | Edge case hardening across 12 files |
| 15 | ~19,271 | 2,319 | Circuit breaker fix, 17 new test files across 20+ modules |
