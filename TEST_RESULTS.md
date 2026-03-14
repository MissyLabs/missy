# TEST_RESULTS

- Timestamp: 2026-03-14
- Test framework: pytest 8.4.1
- Python: 3.12.3

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 4290 |
| Passed | 4290 |
| Failed | 0 |
| Skipped | 7 |
| Warnings | 5 |
| Duration | ~99s |
| Source files | 123 |
| Test files | 114 |
| Coverage | 98% |

## Test Distribution

| Test Area | Tests | Files |
|-----------|-------|-------|
| Agent (runtime, circuit breaker, context, done criteria, learnings, evolution, proactive) | ~520 | 18 |
| Channels (CLI, Discord, Webhook, Voice, voice registry) | ~540 | 22 |
| CLI commands | ~185 | 5 |
| Config | ~30 | 2 |
| Core (session, events, exceptions) | ~30 | 3 |
| Integration (policy enforcement, end-to-end) | ~107 | 2 |
| Memory (SQLite, resilient, sessions, costs) | ~120 | 6 |
| Observability (audit logger, OTEL) | ~40 | 3 |
| Plugins (loader) | ~30 | 1 |
| Policy (network, filesystem, shell) | ~120 | 4 |
| Providers (anthropic, openai, ollama, codex, registry, rate limiter) | ~400 | 8 |
| Scheduler (jobs, parser, manager) | ~100 | 7 |
| Security (sanitizer, secrets, vault, censor, sandbox, fuzz) | ~216 | 6 |
| Skills (registry, base, builtins) | ~80 | 3 |
| Tools (registry, base, builtins, shell, incus, atspi, x11, file ops) | ~420 | 12 |
| Unit (Discord channel, config, gateway, infrastructure) | ~212 | 8 |

## Session 9 Additions (323 new tests)

- **116 security fuzz tests**: unicode evasion, encoding bypass, large input stress, secret format variations, vault corruption recovery, hypothesis property-based invariants
- **48 rate limiter stress tests**: concurrent acquire, token exhaustion, refill accuracy, burst handling, thread safety, hypothesis properties
- **42 gateway/watchdog coverage tests**: PUT/apost/aclose methods, context managers, category forwarding, URL validation, watchdog recovery/threshold/audit errors
- **37 proactive manager tests**: schedule loop stop, file handler, threshold polling, observer stop exception, cooldown, confirmation gate
- **18 voice registry tests**: atomic write failure, purge audio stat/unlink errors, non-file entry handling
- **77 end-to-end integration tests**: security pipeline, policy enforcement chain, memory lifecycle, circuit breaker, cost tracker, tool registry, scheduler, audit events, config reload, multi-layer security

## pytest output
```
4290 passed, 7 skipped, 5 warnings in 98.51s
```
