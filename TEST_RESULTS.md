# TEST_RESULTS

- Timestamp: 2026-03-14
- Test framework: pytest 8.4.1
- Python: 3.12.3

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 3675 |
| Passed | 3675 |
| Failed | 0 |
| Skipped | 3 |
| Warnings | 4 (deprecation) |
| Duration | ~44s |
| Source files | 123 |
| Test files | 99 |
| Coverage | 96% |

## Test Distribution

| Test Area | Tests | Files |
|-----------|-------|-------|
| Agent (runtime, circuit breaker, context, done criteria, learnings, evolution) | ~480 | 17 |
| Channels (CLI, Discord, Webhook, Voice) | ~520 | 21 |
| CLI commands | ~185 | 5 |
| Config | ~30 | 2 |
| Core (session, events, exceptions) | ~30 | 3 |
| Integration (policy enforcement) | ~30 | 1 |
| Memory (SQLite, resilient, sessions, costs) | ~120 | 6 |
| Observability (audit logger, OTEL) | ~40 | 3 |
| Plugins (loader) | ~30 | 1 |
| Policy (network, filesystem, shell) | ~120 | 4 |
| Providers (anthropic, openai, ollama, codex, registry, rate limiter) | ~350 | 7 |
| Scheduler (jobs, parser, manager) | ~100 | 7 |
| Security (sanitizer, secrets, vault, censor, sandbox) | ~100 | 5 |
| Skills (registry, base, builtins) | ~80 | 3 |
| Tools (registry, base, builtins, shell, incus, atspi, x11, file ops) | ~420 | 12 |
| Unit (Discord channel, config, gateway, infrastructure) | ~170 | 7 |

## pytest output
```
3675 passed, 3 skipped, 4 warnings in 44.27s
```
