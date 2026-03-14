# TEST_RESULTS

- Timestamp: 2026-03-14
- Test framework: pytest 8.4.1
- Python: 3.12.3

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 2500 |
| Passed | 2500 |
| Failed | 0 |
| Warnings | 3 (deprecation) |
| Duration | ~37s |
| Source files | 123 |
| Test files | 69 |
| Coverage | 70% |

## Test Distribution

| Test Area | Tests | Files |
|-----------|-------|-------|
| Agent (runtime, circuit breaker, context, etc.) | ~180 | 10 |
| Channels (CLI, Discord, Webhook, Voice) | ~250 | 12 |
| CLI commands | ~160 | 3 |
| Config | ~30 | 2 |
| Core (session, events) | ~20 | 2 |
| Integration (policy enforcement) | ~30 | 1 |
| Memory (SQLite, resilient, sessions, costs) | ~80 | 5 |
| Observability (audit logger) | ~20 | 1 |
| Plugins (loader) | ~30 | 1 |
| Policy (network, filesystem, shell) | ~120 | 4 |
| Providers (anthropic, openai, ollama, codex, registry, rate limiter) | ~350 | 7 |
| Scheduler (jobs, parser, manager) | ~80 | 6 |
| Security (sanitizer, secrets, vault, censor, sandbox) | ~100 | 5 |
| Skills (registry, base, builtins) | ~80 | 3 |
| Tools (registry, base, builtins, shell_exec, calculator) | ~200 | 5 |
| Unit (Discord channel, config, gateway) | ~160 | 5 |

## pytest output
```
2500 passed, 3 warnings in 37.45s
```
