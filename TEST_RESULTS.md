# TEST_RESULTS

- Timestamp: 2026-03-15
- Test framework: pytest 8.4.1
- Python: 3.12.3

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 4891 |
| Passed | 4891 |
| Failed | 0 |
| Skipped | 7 |
| Warnings | 7 |
| Duration | ~134s |
| Source files | 123 |
| Test files | 137 |
| Statements | 11550 |
| Missed lines | 108 |
| Coverage | 99.06% |

## Test Distribution

| Test Area | Tests | Files |
|-----------|-------|-------|
| Agent (runtime, circuit breaker, context, done criteria, learnings, evolution, sub-agent, approval, prompt patches, proactive) | ~743 | 27 |
| Channels (CLI, Discord, Webhook, Voice, voice registry, input validation) | ~606 | 25 |
| CLI commands | ~192 | 6 |
| Config | ~33 | 2 |
| Core (session, events, exceptions) | ~30 | 3 |
| Integration (policy enforcement, end-to-end) | ~107 | 2 |
| MCP (client, manager) | ~54 | 2 |
| Memory (SQLite, resilient, sessions, costs) | ~120 | 6 |
| Observability (audit logger, OTEL) | ~40 | 3 |
| Plugins (loader) | ~30 | 1 |
| Policy (network, filesystem, shell) | ~120 | 4 |
| Providers (anthropic, openai, ollama, codex, registry, rate limiter) | ~400 | 8 |
| Scheduler (jobs, parser, manager) | ~100 | 7 |
| Security (sanitizer, secrets, vault, censor, sandbox, fuzz, X11 injection) | ~231 | 7 |
| Skills (registry, base, builtins) | ~80 | 3 |
| Tools (registry, base, builtins, shell, incus, atspi, x11, browser, file ops, security edges) | ~530 | 14 |
| Unit (Discord channel, config, gateway, infrastructure, coverage gaps) | ~233 | 10 |

## Session 10 Additions (86 new tests)

- **46 ruff lint errors fixed**: import ordering, SIM103/105/117, unused variables
- **7 CLI coverage tests**: proactive callback success/fallback, doctor watchdog/voice/checkpoint exceptions
- **22 browser tools tests**: display setup, session start/close, page helper, registry edge cases
- **19 Discord voice tests**: start paths, join channel-id-not-found, start_listening body, watchdog exception, speech handler branches, resample boundary
- **17 proactive tests**: watchdog import success path, threshold loop inner break, audit publish+exception, file handler with mocked watchdog
- **21 remaining gap tests**: heartbeat loop fire, overnight active hours, webhook log_message, config watcher OSError, Discord voice callback closure, DM policy fallthrough, mention fallback, gateway heartbeat loop

## Session 9 Additions (412 new tests)

- **116 security fuzz tests**: unicode evasion, encoding bypass, large input stress, secret format variations, vault corruption recovery, hypothesis property-based invariants
- **48 rate limiter stress tests**: concurrent acquire, token exhaustion, refill accuracy, burst handling, thread safety, hypothesis properties
- **42 gateway/watchdog coverage tests**: PUT/apost/aclose methods, context managers, category forwarding, URL validation, watchdog recovery/threshold/audit errors
- **37 proactive manager tests**: schedule loop stop, file handler, threshold polling, observer stop exception, cooldown, confirmation gate
- **18 voice registry tests**: atomic write failure, purge audio stat/unlink errors, non-file entry handling
- **77 end-to-end integration tests**: security pipeline, policy enforcement chain, memory lifecycle, circuit breaker, cost tracker, tool registry, scheduler, audit events, config reload, multi-layer security
- **54 incus tools coverage tests**: unreachable fallbacks, network attach/detach, volume operations, profile set/edit, project config, device validation, copy/move flags
- **35 targeted coverage gap tests**: Discord REST error paths, config api_keys fallback, filesystem policy ValueError, skills registry audit, sandbox exceptions, voice command guards

## pytest output
```
4489 passed, 7 skipped, 7 warnings in 102.32s
```
