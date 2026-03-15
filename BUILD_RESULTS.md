# BUILD_RESULTS

- Timestamp: 2026-03-15
- Python version: 3.12.3

## Build Summary

| Metric | Value |
|--------|-------|
| Source files | 123 Python modules |
| Test files | 151 test files |
| Total tests | 5329 |
| Tests passing | 5329 |
| Tests failing | 0 |
| Tests skipped | 7 |
| Warnings | 1 (websockets deprecation) |
| Coverage | 99%+ |
| Lint errors | 0 |

## Architecture

```
missy/                          # 123 Python source files
  core/        - session, events, exceptions
  config/      - settings, YAML loading, hot-reload (watchdog)
  policy/      - network (CIDR/domain/per-category), filesystem, shell + facade
  gateway/     - PolicyHTTPClient
  providers/   - base, anthropic, openai, ollama, codex, registry, rate_limiter
  tools/       - base, registry, 15+ builtin tools (shell, file, web, browser,
                 atspi, x11, calculator, tts, code_evolve, incus, self_create_tool)
  skills/      - base, registry, 6 builtin skills
  plugins/     - base, loader
  scheduler/   - jobs, parser, manager (retry, timezone)
  memory/      - sqlite_store (FTS5, sessions, costs), resilient_store, json_store
  observability/ - audit_logger, otel_exporter
  security/    - sanitizer, secrets, censor, vault (ChaCha20), sandbox (Docker)
  channels/    - base, cli, discord (gateway, rest, voice, commands, config, threads),
                 webhook, voice (server, registry, pairing, presence, stt, tts, edge_client)
  agent/       - runtime (streaming, budget, recovery), circuit_breaker, context,
                 checkpoint, failure_tracker, done_criteria, learnings, prompt_patches,
                 sub_agent, approval, proactive, cost_tracker, watchdog, heartbeat,
                 code_evolution
  cli/         - main (60+ commands via click + rich), wizard, oauth, anthropic_auth
  mcp/         - manager, client (MCP server integration)
```

## CLI Commands (60+)

```
missy init, setup, doctor, ask, run, providers, skills, plugins
missy schedule add/list/pause/resume/remove
missy audit security/recent
missy discord status/probe/register-commands/audit
missy gateway start/status
missy vault set/get/list/delete
missy sessions cleanup/list/rename
missy approvals list
missy patches list/approve/reject
missy mcp list/add/remove
missy devices list/pair/unpair/status/policy
missy voice status/test
missy cost (--session)
missy recover (--abandon-all)
missy evolve list/approve/reject/show
```

## Subsystems Verified

- Policy engine: 3-layer default-deny (network, filesystem, shell)
- Providers: Anthropic, OpenAI, Ollama, Codex with fallback/tiering/rotation
- Channels: CLI, Discord (full WebSocket + REST + voice), Webhook, Voice
- Agent: Multi-step tool loop, streaming, budget enforcement, checkpointing
- Security: Input sanitizer, secrets detector, censor, vault, Docker sandbox
- Memory: SQLite FTS5 with sessions, costs, learnings
- Scheduler: APScheduler with retry, timezone, active hours
- Observability: JSONL audit log + OpenTelemetry
- MCP: Server management with auto-restart health checks
- Code evolution: Propose, test, apply, rollback with approval workflow
