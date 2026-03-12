# Missy — Implementation Gaps

Gap analysis comparing `prompt.md` requirements against the current implementation.
Updated: 2026-03-12.

---

## Missing CLI Commands (prompt.md lines 578-596)

| Required Command | Status | Notes |
|---|---|---|
| `missy gateway start` | **Implemented** | Full WebSocket gateway server (`gateway start --host/--port`) |
| `missy gateway status` | **Implemented** | Shows runtime subsystem health |
| `missy doctor` | **Implemented** | Full health-check command with subsystem probes |

All three originally-missing CLI commands are now present.

---

## Missing Documentation Files (prompt.md lines 388-410)

These docs were explicitly required but still do not exist:

- `ARCHITECTURE.md`
- `CONFIG_REFERENCE.md`
- `PROVIDERS.md`
- `SCHEDULER.md`
- `SKILLS_AND_PLUGINS.md`
- `MEMORY_AND_PERSISTENCE.md`
- `TESTING.md`
- `TROUBLESHOOTING.md`
- `docs/implementation/module-map.md`
- `docs/implementation/agent-loop.md`
- `docs/implementation/policy-engine.md`
- `docs/implementation/network-client.md`
- `docs/implementation/provider-abstraction.md`
- `docs/implementation/scheduler-execution.md`
- `docs/implementation/audit-events.md`
- `docs/implementation/persistence-schema.md`
- `docs/implementation/manifest-schema.md`

Only `docs/THREAT_MODEL.md` and `docs/implementation/discord-channel.md` exist.
Status: **Remaining** — 17 of 17 documentation files still absent.

---

## Missing Features

| Feature | Original Status | Current Status | Notes |
|---|---|---|---|
| `mypy` in dev dependencies | Missing | **Remaining** | Still not in `pyproject.toml` |
| Systemd / service-mode | Missing | **Remaining** | No `.service` files |
| Per-category network allowlists | Missing | **Remaining** | Still a single unified allowlist |
| Approval model for high-risk actions | Missing | **Implemented** | `missy/agent/approval.py`; `missy approvals list` CLI |
| Safe-chat-only / no-tools mode | Missing | **Partial** | Per-device `policy_mode` on voice nodes; not per-CLI-session |
| Separate secrets directory | Missing | **Implemented** | `~/.missy/secrets/` created by `init` and `setup`; `vault` subgroup |
| Bundled first-party skills | Missing | **Remaining** | Still only a `calculator` tool; no bundled skills |
| Discord thread-aware reply handling | Unclear | **Remaining** | Not confirmed implemented |
| Discord typing indicators | Unclear | **Remaining** | Not confirmed implemented |
| Discord attachment/media policy gating | Unclear | **Remaining** | Not confirmed implemented |
| Per-provider policy model | Missing | **Remaining** | Still only `timeout` per-provider; no enable/disable or endpoint binding |
| Discord example configs in docs | Missing | **Remaining** | Still undocumented |
| Weekly schedule `at` keyword | Discrepancy | **Implemented** | `_WEEKLY_PATTERN` accepts optional `at` keyword |

---

## New Capabilities Added Since Original Gap Analysis

The following were not tracked in the original gaps document but have since been implemented:

| Feature | Where |
|---|---|
| Multi-step agentic loop with tool invocation | `missy/agent/runtime.py` — `_tool_loop()` |
| Built-in tools (file_read, file_write, file_delete, list_files, shell_exec, web_fetch) | `missy/tools/builtin/` |
| MCP integration | `missy/mcp/client.py`, `missy/mcp/manager.py`; `missy mcp list/add/remove` |
| Circuit breaker | `missy/agent/circuit_breaker.py` |
| Context window management (7-tier budget) | `missy/agent/context.py` |
| Resilient memory with failover | `missy/memory/resilient.py` |
| Watchdog / subsystem health monitor | `missy/agent/watchdog.py` |
| DONE criteria engine | `missy/agent/done_criteria.py` |
| Cross-task learning | `missy/agent/learnings.py` |
| Prompt self-tuning / patch system | `missy/agent/prompt_patches.py`; `missy patches list/approve/reject` |
| Tiered model routing | `missy/providers/registry.py` — `ModelRouter` |
| Multiple API key rotation | `missy/providers/registry.py` — `rotate_key()` |
| Execution approval flow | `missy/agent/approval.py`; `missy approvals list` |
| Vault / encrypted secrets (ChaCha20-Poly1305) | `missy/security/vault.py`; `missy vault set/get/list/delete` |
| Outbound secret censoring | `missy/security/censor.py` |
| Raw cron expression support | `missy/scheduler/parser.py` |
| Timezone support in scheduler | `missy/scheduler/parser.py` — `tz` parameter |
| Job retry on failure | `missy/scheduler/jobs.py`, `missy/scheduler/manager.py` |
| One-shot future-dated jobs | `missy/scheduler/parser.py` — `DateTrigger` via `_AT_PATTERN` |
| Active-hours window | `missy/scheduler/jobs.py`, `missy/scheduler/manager.py` |
| SQLite FTS memory search | `missy/memory/sqlite_store.py` — `search()` with FTS5 |
| Session compaction | `missy/memory/store.py` — `compact_session()` |
| Sub-agent decomposition | `missy/agent/sub_agent.py` |
| Heartbeat system | `missy/agent/heartbeat.py` |
| Webhook channel | `missy/channels/webhook.py` |
| OpenTelemetry | `missy/observability/otel.py` |
| Config hot-reload | `missy/config/hotreload.py` |
| Agent-authored persistent custom tools | `missy/tools/builtin/self_create_tool.py` |
| Failure alert routing for scheduled jobs | `missy/scheduler/manager.py` |
| Session cleanup CLI | `missy sessions cleanup [--dry-run] [--before DAYS]` |
| Onboarding wizard | `missy/cli/wizard.py`; `missy setup` |
| OpenAI OAuth PKCE flow | `missy/cli/oauth.py` |
| Anthropic setup-token flow (with ToS warning) | `missy/cli/anthropic_auth.py` |
| Voice channel | `missy/channels/voice/` — WebSocket server, STT/TTS, device registry, pairing, presence; `missy devices` and `missy voice` CLI groups |

---

## Remaining Gaps Summary

| Priority | Gap |
|---|---|
| High | 17 missing documentation files |
| Medium | Per-category network allowlists |
| Medium | Per-session capability modes (safe-chat-only, no-tools) — voice nodes only, not CLI/Discord sessions |
| Medium | systemd service files |
| Low | `mypy` in dev dependencies |
| Low | Bundled first-party skills |
| Low | Discord UX gaps (thread replies, typing indicators, attachment gating) |
| Low | Per-provider enable/disable flag and endpoint binding |
| Low | Discord example configurations in docs |
