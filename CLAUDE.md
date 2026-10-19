# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository. Keep this
file a map, not a history: put rationale in `docs/adr/` and reference material
in `docs/reference/`.

## Project overview

**Missy** is a security-first, self-hosted agentic AI assistant for Linux
(Python 3.11+). Every capability (shell, plugins, network) is off until enabled
in `~/.missy/config.yaml`, all outbound HTTP passes one policy gateway, and
every action is audited.

## Commands

```bash
pip install -e ".[dev]"            # extras: voice, otel, vector, vision, desktop, discord_voice
python3 -m pytest tests/ -n auto   # full suite (~24k tests); add -k NAME or a path to narrow
ruff check missy/ tests/ && ruff format missy/ tests/
```

CLI entry point: `missy` (`missy.cli.main:cli`). Full command list:
[docs/reference/cli.md](docs/reference/cli.md); `missy <group> --help` is
authoritative. Annotated config: [docs/reference/config-schema.md](docs/reference/config-schema.md).

## Architecture

```
missy CLI (cli/main.py) ── setup/wizard, config migrate + hot reload
  └─ gateway start ─► AgentRuntime (agent/runtime.py)
        ├─ input: InputSanitizer, SecretsDetector/Censor, PromptDriftDetector
        ├─ policy: PolicyEngine (network/filesystem/shell/REST) + ApprovalGate
        ├─ providers: ProviderRegistry (fallback, key rotation, round-robin accounts)
        ├─ tool loop: ToolRegistry + MCP tools, response guards, cost/budget
        ├─ memory: SQLiteMemoryStore (FTS5), checkpoints, graph/vector stores
        └─ observability: AuditLogger (signed, hash-chained JSONL), OTel
  channels: CLI | Discord | Webhook | Voice (edge nodes) | Screencast
  services in the gateway: scheduler, proactive triggers, heartbeat, web console/API, watchdog
```

| Area | Where | What to know |
|---|---|---|
| Policy | `missy/policy/` | Network (CIDR/domain/per-category hosts, presets), filesystem, shell allow-list (`unrestricted` skips matching), REST method+path rules. |
| HTTP gateway | `missy/gateway/client.py` | `PolicyHTTPClient` is the only outbound HTTP path; never follows redirects automatically. |
| Providers | `missy/providers/` | Anthropic, OpenAI, Ollama, Codex (OAuth), Claude Code CLI (`claude -p`; supersedes acpx). Fallback in `AgentRuntime._call_provider_with_fallback`; per-provider circuit breakers. |
| Agent loop | `missy/agent/runtime.py` | `_tool_loop` runs tools then regex response guards (`response_guards.py`). Guards share `max_guard_retries`; turns have `max_turn_seconds`; delegation is capped tree-wide (`sub_agent.MAX_TOTAL_SUB_RUNS`). |
| Security | `missy/security/` | Prompt-injection regexes (101) require a second signal, not bare phrases; secrets detector (54 patterns); vault (ChaCha20-Poly1305); Ed25519 identity. |
| Memory | `missy/memory/`, `agent/context.py` | Recalled history/memory is scanned and quarantined per entry. |
| Channels | `missy/channels/` | Discord gateway + REST governor; voice WebSocket server for `missy-edge`; screencast. |
| Scheduler | `missy/scheduler/` | APScheduler; `jobs.json` merged under `flock` with CLI writers; daily retention job. |
| MCP | `missy/mcp/` | Tools namespaced `server__tool`; digest pinning re-checked at call time. |
| Video/vision/desktop | `missy/tools/builtin/`, `missy/vision/` | ComfyUI over HTTP (`MISSY_COMFYUI_HOST`, falls back to `127.0.0.1:8199`); OBS/VTube/X11 tools are rate-limited and approval-gated. |

Detailed subsystem rationale (historical): [docs/adr/0000-subsystem-design-history.md](docs/adr/0000-subsystem-design-history.md).
Topic docs live in `docs/` (security, providers, discord, scheduler, operations, testing, threat model).

## Invariants: read before changing these areas

- **SQLite files:** never `os.open`/`os.close` an existing database file
  (closing any descriptor drops SQLite's POSIX locks process-wide). Use
  `missy.core.sqlite_files.ensure_private_db_file`. See ADR-0002.
- **Tests never touch the live `~/.missy`:** `tests/conftest.py` redirects
  `HOME` at import time. Don't bypass it; a parallel run once corrupted the
  live `checkpoints.db`.
- **Injection patterns:** add a second signal (an instruction to the assistant,
  a role delimiter, an evasion target), never a bare everyday phrase, and keep
  `automation/injection_precision.py` in the validation harness at ≤5% benign
  blocks and ≥95% attack detection. See ADR-0001.
- **Response guards:** completion/promise guards apply to action requests;
  on exhaustion keep the model's answer with a notice rather than substituting
  canned text. New guards must count toward the shared budget. See ADR-0003.
- **Gateway startup:** record each subsystem with `_startup_ok/_failed/_disabled`
  so failures surface in `missy gateway status` (reads
  `~/.missy/run/gateway_status.json`) and `/diagnostics`.
- **Deployment:** the live gateway (`systemctl --user restart missy-gateway`)
  imports this working tree, so the checked-out branch becomes production on
  restart. Restart only from `master`.
- **Cross-process files** (`jobs.json`, `devices.json`, `patches.json`,
  `trust.json`): read-modify-write under `flock` with reload-on-change.

## Default file locations (`~/.missy/`)

| Purpose | Path |
|---|---|
| Config / backups | `config.yaml`, `config.d/` |
| Audit log | `audit.jsonl` (rotated, 5 × 50 MB) |
| Memory, checkpoints, graph | `memory.db`, `checkpoints.db`, `graph_memory.db` |
| Jobs, devices, MCP | `jobs.json`, `devices.json`, `mcp.json` |
| Secrets | `secrets/` (vault, web console key, OAuth tokens) |
| Identity | `identity.pem` |
| Patches, playbook, persona | `patches.json`, `playbook.json`, `persona.yaml` |
| Gateway status | `run/gateway_status.json` |
| Media | `captures/`, `videos/` |
| Quarantined/backups | `quarantine/`, `backups/` |

## Tests

`tests/` mirrors `missy/` (agent, api, channels, cli, config, core, gateway,
integration, mcp, memory, observability, policy, providers, scheduler,
security, skills, tools, unit, vision). Coverage gate 90%; per-test timeout
300 s. Prefer injected clocks over `time.sleep`, and lower-bound (`>=`)
assertions over exact counts of detection patterns.

## Related

- Docs site: https://missylabs.github.io/ (source `/home/missy/missylabs.github.io/`).
- Voice edge client: `/home/missy/missy-edge/`; protocol notes in
  [docs/reference/missy-edge.md](docs/reference/missy-edge.md). Changing
  `missy/channels/voice/server.py` requires matching changes there.
- Validation harness: `/home/missy/missy-validation-harness/`.
