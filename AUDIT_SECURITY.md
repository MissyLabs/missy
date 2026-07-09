# AUDIT_SECURITY

- Timestamp: 2026-07-08 21:40:04

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- `missy/api/run_stream.py`: added a bus subscription on `agent.run.complete`
/home/missy/missy/LAST_SESSION_SUMMARY.md:8:  (`_SUMMARY_TOPIC`) that captures `resolved_provider`/`tools_used`/`cost`
/home/missy/missy/LAST_SESSION_SUMMARY.md:13:- `missy/api/operator_controls.py`: added `scheduler.remove_job`, a third
/home/missy/missy/LAST_SESSION_SUMMARY.md:14:  confirmation-gated scheduler control (destructive-flagged) alongside
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:- `missy/api/server.py`: new routes `GET/POST /api/v1/scheduler/jobs`,
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:  `DELETE /api/v1/scheduler/jobs/{id}` (delegates to `scheduler.remove_job`),
/home/missy/missy/LAST_SESSION_SUMMARY.md:22:  schema migration); `cleanup()` now exempts pinned turns via
/home/missy/missy/LAST_SESSION_SUMMARY.md:27:- `missy/api/web_console.py`: new **Scheduled Jobs** panel (list/create/
/home/missy/missy/LAST_SESSION_SUMMARY.md:29:  console's completion handler now shows a provider/tools/cost summary
/home/missy/missy/LAST_SESSION_SUMMARY.md:32:  (`tests/api/test_run_stream.py`), +13 unit tests (new
/home/missy/missy/LAST_SESSION_SUMMARY.md:36:  `AUDIT_CONNECTIVITY.md`, `TEST_EDGE_CASES.md` rewritten for this session's
/home/missy/missy/LAST_SESSION_SUMMARY.md:37:  changes (the security/connectivity audits are now hand-written summaries
/home/missy/missy/LAST_SESSION_SUMMARY.md:39:  session).
/home/missy/missy/LAST_SESSION_SUMMARY.md:49:python3 -m pytest tests/api/test_run_stream.py -q
/home/missy/missy/LAST_SESSION_SUMMARY.md:74:login → CSRF → scheduler job create/list/remove (with/without confirmation)
/home/missy/missy/LAST_SESSION_SUMMARY.md:76:separately verified the run cost/provider/tools_used enrichment through both
/home/missy/missy/LAST_SESSION_SUMMARY.md:79:Full-repo `python3 -m pytest -q` was run before ending the session; see
/home/missy/missy/LAST_SESSION_SUMMARY.md:84:- No dashboard-wide cost/usage panel (aggregate spend) — cost is now visible
/home/missy/missy/LAST_SESSION_SUMMARY.md:87:- Safe controls cover providers, scheduler (pause/resume/remove), and now
/home/missy/missy/LAST_SESSION_SUMMARY.md:88:  memory turns, but not tools/skills/plugins/Discord/voice/vision/webhooks/
/home/missy/missy/LAST_SESSION_SUMMARY.md:89:  secrets/config.
/home/missy/missy/LAST_SESSION_SUMMARY.md:90:- No dashboard-wide offline/reconnecting banner (only the run console has
/home/missy/missy/LAST_SESSION_SUMMARY.md:97:Add a dashboard-wide cost/usage panel backed by
/home/missy/missy/LAST_SESSION_SUMMARY.md:98:`SQLiteMemoryStore.get_total_costs()`, then extend `operator_controls.py`
/home/missy/missy/LAST_SESSION_SUMMARY.md:99:with tool/skill enable-disable controls to keep closing the "full
/home/missy/missy/LAST_SESSION_SUMMARY.md:100:bot-control coverage" gap.
/home/missy/missy/examples/systemd/README.md:8:## Prerequisites
/home/missy/missy/examples/systemd/README.md:14:- A valid `config.yaml` must exist at `~/.missy/config.yaml` (or wherever
/home/missy/missy/examples/systemd/README.md:16:- Any required API keys must be set as environment variables. You can use a
/home/missy/missy/examples/systemd/README.md:43:the instance name becomes the `%i` substitution variable, which is used
/home/missy/missy/examples/systemd/README.md:51:sudo systemctl enable --now missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:54:This enables the service to start on boot and starts it immediately.
/home/missy/missy/examples/systemd/README.md:76:If you change `config.yaml` or environment variables, restart the service:
/home/missy/missy/examples/systemd/README.md:91:sudo systemctl disable missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:102:| `NoNewPrivileges=true` | Prevents the process from gaining new privileges (e.g. via setuid). |
/home/missy/missy/examples/systemd/README.md:106:| `ReadWritePaths=...` | Grants write access to `~/.missy` (for audit logs, jobs, memory) and `~/workspace` (for agent output). |
/home/missy/missy/examples/systemd/README.md:120:- Check that `config.yaml` includes the required domains in `allowed_domains` or `allowed_hosts`.
/home/missy/missy/examples/systemd/README.md:121:- Verify that API key environment variables are set in the service environment.
/home/missy/missy/examples/systemd/README.md:124:- Check that `discord.enabled: true` is set in `config.yaml`.
/home/missy/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:16:| A8 | Per-channel identity cascade | not_started | Persona config extension remains. |
/home/missy/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:19:| A11 | Raw-stream JSONL diagnostics | not_started | A1 exposes `raw_stream_callback`; observability module remains. |
/home/missy/missy/HUMANIZE_STATUS.md:27:| H_A | Variable response timing and typing pauses | not_started | Depends on A7 channel block flushing. |
/home/missy/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy/HUMANIZE_STATUS.md:35:| H_I | Mood state with decay | not_started | First humanize implementation target in sessions 8-9. |
/home/missy/missy/HUMANIZE_STATUS.md:39:- Initialized required loop tracking documents.
/home/missy/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
/home/missy/missy/HUMANIZE_STATUS.md:43:- Expanded `tests/agent/test_runtime_streaming.py`.
/home/missy/missy/HUMANIZE_STATUS.md:45:- Session 2 added the A2 layered tool policy pipeline with profile bundles, group expansion, glob matching, inline `-tool` denies, `alsoAllow`, fail-warning unknown allowlists, and structured trace records.
/home/missy/missy/HUMANIZE_STATUS.md:46:- Session 2 wired `AgentRuntime._get_tools()` to resolve tools through the pipeline and record `_last_tool_policy_decision` for audit/debugging.
/home/missy/missy/HUMANIZE_STATUS.md:47:- Session 2 added `tests/policy/test_tool_policy_pipeline.py` and runtime coverage for policy decisions in `tests/agent/test_runtime_streaming.py`.
/home/missy/missy/HUMANIZE_STATUS.md:48:- Session 3 added config parsing for `tools.*`, `tools.byProvider`, `tools.byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools`.
/home/missy/missy/HUMANIZE_STATUS.md:49:- Session 3 added `build_configured_tool_policy_layers()` and `collect_tool_policy_groups()` so runtime policy resolution now consumes YAML-backed provider/global/agent/sandbox/subagent layers.
/home/missy/missy/HUMANIZE_STATUS.md:50:- Session 3 routed parsed tool policies into CLI-created runtimes for ask/run/gateway/API paths and documented the YAML surface in `docs/configuration.md`.
/home/missy/missy/HUMANIZE_STATUS.md:51:- Session 3 added config, policy-pipeline, and runtime tests for those surfaces, then verified the full test suite and full-repo ruff.
/home/missy/missy/HUMANIZE_STATUS.md:55:1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription` where Missy's providers expose stream events, not only the simple `run_stream()` path.
/home/missy/missy/HUMANIZE_STATUS.md:56:2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
/home/missy/missy/HUMANIZE_STATUS.md:58:4. Add channel/group policy sources on top of the A2 pipeline when Discord/CLI/Web channel identity context is available.
/home/missy/missy/install.sh:29:    echo "Error: Python 3.11+ is required." >&2
/home/missy/missy/install.sh:37:    echo "Error: git is required." >&2
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260709-004527
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy/README.md:31:- **Model tiers** — `fast_model` for quick tasks, `premium_model` for complex reasoning, auto-routed by ModelRouter
/home/missy/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy/README.md:42:- **Cost tracking** — per-session budget caps with `max_spend_usd`
/home/missy/missy/README.md:44:- **Checkpoint recovery** — WAL-mode SQLite checkpointing; `missy recover` resumes incomplete sessions
/home/missy/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy/README.md:46:- **Watchdog** — background subsystem health monitoring with degradation reporting
/home/missy/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy/README.md:53:- **REST API** — Agent-as-a-Service endpoint (`missy api start`) with loopback binding, API key auth, rate limiting
/home/missy/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy/README.md:62:- **Encrypted vault** — ChaCha20-Poly1305 with atomic key creation, `vault://` config references
/home/missy/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy/README.md:74:- **Webhooks** — HTTP ingress with HMAC auth, rate limiting, payload validation
/home/missy/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy/README.md:76:- **Screencast** — browser-based screen capture channel with token authentication and session management
/home/missy/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy/README.md:104:- **Config migration** — auto-upgrades old configs to preset format on startup, backs up first
/home/missy/missy/README.md:105:- **Config plan/rollback** — `missy config diff`, `missy config rollback`, automatic backups (max 5)
/home/missy/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy/README.md:112:- **Cost tracking** — per-session spend monitoring with configurable caps
/home/missy/missy/README.md:122:This clones to `~/.local/share/missy`, creates a venv, installs, and symlinks `missy` into `~/.local/bin`. Requires Python 3.11+ and git.
/home/missy/missy/README.md:124:## Quick Start
/home/missy/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy/README.md:134:missy run    # interactive session
/home/missy/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy/README.md:164:pip install -e ".[discord_voice]" # discord.py[voice] + voice recv; requires system ffmpeg
/home/missy/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy/README.md:211:Missy uses `~/.missy/config.yaml`. API keys go in environment variables or the encrypted vault — never in the config file. Old configs are auto-migrated on startup.
/home/missy/missy/README.md:214:config_version: 2
/home/missy/missy/README.md:217:  default_deny: true
/home/missy/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy/README.md:223:  rest_policies:                 # L7 HTTP method + path controls
/home/missy/missy/README.md:234:  enabled: false
/home/missy/missy/README.md:237:providers:
/home/missy/missy/README.md:238:  anthropic:
/home/missy/missy/README.md:239:    name: anthropic
/home/missy/missy/README.md:246:  enabled: false
/home/missy/missy/README.md:257:See the [full configuration reference](https://missylabs.github.io/configuration/reference/) for all options.
/home/missy/missy/README.md:266:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy/README.md:267:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy/README.md:268:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy/README.md:269:missy providers list                # List providers and availability
/home/missy/missy/README.md:270:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy/README.md:271:missy doctor                        # System health check
/home/missy/missy/README.md:277:# Security & audit
/home/missy/missy/README.md:278:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy/README.md:279:missy audit security                # Policy violations
/home/missy/missy/README.md:283:missy config backups                # List config backups
/home/missy/missy/README.md:284:missy config diff                   # Diff vs latest backup
/home/missy/missy/README.md:285:missy config rollback               # Restore from backup
/home/missy/missy/README.md:286:missy presets list                  # Show built-in network presets
/home/missy/missy/README.md:289:missy discord status | probe | register-commands | audit
/home/missy/missy/README.md:293:missy devices list | pair | unpair | status | policy
/home/missy/missy/README.md:295:# MCP & skills
/home/missy/missy/README.md:297:missy skills                        # List registered skills
/home/missy/missy/README.md:298:missy skills scan                   # Discover SKILL.md files
/home/missy/missy/README.md:301:missy vision devices | capture | inspect | review | doctor
/home/missy/missy/README.md:302:missy vision health | benchmark | validate | memory
/home/missy/missy/README.md:317:missy sessions list | rename | cleanup
/home/missy/missy/README.md:344:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy/README.md:354:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy/README.md:370:| [Getting Started](https://missylabs.github.io/getting-started/) | 5 | Install, quickstart, wizard, first conversation |
/home/missy/missy/README.md:371:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy/README.md:373:| [Architecture](https://missylabs.github.io/architecture/) | 10 | Runtime, context, circuit breaker, progress, playbook, sleep mode, synthesizer, attention, message bus |
/home/missy/missy/README.md:376:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy/README.md:377:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy/README.md:378:| [Missy Edge](https://missylabs.github.io/edge/) | 6 | Hardware, Pi setup, pairing, config, wake word |
/home/missy/missy/README.md:384:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy/README.md:392:├── agent/           Runtime, circuit breaker, context, playbook, consolidation,
/home/missy/missy/README.md:393:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/README.md:396:├── channels/        CLI, Discord, webhooks, voice (WebSocket), screencast (browser)
/home/missy/missy/README.md:398:├── config/          YAML settings, hot-reload, migration, plan/rollback
/home/missy/missy/README.md:401:├── mcp/             MCP server manager, health checks, digest pinning
/home/missy/missy/README.md:404:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/README.md:405:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy/README.md:406:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy/README.md:409:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy/README.md:410:├── plugins/         Security-gated external plugin loader
/home/missy/missy/README.md:411:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy/README.md:412:└── vision/          Camera discovery, capture, analysis, scene memory, health
/home/missy/missy/CLAUDE.md:3:This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
/home/missy/missy/CLAUDE.md:7:**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. Production-grade agent platform with strict security controls, policy enforcement, and full auditability. Python 3.11+.
/home/missy/missy/CLAUDE.md:19:pip install -e ".[discord_voice]" # adds discord.py[voice] + voice recv; requires system ffmpeg
/home/missy/missy/CLAUDE.md:23:python3 -m pytest tests/unit/test_policy_engine.py -v     # single file
/home/missy/missy/CLAUDE.md:36:Secure-by-default: all capabilities (shell, plugins, network) are disabled until explicitly enabled in `~/.missy/config.yaml`.
/home/missy/missy/CLAUDE.md:42:  → missy setup (wizard.py + oauth.py + anthropic_auth.py)
/home/missy/missy/CLAUDE.md:43:  → Config migration (config/migrate.py) — auto-upgrades old configs to preset format
/home/missy/missy/CLAUDE.md:44:  → load_config() (config/settings.py) + ConfigWatcher (config/hotreload.py)
/home/missy/missy/CLAUDE.md:48:       ├─ PolicyEngine (policy/engine.py) + RestPolicy (policy/rest_policy.py)
/home/missy/missy/CLAUDE.md:49:       ├─ AgentIdentity (security/identity.py) — Ed25519 keypair, signs audit events
/home/missy/missy/CLAUDE.md:50:       ├─ TrustScorer (security/trust.py) — 0-1000 reliability tracking per tool/provider
/home/missy/missy/CLAUDE.md:51:       ├─ CircuitBreaker (agent/circuit_breaker.py)
/home/missy/missy/CLAUDE.md:57:       ├─ ProviderRegistry + ModelRouter (providers/registry.py)
/home/missy/missy/CLAUDE.md:58:       ├─ RateLimiter (providers/rate_limiter.py)
/home/missy/missy/CLAUDE.md:59:       ├─ PolicyHTTPClient (gateway/client.py) + InteractiveApproval (agent/interactive_approval.py)
/home/missy/missy/CLAUDE.md:60:       ├─ ToolRegistry (tools/registry.py) + built-in tools
/home/missy/missy/CLAUDE.md:62:       ├─ SkillDiscovery (skills/discovery.py) — SKILL.md dynamic skill loading
/home/missy/missy/CLAUDE.md:68:       ├─ ApprovalGate (agent/approval.py)
/home/missy/missy/CLAUDE.md:72:       ├─ CostTracker (agent/cost_tracker.py) — per-session spend monitoring + budget caps
/home/missy/missy/CLAUDE.md:74:       ├─ FailureTracker (agent/failure_tracker.py) — per-tool failure counts + strategy rotation
/home/missy/missy/CLAUDE.md:75:       ├─ Watchdog (agent/watchdog.py) — background subsystem health monitor
/home/missy/missy/CLAUDE.md:77:       ├─ CodeEvolutionManager (agent/code_evolution.py) — self-evolving code with approval + git rollback
/home/missy/missy/CLAUDE.md:78:       ├─ StructuredOutput (agent/structured_output.py) — Pydantic schema enforcement on LLM responses
/home/missy/missy/CLAUDE.md:85:       ├─ SecurityScanner (security/scanner.py) — installation security auditing
/home/missy/missy/CLAUDE.md:96:  → Browser-based screen capture with token auth + session management
/home/missy/missy/CLAUDE.md:108:**Policy Engine (`missy/policy/`)** — Multi-layer enforcement facade:
/home/missy/missy/CLAUDE.md:109:- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, per-category host allowlists (provider, tool, discord)
/home/missy/missy/CLAUDE.md:110:- `FilesystemPolicyEngine`: Per-path read/write access control
/home/missy/missy/CLAUDE.md:112:- `RestPolicy`: L7 HTTP method + path glob rules per host (e.g. allow GET /repos/**, deny DELETE /**)
/home/missy/missy/CLAUDE.md:113:- Network presets (`missy/policy/presets.py`): `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy/CLAUDE.md:115:**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx; single enforcement point for ALL outbound HTTP. Every request checked against network policy + REST policy before dispatch. `InteractiveApproval` TUI prompts operator on denied requests (y/n/a with session memory).
/home/missy/missy/CLAUDE.md:117:**Providers (`missy/providers/`)** — `BaseProvider` defines the interface (`Message`, `CompletionResponse`, `ToolCall`, `ToolResult`). Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. `ProviderRegistry` handles resolution with fallback. `ProviderConfig` supports API key rotation (`api_keys` list), `fast_model`/`premium_model` tiers.
/home/missy/missy/CLAUDE.md:121:- `DiscordChannel`: Full WebSocket Gateway API with access control (DM allowlist, guild/role policies), slash commands (`/ask`, `/status`, `/model`, `/help`)
/home/missy/missy/CLAUDE.md:122:- `WebhookChannel`: HTTP webhook ingress
/home/missy/missy/CLAUDE.md:123:- `ScreencastChannel`: Browser-based screen capture with token auth (`ScreencastTokenRegistry`) and session management (`SessionManager`)
/home/missy/missy/CLAUDE.md:124:- `VoiceChannel`: WebSocket server (default port 8765) accepting connections from edge nodes (ReSpeaker, Raspberry Pi). Protocol: JSON control frames + binary PCM audio. Device pairing with PBKDF2-hashed tokens. Per-node policy modes: `full`, `safe-chat`, `muted`. STT via faster-whisper, TTS via piper binary.
/home/missy/missy/CLAUDE.md:127:- `CircuitBreaker`: Closed/Open/HalfOpen state machine with exponential backoff (threshold=5, base_timeout=60s, max=300s)
/home/missy/missy/CLAUDE.md:128:- `ContextManager`: Token budget (default 30k) with reserves for system prompt, tool definitions, memory fraction (15%), learnings fraction (5%). Prunes oldest history first.
/home/missy/missy/CLAUDE.md:131:- `AttentionSystem`: 5 brain-inspired subsystems — `AlertingAttention` (urgency keywords), `OrientingAttention` (topic extraction), `SustainedAttention` (focus continuity), `SelectiveAttention` (context filtering), `ExecutiveAttention` (tool prioritization)
/home/missy/missy/CLAUDE.md:132:- `Playbook`: Auto-captures successful tool patterns (task_type + tool_sequence hash), injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals. JSON persistence at `~/.missy/playbook.json`.
/home/missy/missy/CLAUDE.md:133:- `ProgressReporter`: Protocol with `NullReporter`, `AuditReporter`, `CLIReporter`. Called in tool loop for structured progress events.
/home/missy/missy/CLAUDE.md:134:- `InteractiveApproval`: Real-time Rich TUI for policy-denied operations (y=allow once, n=deny, a=allow always). Session-scoped memory. Non-TTY auto-denies.
/home/missy/missy/CLAUDE.md:135:- `DoneCriteria`: Generates verification prompts injected after each tool-call round
/home/missy/missy/CLAUDE.md:136:- `Learnings`: Extracts task_type/outcome/lesson from tool-augmented runs, persisted in SQLite
/home/missy/missy/CLAUDE.md:137:- `PromptPatchManager`: Self-tuning prompt patches with approval workflow (proposed/approved/rejected)
/home/missy/missy/CLAUDE.md:139:- `ApprovalGate`: Human-in-the-loop approval for sensitive operations
/home/missy/missy/CLAUDE.md:140:- `CostTracker`: Per-session cost tracking and budget enforcement (`max_spend_usd`). Raises `BudgetExceededError` when cap hit.
/home/missy/missy/CLAUDE.md:141:- `Checkpoint`: WAL-mode SQLite checkpointing for task state. Enables `missy recover` to resume incomplete sessions.
/home/missy/missy/CLAUDE.md:142:- `FailureTracker`: Per-tool consecutive failure counts. Injects strategy-rotation prompts after repeated failures.
/home/missy/missy/CLAUDE.md:143:- `Watchdog`: Background health monitor for subsystems. Tracks `SubsystemHealth` status and reports degradation.
/home/missy/missy/CLAUDE.md:145:- `CodeEvolutionManager`: Self-evolving code modification engine with approval workflow, git-backed rollback, and `missy evolve` CLI.
/home/missy/missy/CLAUDE.md:146:- `StructuredOutput`: Pydantic model schema enforcement on LLM responses with automatic retry on validation failure.
/home/missy/missy/CLAUDE.md:152:**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`. Digest pinning (`missy mcp pin`) records SHA-256 of tool manifests; mismatches refuse to load.
/home/missy/missy/CLAUDE.md:154:**Skills (`missy/skills/`)** — `SkillDiscovery` scans directories for SKILL.md files (cross-agent portable skill format with YAML frontmatter). `missy skills scan` lists discovered skills. Fuzzy search by name/description.
/home/missy/missy/CLAUDE.md:156:**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions.
/home/missy/missy/CLAUDE.md:162:- `Vault`: ChaCha20-Poly1305 encrypted key-value store. Key file at `~/.missy/secrets/vault.key`, encrypted data at `~/.missy/secrets/vault.enc`. Supports `vault://KEY_NAME` references in config.
/home/missy/missy/CLAUDE.md:163:- `AgentIdentity`: Ed25519 keypair at `~/.missy/identity.pem`. Signs audit events. JWK export.
/home/missy/missy/CLAUDE.md:164:- `TrustScorer`: 0-1000 reliability tracking per tool/provider/MCP server. Success (+10), failure (-50), violation (-200). Warns below threshold.
/home/missy/missy/CLAUDE.md:165:- `PromptDriftDetector`: SHA-256 hashes system prompts at start, verifies before each provider call. Emits `security.prompt_drift` audit event on tamper.
/home/missy/missy/CLAUDE.md:166:- `ContainerSandbox`: Optional Docker-based isolation for tool execution. Per-session containers with `--network=none`, memory/CPU limits. Config: `container: { enabled: true }`.
/home/missy/missy/CLAUDE.md:167:- `LandlockPolicy`: Linux Landlock LSM filesystem policy enforcement via ctypes syscalls. Kernel-level read/write path restrictions complementing userspace policy engine.
/home/missy/missy/CLAUDE.md:168:- `SecurityScanner`: Installation security auditor (`missy security scan`). Checks file permissions, config hygiene, exposed secrets, and reports severity-ranked findings.
/home/missy/missy/CLAUDE.md:176:- `AnalysisPromptBuilder`: Domain-specific prompts (puzzle board-state, painting coaching)
/home/missy/missy/CLAUDE.md:177:- `VisionIntentClassifier`: Audio-triggered vision activation with configurable thresholds
/home/missy/missy/CLAUDE.md:178:- `VisionDoctor`: Diagnostics (opencv, video group, devices, capture test, health)
/home/missy/missy/CLAUDE.md:181:- Agent tools: `vision_capture`, `vision_burst`, `vision_analyze`, `vision_devices`, `vision_scene`
/home/missy/missy/CLAUDE.md:184:**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days. Optional `VectorMemoryStore` with FAISS semantic search (`pip install -e ".[vector]"`). `GraphMemoryStore` provides SQLite-backed entity-relationship graph memory with rule-based pattern matching for structured knowledge retrieval.
/home/missy/missy/CLAUDE.md:188:**Config Migration (`missy/config/migrate.py`)** — Auto-migrates old configs on startup. Detects manual hosts matching presets, replaces with `presets: [...]`, stamps `config_version: 2`. Backs up before modifying. Idempotent.
/home/missy/missy/CLAUDE.md:190:**Config Plan (`missy/config/plan.py`)** — Automatic backups on config writes (max 5, pruned). `missy config rollback/diff/plan/backups` commands.
/home/missy/missy/CLAUDE.md:192:**API Server (`missy/api/`)** — Agent-as-a-Service REST API (`missy api start`). Loopback-only binding by default, API key authentication, rate limiting, and automatic secrets censoring on responses.
/home/missy/missy/CLAUDE.md:194:**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl`. `OtelExporter` sends traces/metrics to an OTLP endpoint when enabled.
/home/missy/missy/CLAUDE.md:200:| Config | `~/.missy/config.yaml` |
/home/missy/missy/CLAUDE.md:201:| Config backups | `~/.missy/config.d/config.yaml.<timestamp>` |
/home/missy/missy/CLAUDE.md:202:| Audit log | `~/.missy/audit.jsonl` |
/home/missy/missy/CLAUDE.md:206:| MCP config | `~/.missy/mcp.json` |
/home/missy/missy/CLAUDE.md:207:| Device registry | `~/.missy/devices.json` |
/home/missy/missy/CLAUDE.md:215:| Persona audit log | `~/.missy/persona_audit.jsonl` |
/home/missy/missy/CLAUDE.md:218:| Skills directory | `~/.missy/skills/` |
/home/missy/missy/CLAUDE.md:228:config_version: 2                    # schema version (auto-migrated on startup)
/home/missy/missy/CLAUDE.md:231:  default_deny: true
/home/missy/missy/CLAUDE.md:233:    - anthropic
/home/missy/missy/CLAUDE.md:238:  provider_allowed_hosts: []      # per-category overrides
/home/missy/missy/CLAUDE.md:239:  tool_allowed_hosts: []
/home/missy/missy/CLAUDE.md:241:  rest_policies:                  # L7 HTTP method + path controls
/home/missy/missy/CLAUDE.md:249:      action: "deny"
/home/missy/missy/CLAUDE.md:256:  enabled: false
/home/missy/missy/CLAUDE.md:259:plugins:
/home/missy/missy/CLAUDE.md:260:  enabled: false
/home/missy/missy/CLAUDE.md:261:  allowed_plugins: []
/home/missy/missy/CLAUDE.md:263:providers:
/home/missy/missy/CLAUDE.md:264:  anthropic:
/home/missy/missy/CLAUDE.md:265:    name: anthropic
/home/missy/missy/CLAUDE.md:272:    enabled: true
/home/missy/missy/CLAUDE.md:275:  enabled: true
/home/missy/missy/CLAUDE.md:280:  enabled: false
/home/missy/missy/CLAUDE.md:286:  otel_enabled: false
/home/missy/missy/CLAUDE.md:293:  enabled: false
/home/missy/missy/CLAUDE.md:308:  # See missy/channels/discord/config.py for full schema
/home/missy/missy/CLAUDE.md:311:  enabled: false
/home/missy/missy/CLAUDE.md:318:  enabled: true
/home/missy/missy/CLAUDE.md:326:  scene_memory_max_sessions: 5
/home/missy/missy/CLAUDE.md:329:audit_log_path: "~/.missy/audit.jsonl"
/home/missy/missy/CLAUDE.md:330:max_spend_usd: 0.0                  # per-session budget cap; 0 = unlimited
/home/missy/missy/CLAUDE.md:336:missy init                          Create default config at ~/.missy/config.yaml
/home/missy/missy/CLAUDE.md:338:missy setup --no-prompt             Non-interactive setup (--provider, --api-key-env, --model)
/home/missy/missy/CLAUDE.md:339:missy ask PROMPT                    Single-turn query (--provider, --session)
/home/missy/missy/CLAUDE.md:340:missy run                           Interactive REPL session (--provider)
/home/missy/missy/CLAUDE.md:341:missy providers list                List configured providers and availability
/home/missy/missy/CLAUDE.md:342:missy providers switch NAME         Switch active provider at runtime
/home/missy/missy/CLAUDE.md:343:missy skills                        List registered skills
/home/missy/missy/CLAUDE.md:344:missy skills scan                   Scan for SKILL.md files (--path)
/home/missy/missy/CLAUDE.md:345:missy presets list                  Show built-in network policy presets
/home/missy/missy/CLAUDE.md:346:missy plugins                       List plugins and their status
/home/missy/missy/CLAUDE.md:347:missy doctor                        System health check
/home/missy/missy/CLAUDE.md:349:missy schedule add                  Add scheduled job (--name, --schedule, --task, --provider)
/home/missy/missy/CLAUDE.md:355:missy audit security                Show recent security events (--limit)
```
