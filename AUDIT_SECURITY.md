# AUDIT_SECURITY

- Timestamp: 2026-07-08 09:31:21

## Current Web TUI Security Notes

- Browser diagnostics are served only from authenticated `/api/v1/diagnostics`.
- Diagnostics are assembled from already-injected server dependencies and pass
  through existing audit redaction helpers before response rendering.
- The Diagnostics panel renders API-derived text through the console escape
  helper before insertion into HTML.
- The endpoint reports presence/count/status posture, not raw API keys,
  cookies, CSRF tokens, provider secrets, or unrestricted filesystem paths.

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Extracted browser operator session storage into `missy/api/web_sessions.py`.
/home/missy/missy/LAST_SESSION_SUMMARY.md:8:- Extracted audit browser filtering, redaction, facets, pagination, stable event
/home/missy/missy/LAST_SESSION_SUMMARY.md:9:  IDs, and event conversion into `missy/api/audit_browser.py`.
/home/missy/missy/LAST_SESSION_SUMMARY.md:10:- Extended authenticated `GET /api/v1/audit` with `offset`, `total`, `limit`,
/home/missy/missy/LAST_SESSION_SUMMARY.md:11:  `has_more`, newest-first paging, and redacted stable `id` values.
/home/missy/missy/LAST_SESSION_SUMMARY.md:12:- Expanded the Web TUI Audit Trail panel with severity, actor, source, redacted
/home/missy/missy/LAST_SESSION_SUMMARY.md:15:- Added API coverage for audit pagination, newest-first ordering, event IDs, and
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  recursive redaction.
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Updated required loop artifacts for the Web TUI primary focus.
/home/missy/missy/LAST_SESSION_SUMMARY.md:45:- Diagnostics panels, run/session streaming viewer, and safe operator controls
/home/missy/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy/HUMANIZE_STATUS.md:35:| H_I | Mood state with decay | not_started | First humanize implementation target in sessions 8-9. |
/home/missy/missy/HUMANIZE_STATUS.md:39:- Initialized required loop tracking documents.
/home/missy/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
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
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260708-122250
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/examples/systemd/README.md:8:## Prerequisites
/home/missy/missy/examples/systemd/README.md:16:- Any required API keys must be set as environment variables. You can use a
/home/missy/missy/examples/systemd/README.md:51:sudo systemctl enable --now missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:54:This enables the service to start on boot and starts it immediately.
/home/missy/missy/examples/systemd/README.md:91:sudo systemctl disable missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:102:| `NoNewPrivileges=true` | Prevents the process from gaining new privileges (e.g. via setuid). |
/home/missy/missy/examples/systemd/README.md:106:| `ReadWritePaths=...` | Grants write access to `~/.missy` (for audit logs, jobs, memory) and `~/workspace` (for agent output). |
/home/missy/missy/examples/systemd/README.md:120:- Check that `config.yaml` includes the required domains in `allowed_domains` or `allowed_hosts`.
/home/missy/missy/examples/systemd/README.md:124:- Check that `discord.enabled: true` is set in `config.yaml`.
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
/home/missy/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy/README.md:53:- **REST API** — Agent-as-a-Service endpoint (`missy api start`) with loopback binding, API key auth, rate limiting
/home/missy/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
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
/home/missy/missy/README.md:217:  default_deny: true
/home/missy/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy/README.md:234:  enabled: false
/home/missy/missy/README.md:237:providers:
/home/missy/missy/README.md:238:  anthropic:
/home/missy/missy/README.md:239:    name: anthropic
/home/missy/missy/README.md:246:  enabled: false
/home/missy/missy/README.md:266:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy/README.md:267:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy/README.md:268:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy/README.md:269:missy providers list                # List providers and availability
/home/missy/missy/README.md:270:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy/README.md:277:# Security & audit
/home/missy/missy/README.md:278:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy/README.md:279:missy audit security                # Policy violations
/home/missy/missy/README.md:286:missy presets list                  # Show built-in network presets
/home/missy/missy/README.md:289:missy discord status | probe | register-commands | audit
/home/missy/missy/README.md:293:missy devices list | pair | unpair | status | policy
/home/missy/missy/README.md:295:# MCP & skills
/home/missy/missy/README.md:297:missy skills                        # List registered skills
/home/missy/missy/README.md:298:missy skills scan                   # Discover SKILL.md files
/home/missy/missy/README.md:302:missy vision health | benchmark | validate | memory
/home/missy/missy/README.md:317:missy sessions list | rename | cleanup
/home/missy/missy/README.md:344:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy/README.md:354:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy/README.md:370:| [Getting Started](https://missylabs.github.io/getting-started/) | 5 | Install, quickstart, wizard, first conversation |
/home/missy/missy/README.md:371:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy/README.md:373:| [Architecture](https://missylabs.github.io/architecture/) | 10 | Runtime, context, circuit breaker, progress, playbook, sleep mode, synthesizer, attention, message bus |
/home/missy/missy/README.md:376:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy/README.md:377:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy/README.md:384:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy/README.md:392:├── agent/           Runtime, circuit breaker, context, playbook, consolidation,
/home/missy/missy/README.md:393:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/README.md:396:├── channels/        CLI, Discord, webhooks, voice (WebSocket), screencast (browser)
/home/missy/missy/README.md:404:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/README.md:405:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy/README.md:406:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy/README.md:409:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy/README.md:410:├── plugins/         Security-gated external plugin loader
/home/missy/missy/README.md:411:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy/HATCHING.md:5:## Quick Start
/home/missy/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:48:| H_A | Delay calculation respects length/complexity/mood/channel caps; quick/fast/asap bypasses long sleeps; channel typing indicator ordering is mocked. |
/home/missy/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
/home/missy/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy/docs/architecture.md:10:Missy is a **security-first**, **local-first**, **multi-provider** AI agent
/home/missy/missy/docs/architecture.md:13:access, filesystem writes, shell execution, plugin loading -- is disabled by
/home/missy/missy/docs/architecture.md:14:default and must be explicitly enabled through a YAML configuration file.
/home/missy/missy/docs/architecture.md:22:   policy engine before any bytes leave the machine.
/home/missy/missy/docs/architecture.md:23:3. **Audit everything** -- every policy decision, provider call, scheduler
/home/missy/missy/docs/architecture.md:24:   execution, and plugin action is recorded as a structured JSONL event.
/home/missy/missy/docs/architecture.md:34:  policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/docs/architecture.md:36:  agent/           Runtime, circuit breaker, context, playbook, consolidation,
/home/missy/missy/docs/architecture.md:37:                   attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/docs/architecture.md:41:  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry + rate limiter
/home/missy/missy/docs/architecture.md:42:  tools/           Tool base class, registry, 18+ built-in tools
/home/missy/missy/docs/architecture.md:43:  skills/          Skill registry + SKILL.md discovery
/home/missy/missy/docs/architecture.md:44:  plugins/         Security-gated external plugin loader and base class
/home/missy/missy/docs/architecture.md:45:  scheduler/       APScheduler integration, human schedule parsing, job persistence
/home/missy/missy/docs/architecture.md:50:  channels/        CLI, Discord (Gateway + REST), webhooks, voice (WebSocket), screencast
/home/missy/missy/docs/architecture.md:64: 2. Config loader         Read YAML, auto-migrate if needed, build MissyConfig
/home/missy/missy/docs/architecture.md:67: 3. Subsystem init        init_policy_engine(cfg)  -- network, filesystem, shell, REST L7
/home/missy/missy/docs/architecture.md:68:        |                 init_audit_logger(cfg.audit_log_path) + AgentIdentity (Ed25519)
/home/missy/missy/docs/architecture.md:69:        |                 init_registry(cfg) -- providers with rate limiter + fallback
/home/missy/missy/docs/architecture.md:71:        |                 init_tool_registry() -- 18+ built-in tools + MCP servers
/home/missy/missy/docs/architecture.md:78:        |                 Resolve provider (with fallback + circuit breaker)
/home/missy/missy/docs/architecture.md:80:        |                 ContextManager builds message list within token budget
/home/missy/missy/docs/architecture.md:82:        |                 Playbook injects proven tool patterns
/home/missy/missy/docs/architecture.md:85:        |                 All HTTP through PolicyHTTPClient -> policy + REST check
/home/missy/missy/docs/architecture.md:93: 8. Post-processing       Learnings extracted from tool-augmented runs
/home/missy/missy/docs/architecture.md:96:        |                 SecretCensor redacts secrets from output
/home/missy/missy/docs/architecture.md:99:        |                 Events signed by AgentIdentity, appended to audit.jsonl
/home/missy/missy/docs/architecture.md:136:|Sleep  | |Checkpoint  | |Circuit  | |Vision     |
/home/missy/missy/docs/architecture.md:161:Every policy dataclass defaults to the most restrictive posture:
/home/missy/missy/docs/architecture.md:163:- `NetworkPolicy.default_deny = True`
/home/missy/missy/docs/architecture.md:164:- `ShellPolicy.enabled = False`
/home/missy/missy/docs/architecture.md:165:- `PluginPolicy.enabled = False`
/home/missy/missy/docs/architecture.md:168:An operator must explicitly add entries to allowlists before any capability is
/home/missy/missy/docs/architecture.md:173:All outbound HTTP traffic -- whether initiated by a provider, a tool, a plugin,
/home/missy/missy/docs/architecture.md:177:`get_policy_engine().check_network(host)`.  If the host is not on an allowlist,
/home/missy/missy/docs/architecture.md:180:The Anthropic and OpenAI providers use their own SDKs for HTTP, but their API
/home/missy/missy/docs/architecture.md:181:hosts must still appear in `network.allowed_hosts` for the initial policy check
/home/missy/missy/docs/architecture.md:182:at the gateway layer.  The Ollama provider routes directly through
/home/missy/missy/docs/architecture.md:191:- `session_id` / `task_id` (correlation)
/home/missy/missy/docs/architecture.md:193:- `category` (one of: `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider`, `security`, `agent`, `tool`, `mcp`, `vision`)
/home/missy/missy/docs/architecture.md:194:- `result` (one of: `allow`, `deny`, `error`)
/home/missy/missy/docs/architecture.md:196:- `policy_rule` (optional rule name)
/home/missy/missy/docs/architecture.md:198:The `AuditLogger` (`missy/observability/audit_logger.py`) wraps the bus's
/home/missy/missy/docs/architecture.md:200:audit log file.
/home/missy/missy/docs/architecture.md:212:  +-> policy/engine
/home/missy/missy/docs/architecture.md:213:  +-> observability/audit_logger + observability/otel
/home/missy/missy/docs/architecture.md:214:  +-> providers/registry
/home/missy/missy/docs/architecture.md:216:  +-> scheduler/manager
/home/missy/missy/docs/architecture.md:217:  +-> plugins/loader
/home/missy/missy/docs/architecture.md:226:  +-> providers/registry + providers/base
/home/missy/missy/docs/architecture.md:227:  +-> core/session + core/events + core/message_bus
/home/missy/missy/docs/architecture.md:228:  +-> tools/registry
/home/missy/missy/docs/architecture.md:229:  +-> agent/attention + agent/context + agent/circuit_breaker
/home/missy/missy/docs/architecture.md:232:  +-> agent/progress + agent/interactive_approval + agent/approval
/home/missy/missy/docs/architecture.md:240:providers/registry
/home/missy/missy/docs/architecture.md:241:  +-> providers/base
/home/missy/missy/docs/architecture.md:242:  +-> providers/anthropic_provider + openai_provider + ollama_provider
/home/missy/missy/docs/architecture.md:243:  +-> providers/rate_limiter
/home/missy/missy/docs/architecture.md:247:  +-> policy/engine + policy/rest_policy
/home/missy/missy/docs/architecture.md:248:  +-> agent/interactive_approval
/home/missy/missy/docs/architecture.md:251:policy/engine
/home/missy/missy/docs/architecture.md:252:  +-> policy/network + policy/filesystem + policy/shell + policy/rest_policy
/home/missy/missy/docs/architecture.md:253:  +-> policy/presets
/home/missy/missy/docs/architecture.md:259:  +-> tools/registry
/home/missy/missy/docs/architecture.md:265:scheduler/manager
/home/missy/missy/docs/architecture.md:266:  +-> scheduler/parser + scheduler/jobs
/home/missy/missy/docs/architecture.md:271:  +-> providers/base (for image formatting)
/home/missy/missy/docs/architecture.md:281:2. `init_policy_engine(cfg)` -- must come first; other subsystems depend on it
/home/missy/missy/docs/architecture.md:282:3. `init_audit_logger(cfg.audit_log_path)` -- wraps the event bus
/home/missy/missy/docs/architecture.md:283:4. `init_registry(cfg)` -- constructs provider instances
/home/missy/missy/docs/architecture.md:300:| Policy engine | `init_policy_engine(cfg)` | `get_policy_engine()` |
/home/missy/missy/docs/architecture.md:301:| Provider registry | `init_registry(cfg)` | `get_registry()` |
/home/missy/missy/docs/architecture.md:302:| Audit logger | `init_audit_logger(path)` | `get_audit_logger()` |
/home/missy/missy/docs/architecture.md:303:| Plugin loader | `init_plugin_loader(cfg)` | `get_plugin_loader()` |
/home/missy/missy/docs/architecture.md:304:| Skill registry | `init_skill_registry()` | `get_skill_registry()` |
/home/missy/missy/docs/architecture.md:305:| Tool registry | `init_tool_registry()` | `get_tool_registry()` |
/home/missy/missy/docs/operations.md:1:# Missy Operations Guide
/home/missy/missy/docs/operations.md:3:This guide covers installation, configuration, day-to-day operation,
/home/missy/missy/docs/operations.md:11:2. [Configuration Guide](#2-configuration-guide)
/home/missy/missy/docs/operations.md:13:4. [Monitoring and Audit Logs](#4-monitoring-and-audit-logs)
/home/missy/missy/docs/operations.md:14:5. [Scheduler Management](#5-scheduler-management)
/home/missy/missy/docs/operations.md:15:6. [Provider Configuration](#6-provider-configuration)
/home/missy/missy/docs/operations.md:22:### Prerequisites
/home/missy/missy/docs/operations.md:26:- API keys for the AI providers you intend to use.
/home/missy/missy/docs/operations.md:60:3. Creates empty `~/.missy/audit.jsonl` and `~/.missy/jobs.json`.
/home/missy/missy/docs/operations.md:67:## 2. Configuration Guide
/home/missy/missy/docs/operations.md:78:# Network policy
/home/missy/missy/docs/operations.md:82:  default_deny: true                  # true = block-by-default (recommended)
/home/missy/missy/docs/operations.md:84:    - "api.anthropic.com"
/home/missy/missy/docs/operations.md:85:    - "api.openai.com"
/home/missy/missy/docs/operations.md:95:# Filesystem policy
/home/missy/missy/docs/operations.md:108:# Shell policy
/home/missy/missy/docs/operations.md:109:# Disabled by default. Enable only if required.
/home/missy/missy/docs/operations.md:112:  enabled: false
/home/missy/missy/docs/operations.md:116:# Plugin policy
/home/missy/missy/docs/operations.md:117:# Disabled by default. Enable only trusted plugins.
/home/missy/missy/docs/operations.md:119:plugins:
/home/missy/missy/docs/operations.md:120:  enabled: false
/home/missy/missy/docs/operations.md:121:  allowed_plugins: []                 # e.g. ["my_plugin_name"]
/home/missy/missy/docs/operations.md:127:providers:
/home/missy/missy/docs/operations.md:128:  anthropic:
/home/missy/missy/docs/operations.md:129:    name: anthropic
/home/missy/missy/docs/operations.md:132:  openai:
/home/missy/missy/docs/operations.md:133:    name: openai
/home/missy/missy/docs/operations.md:136:  ollama:
/home/missy/missy/docs/operations.md:137:    name: ollama
/home/missy/missy/docs/operations.md:146:audit_log_path: "~/.missy/audit.jsonl"
/home/missy/missy/docs/operations.md:154:| `ANTHROPIC_API_KEY` | API key for the Anthropic provider. |
/home/missy/missy/docs/operations.md:155:| `OPENAI_API_KEY` | API key for the OpenAI provider. |
/home/missy/missy/docs/operations.md:165:missy ask --provider ollama "Explain quantum entanglement in simple terms."
/home/missy/missy/docs/operations.md:166:missy ask --session my-session-id "Continue our earlier conversation."
/home/missy/missy/docs/operations.md:169:### Interactive session
/home/missy/missy/docs/operations.md:173:missy run --provider openai
/home/missy/missy/docs/operations.md:176:Type `quit`, `exit`, or `q` to end the session.  Press Ctrl-D (EOF) or
/home/missy/missy/docs/operations.md:192:`--debug` enables Python `DEBUG`-level logging, which includes policy check
/home/missy/missy/docs/operations.md:193:details, provider selection, and audit event publishing. Python application
/home/missy/missy/docs/operations.md:210:The application log captures Python warnings/errors, provider diagnostics, and
/home/missy/missy/docs/operations.md:211:debug logs when enabled. The default path is `~/.missy/missy.log`; override it
/home/missy/missy/docs/operations.md:216:The audit log is a newline-delimited JSON (JSONL) file, by default at
/home/missy/missy/docs/operations.md:217:`~/.missy/audit.jsonl`.  Each line is a self-contained JSON object.
/home/missy/missy/docs/operations.md:224:  "session_id": "uuid-...",
/home/missy/missy/docs/operations.md:225:  "task_id": "uuid-...",
/home/missy/missy/docs/operations.md:227:  "category": "provider",
/home/missy/missy/docs/operations.md:229:  "detail": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
/home/missy/missy/docs/operations.md:230:  "policy_rule": null
/home/missy/missy/docs/operations.md:234:`result` is one of `allow`, `deny`, or `error`.
/home/missy/missy/docs/operations.md:240:missy audit recent
/home/missy/missy/docs/operations.md:243:missy audit recent --limit 20 --category network
/home/missy/missy/docs/operations.md:245:# Show policy violations only
/home/missy/missy/docs/operations.md:246:missy audit security
/home/missy/missy/docs/operations.md:247:missy audit security --limit 100
/home/missy/missy/docs/operations.md:253:# Show all policy denials
/home/missy/missy/docs/operations.md:254:jq 'select(.result == "deny")' ~/.missy/audit.jsonl
/home/missy/missy/docs/operations.md:257:jq 'select(.event_type | startswith("agent.run"))' ~/.missy/audit.jsonl
/home/missy/missy/docs/operations.md:260:jq -r '.category' ~/.missy/audit.jsonl | sort | uniq -c | sort -rn
```
