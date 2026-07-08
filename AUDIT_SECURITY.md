# AUDIT_SECURITY

- Timestamp: 2026-07-08 11:20:20

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260708-122250
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Added scheduler pause/resume to the Web TUI safe controls API.
/home/missy/missy/LAST_SESSION_SUMMARY.md:8:- `/api/v1/controls` now returns `scheduler.pause_job` and
/home/missy/missy/LAST_SESSION_SUMMARY.md:9:  `scheduler.resume_job` controls when the API runtime has an attached
/home/missy/missy/LAST_SESSION_SUMMARY.md:10:  scheduler.
/home/missy/missy/LAST_SESSION_SUMMARY.md:11:- Control execution now validates scheduler targets, requires exact
/home/missy/missy/LAST_SESSION_SUMMARY.md:13:  job state, mutates through `pause_job()` / `resume_job()`, and audits both
/home/missy/missy/LAST_SESSION_SUMMARY.md:14:  allowed and denied attempts as `web.control` events.
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  target labels, provider/schedule metadata, and generic confirmation prompts.
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Added API tests covering scheduler control listing, confirmation denial,
/home/missy/missy/LAST_SESSION_SUMMARY.md:18:  allowed pause/resume, audit filtering, and frontend control hooks.
/home/missy/missy/LAST_SESSION_SUMMARY.md:44:- Safe controls still need tool, channel, and experimental-feature control
/home/missy/missy/LAST_SESSION_SUMMARY.md:45:  surfaces with policy gates and audit coverage.
/home/missy/missy/LAST_SESSION_SUMMARY.md:46:- Run/session streaming viewer is still not implemented.
/home/missy/missy/LAST_SESSION_SUMMARY.md:47:- Live diagnostics probes should be added carefully behind policy and timeout
/home/missy/missy/LAST_SESSION_SUMMARY.md:54:Add the next safe controls slice for tools or channels, keeping the same
/home/missy/missy/LAST_SESSION_SUMMARY.md:55:confirmation, policy, CSRF, and structured audit behavior used for providers
/home/missy/missy/LAST_SESSION_SUMMARY.md:56:and scheduler jobs.
/home/missy/missy/PERSONA.md:5:## Quick Start
/home/missy/missy/PERSONA.md:48:  - Always respect policy engine decisions
/home/missy/missy/PERSONA.md:79:The `BehaviorLayer` injects persona information into the system prompt sent to the AI provider. This includes identity, tone preferences, style rules, and boundaries. The prompt is structured so the LLM naturally adopts the persona.
/home/missy/missy/PERSONA.md:93:- **Response Guidelines** — Context-specific behavioral directives
/home/missy/missy/AUDIT_CONNECTIVITY.md:6:- default-deny network where practical
/home/missy/missy/AUDIT_CONNECTIVITY.md:7:- exact provider endpoints
/home/missy/missy/AUDIT_CONNECTIVITY.md:8:- explicit local Web TUI bind address and origin policy
/home/missy/missy/AUDIT_CONNECTIVITY.md:9:- exact benchmark and provider endpoints
/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/BUILD_STATUS.md:8:stdlib `ApiServer` owns transport, authentication, browser sessions, CSRF,
/home/missy/missy/BUILD_STATUS.md:10:diagnostics, audit browsing, session state, and operator controls remain split
/home/missy/missy/BUILD_STATUS.md:13:This session expanded safe operator controls from provider switching into
/home/missy/missy/BUILD_STATUS.md:14:scheduler job control:
/home/missy/missy/BUILD_STATUS.md:16:- Added scheduler pause/resume controls to `missy/api/operator_controls.py`.
/home/missy/missy/BUILD_STATUS.md:17:- `/api/v1/controls` now lists provider controls plus scheduler job targets
/home/missy/missy/BUILD_STATUS.md:18:  from the runtime-attached scheduler when present.
/home/missy/missy/BUILD_STATUS.md:19:- `/api/v1/controls/scheduler.pause_job` and
/home/missy/missy/BUILD_STATUS.md:20:  `/api/v1/controls/scheduler.resume_job` require exact confirmation strings,
/home/missy/missy/BUILD_STATUS.md:22:  `resume_job()`, and emit structured `web.control` audit events through the
/home/missy/missy/BUILD_STATUS.md:25:  than provider-only, with labels, target names, schedule/provider metadata,
/home/missy/missy/BUILD_STATUS.md:27:- Added focused API tests for scheduler control listing, denied confirmation,
/home/missy/missy/BUILD_STATUS.md:28:  allowed pause/resume mutations, audit allow/deny events, and frontend control
/home/missy/missy/BUILD_STATUS.md:35:| Existing JSON API auth | preserved | `X-API-Key`, bearer token auth, and browser sessions still guard `/api/v1/*`. |
/home/missy/missy/BUILD_STATUS.md:36:| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` use cookie sessions and CSRF. |
/home/missy/missy/BUILD_STATUS.md:37:| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
/home/missy/missy/BUILD_STATUS.md:38:| CSRF protection | implemented | Unsafe browser API calls and logout require the per-session CSRF token; denials are audited. |
/home/missy/missy/BUILD_STATUS.md:39:| Operator dashboard | improved | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
/home/missy/missy/BUILD_STATUS.md:40:| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
/home/missy/missy/BUILD_STATUS.md:41:| Diagnostics API | improved | `/api/v1/diagnostics` reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
/home/missy/missy/BUILD_STATUS.md:42:| Safe controls API | improved | Provider default switching and scheduler pause/resume are confirmed, validated, CSRF-protected for browser sessions, and audited. |
/home/missy/missy/BUILD_STATUS.md:43:| Renderer extraction | improved | Login, message, main console shell, CSS, and JavaScript live in `missy/api/web_console.py`. |
/home/missy/missy/BUILD_STATUS.md:44:| Console tests | expanded | API suite covers auth, CSRF, controls, audit behavior, scheduler controls, and renderer hooks. |
/home/missy/missy/BUILD_STATUS.md:49:  and API, delegating console HTML/JS/CSS to `missy/api/web_console.py`.
/home/missy/missy/BUILD_STATUS.md:50:- `missy/api/operator_controls.py` is now the policy-shaped control boundary
/home/missy/missy/BUILD_STATUS.md:51:  for provider and scheduler mutations. Unknown controls, invalid targets,
/home/missy/missy/BUILD_STATUS.md:52:  missing dependencies, missing confirmation, unavailable providers, unknown
/home/missy/missy/BUILD_STATUS.md:54:- Scheduler controls intentionally use the runtime-attached scheduler instead
/home/missy/missy/BUILD_STATUS.md:55:  of creating a parallel scheduler service.
/home/missy/missy/BUILD_STATUS.md:56:- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
/home/missy/missy/BUILD_STATUS.md:57:  `/sessions`, `/diagnostics`, `/controls`, and `/audit` with same-origin
/home/missy/missy/BUILD_STATUS.md:59:- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
/home/missy/missy/BUILD_STATUS.md:71:1. Expand safe controls to tools, channels, and experimental features with
/home/missy/missy/BUILD_STATUS.md:72:   policy and confirmation gates.
/home/missy/missy/BUILD_STATUS.md:73:2. Add a real session/run viewer with streaming output, tool calls, errors,
/home/missy/missy/BUILD_STATUS.md:76:   reachability checks that do not bypass policy.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:and Odin remain references for operator ergonomics, diagnostics, auditability,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:16:| Secure local Web UI entrypoint | started | `/login` and `/` implemented with cookie sessions and CSRF. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:17:| Explicit authentication/session handling | improved | Browser session storage is extracted into `missy/api/web_sessions.py`. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:18:| Polished dashboard | started | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:19:| Session/run viewer | not_started | Needs streaming output, tool calls, errors, costs, routing, fallback, resume context. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:20:| Audit log browser | improved | `/api/v1/audit` supports filters, facets, file/memory sources, redaction, IDs, totals, offsets, and `has_more`; UI has filters, pagination, and details. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:21:| Diagnostics/doctor views | improved | `/api/v1/diagnostics` covers Web, providers, tools, memory, policy, gateway, Discord, scheduler, runtime, and remediation hints. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Safe operator controls | improved | Provider default switching plus scheduler pause/resume are confirmed, validated, CSRF-protected for browser sessions, and audited. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:24:| Backend Web TUI security | improved | Auth, CSRF, rate limit, hardened headers, audit events, redaction, XSS-resistant dashboard rendering, and redacted audit search are in place. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:25:| Renderer/assets architecture | improved | Login, message, console shell, CSS, and JavaScript live in `missy/api/web_console.py`; server routing is thinner. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:32:| A2 | Layered tool policy pipeline | hardened | Policy surfaces include current security updates and are reflected in diagnostics. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:33:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated identical failing tool calls are fingerprinted and surfaced to the model. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:36:| A6 | Per-provider tool schema normalization | live | Provider schema methods delegate to `normalize_for_provider()` with fallbacks. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:47:Add the next safe controls slice for tools or channels with explicit policy
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:48:gates, confirmation text, denial audit events, and focused API/UI tests.
/home/missy/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file under 5 MB. Move older entries to timestamped archive files before appending more.
/home/missy/missy/HUMANIZE_AUDIT.md:7:| 2026-04-27T16:09:36Z | humanize.loop.initialized | allow | Initialized audit file for the OpenClaw/humanize loop. No opt-in humanistic behavior was activated this session. |
/home/missy/missy/HUMANIZE_AUDIT.md:8:| 2026-04-27T16:09:36Z | openclaw.a1.subscription | allow | Added streaming state machine primitives that can support future timing, tone, apology, and mood integrations without changing tool correctness. |
/home/missy/missy/HUMANIZE_AUDIT.md:9:| 2026-04-27T18:32:16Z | openclaw.a2.tool_policy | allow | Added layered tool availability filtering with trace labels. This gates future humanistic memory tools without changing execution fail-closed policy. |
/home/missy/missy/HUMANIZE_AUDIT.md:10:| 2026-04-27T18:53:28Z | openclaw.a2.config_policy | allow | Routed YAML-backed provider/global/agent/sandbox/subagent tool policy layers into runtime exposure decisions. Execution policy remains fail-closed in the registry. |
/home/missy/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
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
/home/missy/missy/OPENCLAW_PATTERNS.md:11:| A1 | Streaming subscription state machine | tested | `missy/agent/subscription.py:34`, `missy/agent/subscription.py:241`, `missy/agent/runtime.py:620` | `tests/agent/test_subscription.py:8`, `tests/agent/test_runtime_streaming.py:83` | Handles `message_start/update/end`, tool events, compaction events, monotonic delta/full-content reconciliation, split think/final tag stripping, code-span awareness, reply directives, reasoning modes, and block flush points. Runtime wiring currently covers simple streaming. |
/home/missy/missy/OPENCLAW_PATTERNS.md:12:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py:116`, `missy/policy/tool_policy_pipeline.py:176`, `missy/policy/tool_policy_pipeline.py:206`, `missy/config/settings.py:132`, `missy/agent/runtime.py:1093`, `missy/cli/main.py:206`, `missy/security/sandbox.py:72` | `tests/policy/test_tool_policy_pipeline.py:14`, `tests/policy/test_tool_policy_pipeline.py:115`, `tests/config/test_settings.py:141`, `tests/agent/test_runtime_config_edges.py:741`, `tests/agent/test_runtime_streaming.py:119` | Implements profiles, standard layer ordering, group expansion, glob matching, inline `-tool` deny syntax, `alsoAllow`, fail-warning unknown allowlists, trace labels, and YAML-backed provider/global/agent/sandbox/subagent surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/OPENCLAW_PATTERNS.md:13:| A3 | Mutation fingerprinting + sticky lastToolError | not_started | Planned: `missy/agent/mutation_tracking.py`, `missy/agent/runtime.py`, `missy/tools/registry.py` | Planned: `tests/agent/test_mutation_tracking.py` | Needed by H_G apology calibration. |
/home/missy/missy/OPENCLAW_PATTERNS.md:15:| A5 | Auth profile cooldown + fallback | not_started | Planned: `missy/providers/auth_profiles.py`, `missy/providers/registry.py`, `missy/providers/rate_limiter.py` | Planned: `tests/providers/test_auth_profiles.py` | Must honor user-pinned profile without fallback. |
/home/missy/missy/OPENCLAW_PATTERNS.md:16:| A6 | Per-provider tool schema normalization | not_started | Planned: `missy/providers/schema_adapter.py` | Planned: `tests/providers/test_schema_adapter.py` | Gemini scrubbing and Mistral ID rewrite remain. |
/home/missy/missy/OPENCLAW_PATTERNS.md:17:| A7 | Block-reply chunking with flush points | not_started | Planned: `missy/channels/block_chunker.py`, channel adapters, `missy/agent/runtime.py` | Planned: `tests/channels/test_block_chunker.py` | A1 has block buffers and tool-start flush; channel delivery remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:18:| A8 | Per-channel identity cascade | not_started | Planned: `missy/agent/persona.py`, config schema | Planned: `tests/agent/test_persona_identity_cascade.py` | Response prefix and ack reaction cascade remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:20:| A10 | Sub-agent depth + child caps | not_started | Planned: `missy/agent/sub_agent.py`, session persistence, A2 filter | Planned: `tests/agent/test_sub_agent_depth_caps.py` | Depth-aware orchestration filtering remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:31:| H_C personal memory | A2 tool policy, A12 transcript repair | A2 can now gate future personal-memory recall/list/forget tools through runtime and YAML policy layers; A12 remains unimplemented. |
/home/missy/missy/OPENCLAW_PATTERNS.md:49:- Runtime capability profile constants live in `missy/policy/tool_policy_pipeline.py:21` and `missy/policy/tool_policy_pipeline.py:35`.
/home/missy/missy/OPENCLAW_PATTERNS.md:50:- OpenClaw-compatible group expansion, including `group:fs`, is defined at `missy/policy/tool_policy_pipeline.py:71`.
/home/missy/missy/OPENCLAW_PATTERNS.md:51:- `ToolPolicyLayer`, `ToolPolicyTraceStep`, and `ToolPolicyDecision` provide source-labelled audit records at `missy/policy/tool_policy_pipeline.py:116`.
/home/missy/missy/OPENCLAW_PATTERNS.md:52:- `build_configured_tool_policy_layers()` creates turn-specific config-backed layers at `missy/policy/tool_policy_pipeline.py:176`.
/home/missy/missy/OPENCLAW_PATTERNS.md:53:- `build_tool_policy_layers()` still exposes the explicit standard profile → provider → global → agent → group → sandbox → subagent sequence at `missy/policy/tool_policy_pipeline.py:232`.
/home/missy/missy/OPENCLAW_PATTERNS.md:54:- `resolve_tool_policy()` applies `allow`, `also_allow`, `deny`, globs, inline `-tool` denies, and fail-warning unknown allowlists at `missy/policy/tool_policy_pipeline.py:262`.
/home/missy/missy/OPENCLAW_PATTERNS.md:55:- `ToolPolicyConfig` and `AgentPolicyConfig` parse YAML-backed tool policy surfaces at `missy/config/settings.py:132`.
/home/missy/missy/OPENCLAW_PATTERNS.md:56:- `AgentRuntime._get_tools()` delegates capability-mode and config-backed filtering to A2 at `missy/agent/runtime.py:1093`.
/home/missy/missy/OPENCLAW_PATTERNS.md:57:- CLI-created runtimes receive parsed tool policies through `_agent_tool_policy_kwargs()` at `missy/cli/main.py:206`.
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
/home/missy/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
```
