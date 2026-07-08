# AUDIT_SECURITY

- Timestamp: 2026-07-08 11:05:50

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/pyproject.toml:1:[build-system]
/home/missy/missy/pyproject.toml:2:requires = ["setuptools>=68", "wheel"]
/home/missy/missy/pyproject.toml:3:build-backend = "setuptools.build_meta"
/home/missy/missy/pyproject.toml:8:description = "A policy-enforced AI agent framework"
/home/missy/missy/pyproject.toml:9:requires-python = ">=3.11"
/home/missy/missy/pyproject.toml:14:    "anthropic>=0.25",
/home/missy/missy/pyproject.toml:15:    "openai>=1.25",
/home/missy/missy/pyproject.toml:17:    "apscheduler>=3.10",
/home/missy/missy/pyproject.toml:22:    "websockets>=12.0",
/home/missy/missy/pyproject.toml:48:# Discord voice transport/session. Requires system ffmpeg installed.
/home/missy/missy/pyproject.toml:60:# Browser automation (Firefox via Playwright) and GTK/X11 accessibility tools.
/home/missy/missy/pyproject.toml:71:[tool.setuptools.package-data]
/home/missy/missy/pyproject.toml:72:missy = ["py.typed", "channels/screencast/web/*.html"]
/home/missy/missy/pyproject.toml:74:[tool.setuptools.packages.find]
/home/missy/missy/pyproject.toml:78:[tool.pytest.ini_options]
/home/missy/missy/pyproject.toml:82:    "ignore:websockets.legacy is deprecated:DeprecationWarning",
/home/missy/missy/pyproject.toml:87:[tool.black]
/home/missy/missy/pyproject.toml:91:[tool.ruff]
/home/missy/missy/pyproject.toml:95:[tool.ruff.lint]
/home/missy/missy/pyproject.toml:99:[tool.ruff.lint.isort]
/home/missy/missy/pyproject.toml:102:[tool.coverage.run]
/home/missy/missy/pyproject.toml:106:[tool.coverage.report]
/home/missy/missy/pyproject.toml:110:[tool.mypy]
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
/home/missy/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy/install.sh:29:    echo "Error: Python 3.11+ is required." >&2
/home/missy/missy/install.sh:37:    echo "Error: git is required." >&2
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260708-122250
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/AUDIT_SECURITY.md:13:/home/missy/missy/pyproject.toml:1:[build-system]
/home/missy/missy/AUDIT_SECURITY.md:14:/home/missy/missy/pyproject.toml:2:requires = ["setuptools>=68", "wheel"]
/home/missy/missy/AUDIT_SECURITY.md:15:/home/missy/missy/pyproject.toml:3:build-backend = "setuptools.build_meta"
/home/missy/missy/AUDIT_SECURITY.md:16:/home/missy/missy/pyproject.toml:8:description = "A policy-enforced AI agent framework"
/home/missy/missy/AUDIT_SECURITY.md:17:/home/missy/missy/pyproject.toml:9:requires-python = ">=3.11"
/home/missy/missy/AUDIT_SECURITY.md:18:/home/missy/missy/pyproject.toml:14:    "anthropic>=0.25",
/home/missy/missy/AUDIT_SECURITY.md:19:/home/missy/missy/pyproject.toml:15:    "openai>=1.25",
/home/missy/missy/AUDIT_SECURITY.md:20:/home/missy/missy/pyproject.toml:17:    "apscheduler>=3.10",
/home/missy/missy/AUDIT_SECURITY.md:21:/home/missy/missy/pyproject.toml:22:    "websockets>=12.0",
/home/missy/missy/AUDIT_SECURITY.md:22:/home/missy/missy/pyproject.toml:48:# Discord voice transport/session. Requires system ffmpeg installed.
/home/missy/missy/AUDIT_SECURITY.md:23:/home/missy/missy/pyproject.toml:60:# Browser automation (Firefox via Playwright) and GTK/X11 accessibility tools.
/home/missy/missy/AUDIT_SECURITY.md:24:/home/missy/missy/pyproject.toml:71:[tool.setuptools.package-data]
/home/missy/missy/AUDIT_SECURITY.md:25:/home/missy/missy/pyproject.toml:72:missy = ["py.typed", "channels/screencast/web/*.html"]
/home/missy/missy/AUDIT_SECURITY.md:26:/home/missy/missy/pyproject.toml:74:[tool.setuptools.packages.find]
/home/missy/missy/AUDIT_SECURITY.md:27:/home/missy/missy/pyproject.toml:78:[tool.pytest.ini_options]
/home/missy/missy/AUDIT_SECURITY.md:28:/home/missy/missy/pyproject.toml:82:    "ignore:websockets.legacy is deprecated:DeprecationWarning",
/home/missy/missy/AUDIT_SECURITY.md:29:/home/missy/missy/pyproject.toml:87:[tool.black]
/home/missy/missy/AUDIT_SECURITY.md:30:/home/missy/missy/pyproject.toml:91:[tool.ruff]
/home/missy/missy/AUDIT_SECURITY.md:31:/home/missy/missy/pyproject.toml:95:[tool.ruff.lint]
/home/missy/missy/AUDIT_SECURITY.md:32:/home/missy/missy/pyproject.toml:99:[tool.ruff.lint.isort]
/home/missy/missy/AUDIT_SECURITY.md:33:/home/missy/missy/pyproject.toml:102:[tool.coverage.run]
/home/missy/missy/AUDIT_SECURITY.md:34:/home/missy/missy/pyproject.toml:106:[tool.coverage.report]
/home/missy/missy/AUDIT_SECURITY.md:35:/home/missy/missy/pyproject.toml:110:[tool.mypy]
/home/missy/missy/AUDIT_SECURITY.md:36:/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Added scheduler pause/resume to the Web TUI safe controls API.
/home/missy/missy/AUDIT_SECURITY.md:37:/home/missy/missy/LAST_SESSION_SUMMARY.md:8:- `/api/v1/controls` now returns `scheduler.pause_job` and
/home/missy/missy/AUDIT_SECURITY.md:38:/home/missy/missy/LAST_SESSION_SUMMARY.md:9:  `scheduler.resume_job` controls when the API runtime has an attached
/home/missy/missy/AUDIT_SECURITY.md:39:/home/missy/missy/LAST_SESSION_SUMMARY.md:10:  scheduler.
/home/missy/missy/AUDIT_SECURITY.md:40:/home/missy/missy/LAST_SESSION_SUMMARY.md:11:- Control execution now validates scheduler targets, requires exact
/home/missy/missy/AUDIT_SECURITY.md:41:/home/missy/missy/LAST_SESSION_SUMMARY.md:13:  job state, mutates through `pause_job()` / `resume_job()`, and audits both
/home/missy/missy/AUDIT_SECURITY.md:42:/home/missy/missy/LAST_SESSION_SUMMARY.md:14:  allowed and denied attempts as `web.control` events.
/home/missy/missy/AUDIT_SECURITY.md:43:/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  target labels, provider/schedule metadata, and generic confirmation prompts.
/home/missy/missy/AUDIT_SECURITY.md:44:/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Added API tests covering scheduler control listing, confirmation denial,
/home/missy/missy/AUDIT_SECURITY.md:45:/home/missy/missy/LAST_SESSION_SUMMARY.md:18:  allowed pause/resume, audit filtering, and frontend control hooks.
/home/missy/missy/AUDIT_SECURITY.md:46:/home/missy/missy/LAST_SESSION_SUMMARY.md:44:- Safe controls still need tool, channel, and experimental-feature control
/home/missy/missy/AUDIT_SECURITY.md:47:/home/missy/missy/LAST_SESSION_SUMMARY.md:45:  surfaces with policy gates and audit coverage.
/home/missy/missy/AUDIT_SECURITY.md:48:/home/missy/missy/LAST_SESSION_SUMMARY.md:46:- Run/session streaming viewer is still not implemented.
/home/missy/missy/AUDIT_SECURITY.md:49:/home/missy/missy/LAST_SESSION_SUMMARY.md:47:- Live diagnostics probes should be added carefully behind policy and timeout
/home/missy/missy/AUDIT_SECURITY.md:50:/home/missy/missy/LAST_SESSION_SUMMARY.md:54:Add the next safe controls slice for tools or channels, keeping the same
/home/missy/missy/AUDIT_SECURITY.md:51:/home/missy/missy/LAST_SESSION_SUMMARY.md:55:confirmation, policy, CSRF, and structured audit behavior used for providers
/home/missy/missy/AUDIT_SECURITY.md:52:/home/missy/missy/LAST_SESSION_SUMMARY.md:56:and scheduler jobs.
/home/missy/missy/AUDIT_SECURITY.md:53:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:and Odin remain references for operator ergonomics, diagnostics, auditability,
/home/missy/missy/AUDIT_SECURITY.md:54:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:16:| Secure local Web UI entrypoint | started | `/login` and `/` implemented with cookie sessions and CSRF. |
/home/missy/missy/AUDIT_SECURITY.md:55:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:17:| Explicit authentication/session handling | improved | Browser session storage is extracted into `missy/api/web_sessions.py`. |
/home/missy/missy/AUDIT_SECURITY.md:56:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:18:| Polished dashboard | started | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
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
/home/missy/missy/HATCHING.md:5:## Quick Start
/home/missy/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
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
/home/missy/missy/missy/tools/intelligence/candidate_store.py:1:"""Tool candidate lifecycle store.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:3:A *tool candidate* is a structured proposal that arose from pattern
/home/missy/missy/missy/tools/intelligence/candidate_store.py:4:detection.  Every candidate passes through a well-defined lifecycle before
/home/missy/missy/missy/tools/intelligence/candidate_store.py:7:    proposed → experimental → benchmarked → approved → enabled
/home/missy/missy/missy/tools/intelligence/candidate_store.py:8:                                                    ↘ deprecated → disabled
/home/missy/missy/missy/tools/intelligence/candidate_store.py:10:At any point a candidate may be denied (→ disabled) with a reason.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:12:The store is backed by SQLite at ``~/.missy/tool_candidates.db`` and
/home/missy/missy/missy/tools/intelligence/candidate_store.py:13:emits an audit event on every state transition.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:23:import uuid
/home/missy/missy/missy/tools/intelligence/candidate_store.py:33:_DEFAULT_DB = Path("~/.missy/tool_candidates.db")
/home/missy/missy/missy/tools/intelligence/candidate_store.py:37:    """Ordered lifecycle states for a :class:`ToolCandidate`.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:45:    BENCHMARKED = "benchmarked"
/home/missy/missy/missy/tools/intelligence/candidate_store.py:47:    ENABLED = "enabled"
/home/missy/missy/missy/tools/intelligence/candidate_store.py:49:    DISABLED = "disabled"
/home/missy/missy/missy/tools/intelligence/candidate_store.py:53:        """States where the tool is usable by the agent."""
/home/missy/missy/missy/tools/intelligence/candidate_store.py:64:    """Condensed benchmark outcome stored inline on a candidate."""
/home/missy/missy/missy/tools/intelligence/candidate_store.py:66:    provider: str
/home/missy/missy/missy/tools/intelligence/candidate_store.py:72:    schema_score: float = 0.0
/home/missy/missy/missy/tools/intelligence/candidate_store.py:78:            "provider": self.provider,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:84:            "schema_score": self.schema_score,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:96:    """A proposed or active structured tool.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:101:        description: One-sentence description shown in tool schemas.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:102:        schema: JSON Schema dict describing the tool's parameters.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:104:        provenance: Description of how this candidate was generated.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:107:        version: Monotonic integer incremented on schema changes.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:108:        owner: Identity of whoever approved/created the candidate.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:112:        notes: Human-readable notes (denial reason, benchmark summary…).
/home/missy/missy/missy/tools/intelligence/candidate_store.py:113:        benchmark_scores: Per-provider :class:`BenchmarkSummary` instances.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:114:        provider_enabled: Per-provider enablement flag based on benchmarks.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:121:    schema: dict[str, Any]
/home/missy/missy/missy/tools/intelligence/candidate_store.py:123:    provenance: str
/home/missy/missy/missy/tools/intelligence/candidate_store.py:132:    benchmark_scores: list[BenchmarkSummary] = field(default_factory=list)
/home/missy/missy/missy/tools/intelligence/candidate_store.py:133:    provider_enabled: dict[str, bool] = field(default_factory=dict)
/home/missy/missy/missy/tools/intelligence/candidate_store.py:141:        schema: dict[str, Any],
/home/missy/missy/missy/tools/intelligence/candidate_store.py:143:        provenance: str = "",
/home/missy/missy/missy/tools/intelligence/candidate_store.py:151:            id=str(uuid.uuid4()),
/home/missy/missy/missy/tools/intelligence/candidate_store.py:154:            schema=schema,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:156:            provenance=provenance,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:171:            "schema": self.schema,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:173:            "provenance": self.provenance,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:182:            "benchmark_scores": [b.to_dict() for b in self.benchmark_scores],
/home/missy/missy/missy/tools/intelligence/candidate_store.py:183:            "provider_enabled": self.provider_enabled,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:189:        scores_raw = json.loads(row["benchmark_scores_json"] or "[]")
/home/missy/missy/missy/tools/intelligence/candidate_store.py:190:        benchmark_scores = [BenchmarkSummary.from_dict(s) for s in scores_raw]
/home/missy/missy/missy/tools/intelligence/candidate_store.py:195:            schema=json.loads(row["schema_json"] or "{}"),
/home/missy/missy/missy/tools/intelligence/candidate_store.py:197:            provenance=row["provenance"] or "",
/home/missy/missy/missy/tools/intelligence/candidate_store.py:206:            benchmark_scores=benchmark_scores,
/home/missy/missy/missy/tools/intelligence/candidate_store.py:207:            provider_enabled=json.loads(row["provider_enabled_json"] or "{}"),
/home/missy/missy/missy/tools/intelligence/candidate_store.py:213:    """SQLite-backed store for tool candidates with lifecycle tracking.
/home/missy/missy/missy/tools/intelligence/candidate_store.py:223:        self._init_schema()
/home/missy/missy/missy/tools/intelligence/candidate_store.py:229:    def add(self, candidate: ToolCandidate) -> ToolCandidate:
```
