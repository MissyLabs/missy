# AUDIT_SECURITY

- Timestamp: 2026-07-08 21:51:07

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/HATCHING_LOG.md:35:- `initialize_config` — Config file creation or detection
/home/missy/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
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
/home/missy/missy/conftest.py:1:# conftest.py — pytest configuration
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
/home/missy/missy/docs/architecture.md:10:Missy is a **security-first**, **local-first**, **multi-provider** AI agent
/home/missy/missy/docs/architecture.md:11:platform.  It is designed to run entirely under the operator's control on a
/home/missy/missy/docs/architecture.md:13:access, filesystem writes, shell execution, plugin loading -- is disabled by
/home/missy/missy/docs/architecture.md:14:default and must be explicitly enabled through a YAML configuration file.
/home/missy/missy/docs/architecture.md:22:   policy engine before any bytes leave the machine.
/home/missy/missy/docs/architecture.md:23:3. **Audit everything** -- every policy decision, provider call, scheduler
/home/missy/missy/docs/architecture.md:24:   execution, and plugin action is recorded as a structured JSONL event.
/home/missy/missy/docs/architecture.md:33:  config/          YAML settings, hot-reload, migration, plan/rollback
/home/missy/missy/docs/architecture.md:34:  policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/docs/architecture.md:36:  agent/           Runtime, circuit breaker, context, playbook, consolidation,
/home/missy/missy/docs/architecture.md:37:                   attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/docs/architecture.md:41:  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry + rate limiter
/home/missy/missy/docs/architecture.md:42:  tools/           Tool base class, registry, 18+ built-in tools
/home/missy/missy/docs/architecture.md:43:  skills/          Skill registry + SKILL.md discovery
/home/missy/missy/docs/architecture.md:44:  plugins/         Security-gated external plugin loader and base class
/home/missy/missy/docs/architecture.md:45:  scheduler/       APScheduler integration, human schedule parsing, job persistence
/home/missy/missy/docs/architecture.md:50:  channels/        CLI, Discord (Gateway + REST), webhooks, voice (WebSocket), screencast
/home/missy/missy/docs/architecture.md:51:  vision/          Camera discovery, capture, analysis, scene memory, health monitoring
/home/missy/missy/docs/architecture.md:62: 1. CLI (click)           Parse command-line arguments, resolve --config path
/home/missy/missy/docs/architecture.md:64: 2. Config loader         Read YAML, auto-migrate if needed, build MissyConfig
/home/missy/missy/docs/architecture.md:67: 3. Subsystem init        init_policy_engine(cfg)  -- network, filesystem, shell, REST L7
/home/missy/missy/docs/architecture.md:68:        |                 init_audit_logger(cfg.audit_log_path) + AgentIdentity (Ed25519)
/home/missy/missy/docs/architecture.md:69:        |                 init_registry(cfg) -- providers with rate limiter + fallback
/home/missy/missy/docs/architecture.md:71:        |                 init_tool_registry() -- 18+ built-in tools + MCP servers
/home/missy/missy/docs/architecture.md:78:        |                 Resolve provider (with fallback + circuit breaker)
/home/missy/missy/docs/architecture.md:79:        |                 AttentionSystem extracts urgency/topics/focus
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
/home/missy/missy/docs/architecture.md:211:  +-> config/settings + config/migrate + config/hotreload + config/plan
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
/home/missy/missy/docs/architecture.md:244:  +-> config/settings
/home/missy/missy/docs/architecture.md:247:  +-> policy/engine + policy/rest_policy
/home/missy/missy/docs/architecture.md:248:  +-> agent/interactive_approval
/home/missy/missy/docs/architecture.md:251:policy/engine
/home/missy/missy/docs/architecture.md:252:  +-> policy/network + policy/filesystem + policy/shell + policy/rest_policy
/home/missy/missy/docs/architecture.md:253:  +-> policy/presets
/home/missy/missy/docs/architecture.md:254:  +-> config/settings
/home/missy/missy/docs/architecture.md:259:  +-> tools/registry
/home/missy/missy/docs/architecture.md:265:scheduler/manager
/home/missy/missy/docs/architecture.md:266:  +-> scheduler/parser + scheduler/jobs
/home/missy/missy/docs/architecture.md:271:  +-> providers/base (for image formatting)
/home/missy/missy/docs/architecture.md:280:1. `load_config(path)` -- parse YAML into `MissyConfig`
/home/missy/missy/docs/architecture.md:281:2. `init_policy_engine(cfg)` -- must come first; other subsystems depend on it
/home/missy/missy/docs/architecture.md:282:3. `init_audit_logger(cfg.audit_log_path)` -- wraps the event bus
/home/missy/missy/docs/architecture.md:283:4. `init_registry(cfg)` -- constructs provider instances
/home/missy/missy/docs/architecture.md:300:| Policy engine | `init_policy_engine(cfg)` | `get_policy_engine()` |
/home/missy/missy/docs/architecture.md:301:| Provider registry | `init_registry(cfg)` | `get_registry()` |
/home/missy/missy/docs/architecture.md:302:| Audit logger | `init_audit_logger(path)` | `get_audit_logger()` |
/home/missy/missy/docs/architecture.md:303:| Plugin loader | `init_plugin_loader(cfg)` | `get_plugin_loader()` |
/home/missy/missy/docs/architecture.md:304:| Skill registry | `init_skill_registry()` | `get_skill_registry()` |
/home/missy/missy/docs/architecture.md:305:| Tool registry | `init_tool_registry()` | `get_tool_registry()` |
/home/missy/missy/install.sh:29:    echo "Error: Python 3.11+ is required." >&2
/home/missy/missy/install.sh:37:    echo "Error: git is required." >&2
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
/home/missy/missy/docs/README.md:3:## User Guides
/home/missy/missy/docs/README.md:5:| Guide | Description |
/home/missy/missy/docs/README.md:8:| [Configuration](configuration.md) | Complete YAML reference with annotated examples |
/home/missy/missy/docs/README.md:9:| [Providers](providers.md) | Anthropic, OpenAI, Ollama setup and API key management |
/home/missy/missy/docs/README.md:10:| [Discord](discord.md) | Discord bot integration, access control, slash commands |
/home/missy/missy/docs/README.md:11:| [Scheduler](scheduler.md) | Job scheduling with human-friendly syntax |
/home/missy/missy/docs/README.md:12:| [Skills & Plugins](skills-and-plugins.md) | Extension system: tools, skills, plugins |
/home/missy/missy/docs/README.md:17:| Guide | Description |
/home/missy/missy/docs/README.md:20:| [Security](security.md) | Security policy, hardening guide, vulnerability reporting |
/home/missy/missy/docs/README.md:23:| [Testing](testing.md) | Test suite layout, coverage, writing tests |
/home/missy/missy/docs/README.md:27:| Guide | Description |
/home/missy/missy/docs/README.md:29:| [Discord](discord.md) | Discord bot integration, access control, slash commands |
/home/missy/missy/docs/README.md:38:| [Policy Engine](implementation/policy-engine.md) | `missy/policy/` |
/home/missy/missy/docs/README.md:39:| [Provider Abstraction](implementation/provider-abstraction.md) | `missy/providers/` |
/home/missy/missy/docs/README.md:42:| [Audit Events](implementation/audit-events.md) | `missy/observability/` |
/home/missy/missy/docs/README.md:43:| [Persistence Schema](implementation/persistence-schema.md) | `missy/memory/`, `missy/scheduler/` |
/home/missy/missy/docs/README.md:44:| [Scheduler Execution](implementation/scheduler-execution.md) | `missy/scheduler/` |
/home/missy/missy/docs/README.md:46:| [Manifest Schema](implementation/manifest-schema.md) | Plugin/skill manifests |
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260709-004527
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:15:  - Delta streams and full-content resend reconciliation.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:20:  - Reasoning stream mode.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:24:- A1 runtime coverage: `tests/agent/test_runtime_streaming.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:25:  - Existing streaming behavior still yields chunks.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:26:  - Split think tags are stripped in `AgentRuntime.run_stream()`.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:36:- A2 config coverage: `tests/config/test_settings.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:39:- A2 runtime coverage: `tests/agent/test_runtime_streaming.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:41:- A2 runtime coverage: `tests/agent/test_runtime_config_edges.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:48:| H_A | Delay calculation respects length/complexity/mood/channel caps; quick/fast/asap bypasses long sleeps; channel typing indicator ordering is mocked. |
/home/missy/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
```
