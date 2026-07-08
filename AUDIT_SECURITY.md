# AUDIT_SECURITY

- Timestamp: 2026-07-08 14:42:39

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and OpenAI provider scan
```
/home/missy/missy/LOOP_INSTRUCTIONS.md:5:Primary focus: complete OpenAI provider overhaul
/home/missy/missy/LOOP_INSTRUCTIONS.md:6:Branch: overhaul/openai-provider-20260708-172558
/home/missy/missy/LOOP_INSTRUCTIONS.md:8:Requirements:
/home/missy/missy/LOOP_INSTRUCTIONS.md:10:- Treat complete OpenAI provider overhaul as the current primary overhaul.
/home/missy/missy/LOOP_INSTRUCTIONS.md:11:- Keep the design flexible for future non-OpenAI-provider overhauls.
/home/missy/missy/HATCHING_LOG.md:30:| `details` | object | Optional structured metadata |
/home/missy/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Added a provider-neutral `structured_output_kwargs(schema)` hook to
/home/missy/missy/LAST_SESSION_SUMMARY.md:9:- Updated `StructuredOutputRunner` to pass native structured-output kwargs into
/home/missy/missy/LAST_SESSION_SUMMARY.md:10:  every sync and async provider attempt while retaining Missy's Pydantic
/home/missy/missy/LAST_SESSION_SUMMARY.md:11:  validation and retry behavior.
/home/missy/missy/LAST_SESSION_SUMMARY.md:12:- Added OpenAI-native structured output formatting for:
/home/missy/missy/LAST_SESSION_SUMMARY.md:15:- Sanitized OpenAI response-format schema names and preserved schema
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Added tests for the generic structured-output provider hook and OpenAI
/home/missy/missy/LAST_SESSION_SUMMARY.md:18:  Responses/Chat structured-output request construction.
/home/missy/missy/LAST_SESSION_SUMMARY.md:19:- Updated provider docs and provider-abstraction implementation docs.
/home/missy/missy/LAST_SESSION_SUMMARY.md:24:python3 -m ruff format missy/providers/base.py missy/agent/structured_output.py missy/providers/openai_provider.py tests/agent/test_structured_output.py tests/providers/test_openai_provider.py
/home/missy/missy/LAST_SESSION_SUMMARY.md:29:python3 -m pytest tests/agent/test_structured_output.py tests/providers/test_openai_provider.py -q
/home/missy/missy/LAST_SESSION_SUMMARY.md:34:python3 -m pytest tests/providers -q
/home/missy/missy/LAST_SESSION_SUMMARY.md:60:- Native Responses tool/function calling still needs a transcript model that
/home/missy/missy/LAST_SESSION_SUMMARY.md:62:- Streamed provider-native tool-call deltas and final validation remain future
/home/missy/missy/LAST_SESSION_SUMMARY.md:64:- OpenAI diagnostics, embeddings, cost accounting, retry/fallback audit
/home/missy/missy/LAST_SESSION_SUMMARY.md:71:Implement OpenAI provider diagnostics/doctor checks or design the Responses
/home/missy/missy/LAST_SESSION_SUMMARY.md:72:tool/function-call transcript model before adding native Responses tools.
/home/missy/missy/AUDIT_SECURITY.md:11:## Security and OpenAI provider scan
/home/missy/missy/examples/systemd/README.md:8:## Prerequisites
/home/missy/missy/examples/systemd/README.md:16:- Any required API keys must be set as environment variables. You can use a
/home/missy/missy/examples/systemd/README.md:51:sudo systemctl enable --now missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:54:This enables the service to start on boot and starts it immediately.
/home/missy/missy/examples/systemd/README.md:91:sudo systemctl disable missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:102:| `NoNewPrivileges=true` | Prevents the process from gaining new privileges (e.g. via setuid). |
/home/missy/missy/examples/systemd/README.md:106:| `ReadWritePaths=...` | Grants write access to `~/.missy` (for audit logs, jobs, memory) and `~/workspace` (for agent output). |
/home/missy/missy/examples/systemd/README.md:120:- Check that `config.yaml` includes the required domains in `allowed_domains` or `allowed_hosts`.
/home/missy/missy/examples/systemd/README.md:124:- Check that `discord.enabled: true` is set in `config.yaml`.
/home/missy/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/HUMANIZE_STATUS.md:12:| A4 | Compaction retry coordination | not_started | A1 tracks retry state locally; runtime manager work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:19:| A11 | Raw-stream JSONL diagnostics | not_started | A1 exposes `raw_stream_callback`; observability module remains. |
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
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/openai-provider-20260708-172558
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete OpenAI provider overhaul
/home/missy/missy/TEST_RESULTS.md:294:  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
/home/missy/missy/TEST_RESULTS.md:297:  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
/home/missy/missy/TEST_RESULTS.md:300:  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
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
/home/missy/missy/README.md:35:- **Sleep mode** — context consolidation at 80% token capacity: summarizes old turns, extracts key facts, preserves recent context
/home/missy/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy/README.md:41:- **Progress reporting** — structured protocol with Null/Audit/CLI reporter implementations
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
/home/missy/missy/README.md:243:    timeout: 30
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
/home/missy/missy/README.md:318:missy cost                          # Budget status
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
/home/missy/missy/README.md:394:│                    checkpoint, cost tracking, sleeptime, condensers, code evolution
/home/missy/missy/README.md:396:├── channels/        CLI, Discord, webhooks, voice (WebSocket), screencast (browser)
/home/missy/missy/README.md:404:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/README.md:405:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy/README.md:406:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy/README.md:409:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy/README.md:410:├── plugins/         Security-gated external plugin loader
/home/missy/missy/README.md:411:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:15:  - Delta streams and full-content resend reconciliation.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:20:  - Reasoning stream mode.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:23:  - Compaction retry state transitions.
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
/home/missy/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:39:- A2 runtime coverage: `tests/agent/test_runtime_streaming.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:48:| H_A | Delay calculation respects length/complexity/mood/channel caps; quick/fast/asap bypasses long sleeps; channel typing indicator ordering is mocked. |
/home/missy/missy/HUMANIZE_TEST_PLAN.md:51:| H_D | Promised follow-up parser schedules implied time; TopicResume observes idle threshold; rate limit blocks second unsolicited message within 6 hours. |
/home/missy/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
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
/home/missy/missy/HATCHING.md:5:## Quick Start
/home/missy/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy/HATCHING.md:61:All hatching events are logged to `~/.missy/hatching_log.jsonl` in structured JSONL format. Each entry includes timestamp, step name, status, and message.
/home/missy/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
/home/missy/missy/CLAUDE.md:3:This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
/home/missy/missy/CLAUDE.md:7:**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. Production-grade agent platform with strict security controls, policy enforcement, and full auditability. Python 3.11+.
/home/missy/missy/CLAUDE.md:19:pip install -e ".[discord_voice]" # adds discord.py[voice] + voice recv; requires system ffmpeg
/home/missy/missy/CLAUDE.md:23:python3 -m pytest tests/unit/test_policy_engine.py -v     # single file
/home/missy/missy/CLAUDE.md:36:Secure-by-default: all capabilities (shell, plugins, network) are disabled until explicitly enabled in `~/.missy/config.yaml`.
/home/missy/missy/CLAUDE.md:42:  → missy setup (wizard.py + oauth.py + anthropic_auth.py)
/home/missy/missy/CLAUDE.md:48:       ├─ PolicyEngine (policy/engine.py) + RestPolicy (policy/rest_policy.py)
/home/missy/missy/CLAUDE.md:49:       ├─ AgentIdentity (security/identity.py) — Ed25519 keypair, signs audit events
/home/missy/missy/CLAUDE.md:50:       ├─ TrustScorer (security/trust.py) — 0-1000 reliability tracking per tool/provider
/home/missy/missy/CLAUDE.md:51:       ├─ CircuitBreaker (agent/circuit_breaker.py)
/home/missy/missy/CLAUDE.md:53:       ├─ ContextManager (agent/context.py) — token budget with memory/learnings injection
/home/missy/missy/CLAUDE.md:57:       ├─ ProviderRegistry + ModelRouter (providers/registry.py)
/home/missy/missy/CLAUDE.md:58:       ├─ RateLimiter (providers/rate_limiter.py)
/home/missy/missy/CLAUDE.md:59:       ├─ PolicyHTTPClient (gateway/client.py) + InteractiveApproval (agent/interactive_approval.py)
/home/missy/missy/CLAUDE.md:60:       ├─ ToolRegistry (tools/registry.py) + built-in tools
/home/missy/missy/CLAUDE.md:62:       ├─ SkillDiscovery (skills/discovery.py) — SKILL.md dynamic skill loading
/home/missy/missy/CLAUDE.md:68:       ├─ ApprovalGate (agent/approval.py)
/home/missy/missy/CLAUDE.md:72:       ├─ CostTracker (agent/cost_tracker.py) — per-session spend monitoring + budget caps
/home/missy/missy/CLAUDE.md:74:       ├─ FailureTracker (agent/failure_tracker.py) — per-tool failure counts + strategy rotation
/home/missy/missy/CLAUDE.md:77:       ├─ CodeEvolutionManager (agent/code_evolution.py) — self-evolving code with approval + git rollback
/home/missy/missy/CLAUDE.md:78:       ├─ StructuredOutput (agent/structured_output.py) — Pydantic schema enforcement on LLM responses
/home/missy/missy/CLAUDE.md:85:       ├─ SecurityScanner (security/scanner.py) — installation security auditing
/home/missy/missy/CLAUDE.md:96:  → Browser-based screen capture with token auth + session management
/home/missy/missy/CLAUDE.md:108:**Policy Engine (`missy/policy/`)** — Multi-layer enforcement facade:
/home/missy/missy/CLAUDE.md:109:- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, per-category host allowlists (provider, tool, discord)
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
/home/missy/missy/CLAUDE.md:145:- `CodeEvolutionManager`: Self-evolving code modification engine with approval workflow, git-backed rollback, and `missy evolve` CLI.
/home/missy/missy/CLAUDE.md:146:- `StructuredOutput`: Pydantic model schema enforcement on LLM responses with automatic retry on validation failure.
/home/missy/missy/CLAUDE.md:152:**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`. Digest pinning (`missy mcp pin`) records SHA-256 of tool manifests; mismatches refuse to load.
/home/missy/missy/CLAUDE.md:154:**Skills (`missy/skills/`)** — `SkillDiscovery` scans directories for SKILL.md files (cross-agent portable skill format with YAML frontmatter). `missy skills scan` lists discovered skills. Fuzzy search by name/description.
/home/missy/missy/CLAUDE.md:156:**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions.
/home/missy/missy/CLAUDE.md:163:- `AgentIdentity`: Ed25519 keypair at `~/.missy/identity.pem`. Signs audit events. JWK export.
/home/missy/missy/CLAUDE.md:164:- `TrustScorer`: 0-1000 reliability tracking per tool/provider/MCP server. Success (+10), failure (-50), violation (-200). Warns below threshold.
/home/missy/missy/CLAUDE.md:165:- `PromptDriftDetector`: SHA-256 hashes system prompts at start, verifies before each provider call. Emits `security.prompt_drift` audit event on tamper.
/home/missy/missy/CLAUDE.md:166:- `ContainerSandbox`: Optional Docker-based isolation for tool execution. Per-session containers with `--network=none`, memory/CPU limits. Config: `container: { enabled: true }`.
/home/missy/missy/CLAUDE.md:167:- `LandlockPolicy`: Linux Landlock LSM filesystem policy enforcement via ctypes syscalls. Kernel-level read/write path restrictions complementing userspace policy engine.
/home/missy/missy/CLAUDE.md:168:- `SecurityScanner`: Installation security auditor (`missy security scan`). Checks file permissions, config hygiene, exposed secrets, and reports severity-ranked findings.
/home/missy/missy/CLAUDE.md:172:- `CameraHandle`: OpenCV-based capture with warm-up, retry, blank-frame detection
/home/missy/missy/CLAUDE.md:176:- `AnalysisPromptBuilder`: Domain-specific prompts (puzzle board-state, painting coaching)
/home/missy/missy/CLAUDE.md:181:- Agent tools: `vision_capture`, `vision_burst`, `vision_analyze`, `vision_devices`, `vision_scene`
/home/missy/missy/CLAUDE.md:184:**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days. Optional `VectorMemoryStore` with FAISS semantic search (`pip install -e ".[vector]"`). `GraphMemoryStore` provides SQLite-backed entity-relationship graph memory with rule-based pattern matching for structured knowledge retrieval.
/home/missy/missy/CLAUDE.md:192:**API Server (`missy/api/`)** — Agent-as-a-Service REST API (`missy api start`). Loopback-only binding by default, API key authentication, rate limiting, and automatic secrets censoring on responses.
/home/missy/missy/CLAUDE.md:194:**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl`. `OtelExporter` sends traces/metrics to an OTLP endpoint when enabled.
/home/missy/missy/CLAUDE.md:202:| Audit log | `~/.missy/audit.jsonl` |
/home/missy/missy/CLAUDE.md:207:| Device registry | `~/.missy/devices.json` |
/home/missy/missy/CLAUDE.md:215:| Persona audit log | `~/.missy/persona_audit.jsonl` |
/home/missy/missy/CLAUDE.md:218:| Skills directory | `~/.missy/skills/` |
/home/missy/missy/CLAUDE.md:228:config_version: 2                    # schema version (auto-migrated on startup)
/home/missy/missy/CLAUDE.md:231:  default_deny: true
/home/missy/missy/CLAUDE.md:233:    - anthropic
/home/missy/missy/CLAUDE.md:238:  provider_allowed_hosts: []      # per-category overrides
/home/missy/missy/CLAUDE.md:239:  tool_allowed_hosts: []
/home/missy/missy/CLAUDE.md:249:      action: "deny"
/home/missy/missy/CLAUDE.md:256:  enabled: false
/home/missy/missy/CLAUDE.md:259:plugins:
/home/missy/missy/CLAUDE.md:260:  enabled: false
/home/missy/missy/CLAUDE.md:261:  allowed_plugins: []
```
