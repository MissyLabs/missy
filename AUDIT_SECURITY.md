# AUDIT_SECURITY

- Timestamp: 2026-07-08 13:44:43

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and OpenAI provider scan
```
/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/BUILD_STATUS.md:7:Primary focus is back on the OpenAI provider overhaul. The provider layer keeps
/home/missy/missy/BUILD_STATUS.md:8:OpenAI-specific assumptions inside `missy/providers/openai_provider.py` and the
/home/missy/missy/BUILD_STATUS.md:9:shared provider abstraction remains provider-neutral.
/home/missy/missy/BUILD_STATUS.md:11:This session hardened OpenAI message normalization and transcript repair:
/home/missy/missy/BUILD_STATUS.md:16:  `https://` URLs; unsafe schemes are stripped before provider invocation.
/home/missy/missy/BUILD_STATUS.md:17:- Added OpenAI tool-turn validation so assistant tool calls with missing IDs or
/home/missy/missy/BUILD_STATUS.md:18:  names, duplicate tool-call IDs, and orphaned tool-result messages are removed
/home/missy/missy/BUILD_STATUS.md:20:- Added `provider_transcript_repair` audit events for OpenAI transcript repairs
/home/missy/missy/BUILD_STATUS.md:21:  with session/task correlation when available.
/home/missy/missy/BUILD_STATUS.md:23:  API migration target in `docs/providers.md`.
/home/missy/missy/BUILD_STATUS.md:29:| Provider interface compliance | improved | OpenAI still returns canonical `CompletionResponse` and isolates provider-specific message repair internally. |
/home/missy/missy/BUILD_STATUS.md:31:| Network policy integration | in place | SDK client is built with policy-aware HTTP where available. |
/home/missy/missy/BUILD_STATUS.md:33:| Tool schema normalization | in place | OpenAI tool schemas delegate to `schema_adapter.normalize_for_provider()`. |
/home/missy/missy/BUILD_STATUS.md:34:| Tool transcript repair | improved | Invalid/duplicate assistant tool calls and orphan tool results are dropped before request. |
/home/missy/missy/BUILD_STATUS.md:36:| Auditability | improved | Transcript repair emits structured provider audit events. |
/home/missy/missy/BUILD_STATUS.md:37:| Tests | improved | Added OpenAI provider tests for safe vision parts, unsafe image stripping, and orphan tool-result repair audit. |
/home/missy/missy/BUILD_STATUS.md:42:  Missy's existing provider abstraction and OpenAI-compatible `base_url`
/home/missy/missy/BUILD_STATUS.md:44:- OpenAI-specific content and tool-turn validation happens before SDK calls, so
/home/missy/missy/BUILD_STATUS.md:45:  unrelated providers do not inherit OpenAI transcript assumptions.
/home/missy/missy/BUILD_STATUS.md:47:  rewritten into potentially incorrect tool state.
/home/missy/missy/BUILD_STATUS.md:56:- `python3 -m ruff format --check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
/home/missy/missy/BUILD_STATUS.md:57:- `python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
/home/missy/missy/BUILD_STATUS.md:58:- `python3 -m pytest tests/providers/test_openai_provider.py tests/providers/test_openai.py -q`: 42 passed.
/home/missy/missy/BUILD_STATUS.md:59:- `python3 -m pytest tests/providers -q`: 833 passed.
/home/missy/missy/BUILD_STATUS.md:67:   OpenAI-compatible Chat Completions fallback for `base_url` providers.
/home/missy/missy/BUILD_STATUS.md:68:2. Expand OpenAI streaming reconciliation to cover provider-native tool-call
/home/missy/missy/BUILD_STATUS.md:70:3. Add structured output support that can use OpenAI-native response formats
/home/missy/missy/BUILD_STATUS.md:72:4. Add embeddings support if vector-memory workflows need an external OpenAI
/home/missy/missy/BUILD_STATUS.md:73:   embedding backend.
/home/missy/missy/BUILD_STATUS.md:74:5. Add provider diagnostics/doctor checks for OpenAI credentials, model list,
/home/missy/missy/BUILD_STATUS.md:75:   network policy, rate-limit posture, and redaction.
/home/missy/missy/BUILD_STATUS.md:76:6. Extend audit events for retry, rate-limit cooldown, usage/cost recording,
/home/missy/missy/BUILD_STATUS.md:77:   fallback, and provider-side validation denials.
/home/missy/missy/BUILD_STATUS.md:81:- No code blocker for the next OpenAI provider slice.
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Re-centered the loop tracking state on the OpenAI provider overhaul.
/home/missy/missy/LAST_SESSION_SUMMARY.md:11:  image URLs while stripping unsafe schemes before provider invocation.
/home/missy/missy/LAST_SESSION_SUMMARY.md:12:- Added OpenAI transcript repair for invalid assistant tool calls, duplicate
/home/missy/missy/LAST_SESSION_SUMMARY.md:13:  tool-call IDs, and orphaned tool-result messages.
/home/missy/missy/LAST_SESSION_SUMMARY.md:14:- Added `provider_transcript_repair` audit events for OpenAI repair decisions.
/home/missy/missy/LAST_SESSION_SUMMARY.md:15:- Added focused OpenAI provider tests for vision payload preservation, unsafe
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  image URL stripping, and orphan tool-result repair audit.
/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Updated `docs/providers.md` with the current OpenAI adapter behavior and the
/home/missy/missy/LAST_SESSION_SUMMARY.md:23:python3 -m ruff format --check missy/providers/openai_provider.py tests/providers/test_openai_provider.py
/home/missy/missy/LAST_SESSION_SUMMARY.md:28:python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py
/home/missy/missy/LAST_SESSION_SUMMARY.md:33:python3 -m pytest tests/providers/test_openai_provider.py tests/providers/test_openai.py -q
/home/missy/missy/LAST_SESSION_SUMMARY.md:38:python3 -m pytest tests/providers -q
/home/missy/missy/LAST_SESSION_SUMMARY.md:45:  compatibility fallback for OpenAI-compatible providers.
/home/missy/missy/LAST_SESSION_SUMMARY.md:46:- Streaming needs robust tool-call delta reconciliation and final transcript
/home/missy/missy/LAST_SESSION_SUMMARY.md:49:- OpenAI diagnostics, embeddings, cost accounting, and retry/fallback audit
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:7:The active primary focus is the OpenAI provider overhaul. OpenClaw and Odin
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:remain references for provider-turn validation, raw stream diagnostics,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:9:operator ergonomics, safe controls, and auditability; Missy implementation
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:18:| Network policy integration | in place | SDK client attempts policy-aware HTTP wiring. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Tool schema normalization | live | Uses provider schema adapter. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:23:| Tool transcript repair | improved | Invalid/duplicate/orphaned tool turns are removed before SDK call. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:25:| Streaming reconciliation | partial | Text delta streaming exists; tool-call/full-content reconciliation remains. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:27:| Embeddings | not_started | Needed only if external vector workflows require OpenAI embeddings. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:29:| Audit events | improved | Provider invoke/error and transcript repair are covered; retry/fallback/cost events remain. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:36:| A2 | Layered tool policy pipeline | hardened | Provider/tool policy surfaces remain active. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:37:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated failing tool calls are surfaced to the model. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:38:| A4 | Compaction retry coordination | not_started | Manager-level retry coordination remains future work. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:39:| A5 | Auth profile cooldown + fallback | not_started | Preserve pinned provider behavior. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:40:| A6 | Per-provider tool schema normalization | live | OpenAI delegates to schema adapter. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:45:| A11 | Raw-stream JSONL diagnostics | not_started | Relevant to OpenAI streaming and Web TUI run viewer. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:46:| A12 | Transcript dual-repair | improved | OpenAI now repairs invalid/orphaned tool turns before SDK calls. |
/home/missy/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file under 5 MB. Move older entries to timestamped archive files before appending more.
/home/missy/missy/HUMANIZE_AUDIT.md:7:| 2026-04-27T16:09:36Z | humanize.loop.initialized | allow | Initialized audit file for the OpenClaw/humanize loop. No opt-in humanistic behavior was activated this session. |
/home/missy/missy/HUMANIZE_AUDIT.md:8:| 2026-04-27T16:09:36Z | openclaw.a1.subscription | allow | Added streaming state machine primitives that can support future timing, tone, apology, and mood integrations without changing tool correctness. |
/home/missy/missy/HUMANIZE_AUDIT.md:9:| 2026-04-27T18:32:16Z | openclaw.a2.tool_policy | allow | Added layered tool availability filtering with trace labels. This gates future humanistic memory tools without changing execution fail-closed policy. |
/home/missy/missy/HUMANIZE_AUDIT.md:10:| 2026-04-27T18:53:28Z | openclaw.a2.config_policy | allow | Routed YAML-backed provider/global/agent/sandbox/subagent tool policy layers into runtime exposure decisions. Execution policy remains fail-closed in the registry. |
/home/missy/missy/HATCHING_LOG.md:30:| `details` | object | Optional structured metadata |
/home/missy/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
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
/home/missy/missy/AUDIT_SECURITY.md:11:## Security and OpenAI provider scan
/home/missy/missy/AUDIT_SECURITY.md:13:/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/AUDIT_SECURITY.md:14:/home/missy/missy/BUILD_STATUS.md:7:Primary focus is back on the OpenAI provider overhaul. The provider layer keeps
/home/missy/missy/AUDIT_SECURITY.md:15:/home/missy/missy/BUILD_STATUS.md:8:OpenAI-specific assumptions inside `missy/providers/openai_provider.py` and the
/home/missy/missy/AUDIT_SECURITY.md:16:/home/missy/missy/BUILD_STATUS.md:9:shared provider abstraction remains provider-neutral.
/home/missy/missy/AUDIT_SECURITY.md:17:/home/missy/missy/BUILD_STATUS.md:11:This session hardened OpenAI message normalization and transcript repair:
/home/missy/missy/AUDIT_SECURITY.md:18:/home/missy/missy/BUILD_STATUS.md:16:  `https://` URLs; unsafe schemes are stripped before provider invocation.
/home/missy/missy/AUDIT_SECURITY.md:19:/home/missy/missy/BUILD_STATUS.md:17:- Added OpenAI tool-turn validation so assistant tool calls with missing IDs or
/home/missy/missy/AUDIT_SECURITY.md:20:/home/missy/missy/BUILD_STATUS.md:18:  names, duplicate tool-call IDs, and orphaned tool-result messages are removed
/home/missy/missy/AUDIT_SECURITY.md:21:/home/missy/missy/BUILD_STATUS.md:20:- Added `provider_transcript_repair` audit events for OpenAI transcript repairs
/home/missy/missy/AUDIT_SECURITY.md:22:/home/missy/missy/BUILD_STATUS.md:21:  with session/task correlation when available.
/home/missy/missy/AUDIT_SECURITY.md:23:/home/missy/missy/BUILD_STATUS.md:23:  API migration target in `docs/providers.md`.
/home/missy/missy/AUDIT_SECURITY.md:24:/home/missy/missy/BUILD_STATUS.md:29:| Provider interface compliance | improved | OpenAI still returns canonical `CompletionResponse` and isolates provider-specific message repair internally. |
/home/missy/missy/AUDIT_SECURITY.md:25:/home/missy/missy/BUILD_STATUS.md:31:| Network policy integration | in place | SDK client is built with policy-aware HTTP where available. |
/home/missy/missy/AUDIT_SECURITY.md:26:/home/missy/missy/BUILD_STATUS.md:33:| Tool schema normalization | in place | OpenAI tool schemas delegate to `schema_adapter.normalize_for_provider()`. |
/home/missy/missy/AUDIT_SECURITY.md:27:/home/missy/missy/BUILD_STATUS.md:34:| Tool transcript repair | improved | Invalid/duplicate assistant tool calls and orphan tool results are dropped before request. |
/home/missy/missy/AUDIT_SECURITY.md:28:/home/missy/missy/BUILD_STATUS.md:36:| Auditability | improved | Transcript repair emits structured provider audit events. |
/home/missy/missy/AUDIT_SECURITY.md:29:/home/missy/missy/BUILD_STATUS.md:37:| Tests | improved | Added OpenAI provider tests for safe vision parts, unsafe image stripping, and orphan tool-result repair audit. |
/home/missy/missy/AUDIT_SECURITY.md:30:/home/missy/missy/BUILD_STATUS.md:42:  Missy's existing provider abstraction and OpenAI-compatible `base_url`
/home/missy/missy/AUDIT_SECURITY.md:31:/home/missy/missy/BUILD_STATUS.md:44:- OpenAI-specific content and tool-turn validation happens before SDK calls, so
/home/missy/missy/AUDIT_SECURITY.md:32:/home/missy/missy/BUILD_STATUS.md:45:  unrelated providers do not inherit OpenAI transcript assumptions.
/home/missy/missy/AUDIT_SECURITY.md:33:/home/missy/missy/BUILD_STATUS.md:47:  rewritten into potentially incorrect tool state.
/home/missy/missy/AUDIT_SECURITY.md:34:/home/missy/missy/BUILD_STATUS.md:56:- `python3 -m ruff format --check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
/home/missy/missy/AUDIT_SECURITY.md:35:/home/missy/missy/BUILD_STATUS.md:57:- `python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py`: passed.
/home/missy/missy/AUDIT_SECURITY.md:36:/home/missy/missy/BUILD_STATUS.md:58:- `python3 -m pytest tests/providers/test_openai_provider.py tests/providers/test_openai.py -q`: 42 passed.
/home/missy/missy/AUDIT_SECURITY.md:37:/home/missy/missy/BUILD_STATUS.md:59:- `python3 -m pytest tests/providers -q`: 833 passed.
/home/missy/missy/AUDIT_SECURITY.md:38:/home/missy/missy/BUILD_STATUS.md:67:   OpenAI-compatible Chat Completions fallback for `base_url` providers.
/home/missy/missy/AUDIT_SECURITY.md:39:/home/missy/missy/BUILD_STATUS.md:68:2. Expand OpenAI streaming reconciliation to cover provider-native tool-call
/home/missy/missy/AUDIT_SECURITY.md:40:/home/missy/missy/BUILD_STATUS.md:70:3. Add structured output support that can use OpenAI-native response formats
/home/missy/missy/AUDIT_SECURITY.md:41:/home/missy/missy/BUILD_STATUS.md:72:4. Add embeddings support if vector-memory workflows need an external OpenAI
/home/missy/missy/AUDIT_SECURITY.md:42:/home/missy/missy/BUILD_STATUS.md:73:   embedding backend.
/home/missy/missy/AUDIT_SECURITY.md:43:/home/missy/missy/BUILD_STATUS.md:74:5. Add provider diagnostics/doctor checks for OpenAI credentials, model list,
/home/missy/missy/AUDIT_SECURITY.md:44:/home/missy/missy/BUILD_STATUS.md:75:   network policy, rate-limit posture, and redaction.
/home/missy/missy/AUDIT_SECURITY.md:45:/home/missy/missy/BUILD_STATUS.md:76:6. Extend audit events for retry, rate-limit cooldown, usage/cost recording,
/home/missy/missy/AUDIT_SECURITY.md:46:/home/missy/missy/BUILD_STATUS.md:77:   fallback, and provider-side validation denials.
/home/missy/missy/AUDIT_SECURITY.md:47:/home/missy/missy/BUILD_STATUS.md:81:- No code blocker for the next OpenAI provider slice.
/home/missy/missy/AUDIT_SECURITY.md:48:/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Re-centered the loop tracking state on the OpenAI provider overhaul.
/home/missy/missy/AUDIT_SECURITY.md:49:/home/missy/missy/LAST_SESSION_SUMMARY.md:11:  image URLs while stripping unsafe schemes before provider invocation.
/home/missy/missy/AUDIT_SECURITY.md:50:/home/missy/missy/LAST_SESSION_SUMMARY.md:12:- Added OpenAI transcript repair for invalid assistant tool calls, duplicate
/home/missy/missy/AUDIT_SECURITY.md:51:/home/missy/missy/LAST_SESSION_SUMMARY.md:13:  tool-call IDs, and orphaned tool-result messages.
/home/missy/missy/AUDIT_SECURITY.md:52:/home/missy/missy/LAST_SESSION_SUMMARY.md:14:- Added `provider_transcript_repair` audit events for OpenAI repair decisions.
/home/missy/missy/AUDIT_SECURITY.md:53:/home/missy/missy/LAST_SESSION_SUMMARY.md:15:- Added focused OpenAI provider tests for vision payload preservation, unsafe
/home/missy/missy/AUDIT_SECURITY.md:54:/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  image URL stripping, and orphan tool-result repair audit.
/home/missy/missy/AUDIT_SECURITY.md:55:/home/missy/missy/LAST_SESSION_SUMMARY.md:17:- Updated `docs/providers.md` with the current OpenAI adapter behavior and the
/home/missy/missy/AUDIT_SECURITY.md:56:/home/missy/missy/LAST_SESSION_SUMMARY.md:23:python3 -m ruff format --check missy/providers/openai_provider.py tests/providers/test_openai_provider.py
/home/missy/missy/AUDIT_SECURITY.md:57:/home/missy/missy/LAST_SESSION_SUMMARY.md:28:python3 -m ruff check missy/providers/openai_provider.py tests/providers/test_openai_provider.py
/home/missy/missy/AUDIT_SECURITY.md:58:/home/missy/missy/LAST_SESSION_SUMMARY.md:33:python3 -m pytest tests/providers/test_openai_provider.py tests/providers/test_openai.py -q
/home/missy/missy/AUDIT_SECURITY.md:59:/home/missy/missy/LAST_SESSION_SUMMARY.md:38:python3 -m pytest tests/providers -q
/home/missy/missy/AUDIT_SECURITY.md:60:/home/missy/missy/LAST_SESSION_SUMMARY.md:45:  compatibility fallback for OpenAI-compatible providers.
/home/missy/missy/AUDIT_SECURITY.md:61:/home/missy/missy/LAST_SESSION_SUMMARY.md:46:- Streaming needs robust tool-call delta reconciliation and final transcript
/home/missy/missy/AUDIT_SECURITY.md:62:/home/missy/missy/LAST_SESSION_SUMMARY.md:49:- OpenAI diagnostics, embeddings, cost accounting, and retry/fallback audit
/home/missy/missy/AUDIT_SECURITY.md:63:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:7:The active primary focus is the OpenAI provider overhaul. OpenClaw and Odin
/home/missy/missy/AUDIT_SECURITY.md:64:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:remain references for provider-turn validation, raw stream diagnostics,
/home/missy/missy/AUDIT_SECURITY.md:65:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:9:operator ergonomics, safe controls, and auditability; Missy implementation
/home/missy/missy/AUDIT_SECURITY.md:66:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:18:| Network policy integration | in place | SDK client attempts policy-aware HTTP wiring. |
/home/missy/missy/AUDIT_SECURITY.md:67:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Tool schema normalization | live | Uses provider schema adapter. |
/home/missy/missy/AUDIT_SECURITY.md:68:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:23:| Tool transcript repair | improved | Invalid/duplicate/orphaned tool turns are removed before SDK call. |
/home/missy/missy/AUDIT_SECURITY.md:69:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:25:| Streaming reconciliation | partial | Text delta streaming exists; tool-call/full-content reconciliation remains. |
/home/missy/missy/AUDIT_SECURITY.md:70:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:27:| Embeddings | not_started | Needed only if external vector workflows require OpenAI embeddings. |
/home/missy/missy/AUDIT_SECURITY.md:71:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:29:| Audit events | improved | Provider invoke/error and transcript repair are covered; retry/fallback/cost events remain. |
/home/missy/missy/AUDIT_SECURITY.md:72:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:36:| A2 | Layered tool policy pipeline | hardened | Provider/tool policy surfaces remain active. |
/home/missy/missy/AUDIT_SECURITY.md:73:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:37:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated failing tool calls are surfaced to the model. |
/home/missy/missy/AUDIT_SECURITY.md:74:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:38:| A4 | Compaction retry coordination | not_started | Manager-level retry coordination remains future work. |
/home/missy/missy/AUDIT_SECURITY.md:75:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:39:| A5 | Auth profile cooldown + fallback | not_started | Preserve pinned provider behavior. |
/home/missy/missy/AUDIT_SECURITY.md:76:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:40:| A6 | Per-provider tool schema normalization | live | OpenAI delegates to schema adapter. |
/home/missy/missy/AUDIT_SECURITY.md:77:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:45:| A11 | Raw-stream JSONL diagnostics | not_started | Relevant to OpenAI streaming and Web TUI run viewer. |
/home/missy/missy/AUDIT_SECURITY.md:78:/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:46:| A12 | Transcript dual-repair | improved | OpenAI now repairs invalid/orphaned tool turns before SDK calls. |
/home/missy/missy/AUDIT_SECURITY.md:79:/home/missy/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file 
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
```
