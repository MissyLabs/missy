# AUDIT_SECURITY

- Timestamp: 2026-07-09 14:01:04

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and tool-intelligence scan
```
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:7:- Enforced tool-candidate lifecycle transitions in
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:8:  `missy/tools/intelligence/candidate_store.py`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:10:  `missy.tools.intelligence`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:11:- Invalid candidate transitions now raise `ValueError`, preserve the current
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:12:  state, and emit `tool.candidate.transition_denied` with `result="deny"`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:13:- `missy tools candidates approve|enable|deny` now report lifecycle errors
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:15:- Updated candidate-store tests for benchmark-before-approval and added edge
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:16:  cases for rejected direct enable, rejected pre-benchmark approval, disabled
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:20:  tool-intelligence lifecycle and provider-gate documentation.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:25:python3 -m pytest tests/tools/test_candidate_store.py tests/tools/test_candidate_generator.py tests/tools/test_request_tracker.py tests/tools/test_provider_gate.py tests/agent/test_tool_intelligence_wiring.py tests/agent/test_request_tracker_wiring.py tests/cli/test_tool_provider_cli.py tests/cli/test_benchmark_run_cmd.py -q
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:47:  `ToolCandidate` lifecycle records.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:48:- Enabled candidates still need a controlled runtime loading path with
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:49:  schema/provenance/policy gates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:50:- Web/API operator controls do not yet expose candidate lifecycle actions.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:51:- Provider fallback recommendations exist in CLI/provider gate code but are
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:52:  not yet surfaced in runtime responses when a tool is gated off.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:56:Build benchmark-to-candidate reconciliation so real benchmark data can move
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:57:candidate records to `benchmarked`, persist provider enablement flags, and
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:58:make approval decisions reviewable from CLI/API.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:45:- Session 2 added the A2 layered tool policy pipeline with profile bundles, group expansion, glob matching, inline `-tool` denies, `alsoAllow`, fail-warning unknown allowlists, and structured trace records.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:46:- Session 2 wired `AgentRuntime._get_tools()` to resolve tools through the pipeline and record `_last_tool_policy_decision` for audit/debugging.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:47:- Session 2 added `tests/policy/test_tool_policy_pipeline.py` and runtime coverage for policy decisions in `tests/agent/test_runtime_streaming.py`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:48:- Session 3 added config parsing for `tools.*`, `tools.byProvider`, `tools.byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:49:- Session 3 added `build_configured_tool_policy_layers()` and `collect_tool_policy_groups()` so runtime policy resolution now consumes YAML-backed provider/global/agent/sandbox/subagent layers.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:50:- Session 3 routed parsed tool policies into CLI-created runtimes for ask/run/gateway/API paths and documented the YAML surface in `docs/configuration.md`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:51:- Session 3 added config, policy-pipeline, and runtime tests for those surfaces, then verified the full test suite and full-repo ruff.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:55:1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription` where Missy's providers expose stream events, not only the simple `run_stream()` path.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:56:2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:58:4. Add channel/group policy sources on top of the A2 pipeline when Discord/CLI/Web channel identity context is available.
/home/missy/missy-loops/missy/LOOP_HEALTH.md:5:- Branch: overhaul/tools-20260709-145628
/home/missy/missy-loops/missy/LOOP_HEALTH.md:6:- Primary focus: complete tool usage and tool intelligence overhaul
/home/missy/missy-loops/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy-loops/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy-loops/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy-loops/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy-loops/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy-loops/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy-loops/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy-loops/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy-loops/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy-loops/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy-loops/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy-loops/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy-loops/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy-loops/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy-loops/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy-loops/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy-loops/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy-loops/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy-loops/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy-loops/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy-loops/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy-loops/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy-loops/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy-loops/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy-loops/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy-loops/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy-loops/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy-loops/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy-loops/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy-loops/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy-loops/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy-loops/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy-loops/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy-loops/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy-loops/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy-loops/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy-loops/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy-loops/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy-loops/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy-loops/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy-loops/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy-loops/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy-loops/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy-loops/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy-loops/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy-loops/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy-loops/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy-loops/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy-loops/missy/README.md:217:  default_deny: true
/home/missy/missy-loops/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy-loops/missy/README.md:234:  enabled: false
/home/missy/missy-loops/missy/README.md:237:providers:
/home/missy/missy-loops/missy/README.md:238:  anthropic:
/home/missy/missy-loops/missy/README.md:239:    name: anthropic
/home/missy/missy-loops/missy/README.md:246:  enabled: false
/home/missy/missy-loops/missy/README.md:267:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy-loops/missy/README.md:268:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy-loops/missy/README.md:269:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy-loops/missy/README.md:270:missy providers list                # List providers and availability
/home/missy/missy-loops/missy/README.md:271:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy-loops/missy/README.md:273:missy plugins                       # List plugins and their status
/home/missy/missy-loops/missy/README.md:279:# Security & audit
/home/missy/missy-loops/missy/README.md:280:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy-loops/missy/README.md:281:missy audit security                # Policy violations
/home/missy/missy-loops/missy/README.md:283:missy approvals list                # Pending human-in-the-loop approval requests
/home/missy/missy-loops/missy/README.md:297:missy discord status | probe | register-commands | audit
/home/missy/missy-loops/missy/README.md:301:missy devices list | pair | unpair | status | policy
/home/missy/missy-loops/missy/README.md:303:# MCP & skills
/home/missy/missy-loops/missy/README.md:305:missy skills                        # List registered skills
/home/missy/missy-loops/missy/README.md:306:missy skills scan                   # Discover SKILL.md files
/home/missy/missy-loops/missy/README.md:310:missy vision health | benchmark | validate | memory
/home/missy/missy-loops/missy/README.md:352:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy-loops/missy/README.md:362:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy-loops/missy/README.md:379:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy-loops/missy/README.md:382:| [CLI Reference](https://missylabs.github.io/cli/) | 20 | Every command group, including gateway, discord, approvals, patches, sandbox, sessions |
/home/missy/missy-loops/missy/README.md:384:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy-loops/missy/README.md:385:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy-loops/missy/README.md:392:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy-loops/missy/README.md:401:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy-loops/missy/README.md:412:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy-loops/missy/README.md:413:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy-loops/missy/README.md:414:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy-loops/missy/README.md:417:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy-loops/missy/README.md:418:├── plugins/         Security-gated external plugin loader
/home/missy/missy-loops/missy/README.md:419:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy-loops/missy/missy/tools/__init__.py:1:"""Missy tools framework — tool registry, base class, and built-in tools."""
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy-loops/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
/home/missy/missy-loops/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy-loops/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy-loops/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy-loops/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy-loops/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:11:| A1 | Streaming subscription state machine | tested | `missy/agent/subscription.py:34`, `missy/agent/subscription.py:241`, `missy/agent/runtime.py:620` | `tests/agent/test_subscription.py:8`, `tests/agent/test_runtime_streaming.py:83` | Handles `message_start/update/end`, tool events, compaction events, monotonic delta/full-content reconciliation, split think/final tag stripping, code-span awareness, reply directives, reasoning modes, and block flush points. Runtime wiring currently covers simple streaming. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:12:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py:116`, `missy/policy/tool_policy_pipeline.py:176`, `missy/policy/tool_policy_pipeline.py:206`, `missy/config/settings.py:132`, `missy/agent/runtime.py:1093`, `missy/cli/main.py:206`, `missy/security/sandbox.py:72` | `tests/policy/test_tool_policy_pipeline.py:14`, `tests/policy/test_tool_policy_pipeline.py:115`, `tests/config/test_settings.py:141`, `tests/agent/test_runtime_config_edges.py:741`, `tests/agent/test_runtime_streaming.py:119` | Implements profiles, standard layer ordering, group expansion, glob matching, inline `-tool` deny syntax, `alsoAllow`, fail-warning unknown allowlists, trace labels, and YAML-backed provider/global/agent/sandbox/subagent surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:13:| A3 | Mutation fingerprinting + sticky lastToolError | not_started | Planned: `missy/agent/mutation_tracking.py`, `missy/agent/runtime.py`, `missy/tools/registry.py` | Planned: `tests/agent/test_mutation_tracking.py` | Needed by H_G apology calibration. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:15:| A5 | Auth profile cooldown + fallback | not_started | Planned: `missy/providers/auth_profiles.py`, `missy/providers/registry.py`, `missy/providers/rate_limiter.py` | Planned: `tests/providers/test_auth_profiles.py` | Must honor user-pinned profile without fallback. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:16:| A6 | Per-provider tool schema normalization | not_started | Planned: `missy/providers/schema_adapter.py` | Planned: `tests/providers/test_schema_adapter.py` | Gemini scrubbing and Mistral ID rewrite remain. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:17:| A7 | Block-reply chunking with flush points | not_started | Planned: `missy/channels/block_chunker.py`, channel adapters, `missy/agent/runtime.py` | Planned: `tests/channels/test_block_chunker.py` | A1 has block buffers and tool-start flush; channel delivery remains. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:18:| A8 | Per-channel identity cascade | not_started | Planned: `missy/agent/persona.py`, config schema | Planned: `tests/agent/test_persona_identity_cascade.py` | Response prefix and ack reaction cascade remains. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:31:| H_C personal memory | A2 tool policy, A12 transcript repair | A2 can now gate future personal-memory recall/list/forget tools through runtime and YAML policy layers; A12 remains unimplemented. |
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:49:- Runtime capability profile constants live in `missy/policy/tool_policy_pipeline.py:21` and `missy/policy/tool_policy_pipeline.py:35`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:50:- OpenClaw-compatible group expansion, including `group:fs`, is defined at `missy/policy/tool_policy_pipeline.py:71`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:51:- `ToolPolicyLayer`, `ToolPolicyTraceStep`, and `ToolPolicyDecision` provide source-labelled audit records at `missy/policy/tool_policy_pipeline.py:116`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:52:- `build_configured_tool_policy_layers()` creates turn-specific config-backed layers at `missy/policy/tool_policy_pipeline.py:176`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:53:- `build_tool_policy_layers()` still exposes the explicit standard profile → provider → global → agent → group → sandbox → subagent sequence at `missy/policy/tool_policy_pipeline.py:232`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:54:- `resolve_tool_policy()` applies `allow`, `also_allow`, `deny`, globs, inline `-tool` denies, and fail-warning unknown allowlists at `missy/policy/tool_policy_pipeline.py:262`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:55:- `ToolPolicyConfig` and `AgentPolicyConfig` parse YAML-backed tool policy surfaces at `missy/config/settings.py:132`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:56:- `AgentRuntime._get_tools()` delegates capability-mode and config-backed filtering to A2 at `missy/agent/runtime.py:1093`.
/home/missy/missy-loops/missy/OPENCLAW_PATTERNS.md:57:- CLI-created runtimes receive parsed tool policies through `_agent_tool_policy_kwargs()` at `missy/cli/main.py:206`.
/home/missy/missy-loops/missy/docs/architecture.md:10:Missy is a **security-first**, **local-first**, **multi-provider** AI agent
/home/missy/missy-loops/missy/docs/architecture.md:13:access, filesystem writes, shell execution, plugin loading -- is disabled by
/home/missy/missy-loops/missy/docs/architecture.md:14:default and must be explicitly enabled through a YAML configuration file.
/home/missy/missy-loops/missy/docs/architecture.md:22:   policy engine before any bytes leave the machine.
/home/missy/missy-loops/missy/docs/architecture.md:23:3. **Audit everything** -- every policy decision, provider call, scheduler
/home/missy/missy-loops/missy/docs/architecture.md:24:   execution, and plugin action is recorded as a structured JSONL event.
/home/missy/missy-loops/missy/docs/architecture.md:34:  policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy-loops/missy/docs/architecture.md:37:                   attention, progress, approval, persona, behavior, hatching,
/home/missy/missy-loops/missy/docs/architecture.md:41:  providers/       BaseProvider ABC, Anthropic, OpenAI, Ollama, registry + rate limiter
/home/missy/missy-loops/missy/docs/architecture.md:42:  tools/           Tool base class, registry, 18+ built-in tools
/home/missy/missy-loops/missy/docs/architecture.md:43:  skills/          Skill registry + SKILL.md discovery
/home/missy/missy-loops/missy/docs/architecture.md:44:  plugins/         Security-gated external plugin loader and base class
/home/missy/missy-loops/missy/docs/architecture.md:45:  scheduler/       APScheduler integration, human schedule parsing, job persistence
/home/missy/missy-loops/missy/docs/architecture.md:67: 3. Subsystem init        init_policy_engine(cfg)  -- network, filesystem, shell, REST L7
/home/missy/missy-loops/missy/docs/architecture.md:68:        |                 init_audit_logger(cfg.audit_log_path) + AgentIdentity (Ed25519)
/home/missy/missy-loops/missy/docs/architecture.md:69:        |                 init_registry(cfg) -- providers with rate limiter + fallback
/home/missy/missy-loops/missy/docs/architecture.md:71:        |                 init_tool_registry() -- 18+ built-in tools + MCP servers
/home/missy/missy-loops/missy/docs/architecture.md:78:        |                 Resolve provider (with fallback + circuit breaker)
/home/missy/missy-loops/missy/docs/architecture.md:82:        |                 Playbook injects proven tool patterns
/home/missy/missy-loops/missy/docs/architecture.md:85:        |                 All HTTP through PolicyHTTPClient -> policy + REST check
/home/missy/missy-loops/missy/docs/architecture.md:93: 8. Post-processing       Learnings extracted from tool-augmented runs
/home/missy/missy-loops/missy/docs/architecture.md:99:        |                 Events signed by AgentIdentity, appended to audit.jsonl
/home/missy/missy-loops/missy/docs/architecture.md:161:Every policy dataclass defaults to the most restrictive posture:
/home/missy/missy-loops/missy/docs/architecture.md:163:- `NetworkPolicy.default_deny = True`
/home/missy/missy-loops/missy/docs/architecture.md:164:- `ShellPolicy.enabled = False`
/home/missy/missy-loops/missy/docs/architecture.md:165:- `PluginPolicy.enabled = False`
/home/missy/missy-loops/missy/docs/architecture.md:168:An operator must explicitly add entries to allowlists before any capability is
/home/missy/missy-loops/missy/docs/architecture.md:173:All outbound HTTP traffic -- whether initiated by a provider, a tool, a plugin,
/home/missy/missy-loops/missy/docs/architecture.md:177:`get_policy_engine().check_network(host)`.  If the host is not on an allowlist,
/home/missy/missy-loops/missy/docs/architecture.md:180:The Anthropic and OpenAI providers use their own SDKs for HTTP, but their API
/home/missy/missy-loops/missy/docs/architecture.md:181:hosts must still appear in `network.allowed_hosts` for the initial policy check
/home/missy/missy-loops/missy/docs/architecture.md:182:at the gateway layer.  The Ollama provider routes directly through
/home/missy/missy-loops/missy/docs/architecture.md:193:- `category` (one of: `network`, `filesystem`, `shell`, `plugin`, `scheduler`, `provider`, `security`, `agent`, `tool`, `mcp`, `vision`)
/home/missy/missy-loops/missy/docs/architecture.md:194:- `result` (one of: `allow`, `deny`, `error`)
/home/missy/missy-loops/missy/docs/architecture.md:196:- `policy_rule` (optional rule name)
/home/missy/missy-loops/missy/docs/architecture.md:198:The `AuditLogger` (`missy/observability/audit_logger.py`) wraps the bus's
/home/missy/missy-loops/missy/docs/architecture.md:200:audit log file.
/home/missy/missy-loops/missy/docs/architecture.md:212:  +-> policy/engine
/home/missy/missy-loops/missy/docs/architecture.md:213:  +-> observability/audit_logger + observability/otel
/home/missy/missy-loops/missy/docs/architecture.md:214:  +-> providers/registry
/home/missy/missy-loops/missy/docs/architecture.md:216:  +-> scheduler/manager
/home/missy/missy-loops/missy/docs/architecture.md:217:  +-> plugins/loader
/home/missy/missy-loops/missy/docs/architecture.md:226:  +-> providers/registry + providers/base
/home/missy/missy-loops/missy/docs/architecture.md:228:  +-> tools/registry
/home/missy/missy-loops/missy/docs/architecture.md:232:  +-> agent/progress + agent/interactive_approval + agent/approval
/home/missy/missy-loops/missy/docs/architecture.md:240:providers/registry
/home/missy/missy-loops/missy/docs/architecture.md:241:  +-> providers/base
/home/missy/missy-loops/missy/docs/architecture.md:242:  +-> providers/anthropic_provider + openai_provider + ollama_provider
/home/missy/missy-loops/missy/docs/architecture.md:243:  +-> providers/rate_limiter
/home/missy/missy-loops/missy/docs/architecture.md:247:  +-> policy/engine + policy/rest_policy
/home/missy/missy-loops/missy/docs/architecture.md:248:  +-> agent/interactive_approval
/home/missy/missy-loops/missy/docs/architecture.md:251:policy/engine
/home/missy/missy-loops/missy/docs/architecture.md:252:  +-> policy/network + policy/filesystem + policy/shell + policy/rest_policy
/home/missy/missy-loops/missy/docs/architecture.md:253:  +-> policy/presets
/home/missy/missy-loops/missy/docs/architecture.md:259:  +-> tools/registry
/home/missy/missy-loops/missy/docs/architecture.md:265:scheduler/manager
/home/missy/missy-loops/missy/docs/architecture.md:266:  +-> scheduler/parser + scheduler/jobs
/home/missy/missy-loops/missy/docs/architecture.md:271:  +-> providers/base (for image formatting)
/home/missy/missy-loops/missy/docs/architecture.md:281:2. `init_policy_engine(cfg)` -- must come first; other subsystems depend on it
/home/missy/missy-loops/missy/docs/architecture.md:282:3. `init_audit_logger(cfg.audit_log_path)` -- wraps the event bus
/home/missy/missy-loops/missy/docs/architecture.md:283:4. `init_registry(cfg)` -- constructs provider instances
/home/missy/missy-loops/missy/docs/architecture.md:300:| Policy engine | `init_policy_engine(cfg)` | `get_policy_engine()` |
/home/missy/missy-loops/missy/docs/architecture.md:301:| Provider registry | `init_registry(cfg)` | `get_registry()` |
/home/missy/missy-loops/missy/docs/architecture.md:302:| Audit logger | `init_audit_logger(path)` | `get_audit_logger()` |
/home/missy/missy-loops/missy/docs/architecture.md:303:| Plugin loader | `init_plugin_loader(cfg)` | `get_plugin_loader()` |
/home/missy/missy-loops/missy/docs/architecture.md:304:| Skill registry | `init_skill_registry()` | `get_skill_registry()` |
/home/missy/missy-loops/missy/docs/architecture.md:305:| Tool registry | `init_tool_registry()` | `get_tool_registry()` |
/home/missy/missy-loops/missy/docs/README.md:9:| [Providers](providers.md) | Anthropic, OpenAI, Ollama setup and API key management |
/home/missy/missy-loops/missy/docs/README.md:11:| [Scheduler](scheduler.md) | Job scheduling with human-friendly syntax |
/home/missy/missy-loops/missy/docs/README.md:12:| [Skills & Plugins](skills-and-plugins.md) | Extension system: tools, skills, plugins |
/home/missy/missy-loops/missy/docs/README.md:20:| [Security](security.md) | Security policy, hardening guide, vulnerability reporting |
/home/missy/missy-loops/missy/docs/README.md:38:| [Policy Engine](implementation/policy-engine.md) | `missy/policy/` |
/home/missy/missy-loops/missy/docs/README.md:39:| [Provider Abstraction](implementation/provider-abstraction.md) | `missy/providers/` |
/home/missy/missy-loops/missy/docs/README.md:42:| [Audit Events](implementation/audit-events.md) | `missy/observability/` |
/home/missy/missy-loops/missy/docs/README.md:43:| [Persistence Schema](implementation/persistence-schema.md) | `missy/memory/`, `missy/scheduler/` |
/home/missy/missy-loops/missy/docs/README.md:44:| [Scheduler Execution](implementation/scheduler-execution.md) | `missy/scheduler/` |
/home/missy/missy-loops/missy/docs/README.md:46:| [Manifest Schema](implementation/manifest-schema.md) | Plugin/skill manifests |
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:1:"""Provider-aware tool enablement based on benchmark results.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:3::class:`ToolProviderGate` decides whether a given (tool, provider) pair
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:7:1. **Operator overrides** — explicit ``enable``/``disable`` calls persisted
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:9:   an operator can force-enable a tool the benchmarks call weak, or
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:10:   force-disable one the benchmarks call strong.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:11:2. **Benchmark data** — when a (tool, provider) pair has at least
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:12:   ``min_samples`` benchmark runs (from either the direct
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:13:   :class:`~missy.tools.benchmark.runner.BenchmarkRunner` or the
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:14:   :class:`~missy.tools.benchmark.llm_runner.LLMBenchmarkRunner`) and its
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:15:   mean composite score is below ``min_composite``, the tool is treated as
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:16:   disabled for that provider.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:18:With no override and no (or insufficient) benchmark data, a tool is enabled
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:20:requires benchmarking before first use.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:22:This module only decides *availability*; it does not touch tool execution
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:23:policy (still enforced by :class:`~missy.tools.registry.ToolRegistry`) or the
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:24:static config-driven :mod:`~missy.policy.tool_policy_pipeline` layers. Both
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:41:_DEFAULT_OVERRIDES_PATH = Path("~/.missy/tool_provider_overrides.json")
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:46:    """Outcome of a gating check for one (tool, provider) pair.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:49:        enabled: Whether the tool should be exposed to this provider.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:51:            benchmark summary, or "no data").
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:52:        source: One of ``"override"``, ``"benchmark"``, or ``"default"``.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:55:    enabled: bool
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:61:    """Persists explicit operator enable/disable overrides.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:64:    ``~/.missy/tool_provider_overrides.json``) since the data volume is tiny
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:66:    inspection. Every mutation emits an audit event.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:74:    def get(self, tool_name: str, provider_name: str) -> bool | None:
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:75:        """Return the explicit override for *(tool_name, provider_name)*, or ``None``."""
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:77:        entry = data.get(tool_name, {})
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:78:        if provider_name not in entry:
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:80:        return bool(entry[provider_name])
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:84:        tool_name: str,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:85:        provider_name: str,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:86:        enabled: bool,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:89:        """Persist an explicit override and emit an audit event."""
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:92:            data.setdefault(tool_name, {})[provider_name] = bool(enabled)
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:94:        _emit_audit(
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:95:            "tool.provider_gate.override_set",
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:97:                "tool": tool_name,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:98:                "provider": provider_name,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:99:                "enabled": enabled,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:104:            "ProviderGateStore: %s override for tool=%r provider=%r by %r",
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:105:            "enabled" if enabled else "disabled",
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:106:            tool_name,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:107:            provider_name,
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:111:    def clear(self, tool_name: str, provider_name: str, actor: str = "operator") -> bool:
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:112:        """Remove an override, reverting to benchmark-driven/default behavior.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:119:            entry = data.get(tool_name, {})
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:120:            if provider_name not in entry:
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:122:            del entry[provider_name]
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:124:                data.pop(tool_name, None)
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:126:        _emit_audit(
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:127:            "tool.provider_gate.override_cleared",
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:128:            {"tool": tool_name, "provider": provider_name, "actor": actor},
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:133:        """Return all stored overrides as ``{tool_name: {provider_name: enabled}}``."""
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:147:            str(tool): {str(p): bool(v) for p, v in providers.items()}
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:148:            for tool, providers in raw.items()
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:149:            if isinstance(providers, dict)
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:160:    """Combine operator overrides and benchmark data into gating decisions.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:164:            enable/disable calls. Defaults to the module-level singleton.
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:165:        benchmark_store: :class:`~missy.tools.benchmark.benchmark_store.BenchmarkStore`
/home/missy/missy-loops/missy/missy/tools/intelligence/provider_gate.py:167:        min_samples: Minimum benchmark runs before a score is trusted.
```
