# AUDIT_SECURITY

## Validation-harness overhaul findings (curated)

Findings from the 2026-07-09/10 validation-harness/security-review
overhaul (`~/missy-loops/prompt.md`, `~/Missy-security-review.md` pinned
at commit `abb7015`). Branch:
`overhaul/missy-validation-20260710-031406`. Tracked by finding ID,
current reachability, and remediation evidence per the prompt's
requirement — do not overwrite, append new entries as work continues.

### SR-1.2 / SR-1.3 — Unauthenticated code-evolution self-approval

- **Status: partially fixed (interim), not fully remediated.**
- **Reachability found:** live and exploitable via two independent
  paths as of commit `eb04006`/`327595b` (this branch, before the fix
  in commit below):
  1. `missy/tools/builtin/code_evolve.py`'s agent-facing `code_evolve`
     tool exposed `approve`/`apply`/`rollback` actions directly to the
     model's tool-calling loop, with no gate. Worse, the default system
     prompt (`missy/agent/runtime.py::AgentConfig.system_prompt`)
     explicitly instructed the model to call
     `code_evolve(action='approve', ...)` then `action='apply'` as
     steps 3-4 of its own "self-evolution workflow" — actively teaching
     self-approval.
  2. `missy/channels/discord/channel.py::_handle_reaction()` let *any*
     Discord user who reacted ✅ on a proposal message trigger
     `CodeEvolutionManager().approve(proposal_id)` directly, with zero
     admin/owner check.
  3. `CodeEvolutionManager.approve()`/`apply()`/`rollback()`
     (`missy/agent/code_evolution.py`) themselves perform **no
     authentication of any kind** — every one of the above paths worked
     because nothing downstream stopped them either.
- **Remediation evidence (commit on this branch, see git log for
  message "FX-E / SR-1.2/1.3: ..."):**
  - `CodeEvolveTool` no longer dispatches `approve`/`apply`/`rollback`;
    those actions are refused unconditionally before
    `CodeEvolutionManager` is constructed. Test:
    `tests/tools/test_code_evolve.py::TestHumanOperatorOnlyActionsRefused`
    and `tests/tools/test_code_evolve_gap_coverage.py::TestHumanOperatorOnlyActionsNeverConstructManager`
    assert the manager class is never called.
  - Default system prompt rewritten: propose-then-stop, plus a general
    "never bypass a gate" instruction.
  - `_handle_reaction()`'s approve branch no longer calls
    `mgr.approve()`; it refuses and emits
    `discord.evolution.approve_denied` (`deny`). Test:
    `tests/channels/test_discord_evolution_reactions.py::TestHandleReaction::test_approve_reaction_is_refused_not_approved`
    and `::test_approve_reaction_emits_deny_audit_event`.
- **Residual risk (not yet fixed):** `CodeEvolutionManager.approve()`/
  `apply()`/`rollback()` still perform zero authentication of their
  own. The only real trust boundary is that `missy evolve
  approve/apply/rollback` (the CLI, `missy/cli/main.py`) requires an
  interactive shell session on the host. There is no "unforgeable,
  proposal-bound, expiring approval artifact" as SR-1.2 asks for, no
  disposable-sandbox validation before promotion (SR-1.3), and no
  authenticated Web API route (none exists yet, so nothing to fix there
  today — but if one is added it must not reuse this trust gap). If any
  future code path constructs `CodeEvolutionManager` directly (bypassing
  the CLI) without its own authentication check, the same class of bug
  reappears. Do not treat this finding as closed.
- **Related finding, checked and fixed this session:** see SR-1.12
  below — the DM-pairing approval flow had the *identical* bug pattern,
  confirmed and fixed in the same session.

### SR-1.12 — Authenticated Discord pairing

- **Status: fixed (self-approval and any in-band DM approval closed).**
- **Reachability found:** live and directly exploitable — worse than
  SR-1.2/1.3 in one respect, since it required no prior state at all.
  `missy/channels/discord/channel.py::_check_pairing()` processed
  `!pair accept <target_id>` and `!pair deny <target_id>` DM commands
  with **zero check on who sent them**. Any unpaired stranger could DM
  the bot `!pair` (adding themselves to `_pending_pairs`) immediately
  followed by `!pair accept <their-own-user-id>` and grant themselves
  full DM access — a complete, self-service bypass of the pairing gate
  requiring no authorization step at all. The code comment above the
  block ("admin only — simplified") acknowledged the intended design
  but the "admin only" check was never actually implemented.
- **Remediation evidence:** `_check_pairing()` no longer processes
  `!pair accept`/`!pair deny` from DM content at all — those commands
  are now unconditionally refused with a
  `discord.channel.pairing_decision_denied` (`deny`) audit event
  regardless of sender. `accept_pair()`/`deny_pair()`/
  `get_pending_pairs()` remain as the only way to resolve a pending
  request, and are documented as requiring an authenticated operator
  surface (the Web console/API, which shares the same process as the
  Discord channel under `missy gateway start`) — never DM content. New
  tests:
  `tests/channels/test_discord_channel_coverage.py::TestCheckPairingDeny::test_pair_deny_via_dm_is_refused_not_processed`,
  `tests/channels/discord/test_discord_channel_integration.py::TestCheckPairing::test_accept_command_via_dm_is_refused`,
  `::test_deny_command_via_dm_is_refused`,
  `::test_accept_pair_only_available_via_programmatic_api`,
  `tests/unit/test_discord_channel.py::TestPairingWorkflow::test_accept_via_dm_command_is_refused`.
- **Residual risk:** no authenticated Web API/console endpoint has
  actually been wired to call `accept_pair()`/`deny_pair()` yet, so
  there is currently **no way for an operator to approve a pending
  Discord pairing at all** through any surface — this trades the
  vulnerability for a (safe, but incomplete) loss of functionality.
  Wiring a real authenticated approval endpoint (or a `missy discord
  pairs` CLI command backed by shared persistence, mirroring
  `missy devices pair`'s `DeviceRegistry` pattern) is tracked as
  follow-up work. Rate-limiting of pairing *requests* (`!pair` itself)
  and replay/expiration handling per the full SR-1.12 ask are also not
  yet implemented.

---

- Timestamp: 2026-07-09 15:35:26 (raw grep-scan dump below, preserved
  for history; unrelated to the curated findings above — note it was
  captured from a different working tree path, `~/missy-loops/missy/`,
  and largely covers the prior "tool intelligence" overhaul, not this
  session's work)

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and tool-intelligence scan
```
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:7:- Added an opt-in controlled runtime loader for enabled tool candidates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:8:- Added persisted candidate `implementation` metadata with SQLite migration.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:10:- Runtime loading is gated by `tool_intelligence.candidate_runtime.enabled`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:11:- The loader only registers enabled candidates for the active provider when
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:12:  provenance, schema, permissions, provider flags, implementation type, and
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:15:  `{"type": "delegated_tool", "tool": "<registered_tool>"}`.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:16:- Loader allow/deny outcomes emit structured candidate audit events.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:20:- Added tests for loader allow/deny behavior, runtime opt-in wiring, config
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:21:  parsing, and candidate-store implementation persistence.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:26:python3 -m pytest tests/vision/test_frame_eviction_hardening.py::TestCaptureDeadlineAwareSleep tests/tools/test_candidate_loader.py tests/tools/test_candidate_store.py tests/agent/test_tool_intelligence_wiring.py tests/config/test_settings.py -q
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:49:- Runtime loader supports only `delegated_tool`; additional adapters need
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:50:  separate policy, sandboxing, provenance, test, and rollback gates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:51:- Provider fallback recommendations exist in CLI/provider gate code but are
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:52:  not yet surfaced in runtime responses when a tool is gated off.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:53:- Candidate review can import schema-score aggregates, but provider-family
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:54:  schema compatibility reporting is still limited.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:59:Add safe CLI/API/operator controls for setting candidate implementation
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:60:metadata, starting with `delegated_tool`, with typed confirmations and audit
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
/home/missy/missy-loops/missy/LOOP_HEALTH.md:5:- Branch: overhaul/tools-20260709-174109
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
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:7:- default-deny network where practical
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:8:- exact provider endpoints
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:9:- exact benchmark and provider endpoints
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:15:- `delegated_tool` candidates inherit the permissions and policy checks of the
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:17:- Candidates that request network permission must still pass normal tool
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:18:  registry and policy-engine checks before execution.
/home/missy/missy-loops/missy/AUDIT_CONNECTIVITY.md:19:- No provider, Discord, plugin, MCP, benchmark, or external HTTP allowlist was
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
/home/missy/missy-loops/missy/missy/security/container.py:1:"""Container-per-session sandbox for isolated tool execution.
/home/missy/missy-loops/missy/missy/security/container.py:18:      enabled: true
/home/missy/missy-loops/missy/missy/security/container.py:48:        enabled: Master switch for container sandboxing.
/home/missy/missy-loops/missy/missy/security/container.py:52:        network_mode: Docker network mode (``"none"`` disables networking).
/home/missy/missy-loops/missy/missy/security/container.py:55:    enabled: bool = False
/home/missy/missy-loops/missy/missy/security/container.py:67:        enabled=bool(data.get("enabled", False)),
/home/missy/missy-loops/missy/missy/security/container.py:130:            logger.debug("Docker not available — container sandbox disabled")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:1:"""Controlled runtime loader for enabled tool candidates.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:4:does not execute generated code and does not infer behavior from a candidate
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:5:description. The first supported binding is ``delegated_tool``: a candidate
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:6:can become a schema/metadata wrapper around an already-registered tool, while
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:7:the normal :class:`missy.tools.registry.ToolRegistry` policy checks still run
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:8:for both the candidate wrapper and the delegated tool.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:19:from missy.tools.base import BaseTool, ToolPermissions, ToolResult
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:20:from missy.tools.registry import ToolRegistry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:22:from .candidate_store import CandidateStore, ToolCandidate, ToolLifecycleState
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:35:_SUPPORTED_IMPLEMENTATIONS = {"delegated_tool"}
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:40:    """A candidate skipped by the runtime loader."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:42:    candidate_id: str
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:56:    """Runtime wrapper that delegates execution to an existing registered tool."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:58:    def __init__(self, candidate: ToolCandidate, target_tool: str, registry: ToolRegistry) -> None:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:59:        self.name = candidate.name
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:60:        self.description = candidate.description
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:61:        self.permissions = _permissions_from_candidate(candidate.permissions)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:62:        self._schema = {
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:63:            "name": candidate.name,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:64:            "description": candidate.description,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:65:            "parameters": candidate.schema,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:67:        self._candidate_id = candidate.id
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:68:        self._target_tool = target_tool
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:69:        self._registry = registry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:72:    def candidate_id(self) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:73:        return self._candidate_id
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:76:    def target_tool(self) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:77:        return self._target_tool
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:79:    def get_schema(self) -> dict[str, Any]:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:80:        return dict(self._schema)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:83:        return self._registry.execute(self._target_tool, **kwargs)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:87:    """Validate and register enabled candidates with explicit implementations."""
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:89:    def __init__(self, store: CandidateStore, registry: ToolRegistry) -> None:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:91:        self._registry = registry
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:93:    def load_enabled(self, provider_name: str) -> CandidateLoadReport:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:94:        """Load enabled candidates for *provider_name* into the tool registry.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:96:        Candidates are skipped unless they pass lifecycle, schema,
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:97:        provenance, implementation, permission, provider-enable, and conflict
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:98:        checks. Every load or skip emits a structured audit event.
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:102:        for candidate in self._store.list_all(state=ToolLifecycleState.ENABLED, limit=1000):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:103:            reason = self._validate(candidate, provider_name)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:105:                skipped.append(CandidateLoadIssue(candidate.id, candidate.name, reason))
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:106:                _emit_audit("tool.candidate.load_skipped", candidate, provider_name, reason, "deny")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:109:            target_tool = str(candidate.implementation["tool"])
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:110:            self._registry.register(CandidateDelegatedTool(candidate, target_tool, self._registry))
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:111:            loaded.append(candidate.name)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:112:            _emit_audit(
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:113:                "tool.candidate.loaded", candidate, provider_name, f"delegates:{target_tool}"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:117:    def _validate(self, candidate: ToolCandidate, provider_name: str) -> str:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:118:        if candidate.state is not ToolLifecycleState.ENABLED:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:119:            return f"candidate state is {candidate.state.value}, not enabled"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:120:        if not _SAFE_NAME_RE.match(candidate.name):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:121:            return "candidate name is not a safe tool identifier"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:122:        if not candidate.provenance.strip():
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:123:            return "candidate provenance is missing"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:124:        schema_error = _validate_schema(candidate.schema)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:125:        if schema_error:
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:126:            return schema_error
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:127:        permission_error = _validate_permissions(candidate.permissions)
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:130:        impl = candidate.implementation
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:132:            return "candidate implementation is missing"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:134:            return "candidate implementation must be an object"
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:138:        target_tool = str(impl.get("tool") or "")
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:139:        if not _SAFE_NAME_RE.match(target_tool):
/home/missy/missy-loops/missy/missy/tools/intelligence/candidate_loader.py:140:            return "delegated tool target is not a safe tool identifier"
```
