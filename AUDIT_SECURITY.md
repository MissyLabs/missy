# AUDIT_SECURITY

- Timestamp: 2026-07-08 21:11:38

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/openai-provider-20260708-172558
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete OpenAI provider overhaul
/home/missy/missy/PERSONA.md:5:## Quick Start
/home/missy/missy/PERSONA.md:48:  - Always respect policy engine decisions
/home/missy/missy/PERSONA.md:52:  She is knowledgeable, practical, and focused on getting things done
/home/missy/missy/PERSONA.md:79:The `BehaviorLayer` injects persona information into the system prompt sent to the AI provider. This includes identity, tone preferences, style rules, and boundaries. The prompt is structured so the LLM naturally adopts the persona.
/home/missy/missy/PERSONA.md:93:- **Response Guidelines** — Context-specific behavioral directives
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Added `missy/api/run_stream.py`: `RunRegistry`/`RunHandle` execute
/home/missy/missy/LAST_SESSION_SUMMARY.md:9:  `tool.request` / `tool.result` message-bus events into a per-run queue
/home/missy/missy/LAST_SESSION_SUMMARY.md:10:  (strictly session-scoped, redacted via the existing audit redaction
/home/missy/missy/LAST_SESSION_SUMMARY.md:11:  helper), and enforce one in-flight run per session. Reconnecting after a
/home/missy/missy/LAST_SESSION_SUMMARY.md:14:  `GET /api/v1/runs?session_id=`, and `GET /api/v1/runs/{id}/events` (SSE)
/home/missy/missy/LAST_SESSION_SUMMARY.md:15:  to `missy/api/server.py`. The SSE route reuses the standard auth and rate
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:  limiting; a conflicting run-start (409) is recorded as a `web.run` audit
/home/missy/missy/LAST_SESSION_SUMMARY.md:21:  `cost_detail` under a `cost` key, matching what the audit event already
/home/missy/missy/LAST_SESSION_SUMMARY.md:23:- Added an "Ask Missy" run console panel to `missy/api/web_console.py`:
/home/missy/missy/LAST_SESSION_SUMMARY.md:24:  prompt box + live SSE-driven activity log (tool calls, completion,
/home/missy/missy/LAST_SESSION_SUMMARY.md:25:  errors) with accessible markup (`role="log"`, `aria-live="polite"`,
/home/missy/missy/LAST_SESSION_SUMMARY.md:26:  labeled controls, Enter/Shift+Enter handling).
/home/missy/missy/LAST_SESSION_SUMMARY.md:27:- Added 19 unit tests (`tests/api/test_run_stream.py`) and 16 integration
/home/missy/missy/LAST_SESSION_SUMMARY.md:31:  stale — missing `web_console`, `web_sessions`, `audit_browser`,
/home/missy/missy/LAST_SESSION_SUMMARY.md:32:  `diagnostics`, `operator_controls` entirely; now documents all of them
/home/missy/missy/LAST_SESSION_SUMMARY.md:33:  plus the new `run_stream`).
/home/missy/missy/LAST_SESSION_SUMMARY.md:35:  `AUDIT_CONNECTIVITY.md`, and `TEST_EDGE_CASES.md` for the Web TUI focus
/home/missy/missy/LAST_SESSION_SUMMARY.md:36:  (they still described the prior OpenAI-provider-overhaul branch's work).
/home/missy/missy/LAST_SESSION_SUMMARY.md:41:python3 -m pytest tests/api/test_run_stream.py -v
/home/missy/missy/LAST_SESSION_SUMMARY.md:70:Full-repo `python3 -m pytest -q` was run before ending the session; see
/home/missy/missy/LAST_SESSION_SUMMARY.md:76:  (the data already flows through the bus and SSE stream).
/home/missy/missy/LAST_SESSION_SUMMARY.md:77:- No scheduler jobs panel (create/remove) or memory browser panel yet —
/home/missy/missy/LAST_SESSION_SUMMARY.md:78:  these are the largest remaining "full bot-control coverage" gaps.
/home/missy/missy/LAST_SESSION_SUMMARY.md:80:- No dashboard-wide offline/reconnecting banner (only the run console has
/home/missy/missy/LAST_SESSION_SUMMARY.md:87:Render `cost` / `provider` / `tools_used` from the run stream in the
/home/missy/missy/LAST_SESSION_SUMMARY.md:88:console's run log (backend already emits it), then start the scheduler
/home/missy/missy/LAST_SESSION_SUMMARY.md:89:jobs panel to close the next-largest "full bot-control coverage" gap.
/home/missy/missy/AUDIT_CONNECTIVITY.md:7:- Default-deny network where practical (`network.default_deny: true`).
/home/missy/missy/AUDIT_CONNECTIVITY.md:9:  (`ApiConfig.host`); binding elsewhere logs an explicit warning at startup.
/home/missy/missy/AUDIT_CONNECTIVITY.md:10:- No new outbound network endpoints were introduced this session — the run
/home/missy/missy/AUDIT_CONNECTIVITY.md:11:  console and SSE stream are entirely local: browser <-> loopback
/home/missy/missy/AUDIT_CONNECTIVITY.md:13:  the agent itself makes during a run (providers, tools) go through the
/home/missy/missy/AUDIT_CONNECTIVITY.md:16:  `EventSource` sends the session cookie automatically for same-origin
/home/missy/missy/AUDIT_CONNECTIVITY.md:18:  set (default browser same-origin policy applies).
/home/missy/missy/AUDIT_CONNECTIVITY.md:23:## Verification performed this session
/home/missy/missy/AUDIT_CONNECTIVITY.md:27:- Confirmed the SSE route (`/api/v1/runs/{id}/events`) requires the same
/home/missy/missy/AUDIT_CONNECTIVITY.md:28:  `_authenticate()` check (API key or web session) as every other
/home/missy/missy/AUDIT_CONNECTIVITY.md:29:  `/api/v1/*` route before any stream bytes are written — `tests/api/
/home/missy/missy/AUDIT_CONNECTIVITY.md:30:  test_server.py::TestRuns::test_events_stream_requires_auth`.
/home/missy/missy/AUDIT_CONNECTIVITY.md:35:  session's work required none.
/home/missy/missy/OPENCLAW_PATTERNS.md:11:| A1 | Streaming subscription state machine | tested | `missy/agent/subscription.py:34`, `missy/agent/subscription.py:241`, `missy/agent/runtime.py:620` | `tests/agent/test_subscription.py:8`, `tests/agent/test_runtime_streaming.py:83` | Handles `message_start/update/end`, tool events, compaction events, monotonic delta/full-content reconciliation, split think/final tag stripping, code-span awareness, reply directives, reasoning modes, and block flush points. Runtime wiring currently covers simple streaming. |
/home/missy/missy/OPENCLAW_PATTERNS.md:12:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py:116`, `missy/policy/tool_policy_pipeline.py:176`, `missy/policy/tool_policy_pipeline.py:206`, `missy/config/settings.py:132`, `missy/agent/runtime.py:1093`, `missy/cli/main.py:206`, `missy/security/sandbox.py:72` | `tests/policy/test_tool_policy_pipeline.py:14`, `tests/policy/test_tool_policy_pipeline.py:115`, `tests/config/test_settings.py:141`, `tests/agent/test_runtime_config_edges.py:741`, `tests/agent/test_runtime_streaming.py:119` | Implements profiles, standard layer ordering, group expansion, glob matching, inline `-tool` deny syntax, `alsoAllow`, fail-warning unknown allowlists, trace labels, and YAML-backed provider/global/agent/sandbox/subagent surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/OPENCLAW_PATTERNS.md:13:| A3 | Mutation fingerprinting + sticky lastToolError | not_started | Planned: `missy/agent/mutation_tracking.py`, `missy/agent/runtime.py`, `missy/tools/registry.py` | Planned: `tests/agent/test_mutation_tracking.py` | Needed by H_G apology calibration. |
/home/missy/missy/OPENCLAW_PATTERNS.md:15:| A5 | Auth profile cooldown + fallback | not_started | Planned: `missy/providers/auth_profiles.py`, `missy/providers/registry.py`, `missy/providers/rate_limiter.py` | Planned: `tests/providers/test_auth_profiles.py` | Must honor user-pinned profile without fallback. |
/home/missy/missy/OPENCLAW_PATTERNS.md:16:| A6 | Per-provider tool schema normalization | not_started | Planned: `missy/providers/schema_adapter.py` | Planned: `tests/providers/test_schema_adapter.py` | Gemini scrubbing and Mistral ID rewrite remain. |
/home/missy/missy/OPENCLAW_PATTERNS.md:17:| A7 | Block-reply chunking with flush points | not_started | Planned: `missy/channels/block_chunker.py`, channel adapters, `missy/agent/runtime.py` | Planned: `tests/channels/test_block_chunker.py` | A1 has block buffers and tool-start flush; channel delivery remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:18:| A8 | Per-channel identity cascade | not_started | Planned: `missy/agent/persona.py`, config schema | Planned: `tests/agent/test_persona_identity_cascade.py` | Response prefix and ack reaction cascade remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:20:| A10 | Sub-agent depth + child caps | not_started | Planned: `missy/agent/sub_agent.py`, session persistence, A2 filter | Planned: `tests/agent/test_sub_agent_depth_caps.py` | Depth-aware orchestration filtering remains. |
/home/missy/missy/OPENCLAW_PATTERNS.md:21:| A11 | Raw-stream JSONL diagnostics | not_started | Planned: `missy/observability/raw_stream.py`, `missy/agent/subscription.py` | Planned: `tests/observability/test_raw_stream.py` | A1 includes a callback seam for best-effort writes. |
/home/missy/missy/OPENCLAW_PATTERNS.md:23:| A13 | Context-window guard | not_started | Planned: `missy/agent/context_guard.py`, `missy/config/settings.py`, `missy/agent/runtime.py` | Planned: `tests/agent/test_context_guard.py` | 16k block and 32k warning thresholds remain. |
/home/missy/missy/OPENCLAW_PATTERNS.md:29:| H_A timing pauses | A7 block replies, A1 stream state | A1 block buffer/flush primitives exist; A7 not implemented. |
/home/missy/missy/OPENCLAW_PATTERNS.md:30:| H_B tone modulation | A1 message-start prompt timing, A8 identity cascade | Not implemented. Tone must be injected before stream begins. |
/home/missy/missy/OPENCLAW_PATTERNS.md:31:| H_C personal memory | A2 tool policy, A12 transcript repair | A2 can now gate future personal-memory recall/list/forget tools through runtime and YAML policy layers; A12 remains unimplemented. |
/home/missy/missy/OPENCLAW_PATTERNS.md:33:| H_E disagreement | A11 raw stream diagnostics, A9 hooks | Not implemented. |
/home/missy/missy/OPENCLAW_PATTERNS.md:35:| H_G apology/gratitude/hedging | A3 sticky mutation error, A1 stream state | Not implemented. |
/home/missy/missy/OPENCLAW_PATTERNS.md:45:- `AgentRuntime.run_stream()` integration starts at `missy/agent/runtime.py:620`.
/home/missy/missy/OPENCLAW_PATTERNS.md:49:- Runtime capability profile constants live in `missy/policy/tool_policy_pipeline.py:21` and `missy/policy/tool_policy_pipeline.py:35`.
/home/missy/missy/OPENCLAW_PATTERNS.md:50:- OpenClaw-compatible group expansion, including `group:fs`, is defined at `missy/policy/tool_policy_pipeline.py:71`.
/home/missy/missy/OPENCLAW_PATTERNS.md:51:- `ToolPolicyLayer`, `ToolPolicyTraceStep`, and `ToolPolicyDecision` provide source-labelled audit records at `missy/policy/tool_policy_pipeline.py:116`.
/home/missy/missy/OPENCLAW_PATTERNS.md:52:- `build_configured_tool_policy_layers()` creates turn-specific config-backed layers at `missy/policy/tool_policy_pipeline.py:176`.
/home/missy/missy/OPENCLAW_PATTERNS.md:53:- `build_tool_policy_layers()` still exposes the explicit standard profile → provider → global → agent → group → sandbox → subagent sequence at `missy/policy/tool_policy_pipeline.py:232`.
/home/missy/missy/OPENCLAW_PATTERNS.md:54:- `resolve_tool_policy()` applies `allow`, `also_allow`, `deny`, globs, inline `-tool` denies, and fail-warning unknown allowlists at `missy/policy/tool_policy_pipeline.py:262`.
/home/missy/missy/OPENCLAW_PATTERNS.md:55:- `ToolPolicyConfig` and `AgentPolicyConfig` parse YAML-backed tool policy surfaces at `missy/config/settings.py:132`.
/home/missy/missy/OPENCLAW_PATTERNS.md:56:- `AgentRuntime._get_tools()` delegates capability-mode and config-backed filtering to A2 at `missy/agent/runtime.py:1093`.
/home/missy/missy/OPENCLAW_PATTERNS.md:57:- CLI-created runtimes receive parsed tool policies through `_agent_tool_policy_kwargs()` at `missy/cli/main.py:206`.
/home/missy/missy/conftest.py:1:# conftest.py — pytest configuration
/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/BUILD_STATUS.md:7:Primary focus switched this session to the **Web TUI / operator console
/home/missy/missy/BUILD_STATUS.md:8:overhaul** (branch `overhaul/web-tui-20260709-004527`). The console already
/home/missy/missy/BUILD_STATUS.md:9:had a working dashboard, audit browser, diagnostics, and a small set of
/home/missy/missy/BUILD_STATUS.md:10:safe operator controls from earlier sessions on this line of work
/home/missy/missy/BUILD_STATUS.md:11:(`missy/api/server.py`, `web_console.py`, `web_sessions.py`,
/home/missy/missy/BUILD_STATUS.md:12:`audit_browser.py`, `diagnostics.py`, `operator_controls.py`); the highest-
/home/missy/missy/BUILD_STATUS.md:13:value gap against the loop's required-capabilities list was the missing
/home/missy/missy/BUILD_STATUS.md:14:"ask the bot and watch a run stream" workflow — session/run viewer with
/home/missy/missy/BUILD_STATUS.md:15:streaming output, tool calls, and errors. This session built that.
/home/missy/missy/BUILD_STATUS.md:17:### What shipped this session
/home/missy/missy/BUILD_STATUS.md:19:- **`missy/api/run_stream.py`** (new): `RunRegistry` executes
/home/missy/missy/BUILD_STATUS.md:21:  `agent.run.start` / `tool.request` / `tool.result` events from the
/home/missy/missy/BUILD_STATUS.md:23:  by `session_id` so concurrent sessions never cross-contaminate each
/home/missy/missy/BUILD_STATUS.md:24:  other's stream), and enforces exactly one in-flight run per session.
/home/missy/missy/BUILD_STATUS.md:27:  exception messages) pass through the existing `redact_audit_value()`
/home/missy/missy/BUILD_STATUS.md:28:  secret-detector-backed redaction.
/home/missy/missy/BUILD_STATUS.md:31:  `GET /api/v1/runs?session_id=` (list run history for a session), and
/home/missy/missy/BUILD_STATUS.md:32:  `GET /api/v1/runs/{id}/events` (Server-Sent Events stream). The SSE route
/home/missy/missy/BUILD_STATUS.md:34:  `text/event-stream` bytes) but still runs through the same
/home/missy/missy/BUILD_STATUS.md:35:  `_authenticate()` check and the same per-IP rate limiter as every other
/home/missy/missy/BUILD_STATUS.md:37:  SSE connection cannot block other operators' requests (health checks,
/home/missy/missy/BUILD_STATUS.md:38:  audit queries, controls) on the same process.
/home/missy/missy/BUILD_STATUS.md:40:  the same `cost_detail` summary already attached to the audit event, so
/home/missy/missy/BUILD_STATUS.md:41:  the run stream (and future UI) can surface per-run cost.
/home/missy/missy/BUILD_STATUS.md:42:- **`missy/api/web_console.py`**: new "Ask Missy" dashboard panel — a
/home/missy/missy/BUILD_STATUS.md:43:  prompt box that starts a run and renders it live via `EventSource`: tool
/home/missy/missy/BUILD_STATUS.md:44:  calls, completion, and errors stream into an accessible
/home/missy/missy/BUILD_STATUS.md:45:  (`role="log"`, `aria-live="polite"`) activity log, with the final response
/home/missy/missy/BUILD_STATUS.md:47:  watching" control closes the stream without cancelling the run itself.
/home/missy/missy/BUILD_STATUS.md:48:- Tests: 19 new unit tests in `tests/api/test_run_stream.py` (lifecycle,
/home/missy/missy/BUILD_STATUS.md:49:  redaction, cross-session isolation, concurrency/409, late-join/reconnect,
/home/missy/missy/BUILD_STATUS.md:51:  (`TestRuns` in `tests/api/test_server.py`) covering auth, CSRF, polling,
/home/missy/missy/BUILD_STATUS.md:52:  listing, SSE delivery, error propagation, and audit emission on conflict.
/home/missy/missy/BUILD_STATUS.md:57:  missing `web_console`, `web_sessions`, `audit_browser`, `diagnostics`,
/home/missy/missy/BUILD_STATUS.md:58:  `operator_controls` entirely, and now also documents `run_stream`).
/home/missy/missy/BUILD_STATUS.md:64:| Web UI entrypoint + auth/session/CSRF | in place (prior sessions) | Unchanged this session; new routes reuse it as-is. |
/home/missy/missy/BUILD_STATUS.md:65:| Dashboard (status/providers/tools/diagnostics/audit/controls) | in place (prior sessions) | Unchanged this session. |
/home/missy/missy/BUILD_STATUS.md:66:| Session/run viewer with streaming output, tool calls, errors | **new** | `POST /runs` + `GET /runs/{id}/events` (SSE) + console panel. |
/home/missy/missy/BUILD_STATUS.md:67:| Session/run history (resumable context) | **new** | `GET /runs?session_id=` lists prior runs per session. |
/home/missy/missy/BUILD_STATUS.md:69:| Concurrency safety for the API server | improved | `ThreadingHTTPServer` swap; verified against full existing test suite. |
/home/missy/missy/BUILD_STATUS.md:70:| Redaction of run/tool event payloads | in place | Reuses `audit_browser.redact_audit_value()`. |
/home/missy/missy/BUILD_STATUS.md:71:| Audit trail for run conflicts | **new** | `web.run` / `deny` event on 409. |
/home/missy/missy/BUILD_STATUS.md:72:| Accessibility of the new panel | in place | Labeled controls, `role="log"`, keyboard submit, non-color-only status text. |
/home/missy/missy/BUILD_STATUS.md:73:| Tests | improved | +35 tests this session (19 unit + 16 integration); full existing suite re-verified green. |
/home/missy/missy/BUILD_STATUS.md:80:- `RunRegistry` (`missy/api/run_stream.py`) is a new, self-contained
/home/missy/missy/BUILD_STATUS.md:84:  just without bus-sourced tool events) when no bus is initialized.
/home/missy/missy/BUILD_STATUS.md:85:- The run console's client-side code lives entirely in `web_console.py`'s
/home/missy/missy/BUILD_STATUS.md:86:  `console_script()` (vanilla JS, no build step), consistent with the rest
/home/missy/missy/BUILD_STATUS.md:91:- `python3 -m pytest tests/api/test_run_stream.py -v`: 19 passed.
/home/missy/missy/BUILD_STATUS.md:97:- Full-repo `python3 -m pytest -q`: see `TEST_RESULTS.md` for this session's
/home/missy/missy/BUILD_STATUS.md:102:1. Render cost/model-routing/provider-fallback detail in the console's run
/home/missy/missy/BUILD_STATUS.md:104:2. Add a scheduler jobs panel (create/remove, not just pause/resume) and a
/home/missy/missy/BUILD_STATUS.md:106:   bot-control coverage" gaps.
/home/missy/missy/BUILD_STATUS.md:107:3. Add a command palette / global search and a skip link for keyboard-first
/home/missy/missy/BUILD_STATUS.md:109:4. Add dedicated offline/reconnecting banner state for the dashboard as a
/home/missy/missy/BUILD_STATUS.md:118:- None. The next slice (rendering cost/routing detail, or a scheduler/
/home/missy/missy/BUILD_STATUS.md:119:  memory panel) is additive and does not require new backend primitives
/home/missy/missy/BUILD_STATUS.md:124:Render `cost`/`provider`/`tools_used` detail from the run stream in the
/home/missy/missy/BUILD_STATUS.md:125:console's run log, then start the scheduler jobs panel (list/create/remove)
/home/missy/missy/BUILD_STATUS.md:126:to close the largest remaining "full bot-control coverage" gap.
/home/missy/missy/HATCHING.md:5:## Quick Start
/home/missy/missy/HATCHING.md:16:2. **Initialize Config** — Creates `~/.missy/config.yaml` with secure defaults if it doesn't exist
/home/missy/missy/HATCHING.md:17:3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
/home/missy/missy/HATCHING.md:19:5. **Generate Persona** — Creates `~/.missy/persona.yaml` with default personality configuration
/home/missy/missy/HATCHING.md:45:  - initialize_config
/home/missy/missy/HATCHING.md:46:  - verify_providers
/home/missy/missy/HATCHING.md:51:persona_generated: true
/home/missy/missy/HATCHING.md:53:provider_verified: true
/home/missy/missy/HATCHING.md:78:The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
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
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:7:Primary focus switched to **completing the Web TUI / operator console
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:overhaul** (branch `overhaul/web-tui-20260709-004527`). OpenClaw and Odin
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:9:remain references for control-plane ergonomics, live status, run/tool
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:10:visibility, and auditability; Missy's implementation is clean-room and
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:11:Python/vanilla-JS native (server-rendered HTML, no frontend build step).
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:13:The previous branch's OpenAI provider work (native Responses routing,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:14:streaming reconciliation, structured outputs, diagnostics) is preserved as-is
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:15:and is not part of this session's scope.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:21:| Secure local Web UI entrypoint + auth/session | in place | `ApiServer` serves `/` (console) and `/api/v1/*` from one `ThreadingHTTPServer`. Cookie session (`HttpOnly`, `SameSite=Strict`) + API key, CSRF on unsafe browser requests, security headers (CSP, X-Frame-Options, no-store). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Dashboard: runtime status, providers, tools, memory, security posture | in place | `GET /status`, `/providers`, `/tools`, `/diagnostics` rendered as scannable panels with health pills. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:23:| Dashboard: scheduler, cost/usage, queues, jobs | partial | Scheduler pause/resume exists as a *control*; there is no dedicated scheduler jobs panel or cost/usage panel yet. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:24:| Session/run viewer with streaming output, tool calls, errors, provider, resumable context | **new this session** | `POST /api/v1/runs` starts a background run; `GET /api/v1/runs/{id}/events` streams `run.started`, `run.start`, `tool.request`, `tool.result`, and terminal `run.complete`/`run.error` over SSE. `GET /api/v1/runs/{id}` polls status; `GET /api/v1/runs?session_id=` lists run history per session (resumable). Console "Ask Missy" panel drives this live. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:25:| Session/run viewer: costs, model routing, provider fallback | partial | `AGENT_RUN_COMPLETE` bus payload now carries `cost` (this session); the run stream forwards it, but the console UI does not yet render cost/model-routing/fallback detail in the run log. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:26:| Audit log browser: filters, severity, actor/source, subsystem, timestamps, redaction | in place | `audit_browser.py` + console audit panel: result/severity/subsystem/actor/source/query/time-range filters, pagination, redacted detail view. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:27:| Diagnostics/doctor views (providers, tools, memory, policy, gateway, Discord, scheduler) | in place | `diagnostics.py` builds a redacted per-subsystem report consumed by the console and `missy doctor`. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:28:| Safe controls (providers, tools, jobs, channels, experimental features) | partial | Only `provider.set_default` and `scheduler.pause_job`/`resume_job` exist. No tool/channel/feature toggles yet. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:29:| Full bot-control coverage (memory, schedules, skills, plugins, Discord, voice, vision, webhooks, secrets, config) | not_started | Only providers + scheduler pause/resume are wired into `operator_controls.py`; the rest remain CLI-only. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:30:| Guided setup/repair flows | not_started | Diagnostics report remediation strings per failing check, but there is no one-click "apply fix" action. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:31:| Command palette, global search, saved filters, keyboard shortcuts, deep links | not_started | Console has per-panel filters (audit) and Enter-to-send in the run console, but no palette/global search/deep-linking yet. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:34:| Accessibility (semantic HTML, labels, landmarks, focus, skip links, ARIA, keyboard, reduced motion, contrast, no color-only status) | improved | Run console uses `aria-label`/`aria-describedby` on the textarea, `role="log"` + `aria-live="polite"` for streamed events (not a rapid-fire live region — only a handful of events per run), Enter/Shift+Enter keyboard handling, and status text (not just color) for state. No dedicated skip link yet anywhere in the console (pre-existing gap, not introduced this session). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:35:| Visual system: spacing, typography, color, cards/forms, hierarchy, no clipping | in place | New panel reuses the existing dark theme tokens (`--bg`, `--panel`, `--accent`, etc.) and card/pill conventions rather than introducing a new visual language. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:36:| Loading/empty/error/degraded/offline/reconnecting/unauthorized/forbidden/read-only states | partial | Run console has starting/running/complete/error/stopped-watching/connection-lost states. Dashboard-wide offline/reconnecting state is still just "console degraded" text, not a dedicated banner. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:37:| Destructive-action confirmations / undo / rollback | in place (existing) | Operator controls already require typed confirmation tokens; unchanged this session. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:38:| Backend: auth, policy, redaction, CSRF, rate limits, structured audit events | improved | New `/runs*` routes reuse the existing auth/CSRF/rate-limit pipeline; run and tool-call payloads are redacted with the same `redact_audit_value` used for the audit browser; a run-start conflict (409) is recorded as a `web.run` audit denial. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:39:| Tests: security, routing, API behavior, audit filtering, redaction, navigation, control-plane actions | improved | 19 new unit tests (`tests/api/test_run_stream.py`) + 16 new integration tests (`TestRuns` in `tests/api/test_server.py`) covering auth, CSRF, concurrency (409), redaction, SSE framing, late-join/reconnect, and error propagation. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:45:| A1 | Streaming subscription state machine | in place | `AgentSubscription` remains wired into `run_stream()`; the new SSE run viewer is a separate, coarser-grained event stream (run/tool lifecycle, not token deltas) suited to the tool-calling loop where token streaming isn't available. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:46:| A2 | Layered tool policy pipeline | hardened | Unchanged this session. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:47:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:48:| A11 | Raw-stream JSONL diagnostics | partial | The new `/runs/{id}/events` SSE stream is effectively a redacted, per-run JSONL-over-HTTP diagnostic feed for tool calls; a persisted raw-stream JSONL log (separate from SSE) remains future work. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:49:| A12 | Transcript dual-repair | improved | Unchanged this session (OpenAI provider layer). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:56:2. Add a scheduler jobs panel (list + create/remove, not just pause/resume)
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:58:   bot-control coverage" gap.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:60:   keyboard-first navigation.
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
```
