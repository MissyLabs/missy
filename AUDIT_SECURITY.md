# AUDIT_SECURITY

- Timestamp: 2026-07-08 21:58:59

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
/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/BUILD_STATUS.md:7:Primary focus remains the **Web TUI / operator console overhaul** (branch
/home/missy/missy/BUILD_STATUS.md:8:`overhaul/web-tui-20260709-004527`). Last session's "Next Actions" named two
/home/missy/missy/BUILD_STATUS.md:9:concrete items — (1) render cost/provider/tools_used detail in the run
/home/missy/missy/BUILD_STATUS.md:10:console's log, and (2) add a scheduler jobs panel (create/remove) and a
/home/missy/missy/BUILD_STATUS.md:12:"full bot-control coverage" gaps. This session built both.
/home/missy/missy/BUILD_STATUS.md:14:### What shipped this session
/home/missy/missy/BUILD_STATUS.md:16:- **`missy/api/run_stream.py`**: `run_stream.py` previously only forwarded
/home/missy/missy/BUILD_STATUS.md:17:  `agent.run.start`/`tool.request`/`tool.result` bus events; the resolved
/home/missy/missy/BUILD_STATUS.md:18:  provider, tools used, and cost summary that `AgentRuntime.run()` already
/home/missy/missy/BUILD_STATUS.md:21:  `resolved_provider`/`tools_used`/`cost` onto the `RunHandle` before the
/home/missy/missy/BUILD_STATUS.md:26:  returning, so the summary handler always fires before `_execute()` builds
/home/missy/missy/BUILD_STATUS.md:28:- **`missy/api/operator_controls.py`**: added `scheduler.remove_job`, a third
/home/missy/missy/BUILD_STATUS.md:29:  confirmation-gated scheduler control (`confirm: "remove-job:<id>"`,
/home/missy/missy/BUILD_STATUS.md:30:  `destructive: true`) alongside the existing pause/resume controls, with its
/home/missy/missy/BUILD_STATUS.md:31:  own audited target list.
/home/missy/missy/BUILD_STATUS.md:33:  `GET /api/v1/scheduler/jobs` (full job detail listing),
/home/missy/missy/BUILD_STATUS.md:34:  `POST /api/v1/scheduler/jobs` (guarded creation: `name`/`schedule`/`task`
/home/missy/missy/BUILD_STATUS.md:35:  required, `provider`/`description`/`active_hours`/`timezone` optional,
/home/missy/missy/BUILD_STATUS.md:36:  `web.scheduler` audit event on both allow and deny),
/home/missy/missy/BUILD_STATUS.md:37:  `DELETE /api/v1/scheduler/jobs/{id}` (thin REST alias that delegates to the
/home/missy/missy/BUILD_STATUS.md:38:  `scheduler.remove_job` control so both entry points share one
/home/missy/missy/BUILD_STATUS.md:39:  confirmation/audit path),
/home/missy/missy/BUILD_STATUS.md:41:  `POST /api/v1/memory/turns/{id}/pin` (both emit `web.memory` audit events).
/home/missy/missy/BUILD_STATUS.md:46:  key in the turn's existing `metadata` JSON blob — no schema migration).
/home/missy/missy/BUILD_STATUS.md:54:- **`missy/api/web_console.py`**: two new panels — **Scheduled Jobs** (list
/home/missy/missy/BUILD_STATUS.md:55:  with state/schedule/provider, a guarded create form, and a Remove button
/home/missy/missy/BUILD_STATUS.md:57:  box + session filter, pin/unpin and delete per result row). The run
/home/missy/missy/BUILD_STATUS.md:59:  (`provider: ... · tools: ... · cost: $...`) once a run finishes.
/home/missy/missy/BUILD_STATUS.md:60:- Tests: +15 integration tests in `tests/api/test_server.py` (scheduler job
/home/missy/missy/BUILD_STATUS.md:61:  CRUD, `scheduler.remove_job` control, memory turn pin/delete, console
/home/missy/missy/BUILD_STATUS.md:62:  markup/script assertions), +2 unit tests in `tests/api/test_run_stream.py`
/home/missy/missy/BUILD_STATUS.md:67:- Docs: `docs/operations.md` gained scheduler-jobs and memory-browser API
/home/missy/missy/BUILD_STATUS.md:72:  login → CSRF extraction → scheduler job create/list/remove (with and
/home/missy/missy/BUILD_STATUS.md:75:  cost/provider/tools_used enrichment flows through both the poll endpoint
/home/missy/missy/BUILD_STATUS.md:83:| Web UI entrypoint + auth/session/CSRF | in place (prior sessions) | Unchanged this session; new routes reuse it as-is. |
/home/missy/missy/BUILD_STATUS.md:84:| Dashboard (status/providers/tools/diagnostics/audit/controls) | in place (prior sessions) | Unchanged this session. |
/home/missy/missy/BUILD_STATUS.md:85:| Session/run viewer with streaming output, tool calls, errors | in place (prior session) | Unchanged this session. |
/home/missy/missy/BUILD_STATUS.md:86:| Run cost/model-routing surfaced end-to-end | **closed this session** | `resolved_provider`/`tools_used`/`cost` now flow through both the SSE terminal event and the poll endpoint, and render in the console's run log. |
/home/missy/missy/BUILD_STATUS.md:87:| Scheduler jobs panel (list/create/remove) | **new** | `GET/POST /scheduler/jobs`, `DELETE /scheduler/jobs/{id}` (→ `scheduler.remove_job` control), console panel with create form + remove button. |
/home/missy/missy/BUILD_STATUS.md:90:| Concurrency safety for the API server | in place (prior session) | `ThreadingHTTPServer`; unchanged this session. |
/home/missy/missy/BUILD_STATUS.md:91:| Redaction of run/tool/scheduler/memory event payloads | in place | New audit events (`web.scheduler`, `web.memory`) reuse the same `_emit_web_audit`/`redact_audit_value` path as everything else. |
/home/missy/missy/BUILD_STATUS.md:92:| Tests | improved | +30 tests this session (15 integration + 2 run-stream unit + 13 memory unit); full existing suite re-verified green (see Tests section). |
/home/missy/missy/BUILD_STATUS.md:100:- `operator_controls.py` now has three scheduler controls (pause/resume/
/home/missy/missy/BUILD_STATUS.md:102:  (`"<action>-job:<id>"`) and one audit-detail shape; `scheduler.remove_job`
/home/missy/missy/BUILD_STATUS.md:103:  is also reachable via a conventional `DELETE /scheduler/jobs/{id}` for
/home/missy/missy/BUILD_STATUS.md:104:  scripts that prefer REST verbs over the controls envelope.
/home/missy/missy/BUILD_STATUS.md:108:- `RunRegistry` (`missy/api/run_stream.py`) now subscribes to two bus topics
/home/missy/missy/BUILD_STATUS.md:117:- `python3 -m pytest tests/api/test_run_stream.py -q`: 21 passed.
/home/missy/missy/BUILD_STATUS.md:122:- Full-repo `python3 -m pytest -q`: see `TEST_RESULTS.md` for this session's
/home/missy/missy/BUILD_STATUS.md:127:1. Add a dashboard-wide cost/usage panel (aggregate spend across sessions
/home/missy/missy/BUILD_STATUS.md:130:2. Add a command palette / global search and a skip link for keyboard-first
/home/missy/missy/BUILD_STATUS.md:132:3. Extend safe controls to tools/skills/plugins and Discord channel/guild
/home/missy/missy/BUILD_STATUS.md:133:   allowlists — memory and scheduler are now covered; those remain
/home/missy/missy/BUILD_STATUS.md:135:4. Add a dedicated offline/reconnecting banner for the dashboard as a whole
/home/missy/missy/BUILD_STATUS.md:143:- None. The next slice (a cost/usage panel, or extending safe controls to
/home/missy/missy/BUILD_STATUS.md:144:  another subsystem) is additive and does not require new backend
/home/missy/missy/BUILD_STATUS.md:149:Add a dashboard-wide cost/usage panel backed by
/home/missy/missy/BUILD_STATUS.md:150:`SQLiteMemoryStore.get_total_costs()`, then extend `operator_controls.py`
/home/missy/missy/BUILD_STATUS.md:151:with tool/skill enable-disable controls to keep closing the "full
/home/missy/missy/BUILD_STATUS.md:152:bot-control coverage" gap.
/home/missy/missy/conftest.py:1:# conftest.py — pytest configuration
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:7:Primary focus remains **completing the Web TUI / operator console overhaul**
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:(branch `overhaul/web-tui-20260709-004527`). OpenClaw and Odin remain
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:9:references for control-plane ergonomics, live status, run/tool visibility,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:10:and auditability; Missy's implementation is clean-room and Python/vanilla-JS
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:11:native (server-rendered HTML, no frontend build step).
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:13:The previous branch's OpenAI provider work (native Responses routing,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:14:streaming reconciliation, structured outputs, diagnostics) is preserved as-is
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:15:and is not part of this session's scope.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:21:| Secure local Web UI entrypoint + auth/session | in place | `ApiServer` serves `/` (console) and `/api/v1/*` from one `ThreadingHTTPServer`. Cookie session (`HttpOnly`, `SameSite=Strict`) + API key, CSRF on unsafe browser requests, security headers (CSP, X-Frame-Options, no-store). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Dashboard: runtime status, providers, tools, memory, security posture | in place | `GET /status`, `/providers`, `/tools`, `/diagnostics` rendered as scannable panels with health pills. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:23:| Dashboard: scheduler, cost/usage, queues, jobs | improved this session | New **Scheduled Jobs** console panel lists every job (`GET /api/v1/scheduler/jobs`), creates new ones via a guarded form (`POST /api/v1/scheduler/jobs`), and removes them through the existing safe-controls confirmation flow. Cost/usage still has no dedicated panel — cost is now visible per-run (see below) but not aggregated dashboard-wide. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:24:| Session/run viewer with streaming output, tool calls, errors, provider, resumable context | in place | `POST /api/v1/runs` starts a background run; `GET /api/v1/runs/{id}/events` streams `run.started`, `run.start`, `tool.request`, `tool.result`, and terminal `run.complete`/`run.error` over SSE. `GET /api/v1/runs/{id}` polls status; `GET /api/v1/runs?session_id=` lists run history per session (resumable). Console "Ask Missy" panel drives this live. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:25:| Session/run viewer: costs, model routing, provider fallback | **closed this session** | `run_stream.py` now subscribes to `agent.run.complete` and folds `resolved_provider`/`tools_used`/`cost` into the terminal `run.complete` SSE event and the `GET /api/v1/runs/{id}` poll response; the console's run log renders a one-line summary (`provider: ... · tools: ... · cost: $...`) once a run finishes. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:26:| Audit log browser: filters, severity, actor/source, subsystem, timestamps, redaction | in place | `audit_browser.py` + console audit panel: result/severity/subsystem/actor/source/query/time-range filters, pagination, redacted detail view. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:27:| Diagnostics/doctor views (providers, tools, memory, policy, gateway, Discord, scheduler) | in place | `diagnostics.py` builds a redacted per-subsystem report consumed by the console and `missy doctor`. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:28:| Safe controls (providers, tools, jobs, channels, experimental features) | improved this session | `provider.set_default`, `scheduler.pause_job`/`resume_job`, and now `scheduler.remove_job` (confirmation-gated, audited, `destructive: true` flagged for the UI). No tool/channel/feature toggles yet. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:29:| Full bot-control coverage (memory, schedules, skills, plugins, Discord, voice, vision, webhooks, secrets, config) | improved this session | Memory turns now support pin (`POST /api/v1/memory/turns/{id}/pin`) and permanent delete (`DELETE /api/v1/memory/turns/{id}`), both audited; scheduler jobs support full list/create/remove. Skills, plugins, Discord, voice, vision, webhooks, secrets, and config remain CLI-only. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:30:| Guided setup/repair flows | not_started | Diagnostics report remediation strings per failing check, but there is no one-click "apply fix" action. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:31:| Command palette, global search, saved filters, keyboard shortcuts, deep links | not_started | Console has per-panel filters (audit, memory search) and Enter-to-send in the run console, but no palette/global search/deep-linking yet. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:33:| Responsive design (desktop + mobile) | in place | Existing `@media` breakpoints cover the grid/panels; the new scheduler/memory panels reuse the same grid/typography system. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:34:| Accessibility (semantic HTML, labels, landmarks, focus, skip links, ARIA, keyboard, reduced motion, contrast, no color-only status) | improved | Run console uses `aria-label`/`aria-describedby`, `role="log"` + `aria-live="polite"` for streamed events, Enter/Shift+Enter keyboard handling, and status text (not just color) for state. New scheduler-form and memory-search inputs all carry explicit `aria-label`s. No dedicated skip link yet anywhere in the console (pre-existing gap, not introduced this session). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:35:| Visual system: spacing, typography, color, cards/forms, hierarchy, no clipping | in place | New panels reuse the existing dark theme tokens (`--bg`, `--panel`, `--accent`, etc.) and card/pill/row conventions rather than introducing a new visual language; destructive buttons get a distinct `.danger` treatment (not color alone — also labeled "Remove"/"Delete"). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:36:| Loading/empty/error/degraded/offline/reconnecting/unauthorized/forbidden/read-only states | partial | Run console has starting/running/complete/error/stopped-watching/connection-lost states. Scheduler and memory panels have empty states ("No scheduled jobs yet.", "Search memory to see results."). Dashboard-wide offline/reconnecting state is still just "console degraded" text, not a dedicated banner. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:37:| Destructive-action confirmations / undo / rollback | in place | Operator controls (including the new `scheduler.remove_job`) require typed confirmation tokens server-side and a `window.confirm()` prompt client-side; memory-turn delete also confirms client-side. No undo — deletions are permanent by design (matches `clear_session`/`cleanup` semantics). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:38:| Backend: auth, policy, redaction, CSRF, rate limits, structured audit events | improved | New `/scheduler/jobs*` and `/memory/turns/*` routes reuse the existing auth/CSRF/rate-limit pipeline; job creation and memory pin/delete emit `web.scheduler`/`web.memory` audit events; job removal reuses the existing `web.control` audit path via `scheduler.remove_job`. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:39:| Tests: security, routing, API behavior, audit filtering, redaction, navigation, control-plane actions | improved | +15 integration tests in `tests/api/test_server.py` (scheduler CRUD, memory turn pin/delete, `scheduler.remove_job` control, console markup/script assertions) and +2 unit tests in `tests/api/test_run_stream.py` (provider/tools/cost enrichment) and +13 unit tests in `tests/memory/test_turn_pin_delete.py` (delete/pin, pinned-cleanup exemption, `ResilientMemoryStore` delegation and failure fallback). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:45:| A1 | Streaming subscription state machine | in place | `AgentSubscription` remains wired into `run_stream()`; the SSE run viewer is a separate, coarser-grained event stream (run/tool lifecycle, not token deltas) suited to the tool-calling loop where token streaming isn't available. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:46:| A2 | Layered tool policy pipeline | hardened | Unchanged this session. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:47:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:48:| A11 | Raw-stream JSONL diagnostics | partial | The `/runs/{id}/events` SSE stream is effectively a redacted, per-run JSONL-over-HTTP diagnostic feed for tool calls, now including the cost/provider/tools summary; a persisted raw-stream JSONL log (separate from SSE) remains future work. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:49:| A12 | Transcript dual-repair | improved | Unchanged this session (OpenAI provider layer). |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:53:1. Add a dashboard-wide cost/usage panel (aggregate spend across sessions,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:57:   keyboard-first navigation.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:58:3. Add safe controls for tool/skill/plugin enable-disable and Discord
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:59:   channel/guild allowlist edits to keep closing the "full bot-control
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:60:   coverage" gap — memory and scheduler are now covered, tools/skills/
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:61:   plugins/Discord/voice/vision/webhooks/secrets/config are not.
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:62:4. Add a dashboard-wide offline/reconnecting banner (currently only the run
/home/missy/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file under 5 MB. Move older entries to timestamped archive files before appending more.
/home/missy/missy/HUMANIZE_AUDIT.md:7:| 2026-04-27T16:09:36Z | humanize.loop.initialized | allow | Initialized audit file for the OpenClaw/humanize loop. No opt-in humanistic behavior was activated this session. |
/home/missy/missy/HUMANIZE_AUDIT.md:8:| 2026-04-27T16:09:36Z | openclaw.a1.subscription | allow | Added streaming state machine primitives that can support future timing, tone, apology, and mood integrations without changing tool correctness. |
/home/missy/missy/HUMANIZE_AUDIT.md:9:| 2026-04-27T18:32:16Z | openclaw.a2.tool_policy | allow | Added layered tool availability filtering with trace labels. This gates future humanistic memory tools without changing execution fail-closed policy. |
/home/missy/missy/HUMANIZE_AUDIT.md:10:| 2026-04-27T18:53:28Z | openclaw.a2.config_policy | allow | Routed YAML-backed provider/global/agent/sandbox/subagent tool policy layers into runtime exposure decisions. Execution policy remains fail-closed in the registry. |
/home/missy/missy/HATCHING_LOG.md:35:- `initialize_config` — Config file creation or detection
/home/missy/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
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
```
