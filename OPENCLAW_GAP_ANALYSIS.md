# OpenClaw Gap Analysis

Last updated: 2026-07-09 01:30 EDT

## Current Focus

Primary focus remains **completing the Web TUI / operator console overhaul**
(branch `overhaul/web-tui-20260709-004527`). OpenClaw and Odin remain
references for control-plane ergonomics, live status, run/tool visibility,
and auditability; Missy's implementation is clean-room and Python/vanilla-JS
native (server-rendered HTML, no frontend build step).

The previous branch's OpenAI provider work (native Responses routing,
streaming reconciliation, structured outputs, diagnostics) is preserved as-is
and is not part of this session's scope.

## Web TUI / Operator Console Capability Status

| Capability (from loop spec) | Status | Notes |
|---|---|---|
| Secure local Web UI entrypoint + auth/session | in place | `ApiServer` serves `/` (console) and `/api/v1/*` from one `ThreadingHTTPServer`. Cookie session (`HttpOnly`, `SameSite=Strict`) + API key, CSRF on unsafe browser requests, security headers (CSP, X-Frame-Options, no-store). |
| Dashboard: runtime status, providers, tools, memory, security posture | in place | `GET /status`, `/providers`, `/tools`, `/diagnostics` rendered as scannable panels with health pills. |
| Dashboard: scheduler, cost/usage, queues, jobs | improved this session | New **Scheduled Jobs** console panel lists every job (`GET /api/v1/scheduler/jobs`), creates new ones via a guarded form (`POST /api/v1/scheduler/jobs`), and removes them through the existing safe-controls confirmation flow. Cost/usage still has no dedicated panel — cost is now visible per-run (see below) but not aggregated dashboard-wide. |
| Session/run viewer with streaming output, tool calls, errors, provider, resumable context | in place | `POST /api/v1/runs` starts a background run; `GET /api/v1/runs/{id}/events` streams `run.started`, `run.start`, `tool.request`, `tool.result`, and terminal `run.complete`/`run.error` over SSE. `GET /api/v1/runs/{id}` polls status; `GET /api/v1/runs?session_id=` lists run history per session (resumable). Console "Ask Missy" panel drives this live. |
| Session/run viewer: costs, model routing, provider fallback | **closed this session** | `run_stream.py` now subscribes to `agent.run.complete` and folds `resolved_provider`/`tools_used`/`cost` into the terminal `run.complete` SSE event and the `GET /api/v1/runs/{id}` poll response; the console's run log renders a one-line summary (`provider: ... · tools: ... · cost: $...`) once a run finishes. |
| Audit log browser: filters, severity, actor/source, subsystem, timestamps, redaction | in place | `audit_browser.py` + console audit panel: result/severity/subsystem/actor/source/query/time-range filters, pagination, redacted detail view. |
| Diagnostics/doctor views (providers, tools, memory, policy, gateway, Discord, scheduler) | in place | `diagnostics.py` builds a redacted per-subsystem report consumed by the console and `missy doctor`. |
| Safe controls (providers, tools, jobs, channels, experimental features) | improved this session | `provider.set_default`, `scheduler.pause_job`/`resume_job`, and now `scheduler.remove_job` (confirmation-gated, audited, `destructive: true` flagged for the UI). No tool/channel/feature toggles yet. |
| Full bot-control coverage (memory, schedules, skills, plugins, Discord, voice, vision, webhooks, secrets, config) | improved this session | Memory turns now support pin (`POST /api/v1/memory/turns/{id}/pin`) and permanent delete (`DELETE /api/v1/memory/turns/{id}`), both audited; scheduler jobs support full list/create/remove. Skills, plugins, Discord, voice, vision, webhooks, secrets, and config remain CLI-only. |
| Guided setup/repair flows | not_started | Diagnostics report remediation strings per failing check, but there is no one-click "apply fix" action. |
| Command palette, global search, saved filters, keyboard shortcuts, deep links | not_started | Console has per-panel filters (audit, memory search) and Enter-to-send in the run console, but no palette/global search/deep-linking yet. |
| Live updates without jarring layout shifts | improved | Dashboard polls every 15s (unchanged); the run console gets true push updates via SSE for the workflow that most needs immediacy. |
| Responsive design (desktop + mobile) | in place | Existing `@media` breakpoints cover the grid/panels; the new scheduler/memory panels reuse the same grid/typography system. |
| Accessibility (semantic HTML, labels, landmarks, focus, skip links, ARIA, keyboard, reduced motion, contrast, no color-only status) | improved | Run console uses `aria-label`/`aria-describedby`, `role="log"` + `aria-live="polite"` for streamed events, Enter/Shift+Enter keyboard handling, and status text (not just color) for state. New scheduler-form and memory-search inputs all carry explicit `aria-label`s. No dedicated skip link yet anywhere in the console (pre-existing gap, not introduced this session). |
| Visual system: spacing, typography, color, cards/forms, hierarchy, no clipping | in place | New panels reuse the existing dark theme tokens (`--bg`, `--panel`, `--accent`, etc.) and card/pill/row conventions rather than introducing a new visual language; destructive buttons get a distinct `.danger` treatment (not color alone — also labeled "Remove"/"Delete"). |
| Loading/empty/error/degraded/offline/reconnecting/unauthorized/forbidden/read-only states | partial | Run console has starting/running/complete/error/stopped-watching/connection-lost states. Scheduler and memory panels have empty states ("No scheduled jobs yet.", "Search memory to see results."). Dashboard-wide offline/reconnecting state is still just "console degraded" text, not a dedicated banner. |
| Destructive-action confirmations / undo / rollback | in place | Operator controls (including the new `scheduler.remove_job`) require typed confirmation tokens server-side and a `window.confirm()` prompt client-side; memory-turn delete also confirms client-side. No undo — deletions are permanent by design (matches `clear_session`/`cleanup` semantics). |
| Backend: auth, policy, redaction, CSRF, rate limits, structured audit events | improved | New `/scheduler/jobs*` and `/memory/turns/*` routes reuse the existing auth/CSRF/rate-limit pipeline; job creation and memory pin/delete emit `web.scheduler`/`web.memory` audit events; job removal reuses the existing `web.control` audit path via `scheduler.remove_job`. |
| Tests: security, routing, API behavior, audit filtering, redaction, navigation, control-plane actions | improved | +15 integration tests in `tests/api/test_server.py` (scheduler CRUD, memory turn pin/delete, `scheduler.remove_job` control, console markup/script assertions) and +2 unit tests in `tests/api/test_run_stream.py` (provider/tools/cost enrichment) and +13 unit tests in `tests/memory/test_turn_pin_delete.py` (delete/pin, pinned-cleanup exemption, `ResilientMemoryStore` delegation and failure fallback). |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | in place | `AgentSubscription` remains wired into `run_stream()`; the SSE run viewer is a separate, coarser-grained event stream (run/tool lifecycle, not token deltas) suited to the tool-calling loop where token streaming isn't available. |
| A2 | Layered tool policy pipeline | hardened | Unchanged this session. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
| A11 | Raw-stream JSONL diagnostics | partial | The `/runs/{id}/events` SSE stream is effectively a redacted, per-run JSONL-over-HTTP diagnostic feed for tool calls, now including the cost/provider/tools summary; a persisted raw-stream JSONL log (separate from SSE) remains future work. |
| A12 | Transcript dual-repair | improved | Unchanged this session (OpenAI provider layer). |

## Recommended Next Slice

1. Add a dashboard-wide cost/usage panel (aggregate spend across sessions,
   not just per-run) using `SQLiteMemoryStore.get_total_costs()` /
   `CostTracker`, which nothing in the console surfaces yet.
2. Add a command palette / global search and a skip link for
   keyboard-first navigation.
3. Add safe controls for tool/skill/plugin enable-disable and Discord
   channel/guild allowlist edits to keep closing the "full bot-control
   coverage" gap — memory and scheduler are now covered, tools/skills/
   plugins/Discord/voice/vision/webhooks/secrets/config are not.
4. Add a dashboard-wide offline/reconnecting banner (currently only the run
   console has explicit connection-lost handling).
