# OpenClaw Gap Analysis

Last updated: 2026-07-09 00:55 EDT

## Current Focus

Primary focus switched to **completing the Web TUI / operator console
overhaul** (branch `overhaul/web-tui-20260709-004527`). OpenClaw and Odin
remain references for control-plane ergonomics, live status, run/tool
visibility, and auditability; Missy's implementation is clean-room and
Python/vanilla-JS native (server-rendered HTML, no frontend build step).

The previous branch's OpenAI provider work (native Responses routing,
streaming reconciliation, structured outputs, diagnostics) is preserved as-is
and is not part of this session's scope.

## Web TUI / Operator Console Capability Status

| Capability (from loop spec) | Status | Notes |
|---|---|---|
| Secure local Web UI entrypoint + auth/session | in place | `ApiServer` serves `/` (console) and `/api/v1/*` from one `ThreadingHTTPServer`. Cookie session (`HttpOnly`, `SameSite=Strict`) + API key, CSRF on unsafe browser requests, security headers (CSP, X-Frame-Options, no-store). |
| Dashboard: runtime status, providers, tools, memory, security posture | in place | `GET /status`, `/providers`, `/tools`, `/diagnostics` rendered as scannable panels with health pills. |
| Dashboard: scheduler, cost/usage, queues, jobs | partial | Scheduler pause/resume exists as a *control*; there is no dedicated scheduler jobs panel or cost/usage panel yet. |
| Session/run viewer with streaming output, tool calls, errors, provider, resumable context | **new this session** | `POST /api/v1/runs` starts a background run; `GET /api/v1/runs/{id}/events` streams `run.started`, `run.start`, `tool.request`, `tool.result`, and terminal `run.complete`/`run.error` over SSE. `GET /api/v1/runs/{id}` polls status; `GET /api/v1/runs?session_id=` lists run history per session (resumable). Console "Ask Missy" panel drives this live. |
| Session/run viewer: costs, model routing, provider fallback | partial | `AGENT_RUN_COMPLETE` bus payload now carries `cost` (this session); the run stream forwards it, but the console UI does not yet render cost/model-routing/fallback detail in the run log. |
| Audit log browser: filters, severity, actor/source, subsystem, timestamps, redaction | in place | `audit_browser.py` + console audit panel: result/severity/subsystem/actor/source/query/time-range filters, pagination, redacted detail view. |
| Diagnostics/doctor views (providers, tools, memory, policy, gateway, Discord, scheduler) | in place | `diagnostics.py` builds a redacted per-subsystem report consumed by the console and `missy doctor`. |
| Safe controls (providers, tools, jobs, channels, experimental features) | partial | Only `provider.set_default` and `scheduler.pause_job`/`resume_job` exist. No tool/channel/feature toggles yet. |
| Full bot-control coverage (memory, schedules, skills, plugins, Discord, voice, vision, webhooks, secrets, config) | not_started | Only providers + scheduler pause/resume are wired into `operator_controls.py`; the rest remain CLI-only. |
| Guided setup/repair flows | not_started | Diagnostics report remediation strings per failing check, but there is no one-click "apply fix" action. |
| Command palette, global search, saved filters, keyboard shortcuts, deep links | not_started | Console has per-panel filters (audit) and Enter-to-send in the run console, but no palette/global search/deep-linking yet. |
| Live updates without jarring layout shifts | improved | Dashboard polls every 15s (unchanged); the new run console additionally gets true push updates via SSE for the workflow that most needs immediacy. |
| Responsive design (desktop + mobile) | in place | Existing `@media` breakpoints cover the grid/panels; new run console panel reuses the same grid/typography system and was checked against the same breakpoints. |
| Accessibility (semantic HTML, labels, landmarks, focus, skip links, ARIA, keyboard, reduced motion, contrast, no color-only status) | improved | Run console uses `aria-label`/`aria-describedby` on the textarea, `role="log"` + `aria-live="polite"` for streamed events (not a rapid-fire live region — only a handful of events per run), Enter/Shift+Enter keyboard handling, and status text (not just color) for state. No dedicated skip link yet anywhere in the console (pre-existing gap, not introduced this session). |
| Visual system: spacing, typography, color, cards/forms, hierarchy, no clipping | in place | New panel reuses the existing dark theme tokens (`--bg`, `--panel`, `--accent`, etc.) and card/pill conventions rather than introducing a new visual language. |
| Loading/empty/error/degraded/offline/reconnecting/unauthorized/forbidden/read-only states | partial | Run console has starting/running/complete/error/stopped-watching/connection-lost states. Dashboard-wide offline/reconnecting state is still just "console degraded" text, not a dedicated banner. |
| Destructive-action confirmations / undo / rollback | in place (existing) | Operator controls already require typed confirmation tokens; unchanged this session. |
| Backend: auth, policy, redaction, CSRF, rate limits, structured audit events | improved | New `/runs*` routes reuse the existing auth/CSRF/rate-limit pipeline; run and tool-call payloads are redacted with the same `redact_audit_value` used for the audit browser; a run-start conflict (409) is recorded as a `web.run` audit denial. |
| Tests: security, routing, API behavior, audit filtering, redaction, navigation, control-plane actions | improved | 19 new unit tests (`tests/api/test_run_stream.py`) + 16 new integration tests (`TestRuns` in `tests/api/test_server.py`) covering auth, CSRF, concurrency (409), redaction, SSE framing, late-join/reconnect, and error propagation. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | in place | `AgentSubscription` remains wired into `run_stream()`; the new SSE run viewer is a separate, coarser-grained event stream (run/tool lifecycle, not token deltas) suited to the tool-calling loop where token streaming isn't available. |
| A2 | Layered tool policy pipeline | hardened | Unchanged this session. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
| A11 | Raw-stream JSONL diagnostics | partial | The new `/runs/{id}/events` SSE stream is effectively a redacted, per-run JSONL-over-HTTP diagnostic feed for tool calls; a persisted raw-stream JSONL log (separate from SSE) remains future work. |
| A12 | Transcript dual-repair | improved | Unchanged this session (OpenAI provider layer). |

## Recommended Next Slice

1. Render cost/model-routing/fallback detail in the console's run log (data
   already flows through the bus/SSE pipeline; only the client-side renderer
   needs it).
2. Add a scheduler jobs panel (list + create/remove, not just pause/resume)
   and a memory browser panel (search, pin/delete) to close the "full
   bot-control coverage" gap.
3. Add a command palette / global search and a skip link for
   keyboard-first navigation.
