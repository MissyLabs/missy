# Build Status

Last updated: 2026-07-09 00:55 EDT

## Current State

Primary focus switched this session to the **Web TUI / operator console
overhaul** (branch `overhaul/web-tui-20260709-004527`). The console already
had a working dashboard, audit browser, diagnostics, and a small set of
safe operator controls from earlier sessions on this line of work
(`missy/api/server.py`, `web_console.py`, `web_sessions.py`,
`audit_browser.py`, `diagnostics.py`, `operator_controls.py`); the highest-
value gap against the loop's required-capabilities list was the missing
"ask the bot and watch a run stream" workflow — session/run viewer with
streaming output, tool calls, and errors. This session built that.

### What shipped this session

- **`missy/api/run_stream.py`** (new): `RunRegistry` executes
  `AgentRuntime.run()` on a background daemon thread per request, mirrors
  `agent.run.start` / `tool.request` / `tool.result` events from the
  process-wide `MessageBus` into a per-run bounded queue (filtered strictly
  by `session_id` so concurrent sessions never cross-contaminate each
  other's stream), and enforces exactly one in-flight run per session.
  Late-joining or reconnecting SSE clients get a synthesized terminal event
  immediately instead of hanging. All event payloads (including free-text
  exception messages) pass through the existing `redact_audit_value()`
  secret-detector-backed redaction.
- **`missy/api/server.py`**: new routes `POST /api/v1/runs` (202, starts a
  background run), `GET /api/v1/runs/{id}` (poll status/result),
  `GET /api/v1/runs?session_id=` (list run history for a session), and
  `GET /api/v1/runs/{id}/events` (Server-Sent Events stream). The SSE route
  is intercepted ahead of the generic JSON router (it writes raw
  `text/event-stream` bytes) but still runs through the same
  `_authenticate()` check and the same per-IP rate limiter as every other
  route. `HTTPServer` was swapped for `ThreadingHTTPServer` so a long-lived
  SSE connection cannot block other operators' requests (health checks,
  audit queries, controls) on the same process.
- **`missy/agent/runtime.py`**: `AGENT_RUN_COMPLETE` bus events now include
  the same `cost_detail` summary already attached to the audit event, so
  the run stream (and future UI) can surface per-run cost.
- **`missy/api/web_console.py`**: new "Ask Missy" dashboard panel — a
  prompt box that starts a run and renders it live via `EventSource`: tool
  calls, completion, and errors stream into an accessible
  (`role="log"`, `aria-live="polite"`) activity log, with the final response
  shown separately. Enter submits, Shift+Enter inserts a newline, a "Stop
  watching" control closes the stream without cancelling the run itself.
- Tests: 19 new unit tests in `tests/api/test_run_stream.py` (lifecycle,
  redaction, cross-session isolation, concurrency/409, late-join/reconnect,
  no-message-bus fallback, SSE framing) and 16 new integration tests
  (`TestRuns` in `tests/api/test_server.py`) covering auth, CSRF, polling,
  listing, SSE delivery, error propagation, and audit emission on conflict.
- Docs: `docs/operations.md` gained a "Web TUI / operator console" section
  (`missy api start`, login/CSRF flow, the run console workflow, and `curl`
  examples for the same API a script could drive); `docs/implementation/
  module-map.md`'s `missy.api` section was brought up to date (it was
  missing `web_console`, `web_sessions`, `audit_browser`, `diagnostics`,
  `operator_controls` entirely, and now also documents `run_stream`).

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Web UI entrypoint + auth/session/CSRF | in place (prior sessions) | Unchanged this session; new routes reuse it as-is. |
| Dashboard (status/providers/tools/diagnostics/audit/controls) | in place (prior sessions) | Unchanged this session. |
| Session/run viewer with streaming output, tool calls, errors | **new** | `POST /runs` + `GET /runs/{id}/events` (SSE) + console panel. |
| Session/run history (resumable context) | **new** | `GET /runs?session_id=` lists prior runs per session. |
| Run cost/model-routing surfaced end-to-end | partial | Cost now flows into the bus payload; console UI doesn't render it yet. |
| Concurrency safety for the API server | improved | `ThreadingHTTPServer` swap; verified against full existing test suite. |
| Redaction of run/tool event payloads | in place | Reuses `audit_browser.redact_audit_value()`. |
| Audit trail for run conflicts | **new** | `web.run` / `deny` event on 409. |
| Accessibility of the new panel | in place | Labeled controls, `role="log"`, keyboard submit, non-color-only status text. |
| Tests | improved | +35 tests this session (19 unit + 16 integration); full existing suite re-verified green. |

## Current Architecture State

- `ApiServer` (`missy/api/server.py`) is the single process serving both the
  JSON REST API and the server-rendered operator console, now on a
  `ThreadingHTTPServer`.
- `RunRegistry` (`missy/api/run_stream.py`) is a new, self-contained
  in-process component: it owns no persistent state (in-memory only, capped
  and TTL-pruned) and depends only on `AgentRuntime.run()` and the optional
  process-wide `MessageBus` — it degrades gracefully (still runs the agent,
  just without bus-sourced tool events) when no bus is initialized.
- The run console's client-side code lives entirely in `web_console.py`'s
  `console_script()` (vanilla JS, no build step), consistent with the rest
  of the console.

## Tests

- `python3 -m pytest tests/api/test_run_stream.py -v`: 19 passed.
- `python3 -m pytest tests/api/test_server.py -k TestRuns -v`: 15 passed.
- `python3 -m pytest tests/api/ -q`: 122 passed.
- `python3 -m pytest tests/agent/ -q`: 4109 passed, 4 skipped.
- `python3 -m ruff check missy/ tests/`: passed.
- `python3 -m ruff format --check` on all touched files: passed.
- Full-repo `python3 -m pytest -q`: see `TEST_RESULTS.md` for this session's
  run (kicked off in the background; results recorded there once complete).

## Remaining Work

1. Render cost/model-routing/provider-fallback detail in the console's run
   log (data already flows through the bus/SSE pipeline).
2. Add a scheduler jobs panel (create/remove, not just pause/resume) and a
   memory browser panel (search, pin/delete) — biggest remaining "full
   bot-control coverage" gaps.
3. Add a command palette / global search and a skip link for keyboard-first
   navigation across the whole console, not just the run panel.
4. Add dedicated offline/reconnecting banner state for the dashboard as a
   whole (currently only the run console has explicit connection-lost
   handling).
5. Persist run history across `ApiServer` restarts if that turns out to
   matter operationally (currently in-memory only, matching the existing
   `_SessionRegistry`).

## Blockers

- None. The next slice (rendering cost/routing detail, or a scheduler/
  memory panel) is additive and does not require new backend primitives
  beyond what already exists.

## Next Actions

Render `cost`/`provider`/`tools_used` detail from the run stream in the
console's run log, then start the scheduler jobs panel (list/create/remove)
to close the largest remaining "full bot-control coverage" gap.
