# Build Status

Last updated: 2026-07-09 01:40 EDT

## Current State

Primary focus remains the **Web TUI / operator console overhaul** (branch
`overhaul/web-tui-20260709-004527`). Last session's "Next Actions" named two
concrete items — (1) render cost/provider/tools_used detail in the run
console's log, and (2) add a scheduler jobs panel (create/remove) and a
memory browser panel (search/pin/delete) to close the largest remaining
"full bot-control coverage" gaps. This session built both.

### What shipped this session

- **`missy/api/run_stream.py`**: `run_stream.py` previously only forwarded
  `agent.run.start`/`tool.request`/`tool.result` bus events; the resolved
  provider, tools used, and cost summary that `AgentRuntime.run()` already
  publishes on `agent.run.complete` were never picked up. Added a second bus
  subscription (`_SUMMARY_TOPIC = "agent.run.complete"`) that captures
  `resolved_provider`/`tools_used`/`cost` onto the `RunHandle` before the
  synthesized terminal `run.complete` event is pushed, so both the SSE event
  and `RunHandle.to_dict()` (used by `GET /api/v1/runs/{id}`) now carry them.
  Verified no race: `MessageBus.publish()` dispatches synchronously in the
  calling thread, and `runtime.run()` publishes `agent.run.complete` before
  returning, so the summary handler always fires before `_execute()` builds
  the terminal event.
- **`missy/api/operator_controls.py`**: added `scheduler.remove_job`, a third
  confirmation-gated scheduler control (`confirm: "remove-job:<id>"`,
  `destructive: true`) alongside the existing pause/resume controls, with its
  own audited target list.
- **`missy/api/server.py`**: new routes —
  `GET /api/v1/scheduler/jobs` (full job detail listing),
  `POST /api/v1/scheduler/jobs` (guarded creation: `name`/`schedule`/`task`
  required, `provider`/`description`/`active_hours`/`timezone` optional,
  `web.scheduler` audit event on both allow and deny),
  `DELETE /api/v1/scheduler/jobs/{id}` (thin REST alias that delegates to the
  `scheduler.remove_job` control so both entry points share one
  confirmation/audit path),
  `DELETE /api/v1/memory/turns/{id}` and
  `POST /api/v1/memory/turns/{id}/pin` (both emit `web.memory` audit events).
  Memory search/history responses now include each turn's `id` and `pinned`
  state so the console can wire up per-row actions.
- **`missy/memory/sqlite_store.py`**: added `delete_turn()` (removes the row
  and its FTS5 index entry) and `set_turn_pinned()` (sets/clears a `pinned`
  key in the turn's existing `metadata` JSON blob — no schema migration).
  `cleanup()` now excludes pinned turns via
  `json_extract(metadata, '$.pinned')` so an operator can preserve specific
  memories past the normal retention window.
- **`missy/memory/resilient.py`**: `ResilientMemoryStore` delegates both new
  methods to the primary store; `delete_turn()` also prunes the in-memory
  fallback cache even when the primary raises, so a crash-looped primary
  can't resurrect a deleted turn from the cache.
- **`missy/api/web_console.py`**: two new panels — **Scheduled Jobs** (list
  with state/schedule/provider, a guarded create form, and a Remove button
  wired to the confirmation flow) and **Memory Browser** (debounced search
  box + session filter, pin/unpin and delete per result row). The run
  console's `run.complete` handler now renders a one-line summary
  (`provider: ... · tools: ... · cost: $...`) once a run finishes.
- Tests: +15 integration tests in `tests/api/test_server.py` (scheduler job
  CRUD, `scheduler.remove_job` control, memory turn pin/delete, console
  markup/script assertions), +2 unit tests in `tests/api/test_run_stream.py`
  (summary-field enrichment, including the late-join/reconnect path), +13
  unit tests in `tests/memory/test_turn_pin_delete.py` (delete/pin,
  pinned-cleanup exemption, `ResilientMemoryStore` delegation and
  primary-failure fallback).
- Docs: `docs/operations.md` gained scheduler-jobs and memory-browser API
  documentation plus the run cost/routing enrichment; `docs/implementation/
  module-map.md`'s `missy.api`/`missy.memory.sqlite_store` sections were
  updated for the new routes and store methods.
- Manually verified end-to-end against a live (non-mocked) `ApiServer`:
  login → CSRF extraction → scheduler job create/list/remove (with and
  without confirmation) → memory search/pin/delete → confirmed deletion
  removes the turn from search results; separately verified the run
  cost/provider/tools_used enrichment flows through both the poll endpoint
  and the terminal SSE event using a fake runtime that publishes
  `agent.run.complete`.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Web UI entrypoint + auth/session/CSRF | in place (prior sessions) | Unchanged this session; new routes reuse it as-is. |
| Dashboard (status/providers/tools/diagnostics/audit/controls) | in place (prior sessions) | Unchanged this session. |
| Session/run viewer with streaming output, tool calls, errors | in place (prior session) | Unchanged this session. |
| Run cost/model-routing surfaced end-to-end | **closed this session** | `resolved_provider`/`tools_used`/`cost` now flow through both the SSE terminal event and the poll endpoint, and render in the console's run log. |
| Scheduler jobs panel (list/create/remove) | **new** | `GET/POST /scheduler/jobs`, `DELETE /scheduler/jobs/{id}` (→ `scheduler.remove_job` control), console panel with create form + remove button. |
| Memory browser panel (search/pin/delete) | **new** | Reuses existing `GET /memory/search`; new `POST /memory/turns/{id}/pin` and `DELETE /memory/turns/{id}`; console panel with debounced search + per-row pin/delete. |
| Pinned-turn retention override | **new** | `SQLiteMemoryStore.cleanup()` now spares turns with `metadata.pinned = true`. |
| Concurrency safety for the API server | in place (prior session) | `ThreadingHTTPServer`; unchanged this session. |
| Redaction of run/tool/scheduler/memory event payloads | in place | New audit events (`web.scheduler`, `web.memory`) reuse the same `_emit_web_audit`/`redact_audit_value` path as everything else. |
| Tests | improved | +30 tests this session (15 integration + 2 run-stream unit + 13 memory unit); full existing suite re-verified green (see Tests section). |

## Current Architecture State

- `ApiServer` (`missy/api/server.py`) is still the single process serving
  both the JSON REST API and the server-rendered operator console. Scheduler
  and memory-turn routes follow the same closure-captured-dependency pattern
  as every other route — no new global state.
- `operator_controls.py` now has three scheduler controls (pause/resume/
  remove) sharing one confirmation-token convention
  (`"<action>-job:<id>"`) and one audit-detail shape; `scheduler.remove_job`
  is also reachable via a conventional `DELETE /scheduler/jobs/{id}` for
  scripts that prefer REST verbs over the controls envelope.
- `SQLiteMemoryStore` still has no dedicated `pinned` column — the flag
  lives inside the existing `metadata` JSON blob, avoiding a migration while
  giving `cleanup()` a cheap `json_extract()` check.
- `RunRegistry` (`missy/api/run_stream.py`) now subscribes to two bus topics
  per run instead of one group: `_RUN_TOPICS` (forwarded verbatim) and
  `_SUMMARY_TOPIC` (folded into the terminal event, not forwarded verbatim).
  Both are cleanly unsubscribed in the `finally` block alongside the
  existing topics.

## Tests

- `python3 -m pytest tests/memory/test_turn_pin_delete.py -q`: 13 passed.
- `python3 -m pytest tests/api/test_run_stream.py -q`: 21 passed.
- `python3 -m pytest tests/api/test_server.py -q`: 118 passed.
- `python3 -m pytest tests/api/ tests/memory/ -q`: 738 passed, 7 skipped.
- `python3 -m ruff check missy/ tests/`: passed (repo-wide).
- `python3 -m ruff format --check missy/ tests/`: passed (733 files, repo-wide).
- Full-repo `python3 -m pytest -q`: see `TEST_RESULTS.md` for this session's
  run.

## Remaining Work

1. Add a dashboard-wide cost/usage panel (aggregate spend across sessions
   via `SQLiteMemoryStore.get_total_costs()` / `CostTracker`, not just the
   now-visible per-run cost).
2. Add a command palette / global search and a skip link for keyboard-first
   navigation across the whole console.
3. Extend safe controls to tools/skills/plugins and Discord channel/guild
   allowlists — memory and scheduler are now covered; those remain
   CLI-only.
4. Add a dedicated offline/reconnecting banner for the dashboard as a whole
   (currently only the run console has explicit connection-lost handling).
5. Persist run history across `ApiServer` restarts if that turns out to
   matter operationally (currently in-memory only, matching the existing
   `_SessionRegistry`).

## Blockers

- None. The next slice (a cost/usage panel, or extending safe controls to
  another subsystem) is additive and does not require new backend
  primitives beyond what already exists.

## Next Actions

Add a dashboard-wide cost/usage panel backed by
`SQLiteMemoryStore.get_total_costs()`, then extend `operator_controls.py`
with tool/skill enable-disable controls to keep closing the "full
bot-control coverage" gap.
