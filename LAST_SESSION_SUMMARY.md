# LAST_SESSION_SUMMARY

Date: 2026-07-09

## Changed

- Added `missy/api/run_stream.py`: `RunRegistry`/`RunHandle` execute
  `AgentRuntime.run()` on a background thread, mirror `agent.run.start` /
  `tool.request` / `tool.result` message-bus events into a per-run queue
  (strictly session-scoped, redacted via the existing audit redaction
  helper), and enforce one in-flight run per session. Reconnecting after a
  run finishes returns the terminal state immediately instead of hanging.
- Added `POST /api/v1/runs`, `GET /api/v1/runs/{id}`,
  `GET /api/v1/runs?session_id=`, and `GET /api/v1/runs/{id}/events` (SSE)
  to `missy/api/server.py`. The SSE route reuses the standard auth and rate
  limiting; a conflicting run-start (409) is recorded as a `web.run` audit
  denial.
- Swapped `HTTPServer` for `ThreadingHTTPServer` in `ApiServer` so a
  long-lived SSE connection can't block other requests on the same process.
- `AGENT_RUN_COMPLETE` bus events (`missy/agent/runtime.py`) now carry
  `cost_detail` under a `cost` key, matching what the audit event already
  recorded.
- Added an "Ask Missy" run console panel to `missy/api/web_console.py`:
  prompt box + live SSE-driven activity log (tool calls, completion,
  errors) with accessible markup (`role="log"`, `aria-live="polite"`,
  labeled controls, Enter/Shift+Enter handling).
- Added 19 unit tests (`tests/api/test_run_stream.py`) and 16 integration
  tests (`TestRuns` in `tests/api/test_server.py`).
- Updated `docs/operations.md` (new "Web TUI / operator console" section)
  and `docs/implementation/module-map.md` (the `missy.api` section was
  stale — missing `web_console`, `web_sessions`, `audit_browser`,
  `diagnostics`, `operator_controls` entirely; now documents all of them
  plus the new `run_stream`).
- Rewrote `OPENCLAW_GAP_ANALYSIS.md`, `AUDIT_SECURITY.md`,
  `AUDIT_CONNECTIVITY.md`, and `TEST_EDGE_CASES.md` for the Web TUI focus
  (they still described the prior OpenAI-provider-overhaul branch's work).

## Verification

```text
python3 -m pytest tests/api/test_run_stream.py -v
19 passed
```

```text
python3 -m pytest tests/api/test_server.py -k TestRuns -v
15 passed
```

```text
python3 -m pytest tests/api/ -q
122 passed
```

```text
python3 -m pytest tests/agent/ -q
4109 passed, 4 skipped
```

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check <all touched files>
All checks passed!
```

Full-repo `python3 -m pytest -q` was run before ending the session; see
`TEST_RESULTS.md` for the exact count.

## Remains

- Console doesn't yet render cost/model-routing/fallback detail per run
  (the data already flows through the bus and SSE stream).
- No scheduler jobs panel (create/remove) or memory browser panel yet —
  these are the largest remaining "full bot-control coverage" gaps.
- No command palette / global search / skip link across the console.
- No dashboard-wide offline/reconnecting banner (only the run console has
  explicit connection-lost handling).
- Run history is in-memory only (matches the existing `_SessionRegistry`
  pattern) — does not survive an `ApiServer` restart.

## First Next Step

Render `cost` / `provider` / `tools_used` from the run stream in the
console's run log (backend already emits it), then start the scheduler
jobs panel to close the next-largest "full bot-control coverage" gap.
