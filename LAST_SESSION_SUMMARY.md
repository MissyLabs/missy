# LAST_SESSION_SUMMARY

Date: 2026-07-09

## Changed

- `missy/api/run_stream.py`: added a bus subscription on `agent.run.complete`
  (`_SUMMARY_TOPIC`) that captures `resolved_provider`/`tools_used`/`cost`
  onto the `RunHandle` and folds them into both the terminal SSE
  `run.complete` event and `RunHandle.to_dict()` (so `GET /api/v1/runs/{id}`
  also carries them). Confirmed no race with `runtime.run()`'s synchronous
  `MessageBus.publish()`.
- `missy/api/operator_controls.py`: added `scheduler.remove_job`, a third
  confirmation-gated scheduler control (destructive-flagged) alongside
  pause/resume.
- `missy/api/server.py`: new routes `GET/POST /api/v1/scheduler/jobs`,
  `DELETE /api/v1/scheduler/jobs/{id}` (delegates to `scheduler.remove_job`),
  `DELETE /api/v1/memory/turns/{id}`, `POST /api/v1/memory/turns/{id}/pin`.
  Memory search/history responses now include each turn's `id` and `pinned`.
- `missy/memory/sqlite_store.py`: added `delete_turn()` and
  `set_turn_pinned()` (flag lives in the existing `metadata` JSON blob, no
  schema migration); `cleanup()` now exempts pinned turns via
  `json_extract(metadata, '$.pinned')`.
- `missy/memory/resilient.py`: `ResilientMemoryStore` delegates both new
  methods, pruning its in-memory cache on delete even when the primary
  store fails.
- `missy/api/web_console.py`: new **Scheduled Jobs** panel (list/create/
  remove) and **Memory Browser** panel (search/pin/delete); the run
  console's completion handler now shows a provider/tools/cost summary
  line.
- Tests: +15 integration tests (`tests/api/test_server.py`), +2 unit tests
  (`tests/api/test_run_stream.py`), +13 unit tests (new
  `tests/memory/test_turn_pin_delete.py`).
- Docs: `docs/operations.md` and `docs/implementation/module-map.md` updated
  for the new routes/methods. `OPENCLAW_GAP_ANALYSIS.md`, `AUDIT_SECURITY.md`,
  `AUDIT_CONNECTIVITY.md`, `TEST_EDGE_CASES.md` rewritten for this session's
  changes (the security/connectivity audits are now hand-written summaries
  of the *new* attack surface rather than stale grep dumps from an earlier
  session).

## Verification

```text
python3 -m pytest tests/memory/test_turn_pin_delete.py -q
13 passed
```

```text
python3 -m pytest tests/api/test_run_stream.py -q
21 passed
```

```text
python3 -m pytest tests/api/test_server.py -q
118 passed
```

```text
python3 -m pytest tests/api/ tests/memory/ -q
738 passed, 7 skipped
```

```text
python3 -m ruff check missy/ tests/
All checks passed!
```

```text
python3 -m ruff format --check missy/ tests/
733 files already formatted
```

Also manually verified end-to-end against a live (non-mocked) `ApiServer`:
login → CSRF → scheduler job create/list/remove (with/without confirmation)
→ memory search/pin/delete → deletion confirmed via re-search; and
separately verified the run cost/provider/tools_used enrichment through both
the poll endpoint and the terminal SSE event.

Full-repo `python3 -m pytest -q` was run before ending the session; see
`TEST_RESULTS.md` for the exact count.

## Remains

- No dashboard-wide cost/usage panel (aggregate spend) — cost is now visible
  per-run but not aggregated.
- No command palette / global search / skip link across the console.
- Safe controls cover providers, scheduler (pause/resume/remove), and now
  memory turns, but not tools/skills/plugins/Discord/voice/vision/webhooks/
  secrets/config.
- No dashboard-wide offline/reconnecting banner (only the run console has
  explicit connection-lost handling).
- Run history is still in-memory only (matches the existing
  `_SessionRegistry` pattern) — does not survive an `ApiServer` restart.

## First Next Step

Add a dashboard-wide cost/usage panel backed by
`SQLiteMemoryStore.get_total_costs()`, then extend `operator_controls.py`
with tool/skill enable-disable controls to keep closing the "full
bot-control coverage" gap.
