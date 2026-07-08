# Build Status

Last updated: 2026-07-08

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The console is still layered onto the existing stdlib `ApiServer` and preserves the API-key JSON contract while adding browser sessions, CSRF, hardened HTML responses, and a denser local operator dashboard.

This session added the first authenticated audit browser slice:

- `/api/v1/audit` returns authenticated, redacted audit events with filters for `event_type`, `category`, `result`, `session_id`, `task_id`, `severity`, `actor`, `source`, `subsystem`, `action`, `since`, `until`, and free-text `q`.
- The Web TUI dashboard now includes an Audit Trail panel with result/subsystem filters.
- Browser login success/failure, logout success/failure, and browser API CSRF denials publish structured audit events through the existing audit bus.
- JSON-derived dashboard strings are escaped before insertion into `innerHTML`.
- Audit response values are recursively redacted before returning to the browser.
- `SimpleVectorizer` now uses stable BLAKE2b bucket hashing instead of process-randomized `hash()`, removing nondeterministic vector collisions in the full suite.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` are served by `missy/api/server.py`. |
| Web session handling | implemented | In-memory session store, cookie auth, expiry, and logout revocation. |
| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, security posture, and audit trail are shown. |
| Audit log browser API | started | Authenticated `/api/v1/audit` supports filtering, facets, file-backed events, memory fallback, and redaction. |
| Web security audit events | started | Login, logout, and CSRF events include actor/source/subsystem/action/severity metadata. |
| Console security tests | expanded | API suite covers audit auth, filtering, redaction, Web UI event emission, CSRF, cookies, and logout. |
| Vector memory determinism | fixed | Stable hashing makes vector encoding deterministic across Python hash seeds. |

## Current Architecture State

- The Web TUI currently lives inside `ApiServer`; this is still coherent for the stdlib/no-build-pipeline constraint but is becoming large.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`, `/sessions`, and `/audit` with same-origin credentials and the embedded CSRF token.
- Audit browser reads from `AuditLogger` when initialized and falls back to the process event bus for lightweight/test server usage.
- Redaction is server-side and recursive; the browser is treated as a presentation surface, not a security boundary.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not touched.

## Tests

- `python3 -m pytest tests/api/test_server.py -q`: 74 passed.
- `python3 -m pytest tests/memory/test_vector_store_coverage.py::TestSimpleVectorizer -q`: 8 passed.
- `python3 -m pytest tests/api/test_server.py tests/memory/test_vector_store_coverage.py::TestSimpleVectorizer -q`: 82 passed.
- `python3 -m pytest -q`: 20452 passed, 13 skipped in 376.44s.
- `python3 -m ruff check .`: passed.
- `python3 -m ruff format --check .`: 726 files already formatted.

## Remaining Work

1. Split Web TUI rendering/session/audit helpers out of `missy/api/server.py` into clearer internal modules while preserving behavior.
2. Expand the audit browser UI with timestamp search, actor/source fields, severity chips, pagination, and event-detail drilldown.
3. Add diagnostics/doctor panels for Discord, providers, scheduler, tools, memory, gateway, policy, and network posture.
4. Add safe policy-gated controls for providers, tools, scheduled jobs, channels, and experimental features.
5. Add run/session viewer with streaming output, tool calls, errors, model routing, fallback, costs, and resumable context.
6. Add responsive visual regression coverage or Playwright smoke checks once a browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
