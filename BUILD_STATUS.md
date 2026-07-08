# Build Status

Last updated: 2026-07-08 09:31:21 EDT

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The
console still runs on the stdlib `ApiServer` and preserves the API-key JSON API
contract, while browser access uses explicit login, hardened cookies, CSRF for
unsafe browser-authenticated API calls, no-store responses, and server-side
audit redaction.

This session added the first diagnostics/doctor slice on top of the existing
Web TUI architecture:

- Added `missy/api/diagnostics.py` as a redacted diagnostics view-model module.
- Added authenticated `GET /api/v1/diagnostics` for Web entrypoint, providers,
  tools, memory, policy, scheduler, and runtime posture.
- Added a dense Diagnostics panel to the browser console.
- Added API tests proving diagnostics auth, redaction, initialized
  default-deny policy reporting, and elevated tool permission summaries.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` remain served by `missy/api/server.py`. |
| Web session handling | extracted | `WebSession` and `WebSessionStore` now live in `missy/api/web_sessions.py`. |
| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, security posture, and richer audit trail are shown. |
| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
| Audit browser UI | improved | Search, actor/source, severity, timestamp filters, pagination, and event details are available. |
| Diagnostics API | started | Authenticated `/api/v1/diagnostics` reports redacted local posture across Web, providers, tools, memory, policy, scheduler, and runtime. |
| Diagnostics UI | started | Browser console now includes a compact Diagnostics panel. |
| Console security tests | expanded | API suite covers audit auth, filtering, redaction, Web UI event emission, CSRF, cookies, logout, and pagination. |

## Current Architecture State

- `missy/api/server.py` is still the HTTP router and renderer, but browser
  session lifecycle and audit-browser query logic are now separate internal API
  modules.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
  `/sessions`, and `/audit` with same-origin credentials and the embedded CSRF
  token.
- Audit browser reads from `AuditLogger` when initialized and falls back to the
  process event bus for lightweight/test server usage.
- Audit matching is performed against redacted records to avoid search-based
  secret disclosure.
- Diagnostics are built from already-injected server dependencies and redact
  summaries before returning them to the operator console.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
  touched.

## Tests

- `python3 -m ruff check .`: passed.
- `python3 -m ruff format --check .`: 729 files already formatted.
- `python3 -m pytest tests/api/test_server.py -q`: 77 passed.
- `python3 -m pytest -q`: 20455 passed, 13 skipped in 389.39s.

## Remaining Work

1. Continue extracting Web TUI rendering and frontend assets out of
   `missy/api/server.py` into clearer internal modules.
2. Deepen diagnostics/doctor panels with Discord, gateway, network probes,
   policy explanations, and actionable remediation.
3. Add safe policy-gated controls for providers, tools, scheduled jobs,
   channels, and experimental features.
4. Add run/session viewer with streaming output, tool calls, errors, model
   routing, fallback, costs, and resumable context.
5. Add responsive visual regression coverage or Playwright smoke checks once a
   browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
