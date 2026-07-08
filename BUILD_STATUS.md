# Build Status

Last updated: 2026-07-08 10:19:19 EDT

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The
stdlib `ApiServer` still owns HTTP routing for the local Web UI and JSON API,
with API-key auth, browser cookie sessions, CSRF checks for unsafe browser API
calls, hardened cookies, no-store responses, rate limits, and server-side
redaction.

This session added the first safe operator controls slice and started renderer
extraction:

- Added `missy/api/operator_controls.py` for explicit, confirmed operator
  control view models and execution.
- Added authenticated `/api/v1/controls` and confirmed
  `/api/v1/controls/provider.set_default` endpoints.
- Enforced known provider target validation, availability checks, exact
  confirmation strings, existing CSRF protection for browser POSTs, and
  structured allow/deny audit events for every control attempt.
- Added a compact Web TUI Controls panel for switching the in-process default
  provider when another registered provider is available.
- Extracted login/message rendering and the shared console stylesheet into
  `missy/api/web_console.py`, reducing embedded renderer responsibility in
  `missy/api/server.py`.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` remain served by `missy/api/server.py`. |
| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
| Diagnostics API | improved | `/api/v1/diagnostics` reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
| Safe controls API | started | Provider default switching is confirmed, validated, CSRF-protected for browser sessions, and audited. |
| Renderer extraction | started | Login/message pages and CSS are now in `missy/api/web_console.py`. |
| Console security tests | expanded | API suite covers controls auth, CSRF, confirmation, audit, and provider target state. |

## Current Architecture State

- `missy/api/server.py` remains the HTTP router and embedded console page
  renderer, while session lifecycle, audit querying, diagnostics, controls, and
  shared Web templates are separated into focused modules.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
  `/sessions`, `/diagnostics`, `/controls`, and `/audit` with same-origin
  credentials and the embedded CSRF token.
- Operator controls are default-deny for unknown controls and invalid targets,
  require exact confirmation for mutations, and audit both denied and allowed
  attempts through the existing event bus.
- Diagnostics remain read-only view models built from injected server
  dependencies plus the active policy engine config reference.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
  touched.

## Tests

- `python3 -m ruff check .`: passed.
- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m pytest tests/api/test_server.py -q`: 83 passed.
- `python3 -m pytest -q`: 20461 passed, 13 skipped in 393.49s.

## Remaining Work

1. Continue extracting Web TUI rendering and frontend assets out of
   `missy/api/server.py`, especially the main console HTML/JavaScript.
2. Expand safe controls beyond provider default switching to tools, scheduled
   jobs, channels, and experimental features with policy and confirmation
   gates.
3. Add run/session viewer with streaming output, tool calls, errors, model
   routing, fallback, costs, and resumable context.
4. Add deeper live probes where safe and useful, especially gateway/network
   reachability checks that do not bypass policy.
5. Add responsive visual regression coverage or Playwright smoke checks once a
   browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
