# Build Status

Last updated: 2026-07-08 10:38:28 EDT

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The
stdlib `ApiServer` still owns HTTP routing, authentication, CSRF enforcement,
rate limiting, response headers, and JSON API dispatch. Browser session state,
audit querying, diagnostics, safe controls, and Web console rendering are now
split into focused modules.

This session completed the main console renderer extraction:

- Moved the authenticated operator console HTML shell from `missy/api/server.py`
  into `missy/api/web_console.py`.
- Added `render_console()` and `console_script()` so the browser UI shell, CSS,
  and JavaScript live behind a dedicated renderer boundary.
- Kept the existing authenticated dashboard behavior unchanged: runtime status,
  providers, tools, sessions, diagnostics, controls, security posture, and audit
  browsing still load through authenticated `/api/v1/*` calls with CSRF headers.
- Added direct renderer tests for CSRF token escaping, required Web TUI hooks,
  client-side audit detail rendering, and confirmed control POST wiring.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key`, bearer token auth, and browser sessions still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` remain served by `missy/api/server.py`. |
| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
| CSRF protection | implemented | Unsafe browser API calls and logout require the per-session CSRF token; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
| Diagnostics API | improved | `/api/v1/diagnostics` reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
| Safe controls API | started | Provider default switching is confirmed, validated, CSRF-protected for browser sessions, and audited. |
| Renderer extraction | improved | Login, message, main console shell, CSS, and JavaScript now live in `missy/api/web_console.py`. |
| Console tests | expanded | API suite covers auth, CSRF, controls, audit behavior, and renderer escaping/hooks. |

## Current Architecture State

- `missy/api/server.py` is now closer to a transport/router layer for the Web
  TUI and API. It delegates Web rendering to `missy/api/web_console.py`.
- `missy/api/web_console.py` owns the embedded browser console shell and script
  while preserving server-side escaping for the CSRF token.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
  `/sessions`, `/diagnostics`, `/controls`, and `/audit` with same-origin
  credentials and the embedded CSRF token.
- Operator controls remain default-deny for unknown controls and invalid
  targets, require exact confirmation for mutations, and audit both denied and
  allowed attempts through the existing event bus.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
  touched.

## Tests

- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m ruff check .`: passed.
- `python3 -m pytest tests/api/test_server.py -q`: 85 passed.
- `python3 -m pytest -q`: 20463 passed, 13 skipped in 381.13s.

## Remaining Work

1. Expand safe controls beyond provider default switching to tools, scheduled
   jobs, channels, and experimental features with policy and confirmation gates.
2. Add a real session/run viewer with streaming output, tool calls, errors,
   costs, model routing, fallback, and resumable context.
3. Split the embedded console JavaScript/CSS further if the stdlib server gains
   static asset support with compatible CSP and cache behavior.
4. Add deeper live probes where safe and useful, especially gateway/network
   reachability checks that do not bypass policy.
5. Add responsive visual regression coverage or Playwright smoke checks once a
   browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
