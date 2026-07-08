# Build Status

Last updated: 2026-07-08 10:57:27 EDT

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The
stdlib `ApiServer` owns transport, authentication, browser sessions, CSRF,
rate limiting, security headers, and JSON API dispatch. Web rendering,
diagnostics, audit browsing, session state, and operator controls remain split
behind focused modules.

This session expanded safe operator controls from provider switching into
scheduler job control:

- Added scheduler pause/resume controls to `missy/api/operator_controls.py`.
- `/api/v1/controls` now lists provider controls plus scheduler job targets
  from the runtime-attached scheduler when present.
- `/api/v1/controls/scheduler.pause_job` and
  `/api/v1/controls/scheduler.resume_job` require exact confirmation strings,
  validate target IDs, enforce current job state, call `pause_job()` /
  `resume_job()`, and emit structured `web.control` audit events through the
  existing server path.
- Updated the browser console control renderer so controls are generic rather
  than provider-only, with labels, target names, schedule/provider metadata,
  and generic confirmation prompts.
- Added focused API tests for scheduler control listing, denied confirmation,
  allowed pause/resume mutations, audit allow/deny events, and frontend control
  data hooks.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key`, bearer token auth, and browser sessions still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` use cookie sessions and CSRF. |
| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
| CSRF protection | implemented | Unsafe browser API calls and logout require the per-session CSRF token; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, diagnostics, controls, security posture, and audit trail are shown. |
| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
| Diagnostics API | improved | `/api/v1/diagnostics` reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
| Safe controls API | improved | Provider default switching and scheduler pause/resume are confirmed, validated, CSRF-protected for browser sessions, and audited. |
| Renderer extraction | improved | Login, message, main console shell, CSS, and JavaScript live in `missy/api/web_console.py`. |
| Console tests | expanded | API suite covers auth, CSRF, controls, audit behavior, scheduler controls, and renderer hooks. |

## Current Architecture State

- `missy/api/server.py` remains the thin transport/router layer for the Web TUI
  and API, delegating console HTML/JS/CSS to `missy/api/web_console.py`.
- `missy/api/operator_controls.py` is now the policy-shaped control boundary
  for provider and scheduler mutations. Unknown controls, invalid targets,
  missing dependencies, missing confirmation, unavailable providers, unknown
  jobs, and wrong job state fail closed.
- Scheduler controls intentionally use the runtime-attached scheduler instead
  of creating a parallel scheduler service.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
  `/sessions`, `/diagnostics`, `/controls`, and `/audit` with same-origin
  credentials and the embedded CSRF token.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
  touched.

## Tests

- `python3 -m ruff format --check .`: 731 files already formatted.
- `python3 -m ruff check .`: passed.
- `python3 -m pytest tests/api/test_server.py -q`: 88 passed.
- `python3 -m pytest -q`: 20466 passed, 13 skipped in 387.83s.

## Remaining Work

1. Expand safe controls to tools, channels, and experimental features with
   policy and confirmation gates.
2. Add a real session/run viewer with streaming output, tool calls, errors,
   costs, model routing, fallback, and resumable context.
3. Add deeper live probes where safe and useful, especially gateway/network
   reachability checks that do not bypass policy.
4. Split embedded console JavaScript/CSS further if stdlib static asset support
   can preserve CSP and cache behavior.
5. Add responsive visual regression coverage or Playwright smoke checks once a
   browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
