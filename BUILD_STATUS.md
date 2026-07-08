# Build Status

Last updated: 2026-07-08 09:59:08 EDT

## Current State

Primary focus remains the Web TUI and browser operator console overhaul. The
stdlib `ApiServer` still owns the local Web UI and JSON API, with API-key auth,
browser login sessions, CSRF checks for unsafe browser API calls, hardened
cookies, no-store responses, and server-side redaction.

This session deepened the operator diagnostics/doctor slice:

- Added gateway diagnostics for the policy-enforcing HTTP client, response size
  cap, and active policy binding.
- Added Discord readiness diagnostics from the active policy/config object:
  integration state, account token presence, routing posture, REST/Gateway
  network prerequisites, and Discord voice tool visibility.
- Added optional remediation hints to diagnostics checks and rendered those
  hints in the Web TUI Diagnostics panel.
- Exposed policy network scope checks for provider/tool allowlists and REST
  method/path rule coverage.
- Preserved the initialized config on `PolicyEngine` for read-only diagnostics
  without reloading config files or exposing secrets.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` remain served by `missy/api/server.py`. |
| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies; denials are audited. |
| Operator dashboard | improved | Runtime, providers, tools, sessions, security posture, audit trail, and diagnostics are shown. |
| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
| Audit browser UI | improved | Search, actor/source, severity, timestamp filters, pagination, and event details are available. |
| Diagnostics API | improved | `/api/v1/diagnostics` now reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
| Diagnostics UI | improved | Browser console now displays compact doctor rows plus first remediation hints. |
| Console security tests | expanded | API suite covers auth, CSRF, audit, diagnostics redaction, gateway posture, and Discord policy readiness. |

## Current Architecture State

- `missy/api/server.py` remains the HTTP router and embedded renderer, while
  session lifecycle, audit querying, and diagnostics view models are separated.
- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
  `/sessions`, `/diagnostics`, and `/audit` with same-origin credentials and
  the embedded CSRF token.
- Diagnostics are pure read-only view models built from injected server
  dependencies plus the active policy engine config reference.
- Diagnostic checks redact summaries and remediation text before returning data
  to the browser/API client.
- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
  touched.

## Tests

- `python3 -m ruff check .`: passed.
- `python3 -m ruff format --check .`: 729 files already formatted.
- `python3 -m pytest tests/api/test_server.py -q`: 78 passed.
- `python3 -m pytest -q`: 20456 passed, 13 skipped in 391.33s.

## Remaining Work

1. Continue extracting Web TUI rendering and frontend assets out of
   `missy/api/server.py` into clearer internal modules.
2. Add safe policy-gated controls for providers, tools, scheduled jobs,
   channels, and experimental features.
3. Add run/session viewer with streaming output, tool calls, errors, model
   routing, fallback, costs, and resumable context.
4. Add deeper live probes where safe and useful, especially gateway/network
   reachability checks that do not bypass policy.
5. Add responsive visual regression coverage or Playwright smoke checks once a
   browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
