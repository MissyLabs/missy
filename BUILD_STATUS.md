# Build Status

Last updated: 2026-07-08

## Current State

Primary focus is now the Web TUI and browser operator console overhaul. The repository has a first secure local browser console layered onto the existing stdlib REST API server without replacing the API-key JSON contract.

The current console is intentionally local-first and coherent:

- `/login` renders a browser login form backed by the configured API key.
- Authenticated browser sessions use an HttpOnly, SameSite=Strict cookie and an in-memory session store with expiry.
- `/` renders a polished operator dashboard that fetches runtime status, providers, tools, recent sessions, and security posture.
- Browser-authenticated unsafe API requests require `X-CSRF-Token`; API-key requests continue to work without browser CSRF.
- HTML responses include CSP, frame denial, no-store caching, nosniff, and referrer policy headers.

## Completed Work

| Area | Status | Notes |
|---|---|---|
| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
| Browser operator entrypoint | implemented | `/login`, `/logout`, and `/` are served by `missy/api/server.py`. |
| Web session handling | implemented | In-memory session store, cookie auth, expiry, and logout revocation. |
| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies. |
| Operator dashboard | implemented | Dense responsive dashboard for runtime, providers, tools, sessions, and security posture. |
| Console security tests | implemented | API suite covers redirects, login failure, hardened cookies, CSRF, API reads, and logout. |
| Prior tool-intelligence work | retained | Request tracking, mutation fingerprinting, schema adapter wiring, and benchmark CLI remain present. |
| Prior Discord diagnostics work | retained | Gateway lifecycle diagnostics, voice/image safety, and CLI diagnostics remain present. |

## Current Architecture State

- The Web TUI currently lives inside the existing `ApiServer` and uses the same injected runtime, memory, provider registry, and tool registry dependencies as the JSON API.
- The dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`, and `/sessions` with same-origin credentials and the embedded CSRF token.
- The API remains dependency-light and stdlib-only; no frontend build pipeline has been introduced.
- `LOOP_INSTRUCTIONS.md` is modified in the working tree from outside this session and was not touched.

## Tests

- `python3 -m pytest tests/api/test_server.py -q`: 71 passed.
- `python3 -m pytest -q`: 20449 passed, 13 skipped in 364.27s.
- `python3 -m ruff check .`: passed.
- `python3 -m ruff format --check .`: 726 files already formatted.

## Remaining Work

1. Split the growing Web TUI rendering/session helpers out of `missy/api/server.py` into clearer internal modules while preserving the current external behavior.
2. Add audit log browser APIs and UI filters for severity, actor/source, subsystem, action, result, and timestamps.
3. Add diagnostics/doctor panels for Discord, providers, scheduler, tools, memory, gateway, policy, and network posture.
4. Add safe policy-gated controls for providers, tools, scheduled jobs, channels, and experimental features.
5. Add run/session viewer with streaming output, tool calls, errors, model routing, fallback, costs, and resumable context.
6. Add structured audit events for browser login success/failure, logout, CSRF denial, and privileged UI actions.
7. Add responsive visual regression coverage or Playwright smoke checks once a browser test dependency is available.

## Blockers

- No code blocker remains for the current Web TUI slice.
