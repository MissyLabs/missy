# TEST_RESULTS

- Timestamp: 2026-07-08
- Branch: overhaul/web-tui-20260708-122250
- Context: Web TUI/operator console browser entrypoint

## Focused Verification

```text
python3 -m pytest tests/api/test_server.py -q
71 passed in 8.20s
```

Coverage included:

- API-key and bearer-token authentication still work.
- Unauthenticated browser root redirects to `/login`.
- Login page returns hardened browser security headers.
- Invalid login does not set a session cookie.
- Valid login sets an HttpOnly, SameSite=Strict cookie.
- Authenticated console renders with an embedded CSRF token.
- Cookie-authenticated API reads work.
- Unsafe cookie-authenticated API requests require `X-CSRF-Token`.
- Logout requires CSRF and revokes the browser session.

## Full Verification

```text
python3 -m pytest -q
20449 passed, 13 skipped in 364.27s (0:06:04)
```

```text
python3 -m ruff check .
All checks passed!
```

```text
python3 -m ruff format --check .
726 files already formatted
```
