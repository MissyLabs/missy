# AUDIT_SECURITY

- Timestamp: 2026-07-08 08:59:24 EDT

## Security Posture

- Web TUI remains local-first and authenticated by the configured API key.
- Browser sessions use HttpOnly, SameSite=Strict cookies and in-memory expiry.
- Unsafe browser-authenticated API requests require `X-CSRF-Token`.
- HTML responses keep CSP, frame denial, no-store caching, nosniff, and no-referrer headers.
- Audit browser API requires the same API-key/bearer/browser-session authentication as the rest of `/api/v1/*`.
- Audit API responses are recursively redacted server-side before leaving the process.
- Dashboard rendering escapes provider/tool/session/audit strings before insertion into HTML.

## New Audit Events

- `web.login` with `allow` or `deny`.
- `web.logout` with `allow` or `deny`.
- `web.csrf` with `deny` for missing/invalid browser CSRF on unsafe API calls.

Each event includes operator-safe metadata: `actor`, `source`, `subsystem`, `action`, `severity`, and `remote_addr`.

## Remaining Security Work

- Split Web TUI auth/session/audit helpers out of `ApiServer` for easier review.
- Add audit event detail pages with explicit redaction tests for nested lists/dicts and timestamp filters.
- Add policy-gated operator controls with confirmation and structured allow/deny audit trails.
- Add diagnostics panels that expose posture without leaking secrets, tokens, raw paths, or unredacted network targets.
- Add browser-level smoke tests for CSP, XSS-resistant rendering, keyboard navigation, and responsive layout.
