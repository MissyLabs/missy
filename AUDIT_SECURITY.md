# AUDIT_SECURITY

- Timestamp: 2026-07-08
- Branch: overhaul/web-tui-20260708-122250

## Expected Security Posture

- Web TUI access is denied unless the configured API key is provided through the login form.
- Existing JSON API clients still require `X-API-Key`, bearer token, or a valid browser session.
- Browser sessions use unguessable tokens stored server-side and delivered in HttpOnly, SameSite=Strict cookies.
- Unsafe browser-authenticated API calls require a matching `X-CSRF-Token`.
- HTML responses use restrictive security headers: CSP, frame denial, no-store caching, nosniff, and no-referrer.
- The API server still binds to loopback by default and warns on non-loopback binding.
- Secrets in runtime responses remain censored by the existing chat path.

## Web TUI Notes

- The current Web TUI has no privileged enable/disable controls yet; it is read-mostly except for session creation through existing APIs and logout.
- API-key-authenticated clients are intentionally not forced through browser CSRF because they are not ambient browser credentials.
- Browser session state is process-local and expires by idle timeout; restarting the server clears browser sessions.
- Inline CSS/JS is currently allowed by CSP to keep the stdlib-only first console coherent; a later asset split should remove `unsafe-inline`.

## Follow-Up Security Work

- Emit structured audit events for login success, login failure, logout, CSRF denial, and privileged operator actions.
- Add policy-gated controls with explicit confirmation for providers, tools, scheduled jobs, channels, and experiments.
- Add audit log redaction tests for browser-rendered event details.
- Add XSS-focused tests around tool/provider/session names rendered into the dashboard.
- Consider secure-cookie enforcement when serving behind HTTPS or a local TLS terminator.
