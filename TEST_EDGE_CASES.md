# TEST_EDGE_CASES

- Timestamp: 2026-07-08

## Web TUI / Operator Console Edge Cases

- Empty API key configuration rejects API and browser login attempts.
- Browser root redirects to login when no valid session cookie exists.
- Wrong operator key does not issue a browser session cookie.
- Browser session cookies are HttpOnly and SameSite=Strict.
- Browser-authenticated unsafe API calls fail closed without a CSRF token.
- API-key-authenticated unsafe API calls remain compatible and do not require browser CSRF.
- Logout requires CSRF and revokes the in-memory session.
- HTML responses include CSP, frame denial, no-store caching, nosniff, and referrer policy headers.
- Dashboard API failures degrade visibly instead of breaking the first screen.

## Remaining Web TUI Edge Cases To Cover

- Expired browser sessions redirect cleanly and do not allow API calls.
- Login/logout/CSRF-denial audit events are emitted and redacted.
- Audit browser filters combine severity, actor/source, subsystem, action, result, and time ranges correctly.
- Safe operator controls are default-deny and policy-gated.
- Streaming session/run viewer handles partial output, tool-call errors, retry/fallback, and reconnects.
- Mobile dashboard does not clip provider/tool/session rows.

## Cross-Focus Edge Cases

- Diagnostics patterns should stay reusable for Discord, scheduler, provider routing, tool delegation, memory, gateway, policy, and network posture.
- Tool policy visibility, approval, and execution enforcement must stay separate from Web TUI presentation state.
