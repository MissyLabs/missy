# TEST_EDGE_CASES

- Timestamp: 2026-07-08 09:59:08 EDT

Current edge-case focus:

- Web TUI auth/session handling cannot be bypassed.
- Audit log browser filters correctly and redacts secrets.
- Diagnostics views do not leak API keys, Discord tokens, vault references, or
  unsafe paths.
- Diagnostics remediation hints are redacted and only expose operator-safe
  actions.
- Gateway diagnostics must report policy binding without making outbound
  network calls.
- Discord diagnostics must distinguish token presence from token value and must
  identify missing REST/Gateway allowlist prerequisites.
- Destructive actions require explicit confirmation and policy.
- Dashboard handles empty/loading/error states.
- Session viewer handles streaming, tool calls, failures, and costs.
- Provider/tool/channel controls enforce server-side policy.
- Frontend remains responsive on desktop and mobile.
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant.
- Future overhaul compatibility for Discord, tools, scheduling, and provider
  routing remains required.
