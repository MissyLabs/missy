# TEST_EDGE_CASES

- Timestamp: 2026-07-08 09:19:07 EDT

Current edge-case focus:

- Web TUI auth/session handling cannot be bypassed.
- Audit log browser filters correctly and redacts secrets before search/detail
  display.
- Audit pagination remains newest-first, bounded, and stable under both file and
  in-memory event sources.
- Destructive actions require explicit confirmation and policy.
- Dashboard handles empty/loading/error states.
- Session viewer handles streaming, tool calls, failures, and costs.
- Provider/tool/channel controls enforce server-side policy.
- Frontend remains responsive on desktop and mobile.
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant.
- Diagnostics views do not leak tokens, secrets, or unsafe paths.
- Future overhaul compatibility for Discord, tools, scheduling, and provider
  routing.
