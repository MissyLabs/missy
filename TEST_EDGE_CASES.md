# TEST_EDGE_CASES

- Timestamp: 2026-07-08 10:38:28 EDT

Current edge-case focus:

- Web TUI auth/session handling cannot be bypassed.
- CSRF tokens are server-side escaped before insertion into the console shell.
- Browser actions send the CSRF token and unsafe browser API calls require it.
- Audit log browser filters correctly and redacts secrets.
- Destructive or privileged actions require explicit confirmation and policy.
- Dashboard handles empty/loading/error states.
- Session viewer still needs streaming, tool call, failure, and cost coverage.
- Provider/tool/channel controls enforce server-side policy.
- Frontend remains responsive on desktop and mobile.
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant.
- Diagnostics views do not leak tokens, secrets, or unsafe paths.
- Future overhaul compatibility remains needed for Discord, tools, scheduling,
  and provider routing.
