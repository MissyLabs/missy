# TEST_EDGE_CASES

- Timestamp: 2026-07-08 10:19:19 EDT

Current edge-case focus:

- Web TUI auth/session handling cannot be bypassed
- unsafe browser API calls require CSRF, including operator controls
- operator controls require exact confirmation and known safe targets
- provider controls deny unavailable or unknown providers and audit denials
- audit log browser filters correctly and redacts secrets
- destructive or privileged actions require explicit confirmation and policy
- dashboard handles empty/loading/error states
- session viewer handles streaming, tool calls, failures, and costs
- provider/tool/channel controls enforce server-side policy
- frontend remains responsive on desktop and mobile
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant
- diagnostics views do not leak tokens, secrets, or unsafe paths
- future overhaul compatibility for Discord, tools, scheduling, and provider routing
