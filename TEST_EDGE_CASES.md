# TEST_EDGE_CASES

- Timestamp: 2026-07-08 09:31:21

Current edge-case focus:
- Web TUI auth/session handling cannot be bypassed
- audit log browser filters correctly and redacts secrets
- destructive actions require explicit confirmation and policy
- dashboard handles empty/loading/error states
- session viewer handles streaming, tool calls, failures, and costs
- provider/tool/channel controls enforce server-side policy
- frontend remains responsive on desktop and mobile
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant
- diagnostics views do not leak tokens, secrets, or unsafe paths
- diagnostics views require auth and summarize elevated tool/provider/policy
  posture without exposing configured API keys
- future overhaul compatibility for Discord, tools, scheduling, and provider routing
