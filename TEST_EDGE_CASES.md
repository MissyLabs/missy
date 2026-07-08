# TEST_EDGE_CASES

- Timestamp: 2026-07-08 10:57:27 EDT

Current edge-case focus:
- Web TUI auth/session handling cannot be bypassed
- unsafe browser API calls require CSRF
- audit log browser filters correctly and redacts secrets
- destructive or operational controls require explicit confirmation and policy
- provider default switching rejects unavailable or unknown providers
- scheduler controls reject missing scheduler, invalid targets, missing
  confirmation, unknown jobs, already-paused jobs, and already-enabled jobs
- scheduler pause/resume actions emit allow/deny audit events with subsystem,
  action, target, reason, and redacted details
- dashboard handles empty/loading/error states
- session viewer handles streaming, tool calls, failures, and costs
- provider/tool/channel controls enforce server-side policy
- frontend remains responsive on desktop and mobile
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant
- diagnostics views do not leak tokens, secrets, or unsafe paths
- future overhaul compatibility for Discord, tools, scheduling, and provider routing
