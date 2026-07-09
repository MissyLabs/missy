# TEST_EDGE_CASES

- Timestamp: 2026-07-08 22:14:51

Current edge-case focus:
- Web TUI auth/session handling cannot be bypassed
- audit log browser filters correctly and redacts secrets
- destructive actions require explicit confirmation and policy
- dashboard handles empty/loading/error states
- session viewer handles streaming, tool calls, failures, and costs
- provider/tool/channel controls enforce server-side policy
- frontend remains responsive on desktop and mobile
- accessibility basics are enforced: labels, focus states, keyboard navigation, and contrast
- control-plane views cover providers, sessions, tools, scheduler, memory, Discord, vision, policy, logs, and config
- live updates do not cause jarring layout shifts or broken interaction state
- UI APIs reject CSRF/XSS/path traversal style attacks where relevant
- diagnostics views do not leak tokens, secrets, or unsafe paths
- future overhaul compatibility for Discord, tools, scheduling, and provider routing
