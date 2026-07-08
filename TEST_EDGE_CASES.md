# TEST_EDGE_CASES

- Timestamp: 2026-07-08 08:59:24 EDT

Current edge-case focus:

- Web TUI auth/session handling cannot be bypassed.
- Browser-authenticated unsafe API calls require valid CSRF and denials are audited.
- Audit log browser filters by event type, category, result, session/task, severity, actor/source, subsystem, action, timestamps, and free text.
- Audit API responses recursively redact secret-bearing keys and detected token-like strings.
- Dashboard rendering escapes JSON-derived values before `innerHTML` insertion.
- Dashboard handles empty/loading/error states.
- Session viewer still needs streaming, tool-call, failure, routing, fallback, and cost edge coverage.
- Future provider/tool/channel controls must enforce server-side policy and explicit confirmations.
- Diagnostics views must not leak tokens, secrets, unsafe paths, or denied host details beyond redacted operator-safe context.
- Vector memory encoding must remain deterministic across Python hash seeds.
