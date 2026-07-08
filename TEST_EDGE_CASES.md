# TEST_EDGE_CASES

- Timestamp: 2026-07-08

## Tool Intelligence Edge Cases

- Frequent-request detection avoids noisy false positives.
- Generated tool candidates remain disabled until approved.
- Generated tool permissions are least-privilege.
- Tool metadata includes provenance, schema, tests, versions, and lifecycle state.
- Benchmark runs compare provider labels without bypassing registry or policy controls.
- Provider-specific enablement respects benchmark thresholds.
- Unsafe or flaky tools are flagged or disabled per provider.
- Tool schema incompatibilities are detected and reported.
- Benchmark failures do not enable tools accidentally.
- Repeated failing tool calls with identical arguments surface `lastToolError`.

## Discord Edge Cases

- Gateway reconnect and shutdown.
- Heartbeat sent/ACK visibility and stale ACK diagnostics.
- Invalid-session handling for resumable and non-resumable sessions.
- RESUME attempts and RESUMED dispatch audit.
- Slash command registration success/failure diagnostics.
- Discord REST retry, rate-limit, and failure handling.
- DM pairing and unknown sender denial.
- Guild/channel allowlist denial.
- Require-mention filtering.
- Bot-authored message and loop prevention.
- Attachment/media safety.
- Policy-gated tool execution from Discord.

## Cross-Focus Edge Cases

- Diagnostics patterns stay reusable for provider routing, scheduler execution, and tool delegation.
- Tool policy visibility, approval, and execution enforcement stay separate.
