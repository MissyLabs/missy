# Security Audit Notes

Last updated: 2026-04-27

## Session Notes

- No new runtime dependencies were added.
- No network calls were introduced.
- `AgentSubscription` strips hidden thinking/final tags from user-visible streaming output while preserving literal tags inside code spans.
- The A1 raw-stream callback is best-effort and catches callback failures; the A11 file writer is not implemented yet.
- Tool execution, policy checks, and permissions were not changed in this session.

## Follow-up Security Work

- A2 must move ad-hoc runtime tool filtering into an audited policy pipeline with source labels.
- A3 must preserve failed mutating tool fingerprints across unrelated tool successes.
- A12 must synthesize missing tool results with explicit `is_error` semantics.
