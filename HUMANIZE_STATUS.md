# Humanize Status

Last updated: 2026-04-27

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
| --- | --- | --- | --- |
| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
| A2 | Layered tool policy pipeline | not_started | Next Tier 1 target after A1 hardening. |
| A3 | Mutation fingerprinting + sticky lastToolError | not_started | Needed before apology calibration can avoid duplicate apologies. |
| A4 | Compaction retry coordination | not_started | A1 tracks retry state locally; runtime manager work remains. |
| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
| A7 | Block-reply chunking with flush points | not_started | A1 has block buffering primitives; channel chunker remains. |
| A8 | Per-channel identity cascade | not_started | Persona config extension remains. |
| A9 | Before/after hook system | not_started | HookRunner module remains. |
| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
| A11 | Raw-stream JSONL diagnostics | not_started | A1 exposes `raw_stream_callback`; observability module remains. |
| A12 | Transcript dual-repair | not_started | Memory write/read repair remains. |
| A13 | Context-window guard | not_started | Guard module and runtime preflight remain. |

## Humanistic Behavior Status

| ID | Behavior | Status | Notes |
| --- | --- | --- | --- |
| H_A | Variable response timing and typing pauses | not_started | Depends on A7 channel block flushing. |
| H_B | Tone modulation by mood and time of day | not_started | Existing `BehaviorLayer` has basic tone heuristics; new tone module remains. |
| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
| H_D | Proactive check-ins/follow-ups | not_started | Existing `ProactiveManager` will be extended later. |
| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
| H_F | Idle sleeptime thoughts | not_started | Existing `SleeptimeWorker` will be extended later. |
| H_G | Apology/gratitude/hedging | not_started | Existing `BehaviorLayer` will be extended after A3. |
| H_H | Humor and callbacks | not_started | SharedMomentStore remains. |
| H_I | Mood state with decay | not_started | First humanize implementation target in sessions 8-9. |

## Iteration Notes

- Initialized required loop tracking documents.
- Added `missy/agent/subscription.py`.
- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
- Added `tests/agent/test_subscription.py`.
- Expanded `tests/agent/test_runtime_streaming.py`.
- Verified with full `pytest -q`, `ruff check .`, and `ruff format --check .`.

## Next Steps

1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription`, not only the simple `run_stream()` path.
2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
3. Start A2 with `missy/policy/tool_policy_pipeline.py`, then replace runtime capability-mode filtering with the audited pipeline.
4. Keep `OPENCLAW_PATTERNS.md`, `HUMANIZE_STATUS.md`, `BUILD_STATUS.md`, `TEST_RESULTS.md`, and `LAST_SESSION_SUMMARY.md` current before each commit.
