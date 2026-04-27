# OpenClaw Pattern Adoption

Last updated: 2026-04-27

Status values used here: `not_started`, `designed`, `implemented`, `wired`, `tested`, `hardened`.

## Checklist

| ID | Pattern | Status | Missy paths | Test coverage | Notes |
| --- | --- | --- | --- | --- | --- |
| A1 | Streaming subscription state machine | tested | `missy/agent/subscription.py:34`, `missy/agent/subscription.py:241`, `missy/agent/runtime.py:620` | `tests/agent/test_subscription.py:8`, `tests/agent/test_runtime_streaming.py:83` | Handles `message_start/update/end`, tool events, compaction events, monotonic delta/full-content reconciliation, split think/final tag stripping, code-span awareness, reply directives, reasoning modes, and block flush points. Runtime wiring currently covers simple streaming. |
| A2 | Layered tool policy pipeline | not_started | Planned: `missy/policy/tool_policy_pipeline.py`, `missy/tools/registry.py` | Planned: `tests/policy/test_tool_policy_pipeline.py`, `tests/tools/test_registry_policy_edges.py` | Will replace ad-hoc capability filtering and record source labels per filter step. |
| A3 | Mutation fingerprinting + sticky lastToolError | not_started | Planned: `missy/agent/mutation_tracking.py`, `missy/agent/runtime.py`, `missy/tools/registry.py` | Planned: `tests/agent/test_mutation_tracking.py` | Needed by H_G apology calibration. |
| A4 | Compaction retry coordination | not_started | Planned: `missy/agent/compaction.py`, `missy/agent/consolidation.py`, `missy/agent/runtime.py` | Planned: `tests/agent/test_compaction_retry.py` | A1 local compaction flags are present; manager-level retry future remains. |
| A5 | Auth profile cooldown + fallback | not_started | Planned: `missy/providers/auth_profiles.py`, `missy/providers/registry.py`, `missy/providers/rate_limiter.py` | Planned: `tests/providers/test_auth_profiles.py` | Must honor user-pinned profile without fallback. |
| A6 | Per-provider tool schema normalization | not_started | Planned: `missy/providers/schema_adapter.py` | Planned: `tests/providers/test_schema_adapter.py` | Gemini scrubbing and Mistral ID rewrite remain. |
| A7 | Block-reply chunking with flush points | not_started | Planned: `missy/channels/block_chunker.py`, channel adapters, `missy/agent/runtime.py` | Planned: `tests/channels/test_block_chunker.py` | A1 has block buffers and tool-start flush; channel delivery remains. |
| A8 | Per-channel identity cascade | not_started | Planned: `missy/agent/persona.py`, config schema | Planned: `tests/agent/test_persona_identity_cascade.py` | Response prefix and ack reaction cascade remains. |
| A9 | Before/after hook system | not_started | Planned: `missy/agent/hooks.py` | Planned: `tests/agent/test_hooks.py` | Hook failure must be soft. |
| A10 | Sub-agent depth + child caps | not_started | Planned: `missy/agent/sub_agent.py`, session persistence, A2 filter | Planned: `tests/agent/test_sub_agent_depth_caps.py` | Depth-aware orchestration filtering remains. |
| A11 | Raw-stream JSONL diagnostics | not_started | Planned: `missy/observability/raw_stream.py`, `missy/agent/subscription.py` | Planned: `tests/observability/test_raw_stream.py` | A1 includes a callback seam for best-effort writes. |
| A12 | Transcript dual-repair | not_started | Planned: `missy/agent/transcript_repair.py`, `missy/memory/__init__.py` | Planned: `tests/agent/test_transcript_repair.py`, `tests/memory/test_resilient_store_deep.py` | Write-time and read-time repair remain. |
| A13 | Context-window guard | not_started | Planned: `missy/agent/context_guard.py`, `missy/config/settings.py`, `missy/agent/runtime.py` | Planned: `tests/agent/test_context_guard.py` | 16k block and 32k warning thresholds remain. |

## Humanize Integration Map

| Humanize feature | Supporting OpenClaw pattern | Current trace |
| --- | --- | --- |
| H_A timing pauses | A7 block replies, A1 stream state | A1 block buffer/flush primitives exist; A7 not implemented. |
| H_B tone modulation | A1 message-start prompt timing, A8 identity cascade | Not implemented. Tone must be injected before stream begins. |
| H_C personal memory | A2 tool policy, A12 transcript repair | Not implemented. Recall tools should be policy-gated. |
| H_D proactive follow-ups | A10 sub-agent caps, A9 hooks | Not implemented. |
| H_E disagreement | A11 raw stream diagnostics, A9 hooks | Not implemented. |
| H_F sleeptime thoughts | A10 sub-agent runner, A4 compaction awareness | Not implemented. |
| H_G apology/gratitude/hedging | A3 sticky mutation error, A1 stream state | Not implemented. |
| H_H humor/callbacks | A8 channel identity, A12 durable transcripts | Not implemented. |
| H_I mood state | A4 compaction snapshots, H_B tone | Not implemented. |

## A1 Implementation Notes

- `BlockState` stores `thinking`, `final`, and `inline_code` state at `missy/agent/subscription.py:48`.
- `strip_block_tags()` strips split tags and preserves literal tags inside code at `missy/agent/subscription.py:144`.
- `AgentSubscription` owns the event handlers and state fields at `missy/agent/subscription.py:241`.
- Full-content resend reconciliation lives in `_reconcile_update()` at `missy/agent/subscription.py:460`.
- `AgentRuntime.run_stream()` integration starts at `missy/agent/runtime.py:620`.
