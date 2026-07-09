# OpenClaw Gap Analysis

Last updated: 2026-07-09 13:52 EDT

## Current Focus

Primary focus is **tool usage and tool intelligence overhaul**. OpenClaw-style
parity here means Missy should observe repeated work, propose structured tools
when safe, store them with provenance and lifecycle metadata, benchmark them
across providers, gate provider access based on evidence, and expose reviewable
operator diagnostics without bypassing policy.

## Tool Intelligence Capability Status

| Capability | Status | Notes |
|---|---|---|
| Frequent-request detection | in place | `RequestTracker` records completed turns and surfaces repeated normalized patterns. |
| Runtime request-tracker wiring | in place | `AgentRuntime._track_request()` records completed user turns with tool calls and provider metadata. |
| Safe tool-candidate generation | in place | `CandidateGenerator` is opt-in and creates proposed candidates only. Shell proposals remain blocked unless explicitly configured. |
| Candidate storage metadata | in place | `CandidateStore` stores schema, permissions, provenance, examples, owner, version, lifecycle state, notes, benchmark summaries, provider flags, and tags. |
| Candidate lifecycle enforcement | improved this session | Store-level transition matrix now rejects skipped gates and disabled-candidate resurrection, with denied audit events. |
| Benchmark harness | in place | Direct and LLM-mediated benchmark runners score correctness, latency, cost, reliability, safety, schema adherence, tool-call quality, and failure behavior. |
| Provider-aware tool gating | in place | `ToolProviderGate` combines benchmark summaries and operator overrides. Runtime gating is opt-in. |
| Provider fallback recommendation | partial | CLI can recommend the best enabled provider; runtime does not yet surface recommendations when a tool is gated off. |
| CLI diagnostics | in place | Candidate list/show/approve/enable/deny, request stats, benchmark run/results/compare, provider status/enable/disable/clear/recommend. |
| Web/API controls | not_started for candidates | Existing web controls cover other subsystems, but candidate lifecycle actions are CLI-only. |
| Runtime loading of enabled candidates | not_started | Enabled candidates are persisted metadata; a controlled loader still needs schema/provenance/policy gates. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | in place | Existing run streaming remains separate from this session. |
| A2 | Layered tool policy pipeline | hardened | Candidate lifecycle enforcement now complements execution policy rather than replacing it. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
| A11 | Raw diagnostics/audit trail | improved | Invalid lifecycle transitions now emit explicit denied audit events. |
| A12 | Transcript/tool-call repair | partial | Provider schema adapters and benchmark scoring exist; candidate schema compatibility reporting remains future work. |

## Recommended Next Slice

1. Add benchmark-to-candidate reconciliation so benchmark store results update
   candidate benchmark summaries and provider-enabled flags.
2. Add candidate lifecycle commands/API for `experimental`, `deprecated`, and
   benchmark import/review, not just approve/enable/deny.
3. Add controlled runtime loading for enabled candidates with schema,
   permissions, provenance, tests, and rollback checks.
4. Surface provider fallback recommendations in runtime diagnostics when
   provider gating removes tools from a turn.
