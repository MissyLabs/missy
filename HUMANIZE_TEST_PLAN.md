# Humanize Test Plan

Last updated: 2026-04-27

## Strategy

- Test OpenClaw-pattern substrate with deterministic unit tests first, then add integration tests at runtime/channel boundaries.
- Test humanistic behavior as observable state or prompt changes. Avoid snapshotting broad prose.
- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.

## Current Coverage

- A1 subscription unit coverage: `tests/agent/test_subscription.py`
  - Delta streams and full-content resend reconciliation.
  - Divergent full-content fail-open append behavior.
  - Split think tag stripping across chunks.
  - Code-span awareness for literal tags.
  - Enforced final tag mode.
  - Reasoning stream mode.
  - Reply directive parsing.
  - Block flush at `text_end` and before tool execution.
  - Compaction retry state transitions.
- A1 runtime coverage: `tests/agent/test_runtime_streaming.py`
  - Existing streaming behavior still yields chunks.
  - Split think tags are stripped in `AgentRuntime.run_stream()`.
- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
  - `group:fs` expands to OpenClaw-style filesystem aliases.
  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
  - `alsoAllow` can restore matching tools after a restrictive layer.
  - Unknown plugin-only allowlists warn without hiding core tools.
  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
  - Capability-mode layers preserve existing Missy runtime modes.
- A2 runtime coverage: `tests/agent/test_runtime_streaming.py`
  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.

## Planned Coverage By Behavior

| ID | Assertions to add |
| --- | --- |
| H_A | Delay calculation respects length/complexity/mood/channel caps; quick/fast/asap bypasses long sleeps; channel typing indicator ordering is mocked. |
| H_B | Mood and time-of-day heuristics select the expected `ToneProfile`; runtime prompt includes tone fragment once per turn. |
| H_C | Personal fact extraction captures "my X is Y"; recall injects related top-k facts; 90-day decay deprioritizes stale facts; CLI show/forget/pin works. |
| H_D | Promised follow-up parser schedules implied time; TopicResume observes idle threshold; rate limit blocks second unsolicited message within 6 hours. |
| H_E | High-confidence technical correction emits disagreement prompt fragment; subjective preferences do not trigger pushback. |
| H_F | Sleeptime thoughts obey awake hours, 4-hour frequency, designated channel, and active-conversation suppression. |
| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
| H_H | SharedMomentStore persists moments; callback fires for related topic once per conversation and respects 7-day cooldown. |
| H_I | Mood vector decays by elapsed hours, persists to disk, nudges on interaction outcomes, and reset CLI restores defaults. |
