# ADR-0003: One shared guard-retry budget per turn

**Status:** Accepted, 2026-10-18 (`dffa1c2`, `b1bc42a`)

**Context.** The tool loop runs many independent regex response guards
(fabrication, promise-without-action, hedging, web/desktop/video completeness,
and others). Each had its own retry cap and several could replace the final
answer with canned text. Together they allowed roughly 30 retries (~44 provider
calls) per turn with no time limit, and false positives replaced correct
answers ("Running tally: 15", "I'll explain how DNS works") with a refusal that
was then saved into session history.

**Decision.**
- Every guard retry is a text-only round that does not return, so the loop
  counts those rounds and stops at `AgentConfig.max_guard_retries` (default
  4), returning the model's latest answer with a "no tool ran" notice.
- `AgentConfig.max_turn_seconds` (default 600) bounds a turn's wall-clock time.
- Completion and promise guards apply to action requests; claims about
  user-supplied content or the model's own reasoning are not fabrication.
- On exhaustion, guards keep the answer and add a notice instead of canned text.
- Delegation is capped across the whole tree (`MAX_TOTAL_SUB_RUNS`).

**Consequences.** Individual guard caps still apply within the shared budget. A
new guard needs no extra wiring for the budget, but must not overwrite the
answer with canned text and should emit `agent.response.*` audit events for its
retries.
