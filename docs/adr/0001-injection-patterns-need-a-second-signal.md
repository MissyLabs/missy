# ADR-0001: Prompt-injection patterns need a second signal

**Status:** Accepted, 2026-10-18 (`8906081`)

**Context.** `InputSanitizer` matches do more than log: at several sites
(replayed history, recalled memory, Discord shared context, scheduled jobs,
tool output, attachments) a match withholds the content. Many patterns matched
ordinary language, such as "act as a", "start a new conversation", a `system:`
log label, or "display the install instructions". Missy silently lost names,
facts, and log lines users had given it. On the labeled corpus in the
validation harness, 44 of 48 benign items were withheld.

**Decision.** A pattern keeps its attack shape but must also require a second
signal: an instruction aimed at the assistant (`you`, `ignore`, `override`, a
request for its own prompt), a no-rules role ("an AI with no rules"), an
evasion target (base64, ROT13), or a role-delimiter position (a dangling
`system:` at the end of the text). Bare everyday phrases are never patterns.
Detection counts in tests are lower bounds.

**Consequences.** Corpus result: 1/48 benign flagged, 24/24 attacks detected.
The one remaining false positive ("you must always respond in …") is kept
deliberately, because it cannot be distinguished lexically from the same
instruction injected through tool output. New patterns must keep the harness
gate (`automation/injection_precision.py`: ≤5% benign blocks, ≥95% detection)
passing.
