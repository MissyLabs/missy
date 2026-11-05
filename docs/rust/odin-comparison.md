# Independent focused review: continuation, sessions and memory

Reviewed 2026-11-05. This is the requested Odin comparison artifact, not a
claim of full rewrite parity or production readiness.

## Scope and evidence boundaries

Read the Odin development/deployment boundary runbook first. Odin comparison
was **read-only source inspection** under its development checkout's `src/`.
No live configuration, secrets, production databases or services were used;
no Odin tests were run. References below identify mechanisms and design
lessons, not copied implementation or private runtime material.

Missy review covered runtime/context, store/checkpoint/journal/memory,
retrieval, governance approvals/guarded execution, and provider accounting.
Tests used synthetic providers, inert counted effects, temporary state and
local fixture servers. CLI, scheduler and extension integration were outside
this review's ownership. Tests passed on a concurrently integrated working
tree, not an immutable release commit.

## Confirmed high-risk defect, repaired and retested

**Construction-time fingerprints were incorrectly treated as proof that a
shared executor had stayed unchanged.** Runtime captured provider/tool/config
identity once in `Runtime::new`, but did not re-read that identity when an
existing runtime resumed a paused turn or completed preflight. Reconstructing
a new runtime correctly rejected changed policy; mutating the existing
executor bypassed that comparison.

Two adversarial regression tests were added to
`crates/missy-runtime/src/tests/approvals.rs` and actually executed **before**
the repair:

- `in_place_policy_drift_blocks_paused_resume_before_effect`: paused with
  original call ID/arguments, approved the call, changed the shared executor's
  policy revision, then resumed the same runtime. Expected refusal failed.
- `in_place_policy_drift_during_preflight_blocks_batch_prepare`: changed the
  revision from inside a successful preflight. The run incorrectly completed
  instead of interrupting before effect dispatch.

Both tests failed before the fix and passed afterward. The narrow runtime
repair recomputes the **existing** configuration binding at launch/load,
before and after preflight, before each actual dispatch, before generation,
and after a returned completion. Observed drift is refused, not silently
rebound. A stale paused result is no longer advertised as resumable. The
checkpoint format, original binding formula and CLI/scheduler were unchanged.

This adopts a useful Odin lesson: mutable authority must be revalidated at
the point of use, not merely at construction. Odin resume derives tools from
current permissions (`src/discord/turn_resume.py`), while executor access
checks revalidate current target/output authority (`src/tools/executor.py`).
Missy already applied this lesson within `GuardedTools`; the repair extends
the generic runtime fence to other shared executors/providers.

**Limit:** this is drift detection, not an atomic transaction with arbitrary
external systems. An executor must still pin/revalidate its real target and
authorization at dispatch. A dishonest fingerprint, ABA revision, or mutation
after the last check is not made safe by repeated hashes. Keep executors
immutable where possible; use owned target handles or executor-level leases
when mutable authority is unavoidable.

## Mechanism comparison: observed facts versus judgments

### Checkpoints and successful-result rehydration

**Observed Odin source:** durability hooks require response/intents before
execution, RUNNING before effect, individual settlement, batch checkpoints and
preserved deadlines (`src/turn_state/durability.py`). Resume repairs unmatched
tool-result pairs from the ledger without executing them, and rejects
ambiguous duplicate durable identities (`src/discord/turn_resume.py`).

**Observed Missy source and tests:** `Runtime::hydrate` requires exact original
call IDs, names and arguments; settled success is loaded from its journal.
Missing result messages are rebuilt with their original IDs. Request hash,
original prompt, transcript, steps, usage and absolute deadline are checked.
`settled_effect_missing_checkpoint_reply_is_repaired_without_replay` passed:
one continuation provider call, zero tool executions. The mismatched-argument
case blocked both. A final-result checkpoint completed without a second
provider call. Approval pause/resume retained the original batch and did not
regenerate calls while awaiting approval.

**Judgment:** this is proper checkpoint continuation, not transcript-only
reconstruction. It intentionally sacrifices some recoverability when evidence
is incomplete. It is not exactly-once execution across external systems.

### Uncertain effects and retry prevention

**Observed Odin source:** unresolved effect-capable PREPARED/RUNNING/UNKNOWN
operations stop resumed generation, preventing a model from bypassing replay
protection with fresh call IDs. Effect-free observations have a separate
classification (`src/discord/turn_resume.py`).

**Observed Missy tests:** prepared/unknown effects blocked continuation and
fresh-ID execution; unjournaled checkpoint calls did not regenerate; a failed
effect was not retried. All tools are conservatively treated as effect-capable
in the store. Known pre-dispatch refusal is distinguished from UNKNOWN.

**Judgment:** Missy's conservative blanket treatment is safer than guessing
that a tool is read-only. It also means interrupted reads can require manual
handling. Do not loosen it merely to improve apparent continuation success.

### Sessions and cross-user/cross-session isolation

**Observed Odin source:** the session manager is channel-keyed, retains
speaker/participant attribution and summary provenance, and uses monotonic
mutation revisions to prevent stale compaction or auto-resume after the
session advances (`src/sessions/manager.py`, `src/discord/turn_resume.py`).
Search supports explicit user/channel filtering; user-scoped searches do not
fall through to an unfiltered semantic backend. These are shared-channel
conversation semantics, not a universal tenant-isolation contract.

**Observed Missy tests/source:** the complete `(tenant, user, channel, session)`
scope is mandatory and is bound into persistence lookup keys. Retrieval tests
changed every scope component and exercised isolation. Runtime dedup/history
and store request-binding tests passed. Session leases serialize workers;
opening another store does not sweep a healthy owner's state. Memory tools
derive scope from `ToolContext`, not model arguments.

**Judgment:** keep this stricter partition. Do not copy channel-only session
identity or globally broaden a lookup to make recall appear helpful. A user
changing session should not silently recall a previous session. Conversely,
this design does **not** currently promise automatic personal memory spanning
sessions. Shared channels or cross-session personal memory need explicit,
authenticated sharing scopes and separate tests, not a fallback query.

### Memory, retrieval and trust

**Observed Odin source:** personal and explicitly shared memory are separate
namespaces, but the memory handler contains compatibility behavior where a
missing principal on a personal save selects the global namespace. Merged
prompt memory includes global plus personal values
(`src/tools/handlers/state.py`, `src/tools/executor.py`). This is an inspected
code path, not proof that a production caller can omit authentication.

**Observed Missy source/tests:** missing scope fails validation. Memory and
document/graph operations use exact scope predicates, with no personal-to-
global fallback. Historical messages and recalled data are framed as inert
data, never promoted into system/developer instructions. Context compaction
keeps the original current request, complete call/result groups and effect
receipts or stops when evidence cannot fit. FTS/SQL/prompt-injection strings,
UTF-8 citations, corruption, replacement and graph bounds tests passed.

**Judgment:** preserve the no-fallback rule even when it returns no memory.
Trust framing is defense in depth, not proof that a model cannot obey malicious
source text; tool policy and authenticated scope must remain the actual fence.
Default retrieval is explicitly lexical BM25 plus hashed vectors, not a claim
of neural semantic recall. Better conversation recall is still a product
quality question beyond these safety tests.

### Accounting and fallback

**Observed Odin source:** usage rollups distinguish reported, estimated and
unknown values; missing actual cost is not automatically a known zero
(`src/usage/rollup.py`). No equivalent end-to-end hard spending cap was verified
in Odin during this review.

**Observed Missy tests/source:** uncertain failed provider attempts taint the
eventual fallback completion's billing. `Usage::add` cannot restore known
cost after an unknown attempt; checkpoints prohibit unknown-to-known rollback.
Pending-provider recovery marks billing unknown and blocks regeneration even
when the monetary cap is explicitly disabled. Tests passed for fallback/circuit
uncertainty, unknown pricing, missing/malformed usage and unknown-cost dispatch
refusal under an enabled cap.

**Judgment:** do not describe a post-response cap as prepaid enforcement.
`max_cost_microusd = 0` deliberately disables that monetary constraint; unknown
cost remains unknown, not free. This explicit opt-out is distinct from losing
usage and silently granting a fresh budget.

### Cancellation and approvals

**Observed Missy tests:** cancellation requested through another runtime was
observed; cancellation serialized before late success and could not be
overwritten; an already-running effect could still record its truthful receipt
without permitting further dispatch. Waiting approvals retained the original
deadline and consumed counters. Approval tokens were durable and single-use;
binding mismatches, expiration, denial, revocation and competing decisions were
rejected. Whole-batch preflight prevented an approved first effect when another
call still awaited approval. Consumption rollback/audit failure tests passed.

**Judgment:** a cancellation request is not evidence that an external operation
never happened. Retain the receipt and UNKNOWN distinction. Approval is an
additional exact binding, not permission to override the executor's policy.

## Executed verification

Before repair, the existing runtime/store/retrieval/governance suite passed.
The two added drift probes then both failed, demonstrating the gap.

After repair, this command completed successfully:

```text
cargo test -p missy-runtime -p missy-store -p missy-retrieval \
  -p missy-governance -p missy-providers -- --test-threads=4
```

101 tests passed: governance 11, provider fixtures 18, retrieval 17, runtime 28,
store unit 2, cancellation race 2, persistence 16, runtime recovery 7. Doc-test
targets also completed successfully (zero examples). `git diff --check` passed
for the changed runtime files.

This is evidence for the focused safety mechanisms above. It is not a claim
that all extensions, desktop behavior, scheduler integrations, real providers,
or production recovery have been independently accepted here. "Make Missy
good" requires this honesty as much as it requires successful tests.
