# Durable Rust runtime

`Runtime::new(Arc<Store>, Arc<dyn Provider>, Arc<dyn ToolExecutor>, String,
RuntimeConfig) -> anyhow::Result<Runtime>` validates the frozen tool schemas and
binds checkpoints to the system prompt, provider configuration, tool catalog,
executor policy fingerprint and runtime configuration. Production providers must
override `Provider::configuration_fingerprint` with their actual nonsecret
configuration. Executors must include execution and approval policy in
`ToolExecutor::configuration_fingerprint`, not approval decision state.

Public APIs:

- `run(scope: Scope, prompt: String, request_id: String)` asynchronously returns
  `RunResult` after its owned execution task settles. Duplicate request identity
  returns the existing turn, never an implicit replay or resume. Changing the
  prompt under an existing request ID is rejected.
- `start` takes the same arguments and returns the actual durable turn identity
  immediately. Work survives dropping the caller's future and remains subject
  to its original deadline. `status(&Scope, &str)` synchronously returns `Turn`.
- `resume(scope: Scope, turn_id: impl AsRef<str>)` is explicit only. It reloads the
  checkpoint after acquiring a new fenced session lease, refuses any unresolved
  tool, and restores cumulative budgets and the original absolute deadline.
- `cancel(&Scope, &str)` synchronously persists a cancellation request. The
  owning worker observes it across processes. This method requests cancellation;
  `TurnStatus::Cancelled` means execution cleanup and durable settlement finished.
- `store()` exposes the shared store reference.

Scope is trusted server context, never inferred from model arguments. History is
scoped, serialized inert data with provenance, never a new system message. Memory
is exposed only through the scoped executor tools. No automatic global recall.

Each whole batch is schema/ID validated, persisted as exact waiting calls, and
preflighted before the first journal intent. `ApprovalRequired` is known
non-execution: the turn pauses as resumable Interrupted without an UNKNOWN tool.
Explicit resume rechecks the exact original batch, with original usage counters
and deadline, without another provider request. Only a typed waiting checkpoint
whose calls exactly match the last assistant message and have no journal records
can resume without settled journals. Ordinary missing journals still fail closed.
A preflight denial or other error clears the waiting marker, refuses the batch,
and creates no tool journal. Cancellation while waiting dispatches no effects.

After all preflights pass, waiting is durably cleared before journal preparation.
The store has no atomic checkpoint-and-batch-intent API: a crash in this gap is a
safe refusal requiring reconciliation, not automatic regeneration or dispatch.
Each actual dispatch is fenced and RUNNING is durable first. Every execution
error, timeout, or interrupted effect is conservatively UNKNOWN and stops all
generation. There is no runtime side-effect retry. Successful journal results
can repair a crash between result persistence and the transcript checkpoint
without executing again.

The core `ToolDispatchRefused` error is the sole execution-error exception: a
trusted executor guarantees it refused before calling its effectful inner
executor. The current call is settled Failed with an inert `executed: false`
receipt and the turn halts without replay. Other calls already Prepared in that
batch remain conservatively UNKNOWN at lease finalization; no store API exists
to mark an undispatched Prepared intent definitely cancelled. Arbitrary errors
from the actual inner executor remain UNKNOWN.

The executor contract matters: an effectful implementation must cancel and
physically stop/join owned children when its future is dropped. BuiltinTools
implements that contract. The runtime signals cancellation, polls cooperative
cleanup for five seconds, and drops the future before final settlement. Arbitrary
third-party executors that detach effects cannot be made safe by a token alone.

Context compaction preserves whole tool blocks, the exact original request,
source-hash lineage, and code-owned effect receipts. If the newest block or
receipt set cannot fit, execution suspends rather than erasing evidence. The
extractive summary is deliberately not a fabricated model-written account.

Provider token-limit completions continue within bounded cumulative budgets;
exhaustion is Interrupted, never Completed. Partial output is persisted. A final
content marker prevents a crash during final bookkeeping from generating again.
Unknown provider billing (including an interrupted request) poisons cumulative
cost certainty and stops a cost-capped turn. A crash with an unresolved provider
completion marker additionally blocks regeneration even without a cost cap:
there is no durable exact assistant batch to safely dispatch or replace.
An explicit zero cost cap disables
cost enforcement; token, step, continuation and time limits still apply. These
are post-response consumed-usage limits, not a preauthorization guarantee against
one provider request's billing overshoot.

Operational audit events come from the transactional store for admission,
checkpoint, lease, intent, execution state, cancellation and finish. No live
configuration, state directories or services are used by the tests.
