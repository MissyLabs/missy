# Continuation and state design: preserve lessons, not old failure modes

The rewrite was compared against both Missy `d152c54` and Odin development
sources. Odin is a reference for useful invariants, not an infallible template.
No Odin source, services, credentials or live data are modified by this work.

## Observed Python Missy risks motivating the design

- `missy/agent/runtime.py:2853-2856` checkpoints after a complete tool batch and
  suppresses checkpoint failures. A tool's external effect can land before the
  checkpoint, leaving no operation-level receipt after a crash.
- `runtime.py:1960-1973` permits operation without durable state. The Rust
  effect-capable path requires durable admission, intent and settlement.
- The public run path's process-local session lock at `runtime.py:1206-1228`
  is not enough for multiple workers or the checkpoint-resume path around
  `runtime.py:6502`. Rust session ownership is checked transactionally in SQLite.
- Checkpoint claims in `missy/agent/checkpoint.py:265,320,340` do not provide
  the same expired-owner fencing guarantees as an unexpired lease and monotonic
  fence checked on every state write.
- Global learnings written around `missy/memory/sqlite_store.py:824-874` can be
  injected into unrelated contexts around `runtime.py:5035`. Rust requires a
  full scope for every memory read, write, expansion and deletion.
- Resetting loop/deadline/guard counters when loading only old messages is not
  continuation. Rust checkpoints preserve consumed controls, not merely prose.

These are source observations, not a claim that the running Python deployment
has been tested or that each line independently proves an exploitable incident.

## Useful Odin invariants

- `src/turn_state/durability.py:9-28` records the intended write invariant:
  durable proposed tool intents, durable RUNNING transition, per-operation
  settlement, batch checkpoint, guard flags and recovery deadline.
- `src/discord/turn_resume.py:308-323,407-432` refuses model generation when
  effectful operations remain PREPARED, RUNNING or UNKNOWN. This matters because
  an instruction saying "do not retry" cannot stop a model issuing the same
  side effect under a new ID.
- The turn snapshot codec and its round-trip tests show why hidden control state
  must be classified as persisted versus reconstructed. Restoring a transcript
  while accidentally resetting a one-shot retry flag silently changes behavior.

## Odin limitations deliberately not copied

- Its store boot sweep assumes one process and suspends all ACTIVE rows
  (`src/turn_state/store.py:1036-1045`). A second Rust worker opening the database
  must not interrupt a healthy first worker.
- Legacy durability-disabled operation (`durability.py:98-129`) is useful for
  compatibility but is not the Rust mutation safety default.
- A personal-to-global memory fallback when a principal is missing is rejected;
  missing authentication must fail, not broaden visibility.

## Durable state machine

Admission binds a scope and request ID to a payload digest. A session lease
serializes work across processes. Every checkpoint and tool transition verifies
the owner, monotonic fence, live expiry and active turn status within the write
transaction. Cancellation revokes further publication. Expired-owner takeover
marks unsettled operations unknown, never "failed so retry it".

Tool results are reusable only when their identity and full arguments match the
journal. Unsettled operations halt continuation before calling a provider. A
manual decision is required to reconcile external reality. This is at-most-once
automatic dispatch under uncertainty, not a fictional exactly-once distributed
transaction across arbitrary external systems.

Continuation snapshots retain the original request, system/configuration hash,
tool lineage, transcript, consumed steps/tokens/cost and continuation controls.
Historical requests and memories are data, not an implicit backlog. Compaction
must preserve complete call/result groups and current effect receipts or stop
when the safe context cannot fit. Silently forgetting a successful mutation is
not a memory optimization.

## Required adversarial acceptance cases

1. Crash after an external effect but before result persistence: resume performs
   zero new model calls and zero effect replay.
2. PREPARED intent at crash: conservative halt, not a guessed safe retry.
3. Expired owner cannot renew or checkpoint, even before another owner claims.
4. Opening worker B does not suspend worker A.
5. Cancellation fences delayed writes and waits for owned child cleanup.
6. Repeated continuations preserve every consumed counter and immutable hash.
7. Cross-scope canaries never appear in history, memory, direct ID fetches,
   summaries, cancellation or browser output.
8. Missing provider usage or uncertain failed attempts never become known-zero
   spend. Post-response limits must not be advertised as hard prepaid ceilings.
