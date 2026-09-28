# Opt-in durable RepoEval coordinator

`missy.repoeval.coordinator.DurableFoundryService` is a separate SQLite
reference service. It uses `FoundryService._validated_plan()` and the injected
`PlanningAuthority` for exact reviewed plans and rechecks. It does **not** turn
on `FoundryService`'s single-process demo dispatcher. Instantiate it with an
explicitly initialized on-disk `SQLiteOutbox`, a `DurableDispatcher` with a
trusted exact-job authorization callback and injected scheduler, and a job
factory. Both dispatcher and coordinator require `enabled=True`. Importing it
neither submits work nor opens a network connection.

```python
outbox = SQLiteOutbox.initialize(database_path)
dispatcher = DurableDispatcher(outbox, scheduler, authorize_exact_job, enabled=True)
service = DurableFoundryService.initialize(
    database_path, foundation, dispatcher, job_factory,
    verifier=independent_validator, enabled=True,
)
```

`job_factory(plan,parent_run_id,child_run_id,provider_index,repetition)` must
return a batch job with `ID=Name=foundry-{child_run_id}` and metadata binding
`foundry_run_id`, `foundry_parent_run_id`, `foundry_provider_index`, and
`foundry_repetition`. The job factory is **not** itself authority. The
dispatcher's injected `authorize(project, child_run_id, exact_job)` must
recheck the immutable snapshot, pinned worker image, project grant, quotas,
placement and scheduler namespace for these exact bytes. The example
`lambda *_: True` in fake-only tests is not an acceptable live authorization
callback.

`benchmark_plan()` commits a project-scoped plan. `benchmark_start()` commits
the project-scoped idempotency reservation and one independent outbox job for
each provider/repetition; it returns **reserved**, not submitted or verified.
If interrupted between parent and child reservations, repeating the same key
with fresh approval finishes missing reservations. All jobs remain inert until
a separate trusted operator calls `dispatch_pending(principal,run_id)`. That
call rechecks the stored plan before **every** external effect; an expired or
revoked plan blocks outstanding submissions. Outbox markers are committed
before scheduler calls; a lost acknowledgement does not trigger replay.

`reconcile_run()` reads exact scheduler observations through the injected
adapter. `run_status()` reads durable local state but does not initiate
scheduler traffic. `run_cancel()` first persists parent cancellation intent,
then cancels each reserved or exactly observed child and preserves uncertain
stop outcomes. Child reservation and parent cancellation use separate SQLite
transactions: concurrent reservation is rechecked against the tombstone and
cancelled, but a scheduler submission racing that handoff needs independent
exact scheduler reconciliation. A scheduler
`complete` observation means **collecting**, not success. `finalize()` may
mark the parent verified only after every exact child is collecting, a trusted
`verifier(project,run,plan,children,manifest)` independently confirms worker
and validator receipts, every declared artifact kind is present, each stored
object's bytes match its SHA-256 and size, and a trusted clearance callback
approves the bytes. `artifacts()` repeats byte and clearance verification on
every read; unverified outputs are unavailable. Neither callback should treat
worker self-reported booleans or arbitrary URIs as proof. No native validator
or provider adapter is installed here.

The reference has important limits: SQLite coordinates only its own plan/run
tables and the existing outbox; it is not an integrated PostgreSQL production
transaction, migration, retention policy, or scheduler incarnation attestor.
Parent reservation and child reservations occur in separate transactions,
with incomplete reservations surfaced and repairable under fresh authority.
Scheduler completion and trusted validator verification are separate external
observations, not an atomic scheduler/SQLite transaction; a production adapter
needs immutable completion receipts tied to an execution incarnation. The
API's `benchmark/start` returns HTTP 202 with `state=reserved` and no parent
job ID. The client tool verifies the project, reviewed plan, key-derived parent
run ID, and complete bounded reserved child identities before acknowledging
reservation, without claiming scheduler submission or completed execution. Do not
fabricate submission or a single job ID to appease the client. Durable
snapshot capture, multi-run comparison, and report drafting fail closed until
implemented. No real Nomad or provider calls have been authorized here.

Local fake-only regression:

```bash
python -m pytest -q tests/repoeval/test_coordinator.py
```
