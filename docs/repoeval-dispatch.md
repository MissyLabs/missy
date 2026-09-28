# RepoEval durable dispatch foundation

`missy.repoeval.dispatch` is an **opt-in reference outbox**, not production
execution. It does not create a scheduler client, install a worker, expose an
HTTP route, or integrate with `FoundryService`, `FoundryAPI`, or PostgreSQL
`PostgresStore`. The existing `control.py` demo still requires its own explicit
single-process opt-in. A live deployment remains disabled until a reviewed
authority and scheduler adapter are explicitly supplied.

## Contract and transaction boundary

`SQLiteOutbox.initialize(path)` explicitly creates an on-disk SQLite WAL table.
`DurableDispatcher(store, scheduler, authorize, enabled=True)` requires an
operator-owned `authorize(project_id, run_id, exact_job)` callback. Tests use
`lambda *_: True` solely with an in-memory fake scheduler. A real adapter must
recheck project execution rights, approved immutable snapshot, pinned image,
worker identity, resource quotas, node pool, namespace, datacenter, network
and credential isolation against the **exact** planned Nomad JSON immediately
before reservation and before marking a submission attempt. `reserve()` binds
project/key/run/job identity transactionally; run IDs must match `run-*`, and
the job ID must be `foundry-{run_id}` with matching metadata. The job JSON is
hashed and persisted; the key is SHA-256 hashed and never exposed by `status()`.

`dispatch()` atomically marks the reserved row `dispatching` *before* invoking
`scheduler.submit(namespace, job)`. A crash, timeout, or exception after that
marker leaves the effect unknown. **It cannot auto-submit again**. `reconcile()`
performs `scheduler.lookup(namespace, job_id)`, accepts only the exact persisted
job payload and its scope, and records `submitted`, `running`, `collecting`,
`failed`, or `conflict`. A missing job or failed lookup remains `uncertain`;
neither absence nor a successful submit return is proof of verified output.
Scheduler `complete` yields only `collecting`; an independent worker/validator
and artifact evidence path is still required for `verified`.

`cancel()` marks cancellation intent before lookup. A reserved run is cancelled
without a scheduler call. Otherwise it must observe the exact job before a
stop. If cancellation commits while submission is in flight, the submitting
caller reconciles on completion (including an uncertain submit outcome) and
attempts the first stop when the exact job is observed; no second cancel call
is needed. Concurrent reconciliation and cancellation compete for one durable
stop claim. The stop attempt is committed *before* calling `scheduler.stop()`.
Crash/timeout means `stop_uncertain`, not an automatic retry, even if a later
lookup fails. Only a subsequent exact `stopped` observation produces
`cancelled`. A colliding scheduler job produces persistent `conflict` and is
never stopped. This reference lookup checks the persisted job bytes, but a
production adapter must additionally attest an immutable scheduler incarnation
before any stop; an ID and matching mutable payload alone are not sufficient.

## Production integration blockers

This independent SQLite store does **not** atomically share transactions with
`PostgresStore.create_run`, workload plan approval or artifact collection.
Implement and review a single PostgreSQL transaction/outbox against the
project-scoped immutable run, its idempotency key, and current approval evidence
before enabling a real service. Scheduler lookup must be authoritative and
return the exact effective job definition, including namespace and metadata;
the existing `NomadClient` has read methods but no RepoEval adapter attesting
exact effective job bytes or bounded mutation credentials. Do not mistake
`nomad.py` plan construction for a job submitter. Do not use the generic
`NomadManager` execution path as an implicit grant. Nomad job IDs can be
resubmitted or replaced, so production evidence must also bind an immutable
submission incarnation (e.g. create index and namespace) before acting on an
existing ID. A weak lookup cannot satisfy that contract.

The outbox keeps states, not signed scheduler receipts or validated worker
artifacts; it intentionally does not mark any run `verified`. Project-scoped
read/execute permissions must wrap every caller of this internal API.
Production migration must include explicit schema versioning, retention and
backups. No automated cleanup/purge is implemented; deleting a row would erase
the no-replay marker and is unsafe while its external effect may exist.

Tests exercise contention across threads and processes, restart, idempotency
conflicts, revoked authorization, uncertain submission/stop, missing or
mismatched lookup, cancellation, and completion without artifact verification.
