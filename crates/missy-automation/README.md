# Rust automation capabilities

`Scheduler::open(path, Arc<Runtime>)` opens a **separate SQLite scheduler database**.
Pass an absolute filename in an existing canonical directory owned by the process
with mode 0700. Existing database files must be regular, owned, mode 0600 without
hardlinks or symlinks. Opening never starts work, sweeps active leases or resumes
turns. A CLI/gateway must explicitly opt into calling `run_due`.

All methods require a server-trusted `missy_core::Scope`:

- `add(&Scope, NewJob { prompt, interval_seconds, first_due: DateTime<Utc> })`
- `add_spec(&Scope, prompt, ScheduleSpec, after: DateTime<Utc>)` for validated
  `missy-schedule-text` one-shot, anchored interval, daily, weekly or five-field
  cron schedules with explicit IANA timezones
- `list(&Scope)`, `pause(&Scope, id, paused)`, `remove(&Scope, id)`
- `occurrences(&Scope, limit)` includes honest terminal statuses and diagnostics
- `run_due(&Scope, max_workers, CancellationToken)` is asynchronous, bounded to
  128 admissions per invocation and 1..32 worker capacity. Runtime transcripts
  serialize per scope, including across scheduler instances.

UTC persisted due dates have whole-second resolution. Spec-derived dates are
rounded **up** to the next whole UTC second, including after rescheduling, so
fractional one-shot/interval anchors never run early or get stuck repeating the
same persisted due date. Overflow is rejected. The legacy `add` API retains its
existing floor-to-second behavior. `add_spec` returns the exact persisted due date.
Recurring specs advance from the previous due date, not completion time; overdue
occurrences catch up subject to the admission bound. Calendar/cron schedules
follow local wall time and the schedule-text crate's DST policy, not fixed UTC
intervals. Exhausted one-shots are disabled and cannot be unpaused after a terminal
occurrence; create a new job for another occurrence.

Schema v1 databases migrate to v2 without losing legacy jobs or occurrence history.
Creation, version validation and migration run inside one immediate SQLite
transaction, including version reread under the lock, so separate processes cannot
both apply the new column. Opening still does not execute any job.

Each occurrence persists a unique
request ID before runtime admission. The same ID is reused after expired-lease
takeover, relying on Runtime's durable idempotency and unknown-effect refusal.
Scheduler claims use an immediate SQLite transaction, expiring leases, heartbeat,
and monotonic fences. No active claim is swept on startup. Terminal failures are
recorded, not reported as success or retried under a new request ID. Infrastructure
errors with uncertain outcome record `unknown` and pause the job for inspection.
Unpausing a job with an unknown occurrence is refused. An operator must reconcile
the effect and deliberately remove/recreate the job to schedule new effects.
`remove` tombstones the job; it does not abort already admitted work. Cancellation
stops new admissions and drains ongoing runtime calls with heartbeats, rather than
dropping a future in the middle of an effect. Callers must await that drain.

`Plan::compile(&trusted_scope)` validates a serde DAG: bounded nodes, calls and
bytes, unique IDs, complete dependencies, acyclicity and exact scope equality.
Use `Plan::parse_json` for a bounded external JSON entry point before compile.
Arguments remain literal JSON; no substitutions occur. **There is no direct plan
executor**: tool dispatch without Runtime's journal would bypass crash safety.
Compiled plans grant no authority, and policy remains Runtime/ToolExecutor-owned.

`discover_skills(&[canonical_root])` recursively reads explicit `SKILL.md` files,
rejecting links and bounding roots, directories, depth, file count and bytes.
Returned markdown is **untrusted data**, never code execution or automatic
instructions. Roots must not be mutated concurrently during discovery.

Trigger/webhook execution and autonomous planning execution are not implemented
by this crate. No pretend compatibility or hidden Python bridge.
