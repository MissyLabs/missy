# missy-store

Synchronous `Store` operations use a mutex-protected `rusqlite::Connection`.
Independent `Store` instances and processes coordinate through `BEGIN IMMEDIATE`
transactions, a unique `(scope, request_id)` admission binding, and SQLite WAL
with `synchronous=FULL`. All public reads and writes require a validated full
`missy_core::Scope` (tenant, user, channel, session). No global fallback exists.

## Runtime contract

- `admit` deduplicates every request, including terminal and retained requests.
  Reusing its ID for a different prompt is rejected. Retention preserves the ID,
  scope, hash, and terminal tombstone, preventing redelivery from executing again.
- `acquire_lease` serializes a session and issues an increasing fencing number.
  Every live-turn mutation checks the full scope, turn, owner, fence, running
  status, and unexpired database lease inside the same write transaction.
  Expired owners cannot heartbeat, release, checkpoint, settle tools, or finish.
  Lease objects are internal capabilities, never accepted from an HTTP client.
- One runtime writer must serialize its own checkpoint snapshots within a lease.
  Checkpoint writes reject decreasing step/token/cost counters, but are not a
  general compare-and-swap interface for concurrent tasks sharing one lease.
- `tool_begin` commits the full arguments and hash before execution.
  `tool_running` must commit immediately before dispatch. A duplicate successful
  invocation returns its cached JSON result. Name/argument changes under an old
  call ID are rejected. All tools are conservatively effect-capable.
- `unresolved_tools` / `unresolved_effects` include prepared, running, and unknown
  intents. The runtime must refuse **all** resumed model generation while any
  remain, not just refuse retries of the same call ID. `tool_begin` also rejects
  a new ID while an earlier running/unknown call remains. Prepared batch intents
  can coexist before sequential execution. Failed means definitively failed;
  cancellation, timeout, transport loss, or uncertain execution must use
  `OutcomeUnknown`, not `Failed`.
- `recover` is explicit and scope-limited. It marks only expired running turns
  interrupted and unsettled tools unknown. It never replays effects, sweeps
  live workers, or runs automatically when an inspector opens the database.
- Checkpoints persist the original system prompt, current-turn transcript,
  steps, cumulative usage, and versioned runtime provider state. Their full JSON
  is checksum-bound to scope and turn. Context and system lineage mismatches
  fail closed. Runtime owns validation of its opaque provider state (deadlines,
  continuation limits, tool manifests, provider fingerprint, and guard lineage).
- A completed finish atomically publishes checkpoint transcript entries after
  the initial system entry to history, updates terminal state, and releases the
  lease. This transcript must contain only the current turn; recalled session
  history belongs in the runtime's inert framed context, not the transcript.
  Duplicate finish is rejected. Completion is refused with unresolved effects.
- Cancellation requests are durable, fully scoped control-plane requests.
  Workers poll `cancellation_requested` and cancel execution cooperatively.

## Files, memory, audit, retention

New state directories are 0700. Existing parents must be current-user-owned and
not group/world writable. Database and SQLite sidecars are 0600; symlink,
foreign-owner, and hardlinked DB/sidecars are rejected. Existing database files
are inspected with metadata/chmod only, never manually opened and closed, which
would release SQLite POSIX locks process-wide. SQLite opens use NOFOLLOW.
Schema migrations run transactionally; future versions and foreign application
IDs are rejected. The application ID is `0x4d535953`.

Memory CRUD and literal-phrase FTS5 search are parameterized and fully scoped.
Provenance is mandatory; trust is restricted to `untrusted`, `user_provided`, or
`tool_output`. These labels are descriptive data, never executable authority.
Trusted callers assign provenance from authenticated identities, not model
claims. Recalled memory is not promoted into system/developer instructions.

Audit entries are append-only under SQL triggers and chained by SHA-256 per
scope. Verification detects modified chain content or broken links. There is no
signing key, external anchor, or claim of resistance to a database owner who can
drop triggers and rewrite or truncate the entire chain. Checkpoint checksums
likewise detect corruption, not malicious rewriting by the database owner.

Retention deletes old terminal message/checkpoint/event/tool content and scoped
memories, strips terminal turn prompt/results, and retains tombstones plus the
append-only audit chain. Audit payloads contain provenance, IDs, statuses, and
hashes, not message or tool-argument bodies. Audit retention/anchoring outside
this crate is an operator responsibility.

Tests use disposable temp directories and child processes only:

    cargo test -p missy-store
    cargo clippy -p missy-store --all-targets -- -D warnings

`tests/runtime_recovery.rs` is a cross-crate acceptance suite using inert
providers/tools. The dev-only dependency on `missy-runtime` does not introduce a
runtime dependency cycle. It covers stale-checkpoint successful-result repair
without reexecution, prepared/unknown and unjournaled-call generation barriers,
full argument binding during hydration, admission without a checkpoint, final
result crash recovery, and unknown billing after an unsettled provider request.
