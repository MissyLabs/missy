# Native operations

All surfaces are explicit library calls. No listener, collector export, workflow,
or scheduled task runs simply because this crate is linked.

## HTTP integration

`build_router(SharedRuntime, OperationsConfig) -> Result<axum::Router>` and
`serve(SharedRuntime, OperationsConfig) -> Result<()>` are the parent CLI entry
points. `SharedRuntime` is `Arc<missy_runtime::Runtime>`. `serve` rejects any
non-loopback bind address, and uses Ctrl-C graceful HTTP shutdown. Runtime work
already admitted follows Runtime's owned-task semantics, not client connection
lifetime. Process termination can leave interrupted/uncertain effects requiring
Runtime reconciliation; shutdown does not assert that all work completed.

`OperationsConfig::default()` is available. Set `replay_db` to a file inside the
application's private state directory. The file must be owned, regular, single
linked, and mode 0600; its parent must be owned and not writable by others.
No credential literal appears in config. `MISSY_WEBHOOK_HMAC_KEY` and
`MISSY_OPERATIONS_READ_KEY` are the default dedicated environment references;
both require distinct values of 32..4096 non-control bytes.

* `POST /webhook`: exact JSON fields `event_id`, `prompt`; unknown fields fail.
  No caller principal, scope, runtime options or tool payload is accepted.
  The configured scope is authoritative. Maximum raw body is 65,536 bytes;
  prompt bound is 64,000 bytes, additionally bounded by inherited Runtime.
  `Content-Type: application/json` is required. `x-missy-timestamp` is decimal
  Unix seconds, within configured skew (default 300 seconds, maximum 900).
  `x-missy-signature` is 64 hexadecimal characters encoding
  HMAC-SHA256(key, timestamp bytes + `.` + raw request bytes). HMAC verification
  is constant-time. Duplicate authentication header fields fail closed.
* Successful admission returns the **actual durable Runtime turn ID**, status,
  and deduplication flag, never prompt/model output. New delivery is 202, exact
  retry 200, event-ID/body conflict 409. Resigning identical raw bytes with a new
  timestamp is valid; changing JSON whitespace also changes the body hash.
* Replay claims use a FULL-synchronous SQLite immediate transaction and scoped
  event/body hashes. Runtime's own scoped idempotency protects admission.
  There is deliberately no claim of atomicity across two separate databases.
  A crash after claim but before receipt persistence fails closed on retry with
  409 `admission_receipt_uncertain_requires_reconciliation`, rather than
  repeating potentially executed effects. A saved receipt whose Runtime turn
  was removed by retention also fails closed. Never delete a claim merely to
  retry an uncertain effect. Cache capacity is bounded and never silently
  evicts old event IDs; full cache rejects new events with 503. Operators must
  reconcile/rotate their delivery namespace and durable storage deliberately.
* `GET /diagnostics`, `GET /metrics`: require `Authorization: Bearer <read key>`.
  The webhook key is not a read key. Only read-only scoped Store calls occur.
  No prompts, tool arguments, responses, error details, paths, credentials,
  principal labels or arbitrary strings are exposed. Metrics use fixed labels.

## Diagnostics and telemetry

`snapshot(&Store, &Scope)` samples the latest **256 scoped turns** with an
explicit truncation flag. It sums the latest durable checkpoint usage for each
sampled turn using public Store APIs. Missing checkpoints and unknown cost are
reported, not treated as known-free usage. These are **not lifetime totals or
remaining-budget guarantees**. Snapshot reads are not a cross-query database
transaction and can observe concurrent progress. Provider/tool/gateway/etc.
health remains `unknown` absent an owner observation; a successful Store read
does not mean external dependencies are healthy.

`LifecycleRegistry` accepts explicit enum subsystem/state observations with
timestamps. `EventBus` uses bounded Tokio broadcast with explicit `Lagged`
receipts. Events contain only a closed kind enum, time and sequence. It is a
best-effort operational feed, not a durable audit/effect journal. Attach one to
webhook admission with `build_router_with_events`; subscribers are trusted local
code, not remotely supplied scope readers.

`OtlpExporter` emits actual OTLP/HTTP JSON `resourceSpans` at `/v1/traces` with
trace/span IDs, nanosecond timestamps, fixed service/instrumentation names and
closed code-owned attributes. Both span count and encoded body bytes are bounded.
Collector partial rejection is an error. All export requests use the supplied
`PolicyHttpClient`; no direct reqwest, proxy, redirect or DNS bypass exists.
Each export is one-shot and is never retried autonomously. This implements
traces, not a general OpenTelemetry SDK, tracing-subscriber bridge, or fake OTLP
metrics. `export_signed_summary` optionally POSTs a signed redacted diagnostic
summary through the same policy gateway; it is not the complete audit journal.

### Automatic native runtime instrumentation

`TelemetryRuntime::build_client(PolicyConfig, TelemetryConfig)` synchronously
creates an optional bounded export worker in the current Tokio runtime. The
default `endpoint: None` performs no I/O and spawns nothing. Opt in with an exact
`http(s)://collector/v1/traces` endpoint **and** independently allow its origin
through network policy. Private addresses still require explicit policy opt-in.
Configuration denies unknown fields. Queue, batch, encoded bytes, flush interval,
request timeout and shutdown timeout all have checked hard bounds.

Pass `telemetry.observer()` as the sixth argument to `Runtime::new_observed`.
Unlike the standalone exporter, this attaches to actual owned runtime work,
including daemon/webhook turns using that runtime. Existing `Runtime::new`
continues unobserved. Each execution attempt gets one fresh random trace, a root
turn span, and root-parented provider, tool execution, and durable receipt
settlement spans. Multiple provider/tool steps share that trace. Explicit resume
creates a new trace; trace IDs are not persisted or derived from principals.
No prompt, model/tool content, call/turn IDs, tool name, principal, path, URL,
credential or arbitrary error string is present in runtime observation events.
Receipts report successful **persistence**, not successful external tool effects:
an unknown effect can have a failed tool span and successful settlement span.

The observer callback only uses `try_reserve` on a bounded channel and atomic
counters. It never waits, performs I/O, or changes runtime errors. Observer
panics are isolated from execution. Overflow drops the newest span immediately.
Memory is bounded by the configured queue plus one batch and bounded HTTP
request/response bodies. This is **best-effort telemetry**, not durable SQLite
spooling, an audit journal, or lossless delivery. Dropped root spans can leave
orphan child spans; loss is counted rather than hidden. Collector failures,
partial rejection and request timeout count the whole batch as failed and never
retry an uncertain POST. Partial acceptance can therefore undercount delivered
spans; no idempotent collector behavior is assumed.

Keep the handle for `metrics()` and explicit `shutdown().await`; shutdown closes
admission and drains under one overall deadline, returning an error on deadline
or worker failure. `monitor()` provides cloneable read-only counters without
keeping the channel alive, including post-shutdown/timeout diagnostics. Counters
include accepted, pending, exported, failed, dropped and successful/failed batch
totals; `TelemetryMetrics::prometheus()` uses fixed unlabelled metric names.
Dropping the owner does **not** stop a still-live observer. Dropping the last
sender automatically closes and drains the worker under the shutdown deadline.
An owning process must await explicit shutdown before destroying Tokio to
guarantee that final drain is given CPU time. No background task can survive
runtime/process termination, and queued spans are intentionally not durable.

## Portable workflow

`Workflow::parse`, `ordered_nodes`, `execute` support a literal prompt DAG with
1..128 hard maximum nodes, 64KiB serialized input, stable topological ordering,
duplicate/cycle/dependency checks, and scoped per-node idempotency. The caller
supplies the authenticated Scope and existing SharedRuntime. JSON cannot supply
policy, principal, tools, runtime settings, recursive children, substitutions,
or credentials. Node prompts remain literal. Execution is sequential and stops
before dependents on any non-completion. It never auto-resumes uncertain effects.

Workflow aggregate limits are **admission gates**, not a replacement for Runtime
budgets: an active node retains the exact original Runtime bounds and aggregate
usage can overshoot the workflow threshold by one bounded turn. Unknown cost
blocks further node admission. Wall deadline cancels the active turn and stops
the workflow. A repeated workflow ID/node ID reuses the original turn, and
changing its prompt conflicts rather than silently running it again. Callers
requiring hard shared provider budgets must reserve them in Runtime/Store first;
this crate does not claim to provide that accounting reservation.

Human schedule parsing is owned by the parent scheduler/schedule-text crate,
not a second grammar in this crate.

## Verification

`cargo test -p missy-operations` uses only temporary databases, mock providers,
local isolated routers and policy-denied collector destinations. No live state,
production listener or provider credential is used.
Tests include actual Runtime provider/tool execution through the bounded worker
and a local HTTP collector, trace parentage across steps, redaction, counted
overflow, failed exports without retries, owner/observer lifetime, policy denial,
and bounded shutdown against a stalled collector. Runtime regressions also cover
observer panic isolation and unknown-effect versus durable-receipt semantics.
