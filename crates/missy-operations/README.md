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
Exporting is one-shot, never implicit or retried autonomously. This implements
traces, not a general OpenTelemetry SDK, tracing-subscriber bridge, or fake OTLP
metrics. `export_signed_summary` optionally POSTs a signed redacted diagnostic
summary through the same policy gateway; it is not the complete audit journal.

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
