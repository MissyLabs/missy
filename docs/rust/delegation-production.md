# Native delegation

`DelegationTools` provides `delegate_task` inside the parent policy and approval
guards. No service or account is started by this library. The operator enables
it, selects the default runtime ID, explicitly allowlists any alternative IDs,
sets `max_tasks` (1..15) and fixes `failure_policy`. An empty allowlist means only
the default runtime. Unknown IDs fail closed rather than falling back.

## Requests

A stable `operation_id` and exactly one of these are required:

* `tasks`: explicit `{id,prompt,depends_on?,provider?}` nodes. Provider choices
  must be in the operator allowlist or equal the default.
* `prompt` with `strategy=decompose` (default): one actual bounded NoTools
  Runtime planner turn returns strict JSON tasks and explicit dependency IDs.
  The complete response is deserialized with unknown fields denied, then the
  same DAG validator checks IDs, duplicates, dependencies, cycles, task count,
  prompt lengths and configured runtimes. No splitting strings into fake plans.
* `prompt` with `strategy=diverse`: four independent requirements/evidence,
  architecture, risk and verification specialists. Providers rotate over the
  explicit ordered operator allowlist, never over inferred model capabilities.
  `include_synthesis` defaults true; the final lead depends on all specialists.

`synthesis` optionally supplies a bounded explicit synthesis prompt in either
mode. Every child is depth one; children cannot delegate further. All outputs
are untrusted. Dependency status, error and result are framed as JSON data, not
instructions or authority.

FailurePolicy is public: `BlockDependents` (default), `Continue`, `Abort`.
`AgentPool::run_with_policy` implements them. Continue passes terminal failure
data to dependents; it does not bypass unknown-spend budget safety. Abort
cancels active peers and blocks remaining planned work. Requests may repeat
the operator policy but cannot change it. Legacy request spellings
`skip_dependents` and `fail_fast` are accepted by preflight as aliases for the
matching fixed policy; model schema advertises the native fixed spelling.

## Persistence, budgets and cancellation

The outer receipt binds authenticated scope, operation ID, complete request,
operator policy and checked runtime fingerprints before any planning. Planner
spend reserves the full configured Runtime step/token/cost envelope durably
before admission. It uses Runtime::no_tools(), sharing the original provider,
store, system, finite config and global budgets; no naked provider call. Stable
derived planner request/session IDs preserve tenant/user/channel. A planner is
also one agent against the root count, not free extra work.

The validated plan runs under the remaining cumulative root budget and original
deadline. The outer receipt conservatively reserves the entire remaining
envelope before dispatch; completed receipts replace this with actual cumulative
child reservations. Reservations are not refunded on failure/unknown outcome.
Planner and child usage are aggregated. Runtime independently applies its own
durable spend, token, lease, schema, journal and response guards.

Planner admission records the durable turn ID and installs a drop guard that
cancels that exact turn. Polling never readmits a replacement ID. Cancellation,
deadline and dropped futures cancel real Runtime work, not just its waiter.

Private SQLite receipts and per-root nonblocking locks prevent duplicate
coordinators. Reopening/retrying a completed or failed operation reads receipts;
it never replans or reexecutes. Changed requests/configuration are rejected.
Lost active coordinators become interrupted on retry and require operator
investigation; no automatic replay, reconciliation or approval fabrication.

## Capability and capacity boundaries

Every allowlisted child catalog must be a schema/effect-preserving subset of the
parent. Recursive/control-plane tools are denied. Catalog equality alone is not
proof of filesystem/host policy narrowing: integrations must use the identical
guarded base executor, a verified stricter executor, or NoTools. Production
wiring uses NoTools children, so delegated tasks are advisory, not a claim that
they can perform privileged actions. Provider diversity changes provider only,
not permissions or policy.

Production composition requires at least two concurrent store/runtime slots for
nested orchestration. A one-slot direct library pool remains valid for ordinary
non-nested calls; an occupied sole slot fails closed instead of bypassing the
lease/capacity guard. Supporting live nested execution at capacity one is not
claimed here.

## Validation

`CARGO_BUILD_JOBS=2 cargo test -p missy-subagents --lib` uses disposable private
SQLite stores and trait-provider fixtures. Coverage includes real planner
execution and NoTools catalog, durable reservation, strict/injected planner
rejection, provider/policy/limit spoofing, DAG failure policies including reversed
array order, diverse optional synthesis, reopening deduplication, capability
widening rejection, cancellation/drop/deadline of actual provider work, bounded
concurrency and scope isolation. No deployments or production accounts used.

This deliberately hardens legacy behavior: strict structured decomposition
instead of regex splitting, explicit provider allowlists, fixed operator failure
policy, no recursive capability grant and persistent no-replay identities.
It is not byte-for-byte compatibility with legacy arbitrary `agents` metadata
or zero-based dependency indices; native `tasks` use stable string IDs.
