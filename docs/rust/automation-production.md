# Native automation and governance

These are implemented native APIs, not a claim that Robby's unreachable process
was replaced. Builds/tests use no account connections or live state.

## Exact typed DAGs

`TypedPlan` contains `TypedNode` values: id, tool, arguments, optional depends_on,
preconditions, postconditions. `Scheduler::execute_plan(scope, operation_id,
plan)` uses `Runtime::run_tools`, not provider prompting that imitates tool
execution. Runtime owns schemas, policy, approvals, leases, budgets and journals.

Nested references are `{"$ref":{"node":"upstream","pointer":"/field/0"}}`.
They preserve JSON types and add dependency edges. Missing refs/pointers fail
closed. Strings including `${...}` remain literal; no shell interpolation or
expression evaluation. A `$ref` object cannot have siblings. Limits: 128 nodes,
32 preconditions and 32 postconditions per node, 32 nesting levels, 1 MiB plan.
Topological ordering breaks ties by node ID.

Conditions contain optional node, pointer (RFC6901, empty means whole value),
test, negate. Precondition nodes must be upstream. An absent postcondition node
means this node. Test kind is exists, equals (JSON value), contains (string
value), not_empty, is_number. Missing is not equal to JSON null.

The scheduler durably binds full plan + Runtime fingerprint to authenticated
scope/operation ID before dispatch. Per-node IDs derive from scope/root/node.
Same-ID retries return receipts; changed plan/config is rejected. Any failed,
interrupted, admitted or uncertain node stops ALL subsequent effects. Truncated
runtime output cannot satisfy typed conditions and fails closed. Outputs remain
untrusted, even after envelope decoding.

`plan_receipt` inspects progress. After a known approval pause, the operator
explicitly resumes the original Runtime node turn, then calls `reconcile_plan`
with original plan. Reconciliation requires that turn already be complete;
it never resumes/replays an unknown effect. Lost active coordinator receipts
fail closed pending investigation.

Operator adapters can call `model_plan_scope(original_authenticated_scope,
operation_id)` to derive the exact child scope used by the model adapter before
`plan_receipt`, original-turn `Runtime::resume`, and `reconcile_plan`. There is
no separate model-callable approval/reconcile tool.

## Model tools and wiring

Adapters belong INSIDE outer policy/approval guards:

* `AutomationTools::new(inner,scheduler)`: schedule_task, list_schedules,
  update_schedule (pause/resume), delete_schedule (audit tombstone), execute_plan.
* `DelegationTools::new(inner,pool,registry,DelegationPolicy)`: delegate_task only
  when operator enabled. Arguments: operation_id, tasks:[{id,prompt,depends_on?}],
  optional synthesis prompt. Provider/depth/permissions/scope are server-fixed.
* `EvolutionTools::new(inner,registry,policy,operator_fixed_repo)`: code_evolve
  accepts title/diff, proposes only. No repo selection, apply, review, rollback,
  restart, deploy or push. Existing exact-hash operator workflow remains.

Scheduler Runtime must omit execute_plan/delegate_task; construction checks
recursion. Model plans derive child sessions without changing tenant/user/channel
or reentering parent leases. Schedule specs are typed timezone plus
once/interval/cron/daily/weekly. Workers only run for configured scopes.

Delegation allows 15 tasks plus optional synthesis, depth one, existing AgentPool
cumulative reservations/deadline/concurrency limits. Failed dependencies block
dependents. Child schemas/effect flags must be subset of parent's; recursive and
control-plane tools are banned. Catalog comparison cannot prove filesystem/host
policy narrowing: child runtimes MUST use the identical guarded base executor or
an independently verified stricter policy. Never reuse a privileged operator
runtime merely because its tool names match.

Explicit native DAG+synthesis provides deliberate decomposition. Automatic model
decomposition, cross-provider diverse selection and continue-on-failure policies
are not silently reproduced. Native handling deliberately fails closed.

## Triggers and watchdog

TriggerConfig: name, enabled(default false), operator_consent(default false),
kind, bounded prompt, cooldown_seconds>=30, optional active_utc start/end minutes
(half-open, overnight supported). `fire_trigger` claims durably before running
the existing scoped Runtime, never widening permissions or manufacturing approval.

Kinds: heartbeat (suppression observation), schedule, file_change (first revision
is baseline), disk_above (finite percent), load_above (finite normalized load),
idle (threshold>=30 seconds), watchdog (unhealthy observation). Observations must
come from trusted authorized collectors, not model claims. Library construction
does not start watchers or inspect ambient paths/desktop activity.

Claims bind consent/config/runtime fingerprint. Drift requires operator reset.
Duplicate polls/cooldown retries cannot re-admit an occurrence. Every
non-completed/uncertain occurrence blocks future runs until operator
reconciliation; never invent a replacement ID after uncertainty.

`record_health` is provider-free durable watchdog state: consecutive failures,
failure threshold, recovery transitions, hourly unchanged-healthy audit events.
It does not restart or repair services.

Idle-triggered ordinary prompts are NOT legacy SleeptimeWorker's automatic
summary batching, learning extraction, graph/semantic indexing. Those require a
dedicated maintenance implementation, not a renamed prompt.

### Trusted collectors and actual maintenance

`ObservationSource` is operator-configured and is never a model argument:
File{path,allowed_root,max_bytes}, Disk{path,allowed_root}, Load, Idle. Call
`validate_source` at startup and `observe` on each authorized poll. Paths must be
absolute canonical paths beneath an explicit root; symlink components are
rejected. File reads use NOFOLLOW/NONBLOCK and only bounded regular files (at
most 1 MiB). No recursive directory scanning. Disk uses native statvfs; load uses
native getloadavg normalized by available parallelism. Idle means durable turn
activity in the exact supplied scope, not desktop input monitoring. Unfinished
turns prevent idle admission; no turn history is not permission to spend quota.

`MaintenanceConfig` separately enables actual provider-free idle maintenance:
enabled/operator_consent default false, idle_seconds default300 (30..86400),
min_unprocessed_turns default5, batch_size default20 (1..100), excerpt bound
default2048 bytes (128..8192). `Scheduler::run_maintenance` scans at most1000
recent exact-scope turns, skips unfinished/non-idle scopes, and processes bounded
unprocessed batches. It writes deterministic conversation-digest and observed
tool-outcome-learning memories through Store APIs with provenance and trust
`untrusted`, making them FTS searchable. Digests retain source hashes and clipped
excerpts; learnings record confirmed success/uncertainty facts from the journal,
not speculative advice or executable instructions. No provider, network, command
or desktop access occurs. This is extractive maintenance, not LLM summarization.

The automation SQLite write lock serializes coordinators across processes for a
bounded batch. Memory writes have stable source-derived IDs and exact-content
collision checks. If a crash commits Store writes before the automation batch
commit, the next batch recognizes them without duplicate memory writes. Config
drift is rejected rather than silently rewriting processed source. Original
conversation history is not deleted. Graph/embedding indexing and generated
abstract summaries remain separate features, not implied by this baseline.

## Operator review

`ApprovalRegistry::review(scope,id,optional_exact_call)` returns a token-free
ApprovalReview. Supplied call name/id/argument hash must match immutable binding.
Sensitive-key values, including nested objects, are redacted. Key-based redaction
does not guarantee secrets embedded in free-form shell text are removed.
`pending_reviews` filters pending/unexpired rows BEFORE limiting. Decision still
requires authenticated operator identity and exact binding hash; approval never
itself resumes a turn. No model adapter exposes approval decisions.

## Disposable verification

`CARGO_BUILD_JOBS=2 cargo test -p missy-automation -p missy-subagents
-p missy-governance -p missy-evolution --quiet`: 68 passing tests at this change
(automation31, subagents14, governance12, evolution11). Includes no-provider exact
DAG dispatch, refs/conditions/dedup/scope/config guards, approval resume plus
plan reconciliation, consent/cooldown/file-change triggers, watchdog transitions,
fixed-provider delegation+synthesis/dedup/control spoof rejection, token-free
exact-call review, evolution proposal preserving source/HEAD, trusted collector
file-bound/symlink rejection/native disk+load checks, and actual once-only idle
digest/tool-learning writes with no provider calls.
All fixtures use private temporary DBs/repos, no accounts, ~/.missy, services or
desktop sessions.

### Trigger pause/recovery availability correction

`fire_trigger` retains its `Result<Option<RunResult>>` API. An existing admitted
or running occurrence returns a deduplicated `inflight` poll receipt; interrupted,
failed, cancelled, or missing-turn reservations return `paused`. These receipts
perform no provider call, execution, resume, or replay. They deliberately do not
claim usage/resumability (`cost_known=false`, `resumable=false`), and a missing
turn has an empty `turn_id`, not a fabricated identity. Initial successful Runtime
returns still carry the actual Runtime result, including approval interruption.
Execution/admission errors after durable reservation become paused receipts, not
fatal supervisor errors. Invalid config and persistence errors remain errors.

`Scheduler::trigger_receipts(scope,name)` supplies read-only operator snapshots:
operation, request_id, exact turn_scope, optional actual turn_id, actual Store
status (or `outcome_unknown`), and blocked. `trigger_scope(scope,name)` derives the
same child scope as admission. Recovery resolves the exact scoped durable request
ID with `Store::turn_by_request`, never bounded recent-turn history and never
admission. Thus a crash between reserving and receiving a Runtime result cannot
authorize a replacement mutation. Legacy ledgers require no schema migration.

After an operator explicitly approves and resumes the original turn, its actual
completion refreshes the ledger on the next eligible poll, including during
cooldown. Only completion releases future admission, still subject to the
original cooldown and file revision/config checks. Approval by itself releases
nothing. Failed/cancelled turns remain nonfatally paused until explicit operator
reset. `reset_trigger` refuses admitted/running/interrupted turns, missing-turn
reservations and unresolved tool journals; it only clears known terminal turns
with no uncertain effects. A missing-turn reservation requires investigation,
not a generic reset loophole. There is no model-callable reset/resume endpoint.

Disposable regression tests cover a sensitive executor returning ApprovalRequired,
reopened scheduler repeated polls without additional provider/effect calls,
explicit human resume plus completion refresh and subsequent cooldown admission,
failed/cancelled pause and safe reset, and missing/admitted/running reservation
recovery without execution.
