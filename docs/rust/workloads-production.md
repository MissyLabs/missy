# Native durable Nomad workloads

`crates/missy-workloads` implements actual scoped finite batch lifecycle, not
aliases for allocation metadata. Source design follows
`missy/nomad/{models,workloads,benchmark,manager,store}.py`.

## Application API

### Integrated acceptance update

The actual CLI supervisor and model executor now share one `Arc<WorkloadTools>`
through `build_runtime_with_services` in `missy-cli/src/main.rs` and
`production.rs`. The previous double-open attempted to hold the same private
coordinator lock twice and was a real startup blocker; it is fixed.

`cargo test -p missy-cli --test workloads_production_e2e -- --nocapture`
passed actual native gateway child acceptance: authenticated HTTP admission,
mock provider receives all four model-callable definitions, emits a finite
schedule call, execution pauses for exact durable approval, operator approves
and explicitly resumes, shared supervisor tick performs fresh mock Nomad
discovery/planning and optimistic submission, SIGTERM drains, and restart does
not replay the occurrence. The test verifies **zero submissions before
approval**. Disposable state/environment and loopback fixtures only. This is
integrated wiring evidence, not live acceptance or whole-repository parity.

* `WorkloadTools::open(policy: PolicyConfig, config: WorkloadsConfig, state_dir)`
  validates without contacting Nomad. State directory is explicit absolute,
  owner-private, no symlink ancestors. Never reads legacy HOME state.
* `ToolExecutor` exposes `nomad_offload_task`, `nomad_benchmark`,
  `nomad_reconcile`, `nomad_schedule` only when enabled. All conservatively
  require the application's **exact durable human approval** gate. Direct
  embedding is responsible for that gate and normal runtime effect journaling.
* Share one `Arc<WorkloadTools>` between runtime and supervisor. The private
  cross-process state lock intentionally refuses a second coordinator.
* `run_due(now, cancellation, limit)` explicitly ticks finite schedules and
  requests nonpurging stop for expired owned active workloads. No implicit
  background worker starts when the crate is opened. A supervisor must tick
  this shared instance for unattended schedules and wall-time enforcement.

## Real behavior

Templates contain an operator-authored one-group/one-task Docker batch graph,
digest-pinned image, bounded CPU/memory/disk and finite wall time. Host mounts,
devices, privileged Docker config, ambient template secret loaders and arbitrary
command construction are refused. Caller parameters populate only operator
named environment slots. Input/output artifacts are explicit approved HTTP(S)
directory-prefix URLs, passed as data environment fields. Remote container
network authority is the operator's Nomad/template boundary, not a claim that
client HTTP policy sandboxes a remote workload.

Planning performs actual policy-bound node discovery, eligible allowed-node
resource inspection and allocation headroom accounting, then Nomad server
planning. Submission repeats fresh discovery and server planning and uses
`EnforceIndex=true, JobModifyIndex=0` for new unique scoped jobs. A durable
`submission_unknown` record precedes POST. Errors, lost acknowledgement and
restart never cause automatic resubmission. Reconciliation checks exact owner,
scope, original job fields, executable/environment and forbidden remote
authority additions; unknown/missing/changed jobs are not adopted or recreated.

Result collection reads real bounded stdout/stderr from Nomad allocation log
API, parses `MISSY_RESULT_JSON=`, checks expected marker and typed required
result fields and declared outputs, and actually GETs each approved output
artifact to compute byte count/SHA256. It does not claim that an artifact was
verified merely because an allocation says `complete`. Logs/structured results
are censored before return/persistence. Artifact receipts retain URL, bytes and
digest; binary bytes are not retained in the SQLite record.

Benchmarks persist warmup/measured run IDs with immutable version/run metadata,
enforce configured run and parallelism bounds, require fresh authorized
capacity, finish warmups before measured submissions, and compare observed
node identities/resources/architecture. `status` only advances an explicitly
started runset. `results` never starts pending runs. Partial planning can be
continued with the same exact definition/idempotency key. Each remote mutation
is independently saved, so later failure does not reset earlier runs to pending.

Schedules are stored with authenticated scope, exact configuration fingerprint,
finite maximum run count, validated timezone-aware ScheduleSpec and original
data. A claim is saved before planning/submission, and an occurrence key is
used as the child task idempotency key. Missed intervals coalesce; uncertain
claimed occurrences do not replay. Config changes block old schedules rather
than rebinding their authorization to new templates. Pause/resume/remove/list
are scoped. Removing retains a tombstone and does not purge job/history/data.

## Deliberate restrictions and remaining acceptance boundaries

This is a native replacement API, not a byte-compatible Python job schema.
There is no service rollout/scaling, restart/purge, arbitrary host command,
native Nomad Variables secret template, arbitrary registry image or implicit
CLI process. The preexisting integration adapter separately covers pinned
service deploy/scale operations. A full replacement claim must distinguish
these separately governed APIs.

No real Nomad endpoint/account/cluster was used. Nine disposable actual
loopback HTTP integration tests exercise full ToolExecutor paths: successful
plan/submit/log/artifact verification and restart deduplication, foreign scope
rejection, lost-ack reconciliation without repost, remote privileged-config
tamper cancellation denial, warmup/parallel measured progression/comparison,
finite schedule restart no duplicate, fresh capacity/optimistic conflict
refusal, incomplete result/schema denial, config-change schedule blocking,
private state/single-coordinator guards and pre-network policy/schema refusal.
Tests initially exposed tempdir permissions and were corrected to explicitly
private fixture dirs, not relaxed production validation. Targeted `cargo test
-p missy-workloads` passes **9/9**; `cargo clippy -p missy-workloads --all-targets
-- -D warnings` passes. No lint allowances were introduced to hide findings.

No full-workspace build or live process/device action is needed to establish
source implementation. Operator canaries remain external deployment acceptance,
not something inferred from the number of tests.
