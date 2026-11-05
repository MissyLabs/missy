# Native Nomad SERVICE lifecycle

This resource handles exact operator-pinned **service** jobs only. Owned batch,
offload, benchmark, output retrieval, schedules and reconciliation belong to
`missy-workloads`, not this integration. No shell/CLI fallback is used.

## Operator grant

`integrations.nomad` retains its exact `namespace`, `jobs`, `nodes`,
`deployments` and named `scales` allowlists. Additionally, each mutable service
needs a protected `services.<exact job ID>` entry:

```json
{
  "owner": "operator-a",
  "expected_modify_index": 7,
  "rollout_overlap": 1,
  "allow_stop": true,
  "allow_restart": true
}
```

The exact operator-authored Job must have matching ID/Namespace,
`Type: "service"`, and `Meta.missy_service_owner: "operator-a"`. All groups need
explicit counts, tasks with positive CPU/MemoryMB resource reservations, and
explicit datacenters. Node IDs must be individually allowed. Dedicated cores
and device capacity are unsupported and fail closed. No caller-authored Job,
image, command, mounts, environment, node, namespace, replica count, or group is
accepted. Named scale grants select the configured group/count only. Other
pinned workload definitions remain exact and do not gain mutation permission
merely by being in the read allowlist.

For initial creation, index zero requires observed absence. For updates, remote
ownership metadata and the exact operator-pinned positive JobModifyIndex must
both match. Foreign jobs, foreign namespaces, disappeared existing jobs and
concurrent index changes are refused. Following a successful update, inspect
the remote service and approve a new config index before further changes.

## Plan, review, submit

The strict `nomad_mutate` actions are:

| Plan action | Submit action | Arguments |
| --- | --- | --- |
| `plan` | `deploy` | `job`; submit also requires `plan` |
| `plan_scale` | `scale` | named `grant`; submit also requires `plan` |
| `plan_restart` | `restart` | `job`; submit also requires `plan` |
| `plan_stop` | `stop` | `job`; submit also requires `plan` |

Plans return an opaque token, operation, job, namespace, modify index, capacity
summary, Nomad dry-run result and `planned_not_submitted` state. Tokens bind
the exact graph, action/grant and index, expire after 300 seconds, and occupy a
bounded 128-entry process-local store. A token is removed before submit does
any network I/O. There is no implicit plan-and-execute shortcut.
Each submission requires the normal outer durable human approval, private
network/origin policy and operator mutation switch. Configuration grants do
not replace approval. Integration-level opaque plans are not durable approval
records and do not themselves authorize unattended operations.

Scale submits the full pinned service definition with only the reviewed count
changed, not the unfenced `/scale` endpoint. All registrations use
`EnforceIndex: true` and `JobModifyIndex` including create-if-absent index zero.
Fresh ownership/index checks and capacity discovery run again at submit time.
Nomad CAS remains the authority if another writer races the final read.

Restart is a **rolling replacement**, not an unfenced allocation task restart.
It requires `allow_restart` and adds the reserved
`MISSY_SERVICE_RESTART_ID=<plan token>` task environment stamp to force Nomad's
normal rolling deployment of the exact pinned graph. Operators must not set
that reserved environment key. Stop requires `allow_stop`, retains the freshly
observed owned graph, and registers `Stop: true` under CAS. It never purges
history and does not use DELETE, which lacks the required modify-index guard.

## Placement and resource safety

Fresh discovery fetches the node list, individually allowed node details, and
their allocations. Only ready, eligible, non-draining nodes in the pinned
datacenters/pool count. CPU, memory (including task MemoryMaxMB), disk,
reserved node resources, ephemeral group disk and all nonterminal node
allocations, including other owners' jobs, are charged. Missing/invalid
resource data fails closed. Capacity placement is conservative first-fit decreasing:
existing allocations remain charged and **all** desired replacement replicas
plus the configured overlap per group must fit. This can reject a deployment
that an incremental rollout would fit, intentionally. Explicit Canary and
MaxParallel settings cannot exceed the reviewed overlap reservation.

An ANDed node-unique-ID regex restricts scheduler placement to exact configured
nodes without removing any operator constraint. Nomad's real dry-run plan
then evaluates placement/constraints, and any failed group or index race is
refused. Capacity reads are not reservations: other jobs can consume resources
after discovery. A successful submission is therefore
`submitted_not_yet_verified`, never a claim of rollout health or guaranteed
placement. Read status/allocations and validate deployment health afterward.

## Unknown outcomes and restart boundary

There are no internal HTTP mutation retries. Once any registration is
attempted, its `(job, index)` is latched for the lifetime of the integration,
including HTTP conflict, timeout, invalid response or other unknown effect.
The consumed plan cannot be reused and another plan at that index is refused.
Do not treat a connection failure as proof that Nomad did nothing.

Plans and attempted-index latches are deliberately in memory; process restart
invalidates all tokens but does **not** provide durable unknown-effect
reconciliation. Operators must inspect remote state and the outer durable
approval/audit records before reviewing any new plan after a restart or
ambiguous response. Never automatically replay a service tool call. Durable
workload journaling is separately provided by `missy-workloads`; this service
resource must not be represented as providing that capability itself.

## Results and validation

`nomad_read` `results` honestly returns `kind: "allocation_metadata"`,
`stdout_available: false`, plus bounded projected allocation status. It is not
stdout, exit-output validation or a task artifact. Owned batch output belongs
to the workload result resource. Regular status/list responses never expose
task definitions or credentials.

`tests/nomad_service.rs` uses fixture-only HTTP sequences for fresh discovery,
capacity shortfall, foreign ownership, dry-run races, concurrent modify-index
changes, scheduler CAS conflict, unknown effects, concurrent token consumption,
restart and stop. No live cluster, account, or running process replacement is
needed for these repository tests.
