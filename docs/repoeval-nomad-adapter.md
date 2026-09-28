# RepoEval Nomad Scheduler adapter

`missy.repoeval.nomad_adapter.NomadSchedulerAdapter` is a standalone opt-in
implementation of `dispatch.Scheduler`. Nothing constructs or enables it by
default. It takes an **existing authenticated** `NomadClient`, an operator-owned
immutable reservation resolver `(namespace, job_id) -> exact_job`, a reviewed
image digest, and `enabled=True`. Do not provide a resolver that accepts a
caller-supplied job as its own evidence. The enclosing durable dispatcher still
requires trusted project authorization and commits an attempt marker **before**
each scheduler mutation. This adapter neither grants project permissions nor
commits an outbox transaction, and it is not connected to a service here.

Every call enforces both the client policy and this module's finite fixed scope:
namespace `sandbox`, datacenter `dc1`, pool `staging`, Docker batch worker, no
network, read-only root, dropped capabilities, no command/args, volumes,
templates, arbitrary environment or secrets. The supplied job must exactly
equal a freshly regenerated `repoeval.nomad.plan_job` template using the
operator-reviewed image digest. **`submit` always refuses to mutate Nomad, even
when enabled**: the worker lacks sealed snapshot mount/custody, an approval
receipt, and a credential broker. These are hard execution prerequisites, not
things an image digest or permitted pool can substitute for. If later
implemented, use create-only CAS, keep durable no-replay attempt markers and
treat mutation timeouts as unknown rather than retries.

`lookup` requires an exact effective inspected JSON job, admitting only
`CreateIndex`, `ModifyIndex`, `JobModifyIndex` as extra top-level server fields;
unknown defaults fail closed until separately reviewed. It also checks
namespace/ID/create index in Nomad status and requires matching allocation
evidence before reporting terminal complete/failed. It does **not** validate
artifacts or mark a run verified. The existing CLI client cannot distinguish
read failure from not-found reliably, so this adapter propagates read errors
instead of returning a misleading `None`. Real Nomad's effective inspect
response may include other default fields; fake tests passing is not a live
compatibility proof. Never weaken comparison by dropping arbitrary fields.

`stop` requires an independently persisted expected Nomad `CreateIndex`, via
`expected_create_index(namespace, job_id)`. Without that evidence it refuses to
stop. This must be bound to the original immutable submission, not learned
from the job visible during the current stop request. The current SQLite
outbox does **not** persist such an index. Moreover the existing
`NomadClient.stop_job` has no atomic create-index condition: another actor can
replace an ID between the adapter's last read and `stop_job`. **`stop` refuses
to mutate Nomad** until a reviewed implementation has an atomic
incarnation-fenced mutation (or an equivalent externally enforced exclusive
job-ID ownership guarantee); it cannot be made safe by adding another read.
Do not treat these tests as an authorization to mutate the cluster. No purge
is performed.

Tests use a fake subclass of `NomadClient` whose constructor bypasses CLI
configuration and credentials; they never contact a cluster. They exercise
disabled-by-default, refusal of both mutations, placement/template/secret
injection rejection, wrong identity and incarnation, terminal uncertainty,
and missing reservation.
