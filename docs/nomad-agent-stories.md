# Nomad Agent Stories

This document defines the intended outcomes for adding Nomad support to Missy.
It is a product contract for review before implementation, not a promise that
the capabilities described below already exist.

## Product Goal

Missy should be able to turn a user's workload request into a safely placed,
observable Nomad workload. Initially, this supports long-running services.
The longer-term goal is to let Missy offload finite, high-latency work to
additional compute, including scheduled tasks and repeatable benchmark runs.

Missy should choose placement from live evidence, operate only on workloads she
owns, and report actual scheduler and application state instead of treating a
successful submission as a successful workload.

## Terminology

- A **Nomad job** is a workload submitted to the Nomad cluster. It may be a
  long-running `service` job or a finite `batch` job.
- A **Missy scheduled job** is an instruction run by Missy's existing
  APScheduler-based scheduler. It may eventually submit or monitor a Nomad job.
- An **owned job** is a Nomad job created by Missy with stable ownership
  metadata and a retained local operation record. A `missy-` name alone does
  not establish ownership.
- **Placement** is the choice of authorized namespace, node pool, datacenter,
  constraints, and resource requests. It is not merely the choice of a host.

## Guiding Constraints

- Credentials remain outside source control, prompts, job specifications,
  logs, and chat output. Live access requires a completed credential bundle;
  the preparation bundle alone is not sufficient.
- Nomad ACL permissions are a technical ceiling, not blanket authorization.
  Missy manages only owner-authorized workloads and never treats broad token
  access as permission to administer the cluster.
- Missy does not autonomously change ACLs, namespaces, node pools, node
  eligibility or drain state, scheduler configuration, cluster membership,
  PKI, VPN routes, firewalls, DNS, shared ingress, host services, or driver
  hardening.
- Job specifications, image metadata, task output, logs, and web content are
  untrusted data. They cannot expand Missy's authority or alter these rules.
- Initial workloads use the Docker driver, explicit resources, non-root users,
  bounded retry and log policies, and approved images. Production images are
  pinned by digest.
- Destructive actions require proof that the target job is owned by Missy.
  Purging history or deleting persistent data is never an implicit cleanup
  step.

## Phase 1: Access, Discovery, and Placement

### NOMAD-001 — Establish a usable Nomad identity

**Agent story:** As Missy, I need to verify my own TLS identity and Nomad ACL
access before operating on workloads so that I fail safely when my credentials
are incomplete, expired, revoked, or insufficient.

**Acceptance criteria:**

- Missy refuses live operations when the certificate, key, CA, or ACL token is
  absent, unreadable, expired, mismatched, or rejected.
- Certificate verification remains enabled; Missy never uses a TLS-skip option.
- A denied operation identifies the capability and namespace that were denied
  without exposing credential material.
- Missy does not borrow an operator, server, bootstrap, or another agent's
  credentials.

### NOMAD-002 — Discover authorized cluster state

**Agent story:** As Missy, I need to inspect the namespaces, node pools,
datacenters, eligible nodes, drivers, and existing allocations visible to my
identity so that decisions use current cluster state rather than recorded
topology.

**Acceptance criteria:**

- Discovery is read-only and timestamped.
- Results distinguish data that is observed, unavailable, and denied.
- Node records include readiness, eligibility, drain state, pool, datacenter,
  architecture, Docker health, scheduler resources, and relevant allocations
  when the API makes them available.
- Missy never interprets an empty or denied namespace listing as proof that no
  namespaces exist.

### NOMAD-003 — Recommend workload placement

**Agent story:** As Missy, I need to compare a workload's resource,
architecture, persistence, network, and lifecycle requirements with live node
state so that I can explain where the workload should run and why.

**Acceptance criteria:**

- The recommendation names an authorized namespace, node pool, datacenter,
  constraints, resource requests, and eligible candidate nodes.
- Capacity calculations account for all tasks in the group, group count,
  existing reservations, ephemeral disk, and temporary rollout overlap.
- Scheduler capacity is distinguished from observed host utilization; missing
  host data is disclosed rather than invented.
- Placement considers CPU architecture, driver availability, persistent-data
  locality, network reachability, and control-plane headroom—not CPU and memory
  alone.
- Missy reports ambiguity or insufficient capacity instead of weakening
  constraints, evicting other workloads, or guessing.

## Phase 2: Owned Long-Running Workloads

### NOMAD-004 — Draft a safe service job

**Agent story:** As Missy, I need to translate an approved long-running
workload into a reviewable Nomad service specification so that its behavior,
resource use, placement, and recovery policy are explicit.

**Acceptance criteria:**

- The job has a unique `missy-<purpose>` ID plus ownership and provenance
  metadata.
- The specification explicitly defines namespace, pool, datacenter,
  constraints, image, command, non-root identity, resources, logs, networking,
  health checks, restart behavior, rescheduling, and update strategy as
  applicable.
- Secrets are referenced through an approved runtime mechanism and are never
  embedded in the specification or operation journal.
- Stateful jobs require an explicit persistence, backup, restore, placement,
  and migration plan before submission.
- Required ingress, firewall, DNS, or shared-service changes are reported as
  external prerequisites and are not performed implicitly.

### NOMAD-005 — Validate and plan before submission

**Agent story:** As Missy, I need to validate and plan the exact job
specification before submitting it so that syntax, policy, placement, and
concurrent-update failures are visible before cluster state changes.

**Acceptance criteria:**

- Missy validates the exact artifact she intends to submit.
- Missy runs a placement plan, understands the installed CLI's plan exit-code
  semantics, and captures the plan result and check index.
- The plan summary identifies allocations created, replaced, stopped, or left
  unplaced and explains failed placement.
- A changed job index causes Missy to inspect and re-plan rather than force an
  overwrite.
- Validation or planning failure prevents submission.

### NOMAD-006 — Submit an owned service safely

**Agent story:** As Missy, I need to submit a validated service job with
optimistic concurrency so that I can create or update my workload without
overwriting another operator's changes.

**Acceptance criteria:**

- New jobs use a zero check index; updates use the index from the reviewed
  plan.
- Before an update, Missy verifies ownership from metadata and her operation
  record, not from the job name alone.
- Missy retains the reviewed source specification and non-secret provenance
  needed to reproduce or roll back the change.
- Submission records the resulting evaluation and deployment identifiers.
- A successful API response is reported as “submitted,” not “healthy.”

### NOMAD-007 — Observe service rollout and health

**Agent story:** As Missy, I need to follow evaluations, deployments,
allocations, task events, and health checks so that I can report whether the
service actually became healthy and reachable.

**Acceptance criteria:**

- Missy waits for a bounded terminal or stable state and reports timeouts as
  incomplete outcomes.
- Reports distinguish scheduling, image-pull, driver, process, health-check,
  and network-reachability failures.
- Logs are bounded and redacted before being stored or shown.
- The final report includes namespace, job ID, specification provenance, image
  digest, placement, resources, plan/check index, evaluation, allocation and
  deployment IDs, health, endpoint availability, and rollback status.

### NOMAD-008 — Operate an owned service

**Agent story:** As Missy, I need to inspect, restart, scale, and update an
owned service so that routine lifecycle work remains safe and attributable.

**Acceptance criteria:**

- Every mutation repeats ownership verification and uses current job state.
- Updates repeat validation, planning, concurrency checks, and rollout
  observation.
- Scaling includes the additional resource demand and rollout overlap in its
  placement analysis.
- Missy does not exec into tasks or alter another operator's job without a
  specific owner request.
- Repeated failures stop with a diagnosis instead of causing an unbounded
  restart or resubmit loop.

### NOMAD-009 — Stop or clean up an owned service

**Agent story:** As Missy, I need to stop an owned service without destroying
unrelated history or data so that cleanup is deliberate and recoverable where
possible.

**Acceptance criteria:**

- Missy verifies ownership and identifies persistent data before stopping a
  job.
- Stopping, purging job history, and deleting application data are treated as
  separate actions with separate consequences.
- Purge is limited to explicitly disposable owned jobs when the user has
  authorized cleanup.
- Missy never runs global garbage collection or host-level cleanup as a
  substitute for job cleanup.

## Phase 3: Finite Work and Additional Compute

### NOMAD-010 — Submit a one-shot compute task

**Agent story:** As Missy, I need to package a finite unit of work as a Nomad
batch job so that a high-latency operation can run on appropriate additional
compute without tying up the interactive agent process.

**Acceptance criteria:**

- The request defines inputs, expected outputs, resource limits, maximum
  duration, retry policy, idempotency expectations, and cleanup policy.
- Missy selects placement using the same live discovery and planning rules as
  service jobs.
- The batch job has bounded retries and a clear success exit condition.
- Inputs and outputs use an explicitly approved transport or storage location;
  allocation-local disk is not treated as durable artifact storage.
- Missy can cancel her own running task and accurately distinguish cancelled,
  timed out, failed, and completed outcomes.

### NOMAD-011 — Collect results from a one-shot task

**Agent story:** As Missy, I need to collect and validate a batch task's
result so that downstream reasoning uses the produced artifact rather than the
fact that Nomad accepted the job.

**Acceptance criteria:**

- Completion requires a successful allocation exit plus any expected artifact
  or structured result validation.
- Missy records immutable input identifiers, image digest, parameters,
  placement, timestamps, exit status, and output identifiers for
  reproducibility.
- Output and logs are treated as untrusted content and pass through the normal
  security and redaction boundaries before reaching the agent context.
- Cleanup occurs only after required results are retained and verified.

### NOMAD-012 — Offload a high-latency tool operation

**Agent story:** As Missy, I need a controlled way to map an eligible
high-latency tool request to a batch workload so that the conversation remains
responsive while additional compute performs the work.

**Acceptance criteria:**

- Only explicitly supported tool workloads can be offloaded; arbitrary tool
  calls are not converted into cluster jobs.
- The user receives a durable task identifier and can ask for status, result,
  cancellation, or failure details later.
- Interactive execution and offloaded execution preserve equivalent policy,
  input validation, output validation, and audit requirements.
- Duplicate requests use an idempotency key or are clearly identified as
  separate executions.
- Loss or restart of the Missy process does not orphan result tracking.

## Phase 4: Scheduling and Benchmarks

### NOMAD-013 — Schedule a future batch submission

**Agent story:** As Missy, I need to let an existing Missy scheduled job
submit and monitor a finite Nomad batch job so that expensive work can run at a
specified time or cadence on suitable compute.

**Acceptance criteria:**

- The schedule record clearly distinguishes the Missy schedule ID from each
  Nomad execution's job, evaluation, and allocation IDs.
- Pausing the schedule prevents future submissions without silently cancelling
  work already running.
- Overlap behavior is explicit: skip, queue, replace, or allow concurrent runs.
- Each firing performs fresh discovery and planning rather than reusing stale
  placement assumptions.
- Missy records scheduled time, actual submission time, completion time, and
  final outcome.

### NOMAD-014 — Run a reproducible benchmark

**Agent story:** As Missy, I need to execute a benchmark definition as one or
more finite Nomad jobs so that results across repositories, models, images, or
hardware can be compared fairly.

**Acceptance criteria:**

- A benchmark definition pins workload version, image digest, inputs,
  parameters, resource requests, timeout, warm-up policy, repetitions, and
  result schema.
- Runs capture node identity and relevant hardware/architecture metadata.
- Parallelism has an explicit upper bound derived from authorized capacity and
  does not starve protected workloads.
- Partial failures remain visible and are not silently removed from aggregate
  results.
- The result set preserves per-run provenance and identifies comparisons that
  are invalid because execution conditions differed.

### NOMAD-015 — Reconcile interrupted work

**Agent story:** As Missy, I need to reconcile locally tracked work with Nomad
after a restart or connectivity loss so that I do not duplicate tasks, lose
results, or misreport stale state.

**Acceptance criteria:**

- On recovery, Missy queries Nomad by recorded identifiers and ownership
  metadata before taking action.
- Running work is reattached to monitoring; terminal work is finalized once.
- Unknown, missing, or externally modified jobs are surfaced for review rather
  than recreated automatically.
- Reconciliation is idempotent and auditable.

## Cross-Cutting Agent Stories

### NOMAD-016 — Explain and audit every operation

**Agent story:** As Missy, I need to keep a non-secret operation journal so
that users can understand and reproduce what I decided and what the cluster
actually did.

**Acceptance criteria:**

- Discovery, placement, validation, plan, submission, mutation, observation,
  cancellation, and cleanup emit structured audit events.
- The journal includes job IDs, namespaces, source-spec identifiers, image
  digests, plan results, check indexes, scheduler identifiers, and outcomes.
- Credentials, raw secret-bearing specifications, and unredacted logs are
  excluded.
- User-facing responses distinguish confirmed facts, recommendations,
  assumptions, denied visibility, and required external actions.

### NOMAD-017 — Preserve policy boundaries

**Agent story:** As Missy, I need Nomad operations to pass through the same
policy and security controls as interactive and scheduled work so that cluster
access does not become a general-purpose escape hatch.

**Acceptance criteria:**

- Nomad commands or API calls are exposed through narrowly scoped operations,
  not unrestricted shell text supplied by a model or user-controlled content.
- Namespace, ownership, action, image, resource, and workload restrictions are
  checked before each mutation.
- Tool output cannot authorize follow-up actions or change credential and
  infrastructure policy.
- Security denials are visible and do not trigger attempts to bypass the
  management proxy, ACLs, TLS, client hardening, or network policy.

## Proposed Delivery Order

1. **Foundation:** NOMAD-001 through NOMAD-003, NOMAD-016, and NOMAD-017.
2. **Long-running MVP:** NOMAD-004 through NOMAD-009.
3. **Finite compute:** NOMAD-010 through NOMAD-012 and NOMAD-015.
4. **Scheduled and benchmark workloads:** NOMAD-013 and NOMAD-014.

Each stage should include unit tests with a fake Nomad API, policy-denial tests,
and integration tests against an isolated namespace before live workload use.
Destructive lifecycle tests must use disposable jobs created by the test run.

## Decisions to Confirm Before Implementation

1. Which ACL policy will the completed Missy identity use: `coadmin` or the
   narrower `agent` workload policy?
2. Which namespaces, node pools, and datacenters are owner-authorized defaults,
   and which must always be stated by the user?
3. Which container registries and image-signing or image-approval rules are
   acceptable for the first release?
4. Where should batch inputs, durable outputs, and benchmark artifacts live?
5. Which high-latency Missy tools are eligible for the first offloading
   integration?
6. Should autonomous service deployment require a user-visible plan
   confirmation, or is a clean plan within defined limits sufficient?
7. What default CPU, memory, disk, wall-time, retry, concurrency, and spend
   limits should apply to service, batch, scheduled, and benchmark workloads?
8. What notification channel should Missy use for asynchronous completion,
   failure, or required intervention?

## Explicit Non-Goals

- Turning Missy into a Nomad cluster administrator.
- Managing workloads merely because the ACL token can see them.
- Automatically modifying networking, DNS, ingress, storage, or host
  configuration to make a job work.
- Treating Nomad submission, allocation placement, process start, application
  health, and user reachability as equivalent states.
- Providing arbitrary remote shell or unrestricted container execution.
- Using ephemeral allocation storage as an implicit artifact store.
- Hiding failed or incomparable benchmark runs to make aggregate results look
  successful.
