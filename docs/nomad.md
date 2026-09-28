# Nomad Workload Orchestration

Missy's Nomad integration discovers live cluster capacity, plans exact job
artifacts, and manages only workloads whose protected local record matches
their remote ownership metadata. It supports long-running Docker services,
finite batch tasks, scheduled submissions, operator-approved asynchronous
offloads, and bounded benchmark run sets.

The integration is disabled by default. A broad Nomad ACL policy is only a
technical ceiling: it does not authorize Missy to administer the cluster or
operate workloads she does not own.

## Safety model

- TLS verification is mandatory. The CLI receives the ACL token through a
  minimal subprocess environment; tokens and private keys never enter job
  specifications, prompts, tool output, or the operation journal.
- Every mutation is constrained to configured namespaces, node pools, and
  datacenters. Empty allowlists deny mutation.
- Images are digest-pinned by default and can be restricted to approved
  registries. Jobs run as non-root with Linux capabilities dropped.
- Planning uses the exact JSON artifact later submitted, with Nomad's check
  index for optimistic concurrency. Plans expire and cannot be submitted twice.
- Updates and scale operations require matching local and remote ownership,
  fresh discovery, validation, placement, and planning.
- Job output is bounded and redacted. Batch success additionally requires a
  successful allocation plus configured marker, structured-result, and output
  artifact checks.
- Missy never manages ACLs, namespaces, node pools, drains, raft state, PKI,
  host services, firewalls, DNS, shared ingress, or global garbage collection.

## Credential bundle

Place the completed bundle files in a private directory (normally
`~/.missy/nomad`) with directory mode `0700`. `missy-key.pem` and
`missy.token` must have mode `0600`. The required files are:

- `ca.pem`
- `missy.pem`
- `missy-key.pem`
- `missy.token`

Missy rejects missing files, symlinks, path escapes, insecure secret modes,
invalid or expired certificates, a certificate/key mismatch, an unexpected
certificate common name, and malformed tokens. Do not copy bundle contents
into the repository or YAML configuration.

## Configuration

The example below remains inert until its owner-specific scope and approved
images/transports are filled in:

```yaml
network:
  default_deny: true
  allowed_hosts:
    - "nomad.example.com:443"

filesystem:
  allowed_read_paths:
    - "~/.missy/nomad"
  allowed_write_paths:
    - "~/.missy/nomad-state"

shell:
  enabled: true
  allowed_commands: ["nomad"]

nomad:
  enabled: true
  address: "https://nomad.example.com"
  bundle_dir: "~/.missy/nomad"
  state_dir: "~/.missy/nomad-state"
  binary: "nomad"
  identity_cn: "missy"
  owner: "missy-bot"

  allowed_namespaces: ["owner-authorized-namespace"]
  allowed_node_pools: ["owner-authorized-pool"]
  allowed_datacenters: ["dc1"]
  default_namespace: "owner-authorized-namespace"
  default_node_pool: "owner-authorized-pool"
  default_datacenter: "dc1"

  approved_registries: ["registry.example.com"]
  allowed_job_commands: ["/app/check"]
  approved_artifact_prefixes: ["artifact://missy/"]
  approved_secret_reference_prefixes: ["nomad-var://missy/"]
  protected_node_names: ["control-plane-host"]
  require_image_digest: true

  max_cpu_mhz: 8000
  max_memory_mb: 16384
  max_disk_mb: 20480
  max_group_count: 8
  max_parallel_jobs: 4
  max_retry_attempts: 2
  max_wall_time_seconds: 86400
  max_benchmark_runs: 32
  max_benchmark_parallelism: 4
  plan_ttl_seconds: 900
  request_timeout_seconds: 60
  allow_purge: false

  workload_templates:
    repository-check:
      description: "Run the reviewed repository benchmark image."
      allowed_parameters: ["repository", "mode"]
      required_parameters: ["repository"]
      request:
        purpose: "repository check"
        image: "registry.example.com/missy/repository-check@sha256:REPLACE_WITH_64_HEX"
        command: "/app/check"
        cpu_mhz: 1000
        memory_mb: 2048
        disk_mb: 2048
        max_run_seconds: 3600
        retry_attempts: 1
        expected_output: "benchmark complete"
```

The general service/batch tools accept a structured job request. The offload
and benchmark tools are narrower: the caller can only select an
operator-defined `workload_templates` entry and supply its allowlisted
parameters plus artifact identifiers under an approved prefix. They cannot
inject an image, command, or arbitrary tool call.

Secret values are never accepted in ordinary environment fields. A reviewed
request may instead map a secret-shaped environment name to an allowlisted
`nomad-var://path#key` reference. Missy renders a Nomad template that resolves
the value inside the allocation; the journal and job specification retain
only the reference.

## Tool workflow

1. `nomad_discover` reads timestamped namespaces, pools, nodes, drivers,
   reservations, and host pressure visible to the configured identity.
2. `nomad_recommend_placement` explains candidates and capacity without
   changing cluster state.
3. `nomad_plan_job` builds, validates, plans, and stores an exact artifact.
4. `nomad_submit_job` submits that unexpired artifact exactly once. Its result
   says submitted, not healthy.
5. `nomad_job_status` observes evaluation, deployment, allocation, task-event,
   and health state. `nomad_job_result` verifies finite work and collects
   bounded redacted logs.
6. `nomad_plan_scale` creates a reviewed service update; it is submitted with
   `nomad_submit_job`. `nomad_job_action` handles owned restart, stop, cancel,
   and separately gated purge operations.
7. `nomad_offload_task` exposes plan/submit/status/result/cancel around a
   durable task ID and an approved workload template.
8. `nomad_schedule` creates, lists, pauses, resumes, or removes schedules that
   perform fresh discovery and planning on every firing.
9. `nomad_benchmark` plans/starts/observes/collects/cancels a bounded set of
   warm-up and measured runs. Every run remains visible; comparisons are
   marked invalid when measured runs fail, remain incomplete, or execute under
   different observed node conditions.
10. `nomad_reconcile` reattaches durable local records to remote state after a
    process restart or ambiguous mutation. It never recreates missing jobs.

## Batch result protocol

Artifact identifiers are references to owner-approved durable storage, not
allocation-local paths. Missy passes declared inputs, outputs, and parameters
as `MISSY_INPUT_ARTIFACTS_JSON`, `MISSY_OUTPUT_ARTIFACTS_JSON`, and
`MISSY_PARAMETERS_JSON` environment variables.

For structured validation, the workload writes one bounded log line:

```text
MISSY_RESULT_JSON={"score":0.98,"outputs":{"result":"artifact://missy/run/result.json"}}
```

The `outputs` values must exactly match the declared output identifiers, and
configured required fields/types must validate. A completed allocation without
valid results is reported as `result_validation_failed`, not success.

## State and recovery

The private journal at `nomad.state_dir/operations.json` retains exact plans,
source requests, ownership tracking IDs, task and benchmark indexes, scheduler
identifiers, and non-secret provenance. Writes are locked, atomic, mode `0600`,
and size-bounded. Back up this directory with the rest of Missy's private
state; losing it intentionally prevents Missy from mutating jobs whose
ownership she can no longer prove.
