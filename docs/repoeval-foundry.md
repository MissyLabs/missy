# RepoEval Foundry client boundary

**Repository ownership:** `MissyLabs/missy` is the first repository under
evaluation, not the home of the Foundry implementation. Foundry's coordinator,
API server, workload catalog, scheduler adapter, deployment packaging, and core
tests belong in `MissyLabs/repoeval-foundry`. This repository contains only an
opt-in Missy HTTP client, its configuration/hot-reload/gateway integration, and
client-side contract tests. This is code only: no API deployed, credentials
provisioned, client enabled, or evaluation run.

`repoeval_foundry_read` and `repoeval_foundry_mutate` are distinct, deliberately gated
project-scoped tools. The read tool supports list, plan, status, compare, artifacts,
and draft report. The mutation tool alone supports snapshot, start, and cancel.
Granting the read tool through Missy tool policy cannot dispatch a mutation;
`writes_state` is descriptive metadata, not a policy enforcement boundary. Neither is
a Nomad, shell, repository-write, deployment, credential, or arbitrary HTTP
tool. The separate Foundry repository owns the server contract; neither a live
endpoint nor a completed scan or benchmark has been established by these
client tests. The integration defaults to unavailable.
Configuration supports an opt-in authenticated bridge using a protected token
file. `api_available` is only an operator assertion, not a health check; this
document does not claim a real API is deployed or reachable. Registration
separately validates the credential path and fails closed if the authenticated
transport or any prerequisite is invalid.

The config integration must default to:

```yaml
repoeval_foundry:
  enabled: false
  base_url: ""
  project_id: ""
  allowed_hosts: []
  token_file: ""
  api_available: false
```

`enabled` and `api_available` must be YAML booleans. Both true requires a
nonempty `token_file` with a normalized absolute path: no `~`, `.`, `..`, `//`,
whitespace/control characters, or backslashes. Parsing validates path syntax only.
Before registration, the secure helper validates metadata only: a current-UID
regular file, mode `0600`, one hard link, no symlink at the file or any path
component, and a size of 1..4096 bytes. Each request securely reopens and
revalidates the file before reading; contents must match the ASCII bearer-token
grammar `[A-Za-z0-9._~+/-]+=*`, optionally ending in one LF.
Never copy the token into YAML, logs, or raw tool arguments. Load it per request
and protect its parent directories.

Hot reload revokes the existing client if `enabled` or `api_available` becomes
false or if `base_url`, `project_id`, `token_file`, or `allowed_hosts` changes.
Revocation also blocks callers holding a reference to the old tool, even if it
remains in the registry. An unchanged configuration keeps the current client
working. Reload does not grant a new endpoint or credential automatically:
after revocation, enabling a different Foundry identity requires a process
restart. Other tools are unaffected. In-flight requests complete before the
revocation takes effect; calls after the reload completes cannot use the old
client.

Before enabling, an operator must independently validate an authenticated
Foundry HTTP service, its explicit project-bound response contract, route
mapping, and policy evidence. The endpoint accepts only an exact root or `/api`
base (optionally with trailing `/`), a valid hostname, and a valid nonzero port.
The adapter must expose `/projects/{id}` or
`/api/projects/{id}`, never `/api/v1/projects/{id}`. Configure only the origin
or `/api` prefix; the client appends the project-scoped resource route.
The pinned client wire contract wraps responses as
`{"ok": true, "data": ...}`. `list_repositories()` returns a bare `list[str]`,
not repository objects and not a project-tagged document. The client accepts
only unique, validated repository IDs (a single legacy segment or exactly
`owner/repo`, each segment bounded to 128 ASCII characters beginning with an
alphanumeric and otherwise containing only alphanumerics, `.`, `_`, `-`). Dot
segments, escapes, URLs and extra slashes are refused without normalization.
This repository-only grammar does not extend to plan, run, snapshot, project or
other resource IDs. Repository IDs travel in JSON bodies, not URL path
segments. The client labels its scope as coming from the
fixed authenticated `/projects/{configured_project}/repositories` route. The
separate server must check the principal's project against that route. This is
**not** an independent per-item project assertion. The HTTP server must preserve
that authentication and route check before enabling the tool. Snapshot,
plan and run-status records contain `project_id`, which the client requires
to equal its configured project; missing or mismatched scope is refused.
Cancellation acknowledgement v1 intentionally has no `project_id`; the client
binds it to the authenticated project route and matches the returned run ID.
Plans require a trusted staging capacity and policy attestation
provider; without it, planning fails closed. A plan without matched workload,
staging placement and seven explicit policy checks cannot authorize start.
The client cannot infer execution approval from HTTP 200, a plan ID,
repository content, or the caller's own assertion.

When a compatible service exists, configure exactly one project identity and
one endpoint whose hostname appears literally in Foundry `allowed_hosts`; HTTP
is limited to loopback, HTTPS otherwise. Network policy must independently
allow the endpoint through `network.tool_allowed_hosts` (including port as
applicable). Requests use category `tool`; `repoeval_foundry` is not a network
policy category. The policy-aware HTTP client accepts Authorization headers
and does not follow redirects by default. Do not bypass or relax global network
policy. Redirects, arbitrary method/path/header inputs, host suffix matching,
query/fragment credentials and raw provider credentials are not available.
Snapshot/start requests require an idempotency key matching Foundry's 8-128 character
`[A-Za-z0-9._:-]` grammar, and client output redacts secret-shaped strings
including short `sk-...` values under otherwise innocuous metadata keys.
Cancellation is idempotent by project-scoped run ID, with no ignored key sent.

Action arguments (all other arguments refused):

| Action | Arguments | Behavior |
| --- | --- | --- |
| `list` | none | Read project repository IDs before any scoped scan or plan. |
| `plan` | `workload` | Pinned commit/image, registered repository, and bounded execution; response must contain required server staging placement/policy evidence. |
| `snapshot` | `repository_id`, `commit_sha`, `acknowledge_project_scope: true`, `idempotency_key` | Bounded project-registered snapshot **request** at immutable SHA; acknowledgement is not a completed scan or verified snapshot. |
| `start` | `plan_id`, `acknowledge_project_scope: true`, `idempotency_key` | Requires a previously observed server-reviewed staging plan. A verified `reserved` response acknowledges only durable reservation, not scheduler submission or completion; a valid legacy `submitted` response acknowledges submission, not completion. |
| `status` | `resource_type: run|snapshot`, `resource_id` | Status without assuming a submitted request succeeded. |
| `compare` | `run_ids` (2..16 unique) | Fixed POST `/compare`; verified run manifests, `project_id`, stable `comparison_id`, exact run IDs, bounded pairwise comparability and reasons. |
| `artifacts` | `run_id` | Fixed GET `/runs/{run_id}/artifacts`; verified run and trusted independently cleared, digest-checked bytes before bounded metadata only. No raw bytes or URI. |
| `report` | `run_ids` (1..16 unique) | Fixed POST `/report`; verified run manifests, stable `report_id`, `project_id`, exact run IDs, bounded comparability groups, `status: draft`, `published: false`. Never publishes. |
| `cancel` | `run_id` | Project-scoped cancellation, idempotent by run ID; not deletion. |

The `acknowledge_project_scope` flag is merely a caller acknowledgement of the
bounded request, **not** authorization or self-approval. The authenticated
Foundry server permission remains authoritative and this flag cannot override
server authorization, provider registry, staging capacity, quotas,
egress, budget, audit, or platform policy. Mutation responses require explicit
acknowledgement, matching project identity and resource evidence. Snapshot
responses saying `requested` are not proof of execution. Start fails closed
without the required reviewed plan. No live backend success is established by
this client. If a request errors or a response lacks identity, the outcome
remains **unknown** and the same request must not be retried under a new
idempotency key; read status first. The client reports uncertain mutation
rather than inventing execution. The
coordinator's HTTP 202 `reserved` start must carry the exact project and plan,
idempotency-key-derived parent ID, null parent job ID, and a bounded complete
set of uniquely indexed reserved children with derived IDs and preassigned
exact job IDs. Reserved children have not been submitted to the scheduler.
This evidence does not claim dispatch. A 202 cancellation response with matching
project and run identity and state `cancel_pending`, `cancelled`, or `failed`
acknowledges that the request was accepted, not terminal success or deletion.
The versioned cross-repository contract fixture is
`schemas/fixtures/cancellation-contract.json` in both repositories.
Legacy `submitted` with a nonempty valid
job ID remains accepted, but neither form marks `execution_complete` true.
API response bodies and exceptions
are not echoed as error text. Metadata output is bounded and sensitive keys
and credential-looking strings and URIs are redacted. Exact credential echoes
anywhere in a response are refused, even under innocuous keys. Returned URIs are
never fetched. No test contacts a live endpoint.

Read-only result contracts reject missing/mismatched `project_id`, unrelated
resource IDs, duplicate/foreign runs, artifact URI or object data, unbounded
fields and publication claims. Run manifest checks require verified state,
persisted canonical SHA-256 digest, exact project/run/plan/workload identity,
and a recomputed comparability key. Artifact metadata alone is not clearance:
the trusted service must fetch bytes through a project/run/artifact-keyed
reader, verify length and digest against the manifest, then obtain independent
clearance for those exact bytes. Without these injected capabilities, artifact
reads fail closed even for an empty artifact list. Reports are draft summaries,
not raw worker output or publication authority.
