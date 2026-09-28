# RepoEval Foundry client boundary

`repoeval_foundry` is a deliberately gated project-scoped tool. It is **not**
a Nomad, shell, repository-write, deployment, credential, or arbitrary HTTP
tool. Existing Foundry code contains a framework-neutral `FoundryAPI.handle()`
route table and an MCP-style facade, **not a running authenticated HTTP server**.
There is no verified endpoint, no discovered capabilities route, and no
confirmed live scan or benchmark. The integration defaults to unavailable.
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
PR1 `FoundryAPI.handle()` wraps responses as
`{"ok": true, "data": ...}`. `list_repositories()` returns a bare `list[str]`,
not repository objects and not a project-tagged document. The client accepts
only unique, validated repository IDs and labels its scope as coming from the
fixed authenticated `/projects/{configured_project}/repositories` route: PR1
checks the principal's project against that route. This is **not** an
independent per-item project assertion. A future HTTP adapter must preserve
that authentication and route check before enabling the tool. PR1 snapshot,
plan, run status and cancellation records contain `project_id`, which the
client requires to equal its configured project; missing or mismatched scope
is refused. PR1 benchmark plans do return the matched workload, but **do not**
return staging placement or seven explicit policy checks. A PR1 plan response
therefore fails, cannot be saved as approved, and cannot authorize start. This is
deliberate: the client cannot infer execution approval from an optimistic HTTP
200, a returned plan ID, repository content, or the caller's own assertion.

When a compatible service exists, configure exactly one project identity and
one endpoint whose hostname appears literally in Foundry `allowed_hosts`; HTTP
is limited to loopback, HTTPS otherwise. Network policy must independently
allow the endpoint through `network.tool_allowed_hosts` (including port as
applicable). Requests use category `tool`; `repoeval_foundry` is not a network
policy category. The policy-aware HTTP client accepts Authorization headers
and does not follow redirects by default. Do not bypass or relax global network
policy. Redirects, arbitrary method/path/header inputs, host suffix matching,
query/fragment credentials and raw provider credentials are not available.
Mutations require an idempotency key matching Foundry's 8-128 character
`[A-Za-z0-9._:-]` grammar, and client output redacts secret-shaped strings
including short `sk-...` values under otherwise innocuous metadata keys.

Action arguments (all other arguments refused):

| Action | Arguments | Behavior |
| --- | --- | --- |
| `capabilities` | none | Fail closed without wire evidence. |
| `list` | none | Read project repository IDs before any scoped scan or plan. |
| `plan` | `workload` | Pinned commit/image, registered repository, and bounded execution; response must contain required server staging placement/policy evidence. |
| `snapshot` | `repository_id`, `commit_sha`, `self_approve: true`, `idempotency_key` | Bounded project-registered snapshot **request** at immutable SHA; acknowledgement is not a completed scan or verified snapshot. |
| `start` | `plan_id`, `self_approve: true`, `idempotency_key` | Requires a previously observed server-reviewed staging plan. A compatible response is only submission acknowledgement, not completion. |
| `status` | `resource_type: run|snapshot`, `resource_id` | Status without assuming a submitted request succeeded. |
| `compare` / `report` | `run_ids` | Fail closed without wire evidence establishing project scope. No HTTP request is sent. |
| `artifacts` | `run_id` | Fail closed without wire evidence establishing project scope. No HTTP request is sent. |
| `cancel` | `run_id`, `idempotency_key` | Project-scoped cancellation; not deletion. |

Self-approval applies only to bounded project scans and benchmarks and does
not override server authorization, provider registry, staging capacity, quotas,
egress, budget, audit, or platform policy. Mutation responses require explicit
acknowledgement, matching project identity and resource evidence. Snapshot
responses saying `requested` are not proof of execution. Start fails closed
without the required reviewed plan. No live backend success is established by
this client. If a request errors or a response lacks identity,
the outcome remains **unknown** and the same request must not be retried under
a new idempotency key; read status first. API response bodies and exceptions
are not echoed as error text. Metadata output is bounded and sensitive keys
and credential-looking strings and URIs are redacted. Exact credential echoes
anywhere in a response are refused, even under innocuous keys. Returned URIs are
never fetched. No test contacts a live endpoint.
