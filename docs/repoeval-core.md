# RepoEval core in Missy

The offline RepoEval core was ported from `repoeval-foundry` commit
`cdc9ee1c3a1b03ba0de2ca62f5b9de6ed16babf8` into **this repository** as
`missy.repoeval`. This is library code and reviewed example data, **not** a
deployed Foundry service. Importing it does not open a socket, initialize a
provider, submit a job, or modify Missy's agent runtime.

## What is present

| Module | Scope |
| --- | --- |
| `contracts`, `schemas/` | Canonical manifests, hashes, comparability and versioned JSON schema/examples. |
| `scanner` | Bounded, local, clean-checkout/exact-commit Git read and static repository inventory. Never runs repository content. |
| `control` | Project-scoped plans, explicit snapshot/image/provider/prompt/fixture/validator allowlists, idempotency, monotonic run states. `MemoryStore` is single-process demo state, not durable authority. |
| `placement`, `nomad` | Capacity selection and bounded, network-disabled batch **plan** generation, no Nomad client or submission. |
| `provider` | Adapter protocol, explicit registry and authorization, request limits, redacted failures. No real provider adapter or configured credential resolver. Calling a supplied adapter can make a provider call, so this module is not an offline-only execution API. |
| `evaluation`, `report` | Deterministic local fixture evaluation and comparison draft, not independent run attestation. |
| `artifacts`, `storage`, `schema.sql` | Artifact manifests, byte/digest/clearance checks, retention semantics, PostgreSQL schema/adapter boundary; no configured database, object store, scanner or migration. |
| `api`, `mcp` | Framework-neutral route and tool facades over an **injected** service/principal; no HTTP listener, authentication provider, MCP transport or Missy tool registration. |
| `offline_demo`, `workloads/` | Checked-in three-class Missy draft catalog; one constant fake-adapter tool-call demonstration with a disposable clean checkout. Catalog definitions are non-executable drafts: providers empty, image host `registry.invalid`, source commit fixed to historical fixture reference. |

Import paths include `from missy.repoeval import FoundryService, FoundryAPI,
scan_repository`, `from missy.repoeval.nomad import plan_job`, and
`from missy.repoeval.offline_demo import run_offline_demo`. The demo may also
be run with `python -m missy.repoeval.offline_demo --checkout /path/to/clean/git/tree
--commit FULL_LOWERCASE_HEAD_SHA` on an **explicit disposable checkout**. Do
not point it at an active working tree. It runs bounded Git reads only. Its
`fixture_scored` result means neither a verified benchmark nor a run artifact;
`benchmark_state` remains `incomplete` and execution is refused without a
dispatcher. The demo's temporary snapshot/image allowlist is synthetic and
must never be treated as an operator-approved production inventory.

`missy.repoeval.nomad.plan_job()` produces an unsubmitted Nomad JSON job
object with capitalized API fields (`TaskGroups`, `Tasks`, `Resources.CPU`,
`Resources.MemoryMB`, and group `EphemeralDisk.SizeMB`). The bounded
`timeout_seconds` becomes the task group's `MaxRunDuration` in nanoseconds,
which is the Nomad-enforced deadline. `FOUNDRY_TIMEOUT_SECONDS` is merely a
worker hint and is **not** an enforcement mechanism. The plan has zero restart
and reschedule attempts; capacity snapshots and pool choice are advisory until
a separately authorized scheduler validates them. No live Nomad registration
or allocation is performed here.

## Missy client boundary and gaps

Missy's existing `missy.tools.builtin.repoeval_tools.RepoevalFoundryTool` is
an independently gated, network-policy-enforced **remote HTTP client**, not
an import of this local library. Its opt-in project/host/token config and
revocation protection remain unchanged. Do **not** point it at this package
or infer that the package provides its configured server. The core facade has
an in-process `FoundryAPI.handle()` contract and the client implements only
bounded list/plan/snapshot/start/status/cancel behavior over an explicitly
available external endpoint; compare/artifacts/report still fail closed in
the client pending project-bound wire contracts. Local schemas and fixtures
do not authorize execution, new networks, credentials, job submission, or
changing Missy's security configuration.

**Client/core compatibility demonstrated offline:**
`tests/repoeval/test_client_core_contract.py` wires the actual
`RepoevalFoundryTool` to the actual `FoundryAPI.handle()` via a closed,
in-memory HTTP-shaped transport and a disposable fixture-only token file.
It proves project-scoped listing, snapshot **request** and snapshot status
round-trip end-to-end through the client's request validation and response
checks using the bundled catalog's actual `MissyLabs/missy` repository ID,
then verifies that a core plan cannot authorize the client's `start`:
the core response lacks staging placement and the seven explicit policy
checks. No network endpoint, provider or scheduler is contacted. A negative
test establishes that slash-containing URLs, encoded traversal, extra path
segments and traversal names are refused as repository IDs. Only repository
identity admits exactly `owner/repo` (or a legacy single ID); all other
resource IDs still refuse slashes. Missing plan attestations need separate
review before attempting real client/core integration. Do not bypass the
client's checks, invent policy evidence, or treat a snapshot request as a
verified snapshot.

Missing for live operation: a separately reviewed and authorized transport
and authentication layer; real per-project snapshot attestations and approved
executable images; persistent transactional dispatch/outbox and a trusted
Nomad worker; provider adapters/secret custody/budget metering; independent
validator and artifact attestations; production database migrations and
retention reconciliation; and client/server wire-contract tests. The
`FoundryService` live-dispatch gate defaults off. Even if manually opted into
demo dispatch with a caller-supplied dispatcher, its memory store is not a
safe production queue. No service deployment or live integration is part of
this port.

Install package data with `pip install -e '.[repoeval]'` (or install with
`.[dev]` for tests). The `jsonschema` dependency is explicitly optional for
evaluation; other modules do not require it. Run local regression tests:

```bash
python3 -m pytest -q tests/repoeval/
```
