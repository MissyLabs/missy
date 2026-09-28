# RepoEval worker image, design only

**Not deployable yet.** `entrypoint.main()` explicitly refuses without a trusted
custody store, independent digest approval verifier, and provider broker. No
production transport, store adapter, credential custody, or image digest is
provided. The existing `nomad.py` emits only a plan; it does not submit jobs.

The only accepted job input is a 64-character lowercase SHA-256 manifest
digest (optional final newline) at `/run/repoeval/input/manifest.sha256` on a
sealed read-only mount. It is not a URL, JSON blob, path, or job-spec credential.
A trusted custody provider must return canonical manifest bytes for that exact
digest, byte payloads, and independently approved project/provider/budget/timeout
bindings. A separate verifier must authenticate approval of the exact digest
and current project rights. The provider broker must hold secrets outside the
job spec and enforce model, project and cost authorization. All of these
backends are currently **unavailable**, intentionally. A test fixture may
inject them; a production integration must not permit arbitrary plugins/env
module names, user-defined URLs, or caller-supplied approval lambdas.

Result is a single JSON report, without raw prompt/response/credentials, at
`/run/repoeval/result/result.json`. Existing file means refusal, not retry or
overwrite. The result mount must be exclusively writable by the worker,
bounded in size/inodes, and collected securely; stdout is not a transport.
Exit 0 means the validator passed; exit 2 means refused, failed, or result
could not be written. A result alone does not prove a test/patch ran.

The Dockerfile requires a **real verified digest** supplied as `BASE_IMAGE`
at build time. Resolve it from your approved registry and verify its signature,
architecture and security updates; record that digest, build inputs and built
image digest in the approval record. The approved base must contain reviewed,
versioned Python and `jsonschema` (including its dependencies); the build fails
if missing and never installs unpinned dependencies over the network. No digest
is invented here. Build from the repository root with `docker build -f
deploy/repoeval/Dockerfile --build-arg BASE_IMAGE='<approved image>@sha256:<verified
digest>' .` only after approval. The build copies the limited RepoEval runtime,
not the repository or secrets. `USER 65532`; no runtime shell invocation or
privilege escalation. The image alone does not enforce isolation. No offline
Docker build was performed because a verified base image digest with dependencies
has not been supplied.

Before running a real job, implement and independently audit custody and
transport, then require a digest-pinned worker image and sandbox policy:

- No docker socket, host mounts, host networking, privileged mode, extra Linux
  capabilities or credentials in job metadata/env. Read-only root filesystem;
  dedicated sealed input and bounded writable result mounts only.
- Default `network_mode=none`. The provider boundary must use an independently
  authorized broker transport (e.g., host-side broker with verified request
  binding), not turn general network access on inside the worker. No such
  transport currently exists here, so live provider calls remain unavailable.
- Enforce finite external wallclock timeout, CPU, memory, disk and output quotas,
  one attempt/no automatic reschedule, then verify job termination and result
  integrity. Python's elapsed-time check cannot interrupt a blocked provider.
- `nomad.py` plans `readonly_rootfs`, `network_mode=none`, `cap_drop=ALL`,
  finite resources and MaxRunDuration, but does not provide the required sealed
  mount, custody backend, broker transport or collected result. Do not submit
  its plan as a functional RepoEval job.

Offline fixture checks: `python3 -m pytest tests/repoeval/test_entrypoint.py -q`.
