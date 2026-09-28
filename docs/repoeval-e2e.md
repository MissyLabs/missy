# RepoEval Foundry closed-loop opt-in test

The opt-in test `tests/repoeval/test_closed_loop.py` is an intentionally local
integration probe. Run only explicitly:

```bash
python -m pytest -q tests/repoeval/test_closed_loop.py
```

It starts the actual `FoundryHTTPServer` on a randomly assigned `127.0.0.1`
port and calls it through `RepoevalFoundryTool` and `PolicyHTTPClient` with an
explicit loopback-only, default-deny network policy. The 0600 bearer token,
durable SQLite outbox, and report artifact live in pytest's temporary directory.
Server, gateway, policy-global state, and thread are closed/restored after the
test. No provider, scheduler cluster, remote host, repository checkout, Docker
image, live service, or Nomad API is contacted. The fake scheduler is a finite
in-memory object; its one `submit` executes `execute_worker` synchronously with
a one-response fake provider and immutable reviewed fixtures.

The full Tool-driven test checks that planning does not dispatch; HTTP start
durably reserves without submitting; an **explicit** coordinator dispatch produces precisely
one scheduler job and one provider request; scheduler completion alone is not
verification; a separate trusted finalization verifies worker-result evidence,
artifact bytes and checksum; and the client observes `verified` status and
project-bound artifact metadata over the real HTTP wire. Those assertions
describe this controlled fixture, **not** production deployment readiness.

The Tool accepts the coordinator's `state=reserved`, `job_id=null` acknowledgement
only after checking the configured project, reviewed plan, exact run identity
derived from the idempotency key, and bounded, uniquely identified, reserved
children and their preassigned exact job IDs. That response confirms a durable
reservation, **not** scheduler submission or completed execution. The test separately calls the coordinator's
explicit dispatch, then independently finalizes and reads the verified result.
The same-key coordinator lookup in the test confirms persistence and identity;
it is not a workaround for a failed client acknowledgement or a fresh-key retry.

The `response` and `validator-report` artifact records cover every required
kind in this workload. Their bytes are written independently under the test
temporary directory; the injected trusted reader and clearance hook re-read
and match those bytes, while a separate `reconcile` principal and verifier
checks the worker's passed result and report digest. The report is data-only.
The fixture checks repository orientation facts
against an immutable reviewed fixture; it does not execute repository code
or generated code. The approved
worker-image digest in the fixture is a policy identity, not an image pull or
proof of container isolation. Likewise, local fake scheduler observations do
not prove real-world Nomad integrity, sealed provider credentials, network
isolation, or multi-process operational recovery. Those boundaries need
separate authorization and deployment tests.
