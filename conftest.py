# conftest.py — pytest configuration
import contextlib
import socket
from unittest.mock import patch

import pytest

collect_ignore = ["tests/discord_live_test.py"]


@pytest.fixture(autouse=True)
def _stop_sleeptime_workers_after_test():
    """Stop every SleeptimeWorker daemon thread a test's AgentRuntime(s) start.

    SR-4.1: SleeptimeWorker is constructed and started (a real OS thread)
    by every AgentRuntime.__init__ in production, by design (operator-
    confirmed: enabled by default). Most tests construct AgentRuntime
    without ever calling the new AgentRuntime.shutdown(), so across a
    20,000+ test suite these threads accumulate rather than being
    cleaned up -- confirmed via a full-suite run that piled up 96+ live
    `missy-sleeptime` threads and tripped pytest's per-test faulthandler
    timeout. This fixture does not change production behavior (the real
    _make_sleeptime_worker() still runs, still starts a real thread, so
    tests that specifically exercise the wiring -- e.g. asserting
    `rt._sleeptime._thread.is_alive()` -- still see the real thing while
    they run); it only stops each worker once its owning test finishes,
    so leaked threads never accumulate across the suite.
    """
    from missy.agent.runtime import AgentRuntime

    original = AgentRuntime._make_sleeptime_worker
    created: list = []

    def _tracked(self):
        worker = original(self)
        if worker is not None:
            created.append(worker)
        return worker

    AgentRuntime._make_sleeptime_worker = _tracked
    try:
        yield
    finally:
        AgentRuntime._make_sleeptime_worker = original
        for worker in created:
            with contextlib.suppress(Exception):
                worker.stop(timeout=1.0)


@pytest.fixture(autouse=True)
def _deterministic_public_dns_for_policy_tests(request):
    """Keep policy/gateway unit tests offline and deterministic.

    Default-deny hostname checks now require a validated address. Most legacy
    unit tests exercise allowlist matching rather than DNS behavior and used
    invented hostnames, implicitly relying on resolution failure being allowed.
    Give those tests a stable public answer. Tests that exercise DNS behavior
    patch ``socket.getaddrinfo`` explicitly and override this outer patch. The
    pinned-transport integration suite is excluded because it opens a real
    loopback server and verifies the actual resolver call count.
    """
    path = str(request.fspath)
    relevant_roots = ("/tests/policy/", "/tests/gateway/", "/tests/unit/", "/tests/integration/")
    is_policy_or_gateway_test = any(root in path for root in relevant_roots)
    needs_real_network = path.endswith(
        ("test_pinned_transport.py", "test_gateway_cost_checkpoint_webhook_edges.py")
    )
    if not is_policy_or_gateway_test or needs_real_network:
        yield
        return

    public_dns = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 443))]
    with patch("missy.policy.network.socket.getaddrinfo", return_value=public_dns):
        yield
