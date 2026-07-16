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


@pytest.fixture
def deterministic_public_dns(request):
    """Give hostname-policy tests a stable public DNS answer.

    This is deliberately opt-in. Tests that need a fabricated hostname must
    request the fixture explicitly, while DNS/rebinding tests remain isolated
    and cannot silently inherit a global resolver mock. Modules opt in with
    ``pytestmark = pytest.mark.usefixtures("deterministic_public_dns")``.
    """
    assert request.node.get_closest_marker("usefixtures") is not None
    public_dns = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 443))]
    with patch("missy.policy.network.socket.getaddrinfo", return_value=public_dns):
        yield
