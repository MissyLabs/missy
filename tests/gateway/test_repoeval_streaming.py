"""Foundry's policy-gateway body budget, using an offline streaming transport."""

import gzip

import httpx
import pytest

import missy.gateway.client as gateway
from missy.gateway.client import PolicyHTTPClient


@pytest.fixture
def allowed(monkeypatch):
    class Policy:
        def check_network_resolved(self, host, session, task, *, category):
            assert host == "foundry.example" and category == "tool"
            return True, "192.0.2.1"

    monkeypatch.setattr(gateway, "get_policy_engine", lambda: Policy())
    monkeypatch.setattr(gateway, "_interactive_approval", None)


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunk=b"a" * 4096):
        self.chunk = chunk
        self.bytes_read = 0
        self.closed = False

    def __iter__(self):
        while True:
            self.bytes_read += len(self.chunk)
            yield self.chunk

    def close(self):
        self.closed = True


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_bounded_response_stops_stream_before_buffering(allowed, method):
    stream = Chunks()
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, stream=stream)

    client = PolicyHTTPClient(category="tool", timeout=5)
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ValueError, match="size limit"):
            getattr(client, method.lower() + "_limited")(
                "https://foundry.example/api/projects/alpha",
                1024 * 1024,
                headers={"Authorization": "Bearer test-private", "Accept-Encoding": "gzip"},
                follow_redirects=True,
            )
        assert stream.closed
        assert stream.bytes_read <= 1024 * 1024 + 4096
        assert requests[0].headers["Accept-Encoding"] == "identity"
    finally:
        client.close()


def test_compressed_response_refused_without_reading(allowed):
    stream = Chunks(gzip.compress(b"x" * (2 * 1024 * 1024)))
    client = PolicyHTTPClient(category="tool")
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)
        )
    )
    try:
        with pytest.raises(ValueError, match="Compressed"):
            client.get_limited("https://foundry.example/", 1024 * 1024)
        assert stream.bytes_read == 0 and stream.closed
    finally:
        client.close()


def test_trickle_stops_on_total_monotonic_deadline(allowed, monkeypatch):
    tick = [0.0]

    def clock():
        tick[0] += 0.03
        return tick[0]

    monkeypatch.setattr(gateway.time, "monotonic", clock)
    stream = Chunks(b"x")
    client = PolicyHTTPClient(category="tool", timeout=1)
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    )
    try:
        with pytest.raises(httpx.TimeoutException, match="deadline"):
            client.get_limited("https://foundry.example/", 1024 * 1024)
        assert stream.bytes_read < 100 and stream.closed
    finally:
        client.close()


def test_no_redirect_followed_even_if_caller_asks(allowed):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://attacker.example/"})

    client = PolicyHTTPClient(category="tool")
    client._sync_client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        response = client.get_limited(
            "https://foundry.example/",
            1024,
            follow_redirects=True,
            headers={"Authorization": "Bearer test-private"},
        )
        assert response.status_code == 302 and len(requests) == 1
    finally:
        client.close()


def test_denied_policy_never_reaches_transport(monkeypatch):
    from missy.core.exceptions import PolicyViolationError

    class Deny:
        def check_network_resolved(self, *args, **kwargs):
            raise PolicyViolationError("Denied", category="network", detail="Denied")

    monkeypatch.setattr(gateway, "get_policy_engine", lambda: Deny())
    monkeypatch.setattr(gateway, "_interactive_approval", None)
    client = PolicyHTTPClient(category="tool")
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(lambda req: pytest.fail("Policy bypass"))
    )
    try:
        with pytest.raises(PolicyViolationError):
            client.post_limited("https://foundry.example/", 1024, json={"x": 1})
    finally:
        client.close()
