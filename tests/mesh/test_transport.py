"""Tests for gossip transports (F01)."""

from __future__ import annotations

from missy.mesh.envelope import SignedEnvelope
from missy.mesh.transport import (
    GossipTransport,
    HttpGossipTransport,
    InMemoryTransport,
    _Bus,
)


def _env(node, kind="memory.update"):
    return SignedEnvelope.create(
        identity=node.identity, sender=node.peer_id, kind=kind, payload={"k": "v"}
    )


class TestInMemoryTransport:
    def test_broadcast_reaches_other_nodes_not_self(self, node_a, node_b, node_c) -> None:
        bus = _Bus()
        ta = InMemoryTransport(node_a.peer_id, bus)
        tb = InMemoryTransport(node_b.peer_id, bus)
        tc = InMemoryTransport(node_c.peer_id, bus)
        ta.broadcast(_env(node_a))
        assert ta.poll() == []  # sender does not receive its own broadcast
        assert len(tb.poll()) == 1
        assert len(tc.poll()) == 1

    def test_poll_drains_inbox(self, node_a, node_b) -> None:
        bus = _Bus()
        ta = InMemoryTransport(node_a.peer_id, bus)
        tb = InMemoryTransport(node_b.peer_id, bus)
        ta.broadcast(_env(node_a))
        assert len(tb.poll()) == 1
        assert tb.poll() == []  # drained

    def test_satisfies_protocol(self, node_a) -> None:
        assert isinstance(InMemoryTransport(node_a.peer_id), GossipTransport)


class _FakeHttp:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.fail_for: set[str] = set()

    def post(self, url, json=None):
        if url in self.fail_for:
            raise ConnectionError("down")
        self.posts.append((url, json))


class TestHttpGossipTransport:
    def test_broadcasts_to_each_peer_address(self, node_a) -> None:
        http = _FakeHttp()
        t = HttpGossipTransport(http, lambda: ["http://p1", "http://p2/"])
        t.broadcast(_env(node_a))
        urls = [u for u, _ in http.posts]
        assert urls == ["http://p1/mesh/gossip", "http://p2/mesh/gossip"]

    def test_down_peer_does_not_abort_broadcast(self, node_a) -> None:
        http = _FakeHttp()
        http.fail_for = {"http://down/mesh/gossip"}
        t = HttpGossipTransport(http, lambda: ["http://down", "http://up"])
        t.broadcast(_env(node_a))
        # the reachable peer still received it
        assert [u for u, _ in http.posts] == ["http://up/mesh/gossip"]

    def test_receive_and_poll(self, node_a) -> None:
        t = HttpGossipTransport(_FakeHttp(), list)
        env = _env(node_a)
        t.receive(env.to_dict())
        polled = t.poll()
        assert len(polled) == 1
        assert polled[0].sender == node_a.peer_id
        assert t.poll() == []  # drained

    def test_receive_accepts_json_string(self, node_a) -> None:
        import json

        t = HttpGossipTransport(_FakeHttp(), list)
        t.receive(json.dumps(_env(node_a).to_dict()))
        assert len(t.poll()) == 1
