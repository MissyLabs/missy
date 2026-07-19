"""Integration tests for MeshNode (F01)."""

from __future__ import annotations

import pytest

from missy.mesh.envelope import SignedEnvelope
from missy.mesh.node import MeshDelegationError, MeshNode
from missy.mesh.peer_registry import PeerRegistry
from missy.mesh.transport import InMemoryTransport, _Bus


def _mesh(node_self, peers_with_caps):
    """Build a MeshNode whose registry knows the given peers with capabilities."""
    reg = PeerRegistry()
    for peer, caps in peers_with_caps:
        reg.add_peer(peer.public_key, capabilities=set(caps), trust=800)
    return reg


class TestSharedMemory:
    def test_signed_memory_propagates_and_merges(self, node_a, node_b) -> None:
        bus = _Bus()
        reg_a = _mesh(node_a, [(node_b, {"memory.write"})])
        reg_b = _mesh(node_b, [(node_a, {"memory.write"})])
        a = MeshNode(node_a.identity, reg_a, InMemoryTransport(node_a.peer_id, bus))
        b = MeshNode(node_b.identity, reg_b, InMemoryTransport(node_b.peer_id, bus))

        a.publish_memory("status", "green")
        report = b.sync()
        assert report.merged == 1 and report.rejected == 0
        assert b.memory.get("status") == "green"

    def test_unsigned_or_forged_update_rejected(self, node_a, node_b) -> None:
        bus = _Bus()
        reg_b = _mesh(node_b, [(node_a, {"memory.write"})])
        b = MeshNode(node_b.identity, reg_b, InMemoryTransport(node_b.peer_id, bus))
        other = InMemoryTransport("attacker", bus)
        forged = SignedEnvelope(
            sender=node_a.peer_id,
            kind="memory.update",
            payload={"key": "x", "value": "evil", "timestamp": 9e9},
            timestamp=9e9,
            nonce="n",
            signature="00" * 64,
        )
        other.broadcast(forged)
        report = b.sync()
        assert report.rejected == 1 and report.merged == 0
        assert b.memory.get("x") is None

    def test_update_from_uncapable_peer_rejected(self, node_a, node_b) -> None:
        bus = _Bus()
        # node_b knows node_a but did NOT grant memory.write
        reg_b = _mesh(node_b, [(node_a, set())])
        reg_a = _mesh(node_a, [(node_b, {"memory.write"})])
        a = MeshNode(node_a.identity, reg_a, InMemoryTransport(node_a.peer_id, bus))
        b = MeshNode(node_b.identity, reg_b, InMemoryTransport(node_b.peer_id, bus))
        a.publish_memory("k", "v")
        report = b.sync()
        assert report.rejected == 1 and report.merged == 0
        assert "memory.write" in report.reasons[0]

    def test_update_from_unknown_peer_rejected(self, node_a, node_b, node_c) -> None:
        bus = _Bus()
        reg_b = _mesh(node_b, [(node_a, {"memory.write"})])  # doesn't know node_c
        b = MeshNode(node_b.identity, reg_b, InMemoryTransport(node_b.peer_id, bus))
        c_transport = InMemoryTransport(node_c.peer_id, bus)
        env = SignedEnvelope.create(
            identity=node_c.identity,
            sender=node_c.peer_id,
            kind="memory.update",
            payload={"key": "k", "value": "v", "timestamp": 1.0},
        )
        c_transport.broadcast(env)
        report = b.sync()
        assert report.rejected == 1
        assert "unknown sender" in report.reasons[0]

    def test_concurrent_writes_converge(self, node_a, node_b) -> None:
        # Both nodes write the same key; after cross-sync both agree (LWW).
        bus = _Bus()
        reg_a = _mesh(node_a, [(node_b, {"memory.write"})])
        reg_b = _mesh(node_b, [(node_a, {"memory.write"})])
        a = MeshNode(node_a.identity, reg_a, InMemoryTransport(node_a.peer_id, bus))
        b = MeshNode(node_b.identity, reg_b, InMemoryTransport(node_b.peer_id, bus))
        a.publish_memory("k", "from-a")
        b.publish_memory("k", "from-b")
        a.sync()
        b.sync()
        assert a.memory.get("k") == b.memory.get("k")  # converged


class TestDelegation:
    def test_delegate_to_capable_peer(self, node_a, node_b) -> None:
        bus = _Bus()
        reg_a = _mesh(node_a, [(node_b, {"delegate"})])
        a = MeshNode(node_a.identity, reg_a, InMemoryTransport(node_a.peer_id, bus))
        env = a.delegate(node_b.peer_id, "render video")
        assert env.payload["task"] == "render video"
        assert env.kind == "delegate.request"

    def test_delegate_to_unknown_peer_refused(self, node_a, node_b) -> None:
        a = MeshNode(node_a.identity, PeerRegistry(), InMemoryTransport(node_a.peer_id))
        with pytest.raises(MeshDelegationError, match="unknown peer"):
            a.delegate(node_b.peer_id, "task")

    def test_delegate_without_capability_refused(self, node_a, node_b) -> None:
        reg_a = _mesh(node_a, [(node_b, set())])  # known but no delegate grant
        a = MeshNode(node_a.identity, reg_a, InMemoryTransport(node_a.peer_id))
        with pytest.raises(MeshDelegationError, match="lacks capability"):
            a.delegate(node_b.peer_id, "task")

    def test_best_peer_for_picks_highest_trust(self, node_a, node_b, node_c) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_b.public_key, capabilities={"delegate"}, trust=600)
        reg.add_peer(node_c.public_key, capabilities={"delegate"}, trust=950)
        a = MeshNode(node_a.identity, reg, InMemoryTransport(node_a.peer_id))
        assert a.best_peer_for("delegate") == node_c.peer_id

    def test_best_peer_for_none_when_no_capable_peer(self, node_a) -> None:
        a = MeshNode(node_a.identity, PeerRegistry(), InMemoryTransport(node_a.peer_id))
        assert a.best_peer_for("delegate") is None


class TestNodeIdentity:
    def test_peer_id_matches_identity(self, node_a) -> None:
        a = MeshNode(node_a.identity, PeerRegistry(), InMemoryTransport(node_a.peer_id))
        assert a.peer_id == node_a.peer_id
