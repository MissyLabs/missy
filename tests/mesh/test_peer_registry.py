"""Tests for the mesh peer registry + capability grants (F01)."""

from __future__ import annotations

import pytest

from missy.mesh.peer_registry import (
    Peer,
    PeerRegistry,
    UnknownCapabilityError,
)


class TestMembership:
    def test_add_peer_derives_id_from_key(self, node_a) -> None:
        reg = PeerRegistry()
        peer = reg.add_peer(node_a.public_key)
        assert peer.peer_id == node_a.peer_id
        assert node_a.peer_id in reg

    def test_remove_peer(self, node_a) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)
        assert reg.remove_peer(node_a.peer_id) is True
        assert node_a.peer_id not in reg
        assert reg.remove_peer(node_a.peer_id) is False

    def test_get_and_list(self, node_a, node_b) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)
        reg.add_peer(node_b.public_key)
        assert reg.get(node_a.peer_id).peer_id == node_a.peer_id
        assert len(reg.list_peers()) == 2
        assert reg.get("nonexistent") is None


class TestCapabilityGrants:
    def test_fail_closed_unknown_peer(self) -> None:
        assert PeerRegistry().is_allowed("ghost", "memory.read") is False

    def test_fail_closed_without_grant(self, node_a) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)  # no capabilities
        assert reg.is_allowed(node_a.peer_id, "memory.read") is False

    def test_grant_then_allowed(self, node_a) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)
        reg.grant(node_a.peer_id, "memory.read")
        assert reg.is_allowed(node_a.peer_id, "memory.read") is True

    def test_revoke(self, node_a) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key, capabilities={"delegate"})
        reg.revoke(node_a.peer_id, "delegate")
        assert reg.is_allowed(node_a.peer_id, "delegate") is False

    def test_grant_unknown_capability_rejected(self, node_a) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)
        with pytest.raises(UnknownCapabilityError):
            reg.grant(node_a.peer_id, "root.everything")

    def test_add_peer_with_unknown_capability_rejected(self, node_a) -> None:
        with pytest.raises(UnknownCapabilityError):
            PeerRegistry().add_peer(node_a.public_key, capabilities={"bogus"})

    def test_grant_unknown_peer_raises(self) -> None:
        with pytest.raises(KeyError):
            PeerRegistry().grant("ghost", "delegate")

    def test_peers_with_capability(self, node_a, node_b) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key, capabilities={"delegate"})
        reg.add_peer(node_b.public_key, capabilities={"memory.read"})
        assert [p.peer_id for p in reg.peers_with("delegate")] == [node_a.peer_id]


class TestPersistence:
    def test_save_and_reload(self, tmp_path, node_a, node_b) -> None:
        path = tmp_path / "peers.json"
        reg = PeerRegistry(persist_path=str(path))
        reg.add_peer(node_a.public_key, capabilities={"delegate"}, trust=900)
        reg.add_peer(node_b.public_key, address="http://b")

        reloaded = PeerRegistry(persist_path=str(path))
        assert node_a.peer_id in reloaded
        assert reloaded.is_allowed(node_a.peer_id, "delegate") is True
        assert reloaded.get(node_a.peer_id).trust == 900
        assert reloaded.get(node_b.peer_id).address == "http://b"

    def test_no_path_does_not_persist(self, tmp_path, node_a) -> None:
        reg = PeerRegistry()  # no path
        reg.add_peer(node_a.public_key)
        reg.save()  # no-op, must not raise
        assert not list(tmp_path.iterdir())


class TestPeerDataclass:
    def test_to_from_dict_round_trip(self, node_a) -> None:
        peer = Peer(
            peer_id=node_a.peer_id,
            public_key=node_a.public_key,
            address="http://x",
            capabilities={"delegate", "gossip"},
            trust=700,
        )
        restored = Peer.from_dict(peer.to_dict())
        assert restored.capabilities == {"delegate", "gossip"}
        assert restored.trust == 700

    def test_identity_verifies(self, node_a) -> None:
        peer = Peer(peer_id=node_a.peer_id, public_key=node_a.public_key)
        sig = node_a.identity.sign(b"m")
        assert peer.identity().verify(b"m", sig) is True
