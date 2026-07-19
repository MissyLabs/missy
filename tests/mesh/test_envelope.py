"""Tests for signed gossip envelopes (F01)."""

from __future__ import annotations

from missy.mesh.envelope import SignedEnvelope
from missy.mesh.peer_registry import PeerRegistry


def _registry(*nodes) -> PeerRegistry:
    reg = PeerRegistry()
    for n in nodes:
        reg.add_peer(n.public_key)
    return reg


class TestSignVerify:
    def test_created_envelope_verifies(self, node_a) -> None:
        env = SignedEnvelope.create(
            identity=node_a.identity,
            sender=node_a.peer_id,
            kind="memory.update",
            payload={"key": "k", "value": "v"},
        )
        ok, reason = env.verify(_registry(node_a))
        assert ok and reason == "ok"

    def test_unknown_sender_rejected(self, node_a) -> None:
        env = SignedEnvelope.create(
            identity=node_a.identity, sender=node_a.peer_id, kind="k", payload={}
        )
        ok, reason = env.verify(PeerRegistry())  # empty
        assert not ok
        assert "unknown sender" in reason

    def test_tampered_payload_rejected(self, node_a) -> None:
        env = SignedEnvelope.create(
            identity=node_a.identity,
            sender=node_a.peer_id,
            kind="k",
            payload={"amount": 1},
        )
        env.payload["amount"] = 1000000  # tamper after signing
        ok, reason = env.verify(_registry(node_a))
        assert not ok
        assert "verification failed" in reason

    def test_signature_from_impostor_rejected(self, node_a, node_b) -> None:
        # node_b signs but claims to be node_a
        env = SignedEnvelope.create(
            identity=node_b.identity, sender=node_a.peer_id, kind="k", payload={}
        )
        ok, reason = env.verify(_registry(node_a, node_b))
        assert not ok
        assert "verification failed" in reason

    def test_missing_signature_rejected(self, node_a) -> None:
        env = SignedEnvelope(sender=node_a.peer_id, kind="k", payload={})
        ok, reason = env.verify(_registry(node_a))
        assert not ok
        assert "missing signature" in reason

    def test_malformed_signature_rejected(self, node_a) -> None:
        env = SignedEnvelope(sender=node_a.peer_id, kind="k", payload={}, signature="not-hex-zz")
        ok, reason = env.verify(_registry(node_a))
        assert not ok
        assert "malformed signature" in reason


class TestCanonicalization:
    def test_key_order_does_not_affect_verification(self, node_a) -> None:
        env = SignedEnvelope.create(
            identity=node_a.identity,
            sender=node_a.peer_id,
            kind="k",
            payload={"b": 2, "a": 1},
        )
        # Rebuild payload with different insertion order — canonical signing
        # bytes sort keys, so it still verifies.
        env.payload = {"a": 1, "b": 2}
        ok, _ = env.verify(_registry(node_a))
        assert ok

    def test_nonce_is_unique_per_envelope(self, node_a) -> None:
        e1 = SignedEnvelope.create(
            identity=node_a.identity, sender=node_a.peer_id, kind="k", payload={}
        )
        e2 = SignedEnvelope.create(
            identity=node_a.identity, sender=node_a.peer_id, kind="k", payload={}
        )
        assert e1.nonce != e2.nonce


class TestSerialization:
    def test_to_from_dict_round_trip_preserves_verifiability(self, node_a) -> None:
        env = SignedEnvelope.create(
            identity=node_a.identity,
            sender=node_a.peer_id,
            kind="memory.update",
            payload={"key": "k", "value": [1, 2, 3]},
        )
        restored = SignedEnvelope.from_dict(env.to_dict())
        ok, _ = restored.verify(_registry(node_a))
        assert ok
        assert restored.nonce == env.nonce
