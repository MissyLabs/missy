"""Tests for mesh peer identity primitives (F01)."""

from __future__ import annotations

import pytest

from missy.mesh.identity import (
    PeerIdentity,
    decode_public_key,
    encode_public_key,
    local_peer_id,
    local_public_key_raw,
    public_bytes_to_peer_id,
)
from missy.security.identity import AgentIdentity


class TestEncoding:
    def test_encode_decode_round_trips(self) -> None:
        raw = local_public_key_raw(AgentIdentity.generate())
        assert decode_public_key(encode_public_key(raw)) == raw

    def test_peer_id_is_key_fingerprint(self) -> None:
        raw = local_public_key_raw(AgentIdentity.generate())
        assert public_bytes_to_peer_id(raw) == public_bytes_to_peer_id(raw)
        assert len(public_bytes_to_peer_id(raw)) == 64  # sha256 hex


class TestPeerIdentity:
    def test_verifies_signature_from_matching_key(self) -> None:
        ident = AgentIdentity.generate()
        peer = PeerIdentity(local_public_key_raw(ident))
        msg = b"hello mesh"
        sig = ident.sign(msg)
        assert peer.verify(msg, sig) is True

    def test_rejects_signature_from_other_key(self) -> None:
        signer = AgentIdentity.generate()
        other = AgentIdentity.generate()
        peer = PeerIdentity(local_public_key_raw(other))
        assert peer.verify(b"x", signer.sign(b"x")) is False

    def test_rejects_tampered_message(self) -> None:
        ident = AgentIdentity.generate()
        peer = PeerIdentity(local_public_key_raw(ident))
        sig = ident.sign(b"original")
        assert peer.verify(b"tampered", sig) is False

    def test_peer_id_matches_local_peer_id(self) -> None:
        ident = AgentIdentity.generate()
        peer = PeerIdentity(local_public_key_raw(ident))
        assert peer.peer_id == local_peer_id(ident)

    def test_from_encoded(self) -> None:
        ident = AgentIdentity.generate()
        enc = encode_public_key(local_public_key_raw(ident))
        peer = PeerIdentity.from_encoded(enc)
        assert peer.encoded_public_key() == enc

    def test_wrong_key_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            PeerIdentity(b"tooshort")
