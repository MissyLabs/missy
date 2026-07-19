"""Peer identity primitives for the Leyline mesh (F01).

The local node signs with the existing :class:`~missy.security.identity.AgentIdentity`
(Ed25519). This module adds the *peer* side: wrapping a remote peer's raw
Ed25519 public key so the mesh can verify messages that peer signed, and a
stable ``peer_id`` derived from the key (its SHA-256 fingerprint) so identity
is the key, not a self-asserted name.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def public_bytes_to_peer_id(raw: bytes) -> str:
    """Return the canonical peer id (SHA-256 hex of the raw public key)."""
    return hashlib.sha256(raw).hexdigest()


def encode_public_key(raw: bytes) -> str:
    """Encode a raw Ed25519 public key as base64url (no padding)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_public_key(encoded: str) -> bytes:
    """Decode a base64url public key (padding-tolerant) to raw bytes."""
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + pad)


class PeerIdentity:
    """A remote peer's verify-only Ed25519 identity.

    Constructed from the peer's raw public key (or its base64url form). The
    ``peer_id`` is the key fingerprint, so a peer cannot claim an id that
    doesn't match the key it signs with.
    """

    def __init__(self, public_key_raw: bytes) -> None:
        if len(public_key_raw) != 32:
            raise ValueError("Ed25519 public key must be 32 bytes")
        self._raw = public_key_raw
        self._key = Ed25519PublicKey.from_public_bytes(public_key_raw)

    @classmethod
    def from_encoded(cls, encoded: str) -> PeerIdentity:
        return cls(decode_public_key(encoded))

    @property
    def peer_id(self) -> str:
        return public_bytes_to_peer_id(self._raw)

    @property
    def public_key_raw(self) -> bytes:
        return self._raw

    def encoded_public_key(self) -> str:
        return encode_public_key(self._raw)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Return ``True`` iff ``signature`` is a valid Ed25519 sig by this peer."""
        try:
            self._key.verify(signature, message)
            return True
        except Exception:
            return False


def local_public_key_raw(identity) -> bytes:
    """Extract the raw 32-byte public key from an :class:`AgentIdentity`.

    Uses the identity's JWK export (which already base64url-encodes the raw
    public key) so we don't reach into its private attributes.
    """
    return decode_public_key(identity.to_jwk()["x"])


def local_peer_id(identity) -> str:
    """Return the mesh peer id for a local :class:`AgentIdentity`."""
    return public_bytes_to_peer_id(local_public_key_raw(identity))
