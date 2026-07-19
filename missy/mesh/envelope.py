"""Signed gossip envelopes for the Leyline mesh (F01).

Every message crossing the mesh is an Ed25519-signed :class:`SignedEnvelope`.
The signature covers a canonical (sorted-key, separator-normalized) JSON
serialization of the sender/kind/payload/timestamp/nonce, so any tampering —
including reordering payload keys — invalidates it. Verification looks the
sender up in the :class:`~missy.mesh.peer_registry.PeerRegistry`: an envelope
from an unknown peer, or whose signature doesn't match the sender's registered
key, is rejected. This is the mesh's authenticity guarantee.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Any


def _canonical_bytes(
    sender: str, kind: str, payload: dict[str, Any], timestamp: float, nonce: str
) -> bytes:
    """Deterministic byte serialization of the signed fields."""
    body = {
        "sender": sender,
        "kind": kind,
        "payload": payload,
        "timestamp": timestamp,
        "nonce": nonce,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class SignedEnvelope:
    """An authenticated mesh message.

    Attributes:
        sender: The sending peer's id (key fingerprint).
        kind: Message type, e.g. ``memory.update``, ``delegate.request``,
            ``policy.vote``.
        payload: Arbitrary JSON-serializable message body.
        timestamp: Unix seconds when signed.
        nonce: Random hex string for replay disambiguation.
        signature: Hex-encoded Ed25519 signature over the canonical bytes.
    """

    sender: str
    kind: str
    payload: dict[str, Any]
    timestamp: float = 0.0
    nonce: str = ""
    signature: str = ""

    def signing_bytes(self) -> bytes:
        return _canonical_bytes(self.sender, self.kind, self.payload, self.timestamp, self.nonce)

    @classmethod
    def create(
        cls,
        *,
        identity,
        sender: str,
        kind: str,
        payload: dict[str, Any],
        timestamp: float | None = None,
    ) -> SignedEnvelope:
        """Build and sign an envelope with a local ``AgentIdentity``."""
        ts = time.time() if timestamp is None else timestamp
        nonce = secrets.token_hex(8)
        raw = _canonical_bytes(sender, kind, payload, ts, nonce)
        sig = identity.sign(raw).hex()
        return cls(
            sender=sender,
            kind=kind,
            payload=payload,
            timestamp=ts,
            nonce=nonce,
            signature=sig,
        )

    def verify(self, registry) -> tuple[bool, str]:
        """Verify authenticity against a :class:`PeerRegistry`.

        Returns ``(ok, reason)``. Fails closed for an unknown sender, a
        missing signature, or a signature that doesn't match the sender's
        registered key.
        """
        peer = registry.get(self.sender)
        if peer is None:
            return (False, f"unknown sender {self.sender!r}")
        if not self.signature:
            return (False, "missing signature")
        try:
            sig_bytes = bytes.fromhex(self.signature)
        except ValueError:
            return (False, "malformed signature")
        if peer.identity().verify(self.signing_bytes(), sig_bytes):
            return (True, "ok")
        return (False, "signature verification failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "kind": self.kind,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignedEnvelope:
        return cls(
            sender=data["sender"],
            kind=data["kind"],
            payload=data.get("payload", {}),
            timestamp=float(data.get("timestamp", 0.0)),
            nonce=data.get("nonce", ""),
            signature=data.get("signature", ""),
        )
