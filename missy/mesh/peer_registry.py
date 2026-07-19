"""Peer registry + capability grants for the Leyline mesh (F01).

The registry is the mesh's trust anchor: it maps a ``peer_id`` (an Ed25519 key
fingerprint) to that peer's public key, network address, and the **capability
grants** the local node has extended to it. Everything is **fail-closed** — an
unknown peer, or a peer without an explicit grant for a capability, is denied.
One compromised or spoofed peer cannot grant itself capabilities: grants are
local state the operator controls, never asserted by the peer over the wire.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from missy.mesh.identity import PeerIdentity

# Recognized mesh capabilities. A grant is a subset of these strings.
CAPABILITIES = frozenset(
    {
        "memory.read",  # query this node's shared memory/graph
        "memory.write",  # contribute CRDT memory updates
        "delegate",  # ask this node to run a delegated subtask
        "gossip",  # participate in signed gossip / CRDT sync
        "policy.vote",  # cast a vote in a distributed policy quorum
    }
)


@dataclass
class Peer:
    """A known mesh peer and the capabilities granted to it."""

    peer_id: str
    public_key: str  # base64url-encoded raw Ed25519 key
    address: str = ""
    capabilities: set[str] = field(default_factory=set)
    trust: int = 500  # 0-1000, mirrors TrustScorer's scale

    def identity(self) -> PeerIdentity:
        return PeerIdentity.from_encoded(self.public_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "public_key": self.public_key,
            "address": self.address,
            "capabilities": sorted(self.capabilities),
            "trust": self.trust,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Peer:
        return cls(
            peer_id=data["peer_id"],
            public_key=data["public_key"],
            address=data.get("address", ""),
            capabilities=set(data.get("capabilities", [])),
            trust=int(data.get("trust", 500)),
        )


class UnknownCapabilityError(ValueError):
    """Raised when granting a capability outside :data:`CAPABILITIES`."""


class PeerRegistry:
    """Thread-safe registry of mesh peers and their capability grants."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._peers: dict[str, Peer] = {}
        self._lock = threading.RLock()
        self._path = Path(persist_path).expanduser() if persist_path else None
        if self._path and self._path.exists():
            self.load()

    # -- membership -------------------------------------------------------
    def add_peer(
        self,
        public_key: str,
        *,
        address: str = "",
        capabilities: set[str] | None = None,
        trust: int = 500,
    ) -> Peer:
        """Register a peer by its base64url public key. Returns the Peer.

        The ``peer_id`` is derived from the key, so a peer cannot choose it.
        """
        ident = PeerIdentity.from_encoded(public_key)
        caps = set(capabilities or set())
        self._validate_caps(caps)
        peer = Peer(
            peer_id=ident.peer_id,
            public_key=public_key,
            address=address,
            capabilities=caps,
            trust=trust,
        )
        with self._lock:
            self._peers[peer.peer_id] = peer
            self._save_locked()
        return peer

    def remove_peer(self, peer_id: str) -> bool:
        with self._lock:
            existed = self._peers.pop(peer_id, None) is not None
            if existed:
                self._save_locked()
            return existed

    def get(self, peer_id: str) -> Peer | None:
        with self._lock:
            return self._peers.get(peer_id)

    def list_peers(self) -> list[Peer]:
        with self._lock:
            return list(self._peers.values())

    def __contains__(self, peer_id: str) -> bool:
        with self._lock:
            return peer_id in self._peers

    # -- capability grants ------------------------------------------------
    def grant(self, peer_id: str, capability: str) -> None:
        self._validate_caps({capability})
        with self._lock:
            peer = self._require(peer_id)
            peer.capabilities.add(capability)
            self._save_locked()

    def revoke(self, peer_id: str, capability: str) -> None:
        with self._lock:
            peer = self._peers.get(peer_id)
            if peer:
                peer.capabilities.discard(capability)
                self._save_locked()

    def is_allowed(self, peer_id: str, capability: str) -> bool:
        """Fail-closed capability check: unknown peer or no grant → False."""
        with self._lock:
            peer = self._peers.get(peer_id)
            return bool(peer and capability in peer.capabilities)

    def peers_with(self, capability: str) -> list[Peer]:
        with self._lock:
            return [p for p in self._peers.values() if capability in p.capabilities]

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self._peers.values()]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self._path or not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        with self._lock:
            self._peers = {p["peer_id"]: Peer.from_dict(p) for p in data}

    # -- helpers ----------------------------------------------------------
    def _require(self, peer_id: str) -> Peer:
        peer = self._peers.get(peer_id)
        if peer is None:
            raise KeyError(f"unknown peer {peer_id!r}")
        return peer

    @staticmethod
    def _validate_caps(caps: set[str]) -> None:
        unknown = caps - CAPABILITIES
        if unknown:
            raise UnknownCapabilityError(
                f"unknown capabilities {sorted(unknown)}; valid: {sorted(CAPABILITIES)}"
            )
