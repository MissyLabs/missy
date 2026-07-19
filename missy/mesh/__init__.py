"""Leyline P2P Agent Mesh (F01): trust-scoped federation of Missy nodes.

A peer-to-peer layer letting multiple Missy instances form a trust-scoped mesh
that shares memory and delegates work without a central server. The pieces:

- :class:`~missy.mesh.identity.PeerIdentity` — a peer's verify-only Ed25519
  identity; the ``peer_id`` is the key fingerprint, so identity is the key.
- :class:`~missy.mesh.peer_registry.PeerRegistry` — the trust anchor: peers +
  per-peer **capability grants**, fail-closed.
- :class:`~missy.mesh.envelope.SignedEnvelope` — every cross-node message is
  Ed25519-signed over a canonical serialization and verified against the
  registry.
- :class:`~missy.mesh.crdt.LWWMap` — a last-writer-wins CRDT so shared memory
  merges deterministically (commutative/associative/idempotent) with no
  coordinator.
- :class:`~missy.mesh.quorum.PolicyQuorum` — capability-widening actions need a
  threshold of distinct, authenticated, trusted votes, so one compromised node
  can't widen the mesh.
- :class:`~missy.mesh.transport.GossipTransport` — pluggable delivery
  (in-memory for tests; HTTP-through-PolicyHTTPClient for production).
- :class:`~missy.mesh.node.MeshNode` — ties it together: publish/verify/merge
  shared memory and capability-gated delegation, all audited.
"""

from __future__ import annotations

from missy.mesh.crdt import LWWMap, Stamp
from missy.mesh.envelope import SignedEnvelope
from missy.mesh.identity import (
    PeerIdentity,
    local_peer_id,
    local_public_key_raw,
    public_bytes_to_peer_id,
)
from missy.mesh.node import MeshDelegationError, MeshNode, SyncReport
from missy.mesh.peer_registry import (
    CAPABILITIES,
    Peer,
    PeerRegistry,
    UnknownCapabilityError,
)
from missy.mesh.quorum import PolicyQuorum, QuorumResult
from missy.mesh.transport import (
    GossipTransport,
    HttpGossipTransport,
    InMemoryTransport,
)

__all__ = [
    "CAPABILITIES",
    "GossipTransport",
    "HttpGossipTransport",
    "InMemoryTransport",
    "LWWMap",
    "MeshDelegationError",
    "MeshNode",
    "Peer",
    "PeerIdentity",
    "PeerRegistry",
    "PolicyQuorum",
    "QuorumResult",
    "SignedEnvelope",
    "Stamp",
    "SyncReport",
    "UnknownCapabilityError",
    "local_peer_id",
    "local_public_key_raw",
    "public_bytes_to_peer_id",
]
