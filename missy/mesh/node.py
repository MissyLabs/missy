"""MeshNode — the Leyline mesh coordinator (F01).

Ties the pieces together for one node: its local Ed25519 identity, the peer
registry (trust + capability grants), a CRDT-backed shared memory replica, and
a gossip transport. It publishes signed memory updates, ingests and *verifies*
peers' updates before merging them (rejecting anything unsigned, unknown, or
from a peer lacking the relevant capability), and routes capability-gated task
delegation to a peer. Every accepted or rejected cross-node action is emitted
as an audit event, so mesh activity lands in the same tamper-evident log as
everything else.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus
from missy.mesh.crdt import LWWMap
from missy.mesh.envelope import SignedEnvelope
from missy.mesh.identity import local_peer_id
from missy.mesh.peer_registry import PeerRegistry
from missy.mesh.transport import GossipTransport

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Result of a :meth:`MeshNode.sync` pass."""

    merged: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)


class MeshDelegationError(RuntimeError):
    """Raised when a delegation is refused (unknown peer / missing grant)."""


class MeshNode:
    """A single participant in the Leyline mesh."""

    def __init__(
        self,
        identity,
        registry: PeerRegistry,
        transport: GossipTransport,
        *,
        memory: LWWMap | None = None,
    ) -> None:
        self._identity = identity
        self._registry = registry
        self._transport = transport
        self._memory = memory or LWWMap()
        self.peer_id = local_peer_id(identity)

    @property
    def memory(self) -> LWWMap:
        return self._memory

    # -- shared memory ----------------------------------------------------
    def publish_memory(self, key: str, value: Any) -> SignedEnvelope:
        """Set a shared-memory key locally and broadcast a signed update."""
        ts = time.time()
        self._memory.set(key, value, timestamp=ts, peer_id=self.peer_id)
        env = SignedEnvelope.create(
            identity=self._identity,
            sender=self.peer_id,
            kind="memory.update",
            payload={"key": key, "value": value, "timestamp": ts},
        )
        self._transport.broadcast(env)
        self._audit("mesh.memory.publish", {"key": key})
        return env

    def sync(self) -> SyncReport:
        """Poll the transport and merge verified memory updates.

        An envelope is merged only if it verifies (known sender + valid
        signature) and the sender holds the ``memory.write`` capability.
        """
        report = SyncReport()
        for env in self._transport.poll():
            if env.kind != "memory.update":
                continue
            ok, reason = env.verify(self._registry)
            if not ok:
                report.rejected += 1
                report.reasons.append(f"{env.sender}: {reason}")
                self._audit(
                    "mesh.memory.reject", {"sender": env.sender, "reason": reason}, result="deny"
                )
                continue
            if not self._registry.is_allowed(env.sender, "memory.write"):
                report.rejected += 1
                report.reasons.append(f"{env.sender}: not granted memory.write")
                self._audit(
                    "mesh.memory.reject",
                    {"sender": env.sender, "reason": "no memory.write grant"},
                    result="deny",
                )
                continue
            key = env.payload.get("key")
            value = env.payload.get("value")
            ts = float(env.payload.get("timestamp", env.timestamp))
            self._memory.set(key, value, timestamp=ts, peer_id=env.sender)
            report.merged += 1
            self._audit("mesh.memory.merge", {"sender": env.sender, "key": key})
        return report

    # -- delegation -------------------------------------------------------
    def delegate(self, peer_id: str, task: str, *, capability: str = "delegate") -> SignedEnvelope:
        """Route a task to a peer, gated on that peer's capability grant.

        Fails closed: an unknown peer, or one without the required capability,
        raises :class:`MeshDelegationError` — the task is never sent.
        """
        if peer_id not in self._registry:
            self._audit(
                "mesh.delegate.refused", {"peer": peer_id, "reason": "unknown peer"}, result="deny"
            )
            raise MeshDelegationError(f"unknown peer {peer_id!r}")
        if not self._registry.is_allowed(peer_id, capability):
            self._audit(
                "mesh.delegate.refused",
                {"peer": peer_id, "reason": f"no {capability} grant"},
                result="deny",
            )
            raise MeshDelegationError(f"peer {peer_id!r} lacks capability {capability!r}")
        env = SignedEnvelope.create(
            identity=self._identity,
            sender=self.peer_id,
            kind="delegate.request",
            payload={"target": peer_id, "task": task, "capability": capability},
        )
        self._transport.broadcast(env)
        self._audit("mesh.delegate.sent", {"peer": peer_id, "capability": capability})
        return env

    def best_peer_for(self, capability: str) -> str | None:
        """Return the highest-trust peer holding ``capability``, if any."""
        candidates = self._registry.peers_with(capability)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.trust).peer_id

    # -- audit ------------------------------------------------------------
    def _audit(self, event_type: str, detail: dict[str, Any], *, result: str = "allow") -> None:
        try:
            event_bus.publish(
                AuditEvent.now(
                    session_id="mesh",
                    task_id=self.peer_id,
                    event_type=event_type,
                    category="channel",
                    result=result,  # type: ignore[arg-type]
                    detail={"node": self.peer_id, **detail},
                )
            )
        except Exception:  # audit must never break mesh operation
            logger.debug("mesh audit emit failed for %s", event_type, exc_info=True)
