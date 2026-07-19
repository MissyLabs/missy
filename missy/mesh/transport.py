"""Gossip transport abstraction for the Leyline mesh (F01).

The mesh logic (signing, verification, CRDT merge, quorum) is transport-
agnostic: it speaks in :class:`~missy.mesh.envelope.SignedEnvelope`s and leaves
delivery to a :class:`GossipTransport`. Two implementations ship here:

* :class:`InMemoryTransport` — an in-process shared bus. Used for tests and
  for a single-process multi-node simulation.
* :class:`HttpGossipTransport` — POSTs signed envelopes to peers' addresses
  through an injected HTTP client (the production wiring passes
  :class:`~missy.gateway.client.PolicyHTTPClient` so every outbound gossip
  request is still policy-gated). Kept behind the same interface so nothing in
  the mesh core depends on a concrete network stack.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Protocol, runtime_checkable

from missy.mesh.envelope import SignedEnvelope

logger = logging.getLogger(__name__)


@runtime_checkable
class GossipTransport(Protocol):
    """Delivers and receives signed envelopes."""

    def broadcast(self, envelope: SignedEnvelope) -> None: ...

    def poll(self) -> list[SignedEnvelope]:
        """Return and clear envelopes received since the last poll."""
        ...


class InMemoryTransport:
    """A shared in-process bus. Nodes attached to the same bus see each other.

    Each attached node has its own inbox; ``broadcast`` fans an envelope out to
    every *other* node's inbox (never echoing to the sender), and ``poll``
    drains this node's inbox.
    """

    def __init__(self, node_id: str, bus: _Bus | None = None) -> None:
        self.node_id = node_id
        self._bus = bus or _Bus()
        self._bus.attach(node_id)

    @property
    def bus(self) -> _Bus:
        return self._bus

    def broadcast(self, envelope: SignedEnvelope) -> None:
        self._bus.deliver(self.node_id, envelope)

    def poll(self) -> list[SignedEnvelope]:
        return self._bus.drain(self.node_id)


class _Bus:
    """Shared delivery fabric for :class:`InMemoryTransport` nodes."""

    def __init__(self) -> None:
        self._inboxes: dict[str, list[SignedEnvelope]] = {}
        self._lock = threading.Lock()

    def attach(self, node_id: str) -> None:
        with self._lock:
            self._inboxes.setdefault(node_id, [])

    def deliver(self, sender_node: str, envelope: SignedEnvelope) -> None:
        with self._lock:
            for node_id, inbox in self._inboxes.items():
                if node_id != sender_node:
                    inbox.append(envelope)

    def drain(self, node_id: str) -> list[SignedEnvelope]:
        with self._lock:
            items = self._inboxes.get(node_id, [])
            self._inboxes[node_id] = []
            return items


class HttpGossipTransport:
    """Broadcasts envelopes to peer addresses via an injected HTTP client.

    Args:
        http_client: An object exposing ``post(url, json=...)`` — in
            production a :class:`~missy.gateway.client.PolicyHTTPClient`, so
            gossip is policy-gated like any other outbound request.
        peer_addresses: Callable returning the current list of peer base URLs
            to broadcast to (resolved from the :class:`PeerRegistry`).
        gossip_path: Path appended to each peer address for envelope ingress.
    """

    def __init__(
        self,
        http_client,
        peer_addresses,
        *,
        gossip_path: str = "/mesh/gossip",
    ) -> None:
        self._http = http_client
        self._peer_addresses = peer_addresses
        self._gossip_path = gossip_path
        self._inbox: list[SignedEnvelope] = []
        self._lock = threading.Lock()

    def broadcast(self, envelope: SignedEnvelope) -> None:
        body = envelope.to_dict()
        for base in self._peer_addresses():
            url = base.rstrip("/") + self._gossip_path
            try:
                self._http.post(url, json=body)
            except Exception as exc:  # a down peer must not abort the broadcast
                logger.warning("mesh gossip to %s failed: %s", url, exc)

    def receive(self, raw: str | dict) -> None:
        """Ingest an inbound envelope (called by the mesh HTTP ingress route)."""
        data = json.loads(raw) if isinstance(raw, str) else raw
        with self._lock:
            self._inbox.append(SignedEnvelope.from_dict(data))

    def poll(self) -> list[SignedEnvelope]:
        with self._lock:
            items = self._inbox
            self._inbox = []
            return items
