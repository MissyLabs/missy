"""Pairing workflow manager for voice-channel edge nodes.

Handles the full lifecycle of registering a new edge node: from the initial
connection (which creates a *pending* entry in the registry) through operator
approval or rejection, to removal of an already-paired device.

The pairing flow is:

1. An edge node that has no existing auth token calls the voice channel
   endpoint.  The channel calls :meth:`PairingManager.initiate_pairing`.
2. The node is stored with ``paired=False`` and appears in
   ``missy devices list --pending``.
3. An operator runs ``missy devices pair <node_id>`` which calls
   :meth:`PairingManager.approve_pairing`.  The plaintext token is displayed
   once and the node is updated to ``paired=True``.
4. Alternatively, the operator rejects the request via
   :meth:`PairingManager.reject_pairing`.

Audit events are emitted under the ``"plugin"`` category with
``session_id="system"`` and ``task_id="device-pairing"``.

Example::

    from missy.channels.voice.registry import DeviceRegistry
    from missy.channels.voice.pairing import PairingManager

    registry = DeviceRegistry()
    registry.load()
    mgr = PairingManager(registry)

    node_id = mgr.initiate_pairing(
        node_id="",
        friendly_name="Living Room",
        room="living_room",
        ip_address="192.168.1.42",
        hardware_profile={"mic_type": "respeaker", "channels": 4},
    )
    token = mgr.approve_pairing(node_id)
"""

from __future__ import annotations

import logging
import uuid

from missy.channels.voice.registry import DeviceRegistry, EdgeNode
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_PAIRING_SESSION_ID = "system"
_PAIRING_TASK_ID = "device-pairing"


class PairingManager:
    """Orchestrates the pairing lifecycle for voice-channel edge nodes.

    This class is a thin workflow layer on top of :class:`DeviceRegistry`.
    It does not hold any state of its own — all persistence is delegated to
    the registry.

    Args:
        registry: The :class:`DeviceRegistry` instance to use.  It is the
            caller's responsibility to call :meth:`~DeviceRegistry.load`
            before constructing a ``PairingManager``.
    """

    def __init__(self, registry: DeviceRegistry) -> None:
        self._registry = registry

    def initiate_pairing(
        self,
        node_id: str,
        friendly_name: str,
        room: str,
        ip_address: str,
        hardware_profile: dict,
        *,
        policy_mode: str = "full",
        audio_logging: bool = False,
        audio_log_dir: str = "",
        audio_log_retention_days: int = 7,
    ) -> str:
        """Register a new unpaired node and return its assigned ``node_id``.

        If *node_id* is empty or ``None``, a fresh UUID4 is generated.  If a
        node with the supplied *node_id* already exists in the registry this
        method returns early with that existing ``node_id`` without
        overwriting the record.

        The new node is stored with ``paired=False`` and ``status="offline"``
        until the operator approves it.

        Emits a ``voice.pairing.initiated`` audit event.

        Args:
            node_id: Desired node identifier.  Pass ``""`` or ``None`` to
                have one generated automatically.
            friendly_name: Human-readable label for the device.
            room: Room identifier surfaced to the agent as context.
            ip_address: IP address or mDNS hostname reported by the device.
            hardware_profile: Freeform dict describing the device hardware.
            policy_mode: Capability policy for this node.  Defaults to
                ``"full"``.
            audio_logging: Whether to persist audio to disk.  Defaults to
                ``False``.
            audio_log_dir: Directory for audio log files.  Defaults to
                ``""``.
            audio_log_retention_days: Retention period for audio files.
                Defaults to ``7``.

        Returns:
            The ``node_id`` of the newly created (or already-existing) node.
        """
        if not node_id:
            node_id = str(uuid.uuid4())

        # Idempotency: if the node is already registered, skip.
        existing = self._registry.get_node(node_id)
        if existing is not None:
            logger.debug(
                "initiate_pairing: node %r already exists — skipping.", node_id
            )
            return node_id

        node = EdgeNode(
            node_id=node_id,
            friendly_name=friendly_name,
            room=room,
            ip_address=ip_address,
            hardware_profile=hardware_profile,
            paired=False,
            status="offline",
            policy_mode=policy_mode,
            audio_logging=audio_logging,
            audio_log_dir=audio_log_dir,
            audio_log_retention_days=audio_log_retention_days,
        )
        self._registry.add_node(node)

        event_bus.publish(
            AuditEvent.now(
                session_id=_PAIRING_SESSION_ID,
                task_id=_PAIRING_TASK_ID,
                event_type="voice.pairing.initiated",
                category="plugin",
                result="allow",
                detail={
                    "node_id": node_id,
                    "friendly_name": friendly_name,
                    "room": room,
                    "ip_address": ip_address,
                },
            )
        )
        logger.info(
            "Pairing initiated for node %r (%s) in room %r.",
            node_id,
            friendly_name,
            room,
        )
        return node_id

    def approve_pairing(self, node_id: str) -> str:
        """Approve a pending node and return the plaintext auth token.

        The token is generated once and stored only as a PBKDF2 hash in the
        registry.  The plaintext token returned here is the only opportunity
        to retrieve it; it cannot be recovered later.

        Emits a ``voice.pairing.approved`` audit event.

        Args:
            node_id: The pending node to approve.

        Returns:
            The plaintext auth token to be delivered to the edge device.

        Raises:
            ValueError: If *node_id* is not found in the registry, or if the
                node has already been approved (``paired=True``).
        """
        node = self._registry.get_node(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id!r}")
        if node.paired:
            raise ValueError(
                f"Node {node_id!r} is already paired — unpair it first to re-issue a token."
            )

        self._registry.approve_node(node_id)
        token = self._registry.generate_token(node_id)

        event_bus.publish(
            AuditEvent.now(
                session_id=_PAIRING_SESSION_ID,
                task_id=_PAIRING_TASK_ID,
                event_type="voice.pairing.approved",
                category="plugin",
                result="allow",
                detail={"node_id": node_id},
            )
        )
        logger.info("Pairing approved for node %r.", node_id)
        return token

    def reject_pairing(self, node_id: str) -> None:
        """Remove a pending node without approving it.

        Emits a ``voice.pairing.rejected`` audit event.  No-op if the node
        does not exist.

        Args:
            node_id: The pending node to reject.

        Raises:
            ValueError: If the node exists but has already been approved
                (use :meth:`unpair_node` to remove an approved node).
        """
        node = self._registry.get_node(node_id)
        if node is None:
            logger.debug("reject_pairing: node %r not found — no-op.", node_id)
            return
        if node.paired:
            raise ValueError(
                f"Node {node_id!r} is already paired — use unpair_node() to remove it."
            )

        self._registry.remove_node(node_id)

        event_bus.publish(
            AuditEvent.now(
                session_id=_PAIRING_SESSION_ID,
                task_id=_PAIRING_TASK_ID,
                event_type="voice.pairing.rejected",
                category="plugin",
                result="allow",
                detail={"node_id": node_id},
            )
        )
        logger.info("Pairing rejected for node %r.", node_id)

    def list_pending(self) -> list[EdgeNode]:
        """Return nodes that are awaiting operator approval.

        Returns:
            A list of :class:`~missy.channels.voice.registry.EdgeNode`
            instances with ``paired=False``.
        """
        return self._registry.list_pending()

    def unpair_node(self, node_id: str) -> None:
        """Remove a previously approved node from the registry.

        Emits a ``voice.pairing.unpaired`` audit event.  No-op if the node
        does not exist.

        Args:
            node_id: The approved node to remove.
        """
        node = self._registry.get_node(node_id)
        if node is None:
            logger.debug("unpair_node: node %r not found — no-op.", node_id)
            return

        self._registry.remove_node(node_id)

        event_bus.publish(
            AuditEvent.now(
                session_id=_PAIRING_SESSION_ID,
                task_id=_PAIRING_TASK_ID,
                event_type="voice.pairing.unpaired",
                category="plugin",
                result="allow",
                detail={
                    "node_id": node_id,
                    "friendly_name": node.friendly_name,
                    "room": node.room,
                },
            )
        )
        logger.info("Unpaired node %r (%s).", node_id, node.friendly_name)
