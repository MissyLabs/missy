"""Presence and occupancy tracking for voice-channel edge nodes.

:class:`PresenceStore` maintains an in-memory map of per-room presence data
sourced from edge node sensor reports.  On construction it seeds itself from
the ``sensor_data`` fields already stored in the :class:`DeviceRegistry`, so
historical readings survive process restarts (they are persisted by the
registry).

The store does **not** have its own persistence layer.  Long-term durability
is handled by the registry's ``sensor_data`` field; the in-memory dict here
is a fast-lookup projection of that data.

Example::

    from missy.channels.voice.registry import DeviceRegistry
    from missy.channels.voice.presence import PresenceStore

    registry = DeviceRegistry()
    registry.load()
    store = PresenceStore(registry)

    store.update("node-abc", occupancy=True, noise_level=0.3)
    print(store.get_context_summary())
    # "Living Room: occupied | Bedroom: unknown"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from missy.channels.voice.registry import DeviceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PresenceData:
    """Current presence reading for a single edge node.

    Attributes:
        node_id: The unique node identifier.
        room: Room name sourced from the node's registry entry.
        occupancy: ``True`` = room occupied, ``False`` = empty,
            ``None`` = sensor reading not yet available.
        noise_level: Normalised ambient noise level in ``[0.0, 1.0]``,
            or ``None`` if not available.
        wake_word_false_positives: Cumulative count of spurious wake-word
            detections since the last :meth:`~PresenceStore.reset_false_positives`
            call.
        updated_at: Unix timestamp of the most recent update.
    """

    node_id: str
    room: str
    occupancy: bool | None = None
    noise_level: float | None = None
    wake_word_false_positives: int = 0
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PresenceStore:
    """In-memory presence store backed by the device registry.

    The store is seeded from the registry on construction.  All subsequent
    updates are applied both to the in-memory dict and, for ``occupancy`` and
    ``noise_level``, pushed back to the registry's persistent ``sensor_data``
    field so readings survive a restart.

    The store is **not** thread-safe by itself; if it is shared across
    threads, external locking is required.  (The :class:`DeviceRegistry`
    methods it delegates to are themselves thread-safe.)

    Args:
        registry: The :class:`DeviceRegistry` to read and update.
    """

    def __init__(self, registry: DeviceRegistry) -> None:
        self._registry = registry
        self._data: dict[str, PresenceData] = {}
        self._seed_from_registry()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _seed_from_registry(self) -> None:
        """Populate the in-memory map from existing registry sensor_data."""
        for node in self._registry.list_nodes():
            sd = node.sensor_data or {}
            self._data[node.node_id] = PresenceData(
                node_id=node.node_id,
                room=node.room,
                occupancy=sd.get("occupancy"),
                noise_level=sd.get("noise_level"),
                wake_word_false_positives=0,
                updated_at=sd.get("updated_at", time.time()),
            )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(
        self,
        node_id: str,
        occupancy: bool | None = None,
        noise_level: float | None = None,
        wake_word_fp: bool = False,
    ) -> None:
        """Update presence data for *node_id* and sync to the registry.

        Fields not supplied (``None``) are left unchanged.

        Args:
            node_id: The node whose presence data to update.
            occupancy: ``True`` if the room is occupied, ``False`` if empty,
                ``None`` to leave unchanged.
            noise_level: Normalised noise level in ``[0.0, 1.0]``, or
                ``None`` to leave unchanged.
            wake_word_fp: If ``True``, increment the false-positive counter
                by one.

        Note:
            If *node_id* is not yet in the in-memory store (e.g. it was
            registered after the store was created) a new :class:`PresenceData`
            is created on-the-fly using metadata from the registry.  If the
            node is also absent from the registry a ``KeyError`` is raised by
            the underlying :meth:`~DeviceRegistry.update_sensor_data` call.
        """
        existing = self._data.get(node_id)
        if existing is None:
            node = self._registry.get_node(node_id)
            if node is None:
                raise KeyError(f"Node not found in registry: {node_id!r}")
            existing = PresenceData(node_id=node_id, room=node.room)
            self._data[node_id] = existing

        if occupancy is not None:
            existing.occupancy = occupancy
        if noise_level is not None:
            existing.noise_level = noise_level
        if wake_word_fp:
            existing.wake_word_false_positives += 1
        existing.updated_at = time.time()

        # Push occupancy + noise_level back into the registry for persistence.
        try:
            self._registry.update_sensor_data(
                node_id,
                occupancy=existing.occupancy,
                noise_level=existing.noise_level,
            )
        except KeyError:
            logger.warning(
                "update: node %r disappeared from registry during presence update.",
                node_id,
            )

    def reset_false_positives(self, node_id: str) -> None:
        """Reset the wake-word false-positive counter for *node_id* to zero.

        Args:
            node_id: The node whose counter to reset.

        Note:
            No-op if *node_id* is not present in the in-memory store.
        """
        entry = self._data.get(node_id)
        if entry is not None:
            entry.wake_word_false_positives = 0
            logger.debug("reset_false_positives: reset counter for node %r.", node_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> PresenceData | None:
        """Return the :class:`PresenceData` for *node_id*, or ``None``.

        Args:
            node_id: The node identifier to look up.

        Returns:
            The :class:`PresenceData` instance, or ``None`` if unknown.
        """
        return self._data.get(node_id)

    def get_all(self) -> list[PresenceData]:
        """Return a snapshot list of all presence records.

        Returns:
            A list of all :class:`PresenceData` instances currently tracked.
        """
        return list(self._data.values())

    def get_occupied_rooms(self) -> list[str]:
        """Return the names of rooms currently reported as occupied.

        Returns:
            A list of room-name strings where ``occupancy=True``.  The list
            is in an unspecified order.
        """
        return [pd.room for pd in self._data.values() if pd.occupancy is True]

    def get_context_summary(self) -> str:
        """Return a human-readable summary for agent context injection.

        Produces a pipe-delimited string of ``"Room: state"`` pairs.  Unknown
        occupancy is represented as ``"unknown"``.

        Returns:
            A string such as
            ``"Living Room: occupied | Bedroom: empty | Kitchen: unknown"``,
            or ``"(no nodes registered)"`` when the store is empty.

        Example::

            store.update("node-a", occupancy=True)
            store.update("node-b", occupancy=False)
            print(store.get_context_summary())
            # "Living Room: occupied | Bedroom: empty"
        """
        if not self._data:
            return "(no nodes registered)"

        parts: list[str] = []
        for pd in self._data.values():
            if pd.occupancy is True:
                state = "occupied"
            elif pd.occupancy is False:
                state = "empty"
            else:
                state = "unknown"
            parts.append(f"{pd.room}: {state}")

        return " | ".join(parts)
