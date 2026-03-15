"""Edge node device registry for the Missy voice channel.

Maintains a persistent registry of voice-capable edge nodes (e.g. ReSpeaker
devices, Raspberry Pi units) that pair with the Missy voice channel.  The
registry is stored as a JSON file on disk and all mutations are applied
atomically via a write-to-temp-then-rename strategy.  All public methods are
thread-safe through an internal ``threading.RLock``.

Audit events are emitted on the module-level :data:`~missy.core.events.event_bus`
under the ``"plugin"`` category with ``session_id="system"`` and
``task_id="device-registry"``.

Example::

    from missy.channels.voice.registry import DeviceRegistry, EdgeNode

    registry = DeviceRegistry()
    registry.load()
    token = registry.generate_token("node-abc")
    assert registry.verify_token("node-abc", token)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_REGISTRY_SESSION_ID = "system"
_REGISTRY_TASK_ID = "device-registry"
_PBKDF2_ITERATIONS = 100_000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EdgeNode:
    """An edge node device registered with the Missy voice channel.

    Attributes:
        node_id: Unique identifier assigned at pairing time (UUID4 string).
        friendly_name: Human-readable label, e.g. ``"Living Room"``.
        room: Room name surfaced to the agent as context.
        ip_address: Last known IP address or mDNS hostname.
        hardware_profile: Freeform dict describing device hardware, e.g.
            ``{"mic_type": "respeaker", "speaker": True, "channels": 4}``.
        last_seen: Unix timestamp of the last successful contact.
        status: One of ``"online"``, ``"offline"``, or ``"muted"``.
        policy_mode: Capability policy applied to this node — one of
            ``"full"``, ``"safe-chat"``, or ``"muted"``.
        paired: ``True`` once an operator has approved the pairing request.
        token_hash: PBKDF2-HMAC-SHA256 hex digest of the node's auth token.
        audio_logging: Whether audio is persisted to disk for debugging.
        audio_log_dir: Directory path for audio log files.
        audio_log_retention_days: Files older than this many days are purged
            automatically by :meth:`DeviceRegistry.purge_audio_logs`.
        sensor_data: Latest sensor readings: ``occupancy`` (``bool | None``),
            ``noise_level`` (``float | None``), and ``updated_at`` (``float``).
    """

    node_id: str
    friendly_name: str
    room: str
    ip_address: str
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    status: str = "offline"
    policy_mode: str = "full"
    paired: bool = False
    token_hash: str = ""
    audio_logging: bool = False
    audio_log_dir: str = ""
    audio_log_retention_days: int = 7
    sensor_data: dict[str, Any] = field(
        default_factory=lambda: {
            "occupancy": None,
            "noise_level": None,
            "updated_at": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _node_to_dict(node: EdgeNode) -> dict[str, Any]:
    """Convert *node* to a plain dict suitable for JSON serialisation."""
    return asdict(node)


def _node_from_dict(data: dict[str, Any]) -> EdgeNode:
    """Reconstruct an :class:`EdgeNode` from a deserialised dict.

    Unknown keys are silently discarded so that older registry files remain
    loadable after a schema extension.
    """
    valid_fields = EdgeNode.__dataclass_fields__.keys()
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return EdgeNode(**filtered)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DeviceRegistry:
    """Persistent, thread-safe registry of voice-channel edge nodes.

    The registry is backed by a JSON file.  All writes are made atomically:
    the new content is written to a temporary file in the same directory and
    then renamed over the target path, which is an atomic operation on POSIX
    systems.

    Args:
        registry_path: Path to the JSON registry file.  Tilde expansion is
            performed automatically.  Defaults to ``~/.missy/devices.json``.
    """

    def __init__(self, registry_path: str = "~/.missy/devices.json") -> None:
        self._path = Path(registry_path).expanduser()
        self._lock = threading.RLock()
        self._nodes: dict[str, EdgeNode] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the registry from disk.

        If the file does not exist an empty registry is initialised.  Corrupt
        or unreadable files are logged at ERROR level and the registry starts
        empty rather than raising.
        """
        with self._lock:
            if not self._path.exists():
                logger.debug("Registry file %s not found — starting empty.", self._path)
                self._nodes = {}
                return
            try:
                import os
                import stat

                # Validate file ownership and permissions before loading
                st = self._path.stat()
                if st.st_uid != os.getuid():
                    logger.error(
                        "Registry file %s not owned by current user — refusing to load.",
                        self._path,
                    )
                    self._nodes = {}
                    return
                if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                    logger.error(
                        "Registry file %s is group/world-writable — refusing to load.",
                        self._path,
                    )
                    self._nodes = {}
                    return

                raw = self._path.read_text(encoding="utf-8")
                data: list[dict[str, Any]] = json.loads(raw)
                self._nodes = {entry["node_id"]: _node_from_dict(entry) for entry in data}
                logger.debug("Loaded %d node(s) from %s.", len(self._nodes), self._path)
            except Exception:
                logger.exception(
                    "Failed to load device registry from %s — starting empty.", self._path
                )
                self._nodes = {}

    def save(self) -> None:
        """Atomically persist the current registry state to disk.

        The directory is created if it does not already exist.  A
        ``PermissionError`` or ``OSError`` is logged and re-raised so that
        callers can handle it.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                [_node_to_dict(n) for n in self._nodes.values()],
                indent=2,
                ensure_ascii=False,
            )
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent, prefix=".devices_", suffix=".json.tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp_path, self._path)
            except Exception:
                # Best-effort cleanup of the temp file.
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> EdgeNode | None:
        """Return the node with the given *node_id*, or ``None`` if absent.

        Args:
            node_id: The unique node identifier to look up.

        Returns:
            The :class:`EdgeNode` instance, or ``None``.
        """
        with self._lock:
            return self._nodes.get(node_id)

    def add_node(self, node: EdgeNode) -> None:
        """Register *node* in the registry and persist immediately.

        Emits a ``voice.device.registered`` audit event.

        Args:
            node: The :class:`EdgeNode` to add.  If a node with the same
                ``node_id`` already exists it is silently replaced.
        """
        with self._lock:
            self._nodes[node.node_id] = node
            self.save()

        event_bus.publish(
            AuditEvent.now(
                session_id=_REGISTRY_SESSION_ID,
                task_id=_REGISTRY_TASK_ID,
                event_type="voice.device.registered",
                category="plugin",
                result="allow",
                detail={
                    "node_id": node.node_id,
                    "friendly_name": node.friendly_name,
                    "room": node.room,
                    "paired": node.paired,
                },
            )
        )
        logger.info("Registered edge node %r (%s).", node.node_id, node.friendly_name)

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        """Apply a partial update to an existing node and persist.

        Only fields that exist on :class:`EdgeNode` are accepted; unknown
        keyword arguments are silently ignored.

        Args:
            node_id: The target node identifier.
            **kwargs: Field name/value pairs to update.

        Raises:
            KeyError: If *node_id* is not found in the registry.
        """
        valid_fields = EdgeNode.__dataclass_fields__.keys()
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(f"Node not found: {node_id!r}")
            for key, value in kwargs.items():
                if key in valid_fields:
                    object.__setattr__(node, key, value)
                else:
                    logger.debug("update_node: ignoring unknown field %r.", key)
            self.save()

    def remove_node(self, node_id: str) -> None:
        """Delete *node_id* from the registry and persist.

        Emits a ``voice.device.removed`` audit event.  No-op if the node does
        not exist.

        Args:
            node_id: The identifier of the node to remove.
        """
        with self._lock:
            node = self._nodes.pop(node_id, None)
            if node is None:
                logger.debug("remove_node: node %r not found — no-op.", node_id)
                return
            self.save()

        event_bus.publish(
            AuditEvent.now(
                session_id=_REGISTRY_SESSION_ID,
                task_id=_REGISTRY_TASK_ID,
                event_type="voice.device.removed",
                category="plugin",
                result="allow",
                detail={"node_id": node_id, "friendly_name": node.friendly_name},
            )
        )
        logger.info("Removed edge node %r.", node_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_nodes(self) -> list[EdgeNode]:
        """Return all registered nodes (both pending and approved).

        Returns:
            A snapshot list of all :class:`EdgeNode` instances.
        """
        with self._lock:
            return list(self._nodes.values())

    def list_paired(self) -> list[EdgeNode]:
        """Return only approved (``paired=True``) nodes.

        Returns:
            A list of :class:`EdgeNode` instances with ``paired=True``.
        """
        with self._lock:
            return [n for n in self._nodes.values() if n.paired]

    def list_pending(self) -> list[EdgeNode]:
        """Return only nodes awaiting operator approval (``paired=False``).

        Returns:
            A list of :class:`EdgeNode` instances with ``paired=False``.
        """
        with self._lock:
            return [n for n in self._nodes.values() if not n.paired]

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def approve_node(self, node_id: str) -> None:
        """Mark *node_id* as approved and persist.

        Emits a ``voice.device.approved`` audit event.

        Args:
            node_id: The node to approve.

        Raises:
            KeyError: If *node_id* is not present in the registry.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(f"Node not found: {node_id!r}")
            node.paired = True
            self.save()

        event_bus.publish(
            AuditEvent.now(
                session_id=_REGISTRY_SESSION_ID,
                task_id=_REGISTRY_TASK_ID,
                event_type="voice.device.approved",
                category="plugin",
                result="allow",
                detail={"node_id": node_id},
            )
        )
        logger.info("Approved edge node %r.", node_id)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _hash_token(self, node_id: str, token: str) -> str:
        """Return the PBKDF2-HMAC-SHA256 hex digest of *token* salted with *node_id*."""
        raw = hashlib.pbkdf2_hmac(
            "sha256",
            token.encode(),
            node_id.encode(),
            iterations=_PBKDF2_ITERATIONS,
        )
        return raw.hex()

    def generate_token(self, node_id: str) -> str:
        """Generate a fresh auth token for *node_id*, store its hash, and return the plaintext.

        Calling this method again for the same node replaces the previous
        token — the old token will no longer authenticate.

        Args:
            node_id: The node for which to generate a token.

        Returns:
            The 32-byte URL-safe base64-encoded plaintext token.  This value
            is shown exactly once; the registry only stores the hash.

        Raises:
            KeyError: If *node_id* is not present in the registry.
        """
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(node_id, token)
        self.update_node(node_id, token_hash=token_hash)
        logger.debug("Generated new token for node %r.", node_id)
        return token

    def verify_token(self, node_id: str, token: str) -> bool:
        """Return ``True`` if *token* matches the stored hash for *node_id*.

        Uses :func:`hmac.compare_digest` via constant-time comparison to
        prevent timing attacks.

        Args:
            node_id: The node to authenticate.
            token: The plaintext token received from the device.

        Returns:
            ``True`` if the token is valid, ``False`` otherwise (including
            when the node does not exist or has no stored hash).
        """
        import hmac

        node = self.get_node(node_id)
        if node is None or not node.token_hash:
            return False
        candidate = self._hash_token(node_id, token)
        return hmac.compare_digest(candidate, node.token_hash)

    # ------------------------------------------------------------------
    # Presence / sensor data
    # ------------------------------------------------------------------

    def update_sensor_data(
        self,
        node_id: str,
        occupancy: bool | None,
        noise_level: float | None,
    ) -> None:
        """Persist the latest sensor readings for *node_id*.

        Args:
            node_id: The target node.
            occupancy: ``True`` = room occupied, ``False`` = empty,
                ``None`` = unknown.
            noise_level: Normalised ambient noise level in ``[0.0, 1.0]``,
                or ``None`` if not available.

        Raises:
            KeyError: If *node_id* is not present in the registry.
        """
        sensor_data = {
            "occupancy": occupancy,
            "noise_level": noise_level,
            "updated_at": time.time(),
        }
        self.update_node(node_id, sensor_data=sensor_data)

    def mark_online(self, node_id: str, ip_address: str) -> None:
        """Record that *node_id* is reachable at *ip_address*.

        Updates ``status`` to ``"online"``, ``last_seen`` to the current time,
        and ``ip_address`` to the supplied value.

        Args:
            node_id: The node that checked in.
            ip_address: The node's current IP address or mDNS hostname.

        Raises:
            KeyError: If *node_id* is not present in the registry.
        """
        self.update_node(
            node_id,
            status="online",
            last_seen=time.time(),
            ip_address=ip_address,
        )

    def mark_offline(self, node_id: str) -> None:
        """Record that *node_id* is no longer reachable.

        Sets ``status`` to ``"offline"``.

        Args:
            node_id: The node that went offline.

        Raises:
            KeyError: If *node_id* is not present in the registry.
        """
        self.update_node(node_id, status="offline")

    # ------------------------------------------------------------------
    # Audio log housekeeping
    # ------------------------------------------------------------------

    def purge_audio_logs(self) -> int:
        """Delete audio log files older than each node's retention policy.

        Iterates over all nodes with ``audio_logging=True`` and removes files
        in ``audio_log_dir`` whose modification time predates
        ``audio_log_retention_days`` days ago.

        Returns:
            Total number of files deleted across all nodes.

        Note:
            Errors encountered while deleting individual files are logged at
            WARNING level and skipped rather than propagated.
        """
        deleted_total = 0
        now = time.time()

        with self._lock:
            nodes_snapshot = [
                n for n in self._nodes.values() if n.audio_logging and n.audio_log_dir
            ]

        for node in nodes_snapshot:
            cutoff = now - node.audio_log_retention_days * 86_400
            log_dir = Path(node.audio_log_dir).expanduser()
            if not log_dir.is_dir():
                logger.debug(
                    "purge_audio_logs: directory %s for node %r does not exist — skipping.",
                    log_dir,
                    node.node_id,
                )
                continue

            for entry in log_dir.iterdir():
                if not entry.is_file():
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    try:
                        entry.unlink()
                        deleted_total += 1
                        logger.debug("purge_audio_logs: deleted %s (node %r).", entry, node.node_id)
                    except OSError:
                        logger.warning(
                            "purge_audio_logs: could not delete %s for node %r.",
                            entry,
                            node.node_id,
                            exc_info=True,
                        )

        if deleted_total:
            logger.info("purge_audio_logs: deleted %d file(s) total.", deleted_total)
        return deleted_total
