"""Session lifecycle management for the Missy framework."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


class SessionMode(str, enum.Enum):
    """Capability mode for a session."""

    FULL = "full"          # All policy-approved capabilities available.
    NO_TOOLS = "no_tools"  # Tools disabled; LLM chat only.
    SAFE_CHAT = "safe_chat"  # No tools, no skills, no plugins.


@dataclass
class Session:
    """Represents an active Missy agent session.

    Attributes:
        id: Unique session identifier.
        created_at: UTC timestamp of session creation.
        metadata: Arbitrary key/value data attached to the session.
        mode: Capability mode controlling which features are active.
    """

    id: UUID
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    mode: SessionMode = SessionMode.FULL

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("Session.created_at must be timezone-aware.")


class SessionManager:
    """Thread-safe manager for the current agent session.

    A single ``SessionManager`` is intended to be used as a process-level
    singleton.  The active session is stored in a ``threading.local`` slot so
    that each thread can independently hold its own session reference while
    still sharing the same manager instance.

    Example::

        manager = SessionManager()
        session = manager.create_session()
        assert manager.get_current_session() is session
    """

    def __init__(self) -> None:
        self._local: threading.local = threading.local()

    # ------------------------------------------------------------------
    # ID factories
    # ------------------------------------------------------------------

    @staticmethod
    def generate_session_id() -> UUID:
        """Return a new random UUID suitable for use as a session ID."""
        return uuid4()

    @staticmethod
    def generate_task_id() -> UUID:
        """Return a new random UUID suitable for use as a task ID."""
        return uuid4()

    # ------------------------------------------------------------------
    # Session accessors
    # ------------------------------------------------------------------

    def get_current_session(self) -> Session | None:
        """Return the session bound to the current thread, or *None*.

        Returns:
            The active :class:`Session` for this thread, or ``None`` if no
            session has been created yet.
        """
        return getattr(self._local, "session", None)

    def create_session(self, metadata: dict[str, Any] | None = None) -> Session:
        """Create a new session and bind it to the current thread.

        Any previously active session on this thread is replaced.

        Args:
            metadata: Optional mapping of arbitrary data to attach to the
                newly created session.

        Returns:
            The newly created and bound :class:`Session`.
        """
        session = Session(
            id=self.generate_session_id(),
            created_at=datetime.now(tz=UTC),
            metadata=metadata or {},
        )
        self._local.session = session
        return session

    def clear_session(self) -> None:
        """Remove the session bound to the current thread, if any."""
        self._local.session = None
