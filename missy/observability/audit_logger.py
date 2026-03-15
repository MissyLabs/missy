"""Structured audit logging to file.

:class:`AuditLogger` subscribes to every event published on
:data:`~missy.core.events.event_bus` and appends a JSON line to the
configured audit log file.  Because :class:`~missy.core.events.EventBus`
does not support wildcard subscriptions, the logger intercepts events by
wrapping the bus's :meth:`~missy.core.events.EventBus.publish` method at
initialisation time.  The wrapper calls the original implementation first
(preserving all existing behaviour) then forwards the event to
:meth:`AuditLogger._handle_event`.

Only one :class:`AuditLogger` should be active per process.  Use
:func:`init_audit_logger` to create and install the singleton.

Example::

    from missy.observability.audit_logger import init_audit_logger

    logger = init_audit_logger("~/.missy/audit.jsonl")
    recent = logger.get_recent_events(limit=20)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from missy.core.events import AuditEvent, EventBus, event_bus

_module_logger = logging.getLogger(__name__)


class AuditLogger:
    """Subscribes to all events on :data:`event_bus` and writes JSON audit logs.

    Events are appended as newline-delimited JSON (JSONL) to *log_path*.
    The parent directory is created automatically on initialisation.

    The logger wraps :meth:`~missy.core.events.EventBus.publish` on the
    supplied *bus* instance so that every published event — regardless of
    its ``event_type`` — is captured without requiring per-type subscription
    registrations.

    Args:
        log_path: Path to the JSONL audit log file.  Tilde expansion is
            applied automatically.
        bus: The :class:`~missy.core.events.EventBus` instance to attach
            to.  Defaults to the process-level :data:`event_bus` singleton.
    """

    def __init__(
        self,
        log_path: str = "~/.missy/audit.jsonl",
        bus: EventBus | None = None,
    ) -> None:
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._bus: EventBus = bus if bus is not None else event_bus
        self._subscribe()

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        """Wrap the bus's publish method to intercept all events.

        The original :meth:`~missy.core.events.EventBus.publish` is stored
        and called first; events are then forwarded to
        :meth:`_handle_event`.  This is safe to call multiple times — each
        call wraps the *current* publish method, which already includes any
        previously installed wrappers.
        """
        original_publish = self._bus.publish

        def _patched_publish(event: AuditEvent) -> None:
            original_publish(event)
            try:
                self._handle_event(event)
            except Exception:
                _module_logger.exception("AuditLogger failed to handle event %r", event.event_type)

        # Bind the patched method onto the bus instance so it replaces the
        # original for all callers sharing the same bus object.
        import types

        self._bus.publish = types.MethodType(  # type: ignore[method-assign]
            lambda _self, event: _patched_publish(event),
            self._bus,
        )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: AuditEvent) -> None:
        """Write *event* as a JSON line to the audit log file.

        The line includes all :class:`~missy.core.events.AuditEvent` fields
        with the ``timestamp`` serialised to an ISO-8601 string.

        Args:
            event: The audit event to persist.
        """
        record: dict[str, Any] = {
            "timestamp": event.timestamp.isoformat(),
            "session_id": event.session_id,
            "task_id": event.task_id,
            "event_type": event.event_type,
            "category": event.category,
            "result": event.result,
            "detail": event.detail,
            "policy_rule": event.policy_rule,
        }
        try:
            line = json.dumps(record, default=str)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            _module_logger.error(
                "AuditLogger: failed to write event %r to %s: %s",
                event.event_type,
                self.log_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def _read_tail_lines(self, limit: int) -> list[str]:
        """Read the last *limit* non-empty lines from the audit log.

        Uses a seek-from-end strategy to avoid loading the entire file for
        large audit logs.  Falls back to full read for small files.
        """
        file_size = self.log_path.stat().st_size
        if file_size == 0:
            return []
        # Estimate ~1KB per JSON event line; read enough to cover limit.
        read_size = min(file_size, limit * 2048)
        with open(self.log_path, "rb") as fh:
            fh.seek(max(0, file_size - read_size))
            chunk = fh.read().decode("utf-8", errors="replace")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        # When we seeked past the start, the first line may be truncated.
        if file_size > read_size and lines:
            lines = lines[1:]
        return lines[-limit:]

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the last *limit* events from the audit log file.

        Args:
            limit: Maximum number of events to return.  Events are returned
                in chronological order (oldest first within the window).

        Returns:
            A list of dicts, each representing one audit event.  Returns an
            empty list when the log file does not exist or cannot be read.
        """
        if not self.log_path.exists():
            return []

        try:
            lines = self._read_tail_lines(limit)
        except Exception as exc:
            _module_logger.error("Failed to read audit log %s: %s", self.log_path, exc)
            return []

        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _module_logger.warning("Skipping malformed audit log line: %s", exc)
        return events

    def get_policy_violations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return events where ``result == "deny"``.

        Scans the entire audit log from newest to oldest and returns the
        first *limit* matching events in chronological order.

        Args:
            limit: Maximum number of violation events to return.

        Returns:
            A list of dicts representing policy-denial audit events.
        """
        if not self.log_path.exists():
            return []

        try:
            # Read more lines than needed since not all will be violations.
            lines = self._read_tail_lines(limit * 10)
        except Exception as exc:
            _module_logger.error("Failed to read audit log %s: %s", self.log_path, exc)
            return []

        violations: list[dict[str, Any]] = []
        # Iterate newest-first to collect up to *limit* violations efficiently.
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("result") == "deny":
                violations.append(record)
                if len(violations) >= limit:
                    break

        # Return in chronological order.
        violations.reverse()
        return violations


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_audit_logger: AuditLogger | None = None


def init_audit_logger(log_path: str = "~/.missy/audit.jsonl") -> AuditLogger:
    """Initialise and install the process-level :class:`AuditLogger`.

    Calling this function a second time replaces the existing logger with a
    new one targeting *log_path*.

    Args:
        log_path: Destination file for audit events.

    Returns:
        The newly created :class:`AuditLogger` instance.
    """
    global _audit_logger
    _audit_logger = AuditLogger(log_path=log_path)
    return _audit_logger


def get_audit_logger() -> AuditLogger:
    """Return the process-level :class:`AuditLogger`.

    Returns:
        The currently installed :class:`AuditLogger`.

    Raises:
        RuntimeError: When :func:`init_audit_logger` has not yet been called.
    """
    if _audit_logger is None:
        raise RuntimeError(
            "AuditLogger has not been initialised. "
            "Call missy.observability.audit_logger.init_audit_logger() first."
        )
    return _audit_logger
