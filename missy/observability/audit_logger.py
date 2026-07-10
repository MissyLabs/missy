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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from missy.core.events import AuditEvent, EventBus, event_bus
from missy.security.censor import censor_response

_module_logger = logging.getLogger(__name__)


def _redact_detail(value: Any) -> Any:
    """Recursively redact secret-shaped substrings within *value*.

    SR-1.10: audit events were persisted to disk verbatim with no
    redaction of any kind -- full egress URLs with query-string secrets
    (e.g. ``?key=...``, AWS presigned ``X-Amz-Signature=...``) and raw
    provider/gateway exception text are logged on every allowed request.
    ``api/audit_browser.py`` only redacts at *display* time, which
    cannot repair what has already been written to the JSONL file --
    the redaction has to happen before the write, at the one place every
    published :class:`~missy.core.events.AuditEvent` passes through
    (here), rather than requiring every individual publisher (policy
    engines, the HTTP gateway, tools, providers) to remember to redact
    its own ``detail`` dict.

    Args:
        value: Any JSON-serialisable value — typically an
            :attr:`AuditEvent.detail` dict, which may nest further
            dicts/lists of strings.

    Returns:
        A structurally identical value with every string leaf passed
        through :func:`~missy.security.censor.censor_response`.
        Non-string, non-container values (numbers, bools, ``None``) are
        returned unchanged.
    """
    if isinstance(value, str):
        return censor_response(value)
    if isinstance(value, dict):
        return {k: _redact_detail(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        redacted = [_redact_detail(v) for v in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


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
        identity: Any | None = None,
    ) -> None:
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._bus: EventBus = bus if bus is not None else event_bus
        # SR-1.1: an AgentIdentity to sign every persisted event with.
        # Direct construction defaults to unsigned (no implicit key I/O
        # for the many callers -- CLI read-only viewers, tests -- that
        # only ever read the log); init_audit_logger(), the documented
        # production entry point, resolves and passes a real identity by
        # default so the actual running gateway/CLI signs every event.
        self._identity = identity
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

        SR-1.1: when this logger has an :class:`~missy.security.identity.AgentIdentity`
        (see ``identity``/:func:`init_audit_logger`), every field of the
        persisted record is signed here -- the one place every published
        event, of any type, actually reaches disk (this class's whole
        raison d'etre per its own docstring) -- and the signature is
        stored as a sibling top-level ``identity_signature`` field, never
        nested inside the mutable ``detail`` dict. This supersedes the
        old, narrower signing previously done in
        ``AgentRuntime._emit_event`` (signed only 3 fields, embedded the
        signature inside ``detail``, and only covered events emitted via
        that one method).

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
            "detail": _redact_detail(event.detail),
            "policy_rule": event.policy_rule,
        }
        if self._identity is not None:
            try:
                payload = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
                record["identity_signature"] = self._identity.sign(payload).hex()
            except Exception:
                _module_logger.warning(
                    "AuditLogger: failed to sign event %r; writing unsigned",
                    event.event_type,
                    exc_info=True,
                )
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


def _make_default_identity() -> Any | None:
    """Resolve the process-level :class:`~missy.security.identity.AgentIdentity`.

    Graceful degradation matching every other ``_make_*`` subsystem
    factory in this codebase: returns ``None`` (unsigned logging) rather
    than raising when the identity module or key file is unavailable,
    but logs loudly at WARNING -- unlike a missing optional subsystem,
    an audit trail that silently stopped being signed is a security-
    relevant fact an operator should see.
    """
    try:
        from missy.security.identity import AgentIdentity

        return AgentIdentity.load_or_generate()
    except Exception:
        _module_logger.warning(
            "AuditLogger: agent identity unavailable -- audit events will "
            "NOT be signed, and integrity cannot be verified.",
            exc_info=True,
        )
        return None


def init_audit_logger(
    log_path: str = "~/.missy/audit.jsonl", identity: Any | None = None
) -> AuditLogger:
    """Initialise and install the process-level :class:`AuditLogger`.

    Calling this function a second time replaces the existing logger with a
    new one targeting *log_path*. This is the documented production entry
    point (per this module's docstring), so unlike direct
    :class:`AuditLogger` construction, it resolves and attaches a real
    signing identity by default (SR-1.1). Pass an already-loaded
    identity explicitly to reuse one (e.g. the same instance
    :class:`~missy.agent.runtime.AgentRuntime` uses) instead of
    resolving a second one independently.

    Args:
        log_path: Destination file for audit events.
        identity: An :class:`~missy.security.identity.AgentIdentity` to
            sign every persisted event with. When ``None`` (default),
            one is resolved via :func:`_make_default_identity`.

    Returns:
        The newly created :class:`AuditLogger` instance.
    """
    global _audit_logger
    if identity is None:
        identity = _make_default_identity()
    _audit_logger = AuditLogger(log_path=log_path, identity=identity)
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


# ---------------------------------------------------------------------------
# SR-1.1: signature verification
# ---------------------------------------------------------------------------

# Per-line verification outcomes.
#   "valid"     -- signature present and matches the recomputed payload.
#   "tampered"  -- signature present but does not match (record was
#                  edited after signing, or the hex signature itself is
#                  malformed/truncated).
#   "unsigned"  -- no identity_signature field at all (written before
#                  signing was enabled, or with signing unavailable).
#   "malformed" -- the line is not valid JSON at all.
VerificationStatus = Literal["valid", "tampered", "unsigned", "malformed"]


@dataclass(frozen=True)
class AuditLineVerification:
    """The verification outcome for a single audit log line.

    Attributes:
        line_number: 1-indexed line number within the log file.
        status: One of :data:`VerificationStatus`.
        event_type: The record's ``event_type``, when the line parsed as
            JSON (``None`` for a ``"malformed"`` line).
    """

    line_number: int
    status: VerificationStatus
    event_type: str | None = None


def verify_audit_log(log_path: str, identity: Any) -> list[AuditLineVerification]:
    """Verify every signed line in the audit log at *log_path*.

    Recomputes, for each line, the exact canonical payload
    :meth:`AuditLogger._handle_event` signed (every field except
    ``identity_signature`` itself, JSON-serialised with ``sort_keys=True``
    -- key order in the persisted line doesn't matter, since this
    normalises it identically regardless) and checks it against the
    stored signature with *identity*'s public key. This is the
    verification counterpart the security review found entirely absent:
    signing without any verification path provides no actual tamper
    detection, since nothing would ever notice a mismatch.

    Args:
        log_path: Path to the JSONL audit log file.
        identity: An :class:`~missy.security.identity.AgentIdentity`
            (only its public key is used) to verify against -- must be
            the same identity (or one sharing the same keypair) the log
            was signed with, or every line will report ``"tampered"``.

    Returns:
        One :class:`AuditLineVerification` per non-blank line, in file
        order. Returns an empty list if the file does not exist.
    """
    path = Path(log_path).expanduser()
    if not path.exists():
        return []

    results: list[AuditLineVerification] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                results.append(AuditLineVerification(line_number, "malformed"))
                continue

            event_type = record.get("event_type") if isinstance(record, dict) else None
            sig_hex = record.pop("identity_signature", None) if isinstance(record, dict) else None
            if sig_hex is None:
                results.append(AuditLineVerification(line_number, "unsigned", event_type))
                continue

            try:
                signature = bytes.fromhex(sig_hex)
                payload = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
                valid = identity.verify(payload, signature)
            except Exception:
                valid = False

            status: VerificationStatus = "valid" if valid else "tampered"
            results.append(AuditLineVerification(line_number, status, event_type))

    return results
