"""Background agent runs with Server-Sent Events streaming for the Web TUI.

:class:`RunRegistry` executes :meth:`AgentRuntime.run` on a background thread
so the API server can return immediately with a ``run_id`` and let the
browser watch progress (tool calls, completion, errors) over a live stream
instead of blocking on a single synchronous HTTP request. Progress is sourced
from the process-level :class:`~missy.core.message_bus.MessageBus`, which the
runtime already publishes ``agent.run.*`` and ``tool.*`` events to.

Only one run may be in flight per session at a time — this keeps the event
stream unambiguous (no interleaving of two runs' tool calls) and prevents an
operator from firing overlapping runs into the same conversation history.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value

if TYPE_CHECKING:
    from missy.agent.runtime import AgentRuntime
    from missy.core.message_bus import BusMessage

logger = logging.getLogger(__name__)

# Sentinel placed on a run's event queue to signal stream completion.
_STREAM_DONE = object()

# Bus topics forwarded into a run's event stream. Completion/error events are
# emitted directly by this module (from the return value of ``runtime.run``)
# rather than mirrored from the bus, so they are not included here.
_RUN_TOPICS = ("agent.run.start", "tool.request", "tool.result")

# Bus topic carrying the resolved provider, tools used, and cost summary for
# a finished run. Subscribed separately from ``_RUN_TOPICS`` because its
# payload is folded into the synthesized ``run.complete``/``run.error``
# event rather than forwarded verbatim.
_SUMMARY_TOPIC = "agent.run.complete"

_EVENT_NAME_BY_TOPIC = {
    "agent.run.start": "run.start",
    "tool.request": "tool.request",
    "tool.result": "tool.result",
}

_TERMINAL_STATUSES = frozenset({"complete", "error"})

_MAX_QUEUE_EVENTS = 500
_SSE_KEEPALIVE_SECONDS = 15.0
_RUN_TTL_SECONDS = 30 * 60
_MAX_TRACKED_RUNS = 500


class RunConflictError(Exception):
    """Raised when a session already has an in-flight run."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id!r} already has an active run")
        self.session_id = session_id


@dataclass(eq=False)
class RunHandle:
    """Mutable state for a single background run."""

    run_id: str
    session_id: str
    provider: str
    message: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "pending"  # pending -> running -> complete | error
    task_id: str | None = None
    response: str | None = None
    error: str | None = None
    finished_at: str | None = None
    resolved_provider: str = ""
    tools_used: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    _queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=_MAX_QUEUE_EVENTS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "status": self.status,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "response": self.response,
            "error": self.error,
            "resolved_provider": self.resolved_provider,
            "tools_used": self.tools_used,
            "cost": self.cost,
        }

    def push(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(event)


def _default_bus() -> Any | None:
    try:
        from missy.core.message_bus import get_message_bus

        return get_message_bus()
    except Exception:
        return None


class RunRegistry:
    """Starts, tracks, and streams background agent runs."""

    def __init__(self, *, bus_factory: Any = None) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._active_sessions: set[str] = set()
        self._lock = threading.Lock()
        self._bus_factory = bus_factory or _default_bus

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        runtime: AgentRuntime,
        message: str,
        session_id: str,
        provider: str = "",
    ) -> RunHandle:
        """Start *message* running against *runtime* on a background thread.

        Raises:
            RunConflictError: If *session_id* already has a run in flight.
        """
        with self._lock:
            if session_id in self._active_sessions:
                raise RunConflictError(session_id)
            self._active_sessions.add(session_id)
            self._prune_locked()

            run_id = str(uuid.uuid4())
            handle = RunHandle(
                run_id=run_id, session_id=session_id, provider=provider, message=message
            )
            self._runs[run_id] = handle

        thread = threading.Thread(
            target=self._execute,
            args=(runtime, handle),
            name=f"missy-run-{run_id[:8]}",
            daemon=True,
        )
        thread.start()
        return handle

    def get(self, run_id: str) -> RunHandle | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_for_session(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent runs for *session_id*, most-recent first."""
        with self._lock:
            matches = [r for r in self._runs.values() if r.session_id == session_id]
        matches.sort(key=lambda r: r.created_at, reverse=True)
        return [r.to_dict() for r in matches[:limit]]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, runtime: AgentRuntime, handle: RunHandle) -> None:
        bus = self._bus_factory()
        handler = None
        summary_handler = None
        if bus is not None:

            def handler(msg: BusMessage, *, _handle: RunHandle = handle) -> None:
                if msg.payload.get("session_id") != _handle.session_id:
                    return
                self._on_bus_message(_handle, msg)

            def summary_handler(msg: BusMessage, *, _handle: RunHandle = handle) -> None:
                if msg.payload.get("session_id") != _handle.session_id:
                    return
                _handle.resolved_provider = str(msg.payload.get("provider") or "")
                _handle.tools_used = list(msg.payload.get("tools_used") or [])
                _handle.cost = redact_audit_value(dict(msg.payload.get("cost") or {}))

            for topic in _RUN_TOPICS:
                bus.subscribe(topic, handler)
            bus.subscribe(_SUMMARY_TOPIC, summary_handler)

        handle.status = "running"
        handle.push(
            {
                "event": "run.started",
                "data": {
                    "run_id": handle.run_id,
                    "session_id": handle.session_id,
                    "message": redact_audit_value(handle.message),
                },
            }
        )
        try:
            response = runtime.run(handle.message, session_id=handle.session_id)
        except Exception as exc:
            handle.status = "error"
            handle.error = str(redact_audit_value(str(exc)))
            handle.finished_at = datetime.now(UTC).isoformat()
            handle.push(
                {
                    "event": "run.error",
                    "data": {"run_id": handle.run_id, "error": handle.error},
                }
            )
            logger.warning("Background run %s failed: %s", handle.run_id, handle.error)
        else:
            handle.status = "complete"
            # Unlike POST /api/v1/chat (which censors response_text via
            # censor_response() before returning it), this background-run
            # path previously stored/streamed the raw agent response with
            # no redaction at all -- every other field pushed by this same
            # method (handle.message, handle.error, cost) already goes
            # through redact_audit_value(), which uses the same
            # SecretsDetector-backed redaction censor_response() does. If
            # the agent's final answer echoes a credential (e.g. quoting a
            # config value or a leaked API key from its own context), a
            # client polling GET /api/v1/runs/{run_id} or the SSE stream
            # got it unredacted, while the identical content through
            # /chat would have been redacted.
            redacted_response = redact_audit_value(response)
            handle.response = redacted_response
            handle.finished_at = datetime.now(UTC).isoformat()
            handle.push(
                {
                    "event": "run.complete",
                    "data": {
                        "run_id": handle.run_id,
                        "response": redacted_response,
                        "provider": handle.resolved_provider,
                        "tools_used": handle.tools_used,
                        "cost": handle.cost,
                    },
                }
            )
        finally:
            if bus is not None and handler is not None:
                for topic in _RUN_TOPICS:
                    with contextlib.suppress(Exception):
                        bus.unsubscribe(topic, handler)
            if bus is not None and summary_handler is not None:
                with contextlib.suppress(Exception):
                    bus.unsubscribe(_SUMMARY_TOPIC, summary_handler)
            handle.push({"event": "__done__", "data": {}})
            with contextlib.suppress(queue.Full):
                handle._queue.put_nowait(_STREAM_DONE)
            with self._lock:
                self._active_sessions.discard(handle.session_id)

    def _on_bus_message(self, handle: RunHandle, msg: BusMessage) -> None:
        if handle.task_id is None and msg.payload.get("task_id"):
            handle.task_id = str(msg.payload["task_id"])
        event_name = _EVENT_NAME_BY_TOPIC.get(msg.topic, msg.topic)
        data = redact_audit_value(dict(msg.payload))
        handle.push({"event": event_name, "data": data})

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, run_id: str, *, timeout: float = _SSE_KEEPALIVE_SECONDS):
        """Yield ``{"event": ..., "data": ...}`` dicts for *run_id*.

        Blocks between events (with periodic ``ping`` keepalives) until the
        run finishes. A caller that connects *after* the run has already
        finished receives a synthesized terminal event immediately instead
        of hanging, so reconnects and late joins never stall.
        """
        handle = self.get(run_id)
        if handle is None:
            return

        if handle.status in _TERMINAL_STATUSES:
            while True:
                try:
                    item = handle._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _STREAM_DONE and item.get("event") != "__done__":
                    yield item
            yield _terminal_event(handle)
            return

        while True:
            try:
                item = handle._queue.get(timeout=timeout)
            except queue.Empty:
                if handle.status in _TERMINAL_STATUSES:
                    # The run finished, but the terminal queue marker
                    # (__done__ / _STREAM_DONE) may have been silently
                    # dropped by push()'s queue.Full suppression if the
                    # queue was already at capacity when _execute()'s
                    # finally block tried to enqueue it (e.g. a
                    # tool-call-heavy run outpacing this consumer).
                    # Fall back to handle.status -- the same signal the
                    # late-join fast path above already trusts -- instead
                    # of pinging forever.
                    while True:
                        try:
                            queued = handle._queue.get_nowait()
                        except queue.Empty:
                            break
                        if queued is not _STREAM_DONE and queued.get("event") != "__done__":
                            yield queued
                    yield _terminal_event(handle)
                    return
                yield {"event": "ping", "data": {}}
                continue
            if item is _STREAM_DONE:
                return
            if item.get("event") == "__done__":
                return
            yield item

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _prune_locked(self) -> None:
        """Evict old finished runs. Caller must hold ``self._lock``."""
        now = datetime.now(UTC)
        for run_id, run in list(self._runs.items()):
            if run.status not in _TERMINAL_STATUSES or not run.finished_at:
                continue
            with contextlib.suppress(ValueError):
                finished = datetime.fromisoformat(run.finished_at)
                if (now - finished).total_seconds() > _RUN_TTL_SECONDS:
                    self._runs.pop(run_id, None)

        if len(self._runs) <= _MAX_TRACKED_RUNS:
            return
        finished = sorted(
            (r for r in self._runs.values() if r.status in _TERMINAL_STATUSES),
            key=lambda r: r.finished_at or r.created_at,
        )
        overflow = len(self._runs) - _MAX_TRACKED_RUNS
        for run in finished[:overflow]:
            self._runs.pop(run.run_id, None)


def _terminal_event(handle: RunHandle) -> dict[str, Any]:
    if handle.status == "error":
        return {"event": "run.error", "data": {"run_id": handle.run_id, "error": handle.error}}
    return {
        "event": "run.complete",
        "data": {
            "run_id": handle.run_id,
            "response": handle.response,
            "provider": handle.resolved_provider,
            "tools_used": handle.tools_used,
            "cost": handle.cost,
        },
    }


def format_sse(event: dict[str, Any]) -> bytes:
    """Render a single event dict as an SSE wire frame."""
    name = str(event.get("event") or "message")
    payload = json.dumps(event.get("data", {}), default=str)
    lines = [f"event: {name}"]
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    lines.append("")
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")
