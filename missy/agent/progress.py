"""Structured progress reporting protocol for the agent tool loop.

Provides a ``ProgressReporter`` protocol and three implementations:

- ``NullReporter``: No-op (default when no reporter is configured).
- ``AuditReporter``: Emits progress as audit events via the event bus.
- ``CLIReporter``: Prints Rich status updates to stderr.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    """Protocol for structured progress reporting during agent tool loops."""

    def on_start(self, task: str) -> None: ...
    def on_progress(self, pct: float, label: str) -> None: ...
    def on_tool_start(self, name: str) -> None: ...
    def on_tool_done(self, name: str, summary: str) -> None: ...
    def on_iteration(self, i: int, max_iterations: int) -> None: ...
    def on_complete(self, summary: str) -> None: ...
    def on_error(self, error: str) -> None: ...


class NullReporter:
    """No-op reporter — all methods are silent."""

    def on_start(self, task: str) -> None:
        pass

    def on_progress(self, pct: float, label: str) -> None:
        pass

    def on_tool_start(self, name: str) -> None:
        pass

    def on_tool_done(self, name: str, summary: str) -> None:
        pass

    def on_iteration(self, i: int, max_iterations: int) -> None:
        pass

    def on_complete(self, summary: str) -> None:
        pass

    def on_error(self, error: str) -> None:
        pass


class AuditReporter:
    """Emits progress events to the Missy audit event bus."""

    def __init__(self, session_id: str = "", task_id: str = "") -> None:
        self._session_id = session_id
        self._task_id = task_id

    def _emit(self, event_type: str, detail: dict) -> None:
        try:
            from missy.core.events import AuditEvent, event_bus

            event_bus.publish(
                AuditEvent.now(
                    event_type=event_type,
                    category="agent",
                    result="allow",
                    detail=detail,
                    session_id=self._session_id,
                    task_id=self._task_id,
                )
            )
        except Exception:
            pass

    def on_start(self, task: str) -> None:
        self._emit("agent.progress.start", {"task": task})

    def on_progress(self, pct: float, label: str) -> None:
        self._emit("agent.progress.update", {"pct": pct, "label": label})

    def on_tool_start(self, name: str) -> None:
        self._emit("agent.progress.tool_start", {"tool": name})

    def on_tool_done(self, name: str, summary: str) -> None:
        self._emit("agent.progress.tool_done", {"tool": name, "summary": summary})

    def on_iteration(self, i: int, max_iterations: int) -> None:
        self._emit("agent.progress.iteration", {"iteration": i, "max": max_iterations})

    def on_complete(self, summary: str) -> None:
        self._emit("agent.progress.complete", {"summary": summary})

    def on_error(self, error: str) -> None:
        self._emit("agent.progress.error", {"error": error})


class CLIReporter:
    """Prints Rich status messages to stderr."""

    def on_start(self, task: str) -> None:
        print(f"  [progress] Starting: {task}", file=sys.stderr)

    def on_progress(self, pct: float, label: str) -> None:
        print(f"  [progress] {pct:.0f}% — {label}", file=sys.stderr)

    def on_tool_start(self, name: str) -> None:
        print(f"  [progress] Tool: {name}...", file=sys.stderr)

    def on_tool_done(self, name: str, summary: str) -> None:
        print(f"  [progress] Tool: {name} done — {summary}", file=sys.stderr)

    def on_iteration(self, i: int, max_iterations: int) -> None:
        print(f"  [progress] Iteration {i + 1}/{max_iterations}", file=sys.stderr)

    def on_complete(self, summary: str) -> None:
        print(f"  [progress] Complete: {summary}", file=sys.stderr)

    def on_error(self, error: str) -> None:
        print(f"  [progress] Error: {error}", file=sys.stderr)
