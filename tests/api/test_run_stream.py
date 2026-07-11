"""Tests for missy.api.run_stream — background run execution + SSE streaming."""

from __future__ import annotations

import threading
import time

import pytest

from missy.api.run_stream import RunConflictError, RunRegistry, format_sse
from missy.core.message_bus import BusMessage, MessageBus

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class FakeRuntime:
    """Minimal stand-in for AgentRuntime.run that publishes bus events."""

    def __init__(self, bus: MessageBus, *, response: str = "ok", error: Exception | None = None):
        self._bus = bus
        self._response = response
        self._error = error
        self.calls: list[tuple[str, str | None]] = []
        self._gate: threading.Event | None = None

    def block_until(self, gate: threading.Event) -> FakeRuntime:
        self._gate = gate
        return self

    def run(self, message: str, session_id: str | None = None) -> str:
        self.calls.append((message, session_id))
        self._bus.publish(
            BusMessage(
                topic="agent.run.start",
                payload={"session_id": session_id, "task_id": "task-1"},
                source="agent",
            )
        )
        self._bus.publish(
            BusMessage(
                topic="tool.request",
                payload={"tool": "shell_exec", "session_id": session_id, "task_id": "task-1"},
                source="tool:shell_exec",
            )
        )
        self._bus.publish(
            BusMessage(
                topic="tool.result",
                payload={
                    "tool": "shell_exec",
                    "is_error": False,
                    "session_id": session_id,
                    "task_id": "task-1",
                },
                source="tool:shell_exec",
            )
        )
        if self._gate is not None:
            self._gate.wait(timeout=2)
        if self._error is not None:
            raise self._error
        self._bus.publish(
            BusMessage(
                topic="agent.run.complete",
                payload={
                    "session_id": session_id,
                    "task_id": "task-1",
                    "provider": "anthropic",
                    "tools_used": ["shell_exec"],
                    "cost": {"total_cost_usd": 0.0042},
                },
                source="agent",
            )
        )
        return self._response


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


@pytest.fixture
def registry(bus: MessageBus) -> RunRegistry:
    return RunRegistry(bus_factory=lambda: bus)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_start_returns_pending_handle(self, registry: RunRegistry, bus: MessageBus) -> None:
        runtime = FakeRuntime(bus, response="42")
        handle = registry.start(runtime=runtime, message="2+2?", session_id="s1", provider="mock")
        assert handle.session_id == "s1"
        assert handle.run_id
        assert handle.status in {"pending", "running", "complete"}

    def test_run_completes_with_response(self, registry: RunRegistry, bus: MessageBus) -> None:
        runtime = FakeRuntime(bus, response="42")
        handle = registry.start(runtime=runtime, message="2+2?", session_id="s1", provider="mock")
        events = list(registry.stream(handle.run_id))
        assert events[-1]["event"] == "run.complete"
        assert events[-1]["data"]["response"] == "42"
        assert registry.get(handle.run_id).status == "complete"
        assert registry.get(handle.run_id).response == "42"

    def test_response_secrets_are_redacted(self, registry: RunRegistry, bus: MessageBus) -> None:
        """Regression: POST /api/v1/chat censors response_text via
        censor_response() before returning it, but this background-run
        path (used by GET /api/v1/runs/{run_id} and its SSE stream)
        previously stored/streamed the raw agent response with no
        redaction at all -- every other field this same method pushes
        (message, error, cost) already goes through redact_audit_value().
        If the agent's final answer echoes a credential, a client polling
        this endpoint got it unredacted while the identical content
        through /chat would have been redacted.
        """
        secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcd"
        runtime = FakeRuntime(bus, response=f"Here is the key you asked about: {secret}")
        handle = registry.start(runtime=runtime, message="what's the key?", session_id="s1")
        events = list(registry.stream(handle.run_id))
        assert events[-1]["event"] == "run.complete"
        assert secret not in events[-1]["data"]["response"]
        assert "[REDACTED]" in events[-1]["data"]["response"]
        assert secret not in registry.get(handle.run_id).response

    def test_stream_includes_bus_sourced_tool_events(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="do a thing", session_id="s1")
        events = [e["event"] for e in registry.stream(handle.run_id)]
        assert "run.started" in events
        assert "run.start" in events
        assert "tool.request" in events
        assert "tool.result" in events
        assert events[-1] == "run.complete"

    def test_task_id_captured_from_bus(self, registry: RunRegistry, bus: MessageBus) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="hi", session_id="s1")
        list(registry.stream(handle.run_id))
        assert handle.task_id == "task-1"

    def test_run_complete_event_carries_provider_tools_and_cost(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="hi", session_id="s1")
        events = list(registry.stream(handle.run_id))
        complete = next(e for e in events if e["event"] == "run.complete")
        assert complete["data"]["provider"] == "anthropic"
        assert complete["data"]["tools_used"] == ["shell_exec"]
        assert complete["data"]["cost"] == {"total_cost_usd": 0.0042}
        assert handle.to_dict()["resolved_provider"] == "anthropic"

    def test_late_join_terminal_event_carries_summary_fields(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="hi", session_id="s1")
        list(registry.stream(handle.run_id))  # drain once

        events = list(registry.stream(handle.run_id))  # late join / reconnect
        assert events[-1]["event"] == "run.complete"
        assert events[-1]["data"]["provider"] == "anthropic"
        assert events[-1]["data"]["tools_used"] == ["shell_exec"]

    def test_events_for_other_sessions_are_not_leaked(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="hi", session_id="s1")
        # A concurrent, unrelated session publishing on the same bus must not
        # bleed into this run's event stream.
        bus.publish(
            BusMessage(
                topic="tool.request",
                payload={"tool": "other_tool", "session_id": "s-unrelated", "task_id": "t9"},
                source="tool:other_tool",
            )
        )
        events = list(registry.stream(handle.run_id))
        tools = [e["data"].get("tool") for e in events if e["event"] == "tool.request"]
        assert "other_tool" not in tools


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestRunErrors:
    def test_run_error_is_captured_and_redacted(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, error=RuntimeError("boom: sk-ant-api03-secretvalue1234567890"))
        handle = registry.start(runtime=runtime, message="fail please", session_id="s1")
        events = list(registry.stream(handle.run_id))
        assert events[-1]["event"] == "run.error"
        assert "sk-ant-api03" not in events[-1]["data"]["error"]
        assert registry.get(handle.run_id).status == "error"

    def test_get_unknown_run_returns_none(self, registry: RunRegistry) -> None:
        assert registry.get("does-not-exist") is None

    def test_stream_unknown_run_yields_nothing(self, registry: RunRegistry) -> None:
        assert list(registry.stream("does-not-exist")) == []


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestRunConcurrency:
    def test_concurrent_run_on_same_session_rejected(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        gate = threading.Event()
        runtime = FakeRuntime(bus, response="slow").block_until(gate)
        registry.start(runtime=runtime, message="first", session_id="s1")
        time.sleep(0.05)
        with pytest.raises(RunConflictError):
            registry.start(runtime=runtime, message="second", session_id="s1")
        gate.set()

    def test_session_freed_after_run_completes(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="done")
        handle = registry.start(runtime=runtime, message="first", session_id="s1")
        list(registry.stream(handle.run_id))
        # Session should now be free for a new run.
        handle2 = registry.start(runtime=runtime, message="second", session_id="s1")
        assert handle2.run_id != handle.run_id

    def test_different_sessions_run_concurrently(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        gate = threading.Event()
        runtime = FakeRuntime(bus, response="slow").block_until(gate)
        h1 = registry.start(runtime=runtime, message="a", session_id="s1")
        h2 = registry.start(runtime=runtime, message="b", session_id="s2")
        assert h1.run_id != h2.run_id
        gate.set()


# ---------------------------------------------------------------------------
# Late join / reconnect
# ---------------------------------------------------------------------------


class TestLateJoin:
    def test_late_join_after_completion_returns_terminal_event_immediately(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="42")
        handle = registry.start(runtime=runtime, message="2+2?", session_id="s1")
        # Drain fully once (simulates the first client finishing its stream).
        list(registry.stream(handle.run_id))

        started = time.monotonic()
        events = list(registry.stream(handle.run_id))
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert events[-1]["event"] == "run.complete"
        assert events[-1]["data"]["response"] == "42"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListForSession:
    def test_list_for_session_returns_only_matching_runs(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="ok")
        h1 = registry.start(runtime=runtime, message="a", session_id="s1")
        list(registry.stream(h1.run_id))
        h2 = registry.start(runtime=runtime, message="b", session_id="s2")
        list(registry.stream(h2.run_id))

        runs = registry.list_for_session("s1")
        assert len(runs) == 1
        assert runs[0]["run_id"] == h1.run_id

    def test_list_for_session_most_recent_first(
        self, registry: RunRegistry, bus: MessageBus
    ) -> None:
        runtime = FakeRuntime(bus, response="ok")
        h1 = registry.start(runtime=runtime, message="a", session_id="s1")
        list(registry.stream(h1.run_id))
        h2 = registry.start(runtime=runtime, message="b", session_id="s1")
        list(registry.stream(h2.run_id))

        runs = registry.list_for_session("s1")
        assert runs[0]["run_id"] == h2.run_id
        assert runs[1]["run_id"] == h1.run_id

    def test_list_for_unknown_session_is_empty(self, registry: RunRegistry) -> None:
        assert registry.list_for_session("nope") == []


# ---------------------------------------------------------------------------
# No message bus attached
# ---------------------------------------------------------------------------


class TestNoMessageBus:
    def test_run_still_completes_without_bus(self) -> None:
        registry = RunRegistry(bus_factory=lambda: None)

        class SimpleRuntime:
            def run(self, message: str, session_id: str | None = None) -> str:
                return "no-bus-response"

        handle = registry.start(runtime=SimpleRuntime(), message="hi", session_id="s1")
        events = list(registry.stream(handle.run_id))
        assert events[-1]["event"] == "run.complete"
        assert events[-1]["data"]["response"] == "no-bus-response"


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


class TestFormatSse:
    def test_format_sse_basic_shape(self) -> None:
        frame = format_sse({"event": "run.complete", "data": {"response": "hi"}})
        text = frame.decode("utf-8")
        assert text.startswith("event: run.complete\n")
        assert '"response": "hi"' in text
        assert text.endswith("\n\n")

    def test_format_sse_defaults_event_name(self) -> None:
        frame = format_sse({"data": {"x": 1}})
        assert frame.decode("utf-8").startswith("event: message\n")

    def test_format_sse_escapes_embedded_newlines_in_json(self) -> None:
        # json.dumps escapes "\n" inside string values, so the frame stays a
        # single well-formed "data:" line even when the payload text spans
        # multiple logical lines.
        frame = format_sse({"event": "e", "data": {"a": "line1\nline2"}})
        text = frame.decode("utf-8")
        data_lines = [line for line in text.splitlines() if line.startswith("data:")]
        assert len(data_lines) == 1
        assert "\\n" in data_lines[0]
