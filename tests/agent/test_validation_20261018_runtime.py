"""Regression tests for the 2026-10-18 live validation run (runtime defects).

Each test failed before its fix:

1. Iteration-limit finalization kept only the failure text (X11-002).
2. Desktop verification exhaustion replaced the answer with canned text
   (X11-003); clipboard reads now count as verification.
3. The web guard demanded web_fetch with no URL present (WB-001).
4. max_turn_seconds did not bound a single tool call (video_generate).
5. The "no tool ran" notice appeared after a tool ran and failed.
6. The promise guard fired on honest failure reports.
7. Audit events named the failed provider after a fallback.
8. Read-only tools could loop 15-29 times per turn.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.response_guards import (
    detect_promise_without_action,
    find_unmet_web_requests,
    find_unverified_desktop_action,
    is_honest_failure_report,
    unverified_answer_notice,
)
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.config.settings import ProviderConfig
from missy.core.events import event_bus
from missy.core.exceptions import ProviderError
from missy.providers.base import BaseProvider, ToolCall
from missy.providers.registry import ProviderRegistry
from missy.tools.base import _TOOL_DEADLINE, current_tool_deadline
from tests.agent.test_runtime_deep import (
    _make_mock_tool,
    _make_provider,
    _make_registry,
    _make_stop_response,
    _make_tool_call_response,
)


@pytest.fixture(autouse=True)
def _clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


def _tool_registry(behaviours: dict) -> MagicMock:
    """Mock tool registry; *behaviours* maps name -> callable(**kwargs) -> (ok, text)."""
    tools = [_make_mock_tool(name) for name in behaviours]
    reg = MagicMock()
    reg.list_tools.return_value = list(behaviours)
    reg.get.side_effect = lambda name: next((t for t in tools if t.name == name), None)

    def _execute(name: str, **kwargs):
        ok, text = behaviours[name](**kwargs)
        result = MagicMock()
        result.success = ok
        result.output = text if ok else None
        result.error = None if ok else text
        result.security_flags = []
        return result

    reg.execute.side_effect = _execute
    return reg, tools


def _run(provider, tool_reg, prompt, **config):
    registry = _make_registry({"fake": provider})
    events: list[dict] = []
    with (
        patch("missy.agent.runtime.get_registry", return_value=registry),
        patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
    ):
        rt = AgentRuntime(AgentConfig(provider="fake", **config))
        rt._emit_event = lambda **kw: events.append(kw)
        result = rt.run(prompt)
    return result, events


def _ok(text="ok"):
    return lambda **_kw: (True, text)


def _fail(text):
    return lambda **_kw: (False, text)


# ---------------------------------------------------------------------------
# 1. Iteration limit keeps the synthesis
# ---------------------------------------------------------------------------


class TestIterationLimitKeepsSynthesis:
    def test_synthesis_with_unresolved_errors_is_kept_with_notice(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("x11_type", "tc-1", {"text": "hello"}),
            _make_tool_call_response("desktop_focus_window", "tc-2", {"title": "Editor"}),
        ]
        synthesis = "I typed 'hello' successfully, but focusing the Editor window failed."
        provider.complete.return_value = _make_stop_response(synthesis)
        tool_reg, _ = _tool_registry(
            {"x11_type": _ok("typed"), "desktop_focus_window": _fail("no window named Editor")}
        )
        result, _ = _run(provider, tool_reg, "type hello in the editor", max_iterations=2)
        assert synthesis in result
        assert "no window named Editor" in result
        assert result.startswith("Note:")

    def test_empty_synthesis_falls_back_to_error_text(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("desktop_focus_window", "tc-1", {"title": "Editor"}),
            _make_tool_call_response("desktop_focus_window", "tc-2", {"title": "editor"}),
        ]
        provider.complete.return_value = _make_stop_response("")
        tool_reg, _ = _tool_registry(
            {"x11_type": _ok("typed"), "desktop_focus_window": _fail("no window named Editor")}
        )
        result, _ = _run(provider, tool_reg, "type hello in the editor", max_iterations=2)
        assert "following tool call(s) failed" in result
        assert "no window named Editor" in result


# ---------------------------------------------------------------------------
# 2. Clipboard reads verify a desktop action
# ---------------------------------------------------------------------------


class TestDesktopClipboardVerification:
    @pytest.mark.parametrize("reader", ["x11_clipboard_read", "clipboard_get", "x11_clipboard_get"])
    def test_clipboard_read_after_copy_verifies(self, reader):
        assert find_unverified_desktop_action(["x11_key", reader]) is None

    def test_clipboard_write_is_not_verification(self):
        assert find_unverified_desktop_action(["x11_key", "x11_clipboard_set"]) == "x11_key"


# ---------------------------------------------------------------------------
# 3. Web guard only demands a fetch for a concrete target
# ---------------------------------------------------------------------------


class TestWebGuardNeedsConcreteTarget:
    @pytest.mark.parametrize(
        "prompt",
        [
            "Fetch the URL from my notes and summarize it",
            "fetch the url I mentioned yesterday",
            "Fetch the url listed in notes.md",
        ],
    )
    def test_no_url_no_demand(self, prompt):
        assert find_unmet_web_requests(prompt, [], {"web_fetch"}) == []

    @pytest.mark.parametrize(
        "prompt",
        [
            "Fetch this URL: https://example.org/a",
            "fetch the url example.com and tell me the title",
            "fetch url docs.python.org/3/library",
            "Fetch the test URL at localhost:8080/health",
        ],
    )
    def test_concrete_target_still_demands_fetch(self, prompt):
        assert find_unmet_web_requests(prompt, [], {"web_fetch"}) == ["web_fetch"]


# ---------------------------------------------------------------------------
# 4. The turn deadline bounds a single tool call
# ---------------------------------------------------------------------------


class TestTurnDeadlineBoundsToolCalls:
    def test_hung_tool_is_abandoned_at_the_deadline(self):
        release = threading.Event()
        seen_deadline: list[float | None] = []

        def _hang(**_kw):
            seen_deadline.append(current_tool_deadline())
            release.wait(30)
            return True, "finished late"

        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("video_generate", "tc-1", {"prompt": "a cat"}),
            _make_stop_response("The video tool timed out."),
        ]
        tool_reg, _ = _tool_registry({"video_generate": _hang})
        started = time.monotonic()
        try:
            with patch("missy.agent.runtime._TOOL_DEADLINE_GRACE_SECONDS", 0.05):
                result, events = _run(
                    provider, tool_reg, "generate a video of a cat", max_turn_seconds=0.5
                )
        finally:
            release.set()
        assert time.monotonic() - started < 10
        assert seen_deadline and seen_deadline[0] is not None
        timeouts = [e for e in events if e["event_type"] == "agent.tool.deadline_exceeded"]
        assert timeouts and timeouts[0]["detail"]["tool"] == "video_generate"
        assert isinstance(result, str) and result

    def test_worker_inherits_execution_identity_and_parent_deadline(self):
        rt = AgentRuntime(AgentConfig(provider="fake"))
        observed: dict = {}

        def _fake_execute(tool_call, **_kw):
            observed["agent"] = rt._current_agent_execution().get("agent_id")
            observed["deadline"] = current_tool_deadline()
            from missy.providers.base import ToolResult

            return ToolResult(tool_call_id=tool_call.id, name=tool_call.name, content="ok")

        rt._execute_tool = _fake_execute
        parent_deadline = time.monotonic() + 5
        token = _TOOL_DEADLINE.set(parent_deadline)
        try:
            with rt.agent_execution(agent_id="sub-7", name="helper", depth=1):
                tr = rt._execute_tool_with_deadline(
                    ToolCall(id="t", name="calculator", arguments={}),
                    time.monotonic() + 60,
                )
        finally:
            _TOOL_DEADLINE.reset(token)
        assert tr.content == "ok"
        assert observed["agent"] == "sub-7"
        # A sub-agent never gets more time than its parent's call allows.
        assert observed["deadline"] == parent_deadline

    def test_comfyui_polling_stops_at_the_turn_deadline(self):
        from missy.tools.builtin import video_generate as vg

        http = MagicMock()
        pending = MagicMock(status_code=200)
        pending.json.return_value = {}
        http.get.return_value = pending
        token = _TOOL_DEADLINE.set(time.monotonic() + 0.2)
        try:
            with patch.object(vg, "_POLL_INTERVAL_SECONDS", 0.01):
                started = time.monotonic()
                out = vg.VideoGenerateTool._wait_for_completion(http, "http://h", "p1", 3600)
        finally:
            _TOOL_DEADLINE.reset(token)
        assert time.monotonic() - started < 5
        assert isinstance(out, str) and "time budget" in out
        # The job is cancelled server-side rather than left on the GPU.
        assert any(call.args[0].endswith("/queue") for call in http.post.call_args_list)

    def test_tool_timeout_is_clamped_to_the_deadline(self):
        from missy.tools.base import bound_timeout_to_deadline

        assert bound_timeout_to_deadline(3600) == 3600
        token = _TOOL_DEADLINE.set(time.monotonic() + 30)
        try:
            assert bound_timeout_to_deadline(3600) <= 30
        finally:
            _TOOL_DEADLINE.reset(token)


# ---------------------------------------------------------------------------
# 5. The unverified notice reflects whether a tool ran
# ---------------------------------------------------------------------------


class TestAccurateUnverifiedNotice:
    def test_failed_tool_is_not_reported_as_no_tool(self):
        out = unverified_answer_notice("No video was made.", ["video_generate"], [])
        assert "no tool ran" not in out
        assert "video_generate" in out and "failed" in out
        assert out.endswith("No video was made.")

    def test_no_tool_still_says_so(self):
        assert "no tool ran" in unverified_answer_notice("Done.", [], [])

    def test_guard_budget_exhaustion_after_failed_tool(self):
        provider = _make_provider()
        claim = "I checked the logs and everything looks fine."
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("shell_exec", "tc-1", {"command": "journalctl"}),
            *[_make_stop_response(claim)] * 6,
        ]
        tool_reg, _ = _tool_registry({"shell_exec": _fail("permission denied")})
        result, _ = _run(
            provider,
            tool_reg,
            "is everything okay on the server?",
            max_iterations=8,
            max_guard_retries=0,
        )
        assert "no tool ran" not in result


# ---------------------------------------------------------------------------
# 6. Honest failure reports are not unfulfilled promises
# ---------------------------------------------------------------------------


class TestPromiseGuardHonestFailures:
    REPORT = (
        "Generating the video failed: ComfyUI generation failed: CUDA out of memory. "
        "No video was generated."
    )

    def test_failure_report_after_failed_tool_is_exempt(self):
        assert detect_promise_without_action(self.REPORT, [])  # the old behaviour
        assert not detect_promise_without_action(self.REPORT, [], ["video_generate"])

    def test_explicit_task_not_completed_is_exempt(self):
        text = "I'm sorry, I couldn't complete the request: the render server is offline."
        assert is_honest_failure_report(text)
        assert not detect_promise_without_action(text, [])

    def test_real_promise_after_failure_still_fires(self):
        assert detect_promise_without_action(
            "I'll generate the video for you now.", [], ["video_generate"]
        )

    def test_failure_report_is_not_retried_in_the_loop(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("video_generate", "tc-1", {"prompt": "a cat"}),
            _make_stop_response(self.REPORT),
            _make_stop_response(self.REPORT),
        ]
        tool_reg, _ = _tool_registry(
            {"video_generate": _fail("video_generate: ComfyUI generation failed: CUDA OOM")}
        )
        _result, events = _run(provider, tool_reg, "generate a video of a cat", max_iterations=5)
        assert not [e for e in events if "promise_without_action" in e["event_type"]]
        assert tool_reg.execute.call_count == 1


# ---------------------------------------------------------------------------
# 7. Audit records the provider that actually served
# ---------------------------------------------------------------------------


class _ToolProvider(BaseProvider):
    def __init__(self, name: str, responses: list | None = None, fail: bool = False):
        self.name = name
        self._config = ProviderConfig(name=name, model="m", api_key="k-" + name)
        self._api_key = self._config.api_key
        self._responses = list(responses or [])
        self._fail = fail

    def is_available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):
        if self._fail:
            raise ProviderError("connection timed out")
        return _make_stop_response("summary", provider_name=self.name)

    def complete_with_tools(self, messages, tools, system=""):
        if self._fail:
            raise ProviderError("connection timed out")
        return self._responses.pop(0)


class TestEffectiveProviderAudit:
    def test_tool_and_run_events_name_the_fallback_provider(self):
        primary = _ToolProvider("openai-codex", fail=True)
        backup = _ToolProvider(
            "acpx",
            [
                _make_tool_call_response("calculator", "tc-1", {"expression": "1+1"}, "acpx"),
                *[_make_stop_response("1+1 = 2", provider_name="acpx")] * 6,
            ],
        )
        registry = ProviderRegistry()
        registry.register("openai-codex", primary, config=primary._config)
        registry.register("acpx", backup, config=backup._config)
        tool_reg, _ = _tool_registry({"calculator": _ok("2")})
        captured = []
        original = event_bus.publish
        event_bus.publish = lambda evt: (original(evt), captured.append(evt))
        try:
            with (
                patch("missy.agent.runtime.get_registry", return_value=registry),
                patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
            ):
                rt = AgentRuntime(AgentConfig(provider="openai-codex", max_iterations=4))
                rt.run("what is 1+1? use the calculator")
        finally:
            event_bus.publish = original
        tool_events = [e for e in captured if e.event_type.startswith("agent.tool.")]
        assert tool_events
        for event in tool_events:
            assert event.detail["provider"] == "acpx", event.event_type
            assert event.detail["requested_provider"] == "openai-codex"
        complete = [e for e in captured if e.event_type == "agent.run.complete"]
        assert complete and complete[0].detail["provider"] == "acpx"


# ---------------------------------------------------------------------------
# 8. Bounded repeated tool calls
# ---------------------------------------------------------------------------


class TestRepeatedCallControl:
    def test_identical_read_only_calls_are_capped(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            *[
                _make_tool_call_response("memory_search", f"tc-{i}", {"query": "Project X"})
                for i in range(6)
            ],
            _make_stop_response("Nothing about Project X is stored."),
        ]
        tool_reg, _ = _tool_registry({"memory_search": _ok("no results")})
        result, events = _run(
            provider, tool_reg, "what do you remember about project x?", max_iterations=10
        )
        assert tool_reg.execute.call_count == 3
        limited = [e for e in events if e["event_type"] == "agent.tool.repeat_call_limited"]
        assert len(limited) == 3
        assert "Project X" in result

    def test_rephrased_searches_hit_the_per_turn_cap(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            *[
                _make_tool_call_response("memory_search", f"tc-{i}", {"query": f"variant {i}"})
                for i in range(6)
            ],
            _make_stop_response("I could not find it."),
        ]
        tool_reg, _ = _tool_registry({"memory_search": _ok("no results")})
        _run(
            provider,
            tool_reg,
            "find my note",
            max_iterations=10,
            max_search_calls_per_tool=4,
        )
        assert tool_reg.execute.call_count == 4

    def test_state_change_resets_identical_count(self):
        provider = _make_provider()
        read = {"path": "/tmp/x.txt"}
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("file_read", "tc-1", read),
            _make_tool_call_response("file_read", "tc-2", read),
            _make_tool_call_response("file_read", "tc-3", read),
            _make_tool_call_response("file_write", "tc-4", {"path": "/tmp/x.txt", "content": "b"}),
            _make_tool_call_response("file_read", "tc-5", read),
            _make_stop_response("The file now contains b."),
        ]
        tool_reg, _ = _tool_registry({"file_read": _ok("a"), "file_write": _ok("written")})
        _run(provider, tool_reg, "update /tmp/x.txt to b", max_iterations=10)
        names = [c.args[0] for c in tool_reg.execute.call_args_list]
        assert names.count("file_read") == 4

    def test_disabled_controls_allow_repeats(self):
        provider = _make_provider()
        provider.complete_with_tools.side_effect = [
            *[
                _make_tool_call_response("memory_search", f"tc-{i}", {"query": "same"})
                for i in range(5)
            ],
            _make_stop_response("done searching"),
        ]
        tool_reg, _ = _tool_registry({"memory_search": _ok("no results")})
        _run(
            provider,
            tool_reg,
            "search memory",
            max_iterations=10,
            max_identical_tool_calls=0,
            max_search_calls_per_tool=0,
        )
        assert tool_reg.execute.call_count == 5
