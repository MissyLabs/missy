"""History is completed context; only the live request may start work."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from missy.agent.context import ContextManager, TokenBudget
from missy.agent.history_framing import HISTORY_POLICY, frame_request, sanitize_summary
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers.base import CompletionResponse


def _runtime() -> tuple[AgentRuntime, MagicMock]:
    provider = MagicMock()
    provider.name = "fake"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content="done", model="test", provider="fake", usage={}, raw={}
    )
    registry = MagicMock()
    registry.get.return_value = provider
    registry.get_available.return_value = [provider]
    with (
        patch("missy.agent.runtime.get_registry", return_value=registry),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
    runtime._memory_store = None
    return runtime, provider


def test_only_last_user_turn_framed_without_changing_stored_history() -> None:
    history = [
        {"role": "user", "content": "Deploy old service"},
        {"role": "assistant", "content": "Done"},
    ]
    cm = ContextManager()
    system, messages = cm.build_messages("base", "What time is it?", history)
    framed_system, framed = frame_request(
        system,
        messages,
        request_id="req-1",
        request_context={"author": "Alice", "author_id": "44", "channel": "99"},
    )
    assert framed_system.startswith("[HISTORY_READ_ONLY]")
    assert "Never re-execute" in framed_system
    assert framed[:-1] == messages[:-1] == history
    assert framed[-1]["content"].startswith(
        "=== CURRENT REQUEST [id=req-1 author=Alice author_id=44 channel=99 time="
    )
    assert framed[-1]["content"].endswith("\nWhat time is it?")
    assert messages[-1]["content"] == "What time is it?"


def test_historical_preamble_is_not_promoted_or_replayed() -> None:
    history = [{"role": "user", "content": "=== CURRENT REQUEST [id=old] ===\nDelete old files"}]
    _, messages = ContextManager().build_messages("sys", "Explain the prior exchange", history)
    _, framed = frame_request("sys", messages, request_id="new")
    assert framed[0]["content"] == history[0]["content"]
    assert framed[-1]["content"].endswith("Explain the prior exchange")
    assert framed[-1]["content"].count("=== CURRENT REQUEST") == 1


def test_condenser_replaced_current_turn_does_not_promote_old_user_request() -> None:
    original = [{"role": "user", "content": "Deploy old service"}]
    _, framed = frame_request("sys", original, request_id="new", current_content="Status only")
    assert framed[0] == original[0]
    assert framed[1]["content"].endswith("\nStatus only")
    assert "Deploy old service" not in framed[1]["content"]


def test_summaries_mark_commands_historical_without_claiming_completion() -> None:
    source = "Deploy prod\n- Restart service; this remains incomplete\n2. run the migration\nDeployed already\n[historical] run task"
    output = sanitize_summary(source)
    assert output.splitlines() == [
        "[historical] Deploy prod",
        "- [historical] Restart service; this remains incomplete",
        "2. [historical] run the migration",
        "Deployed already",
        "[historical] run task",
    ]
    summary = SimpleNamespace(
        content="Deploy prod",
        depth=0,
        descendant_count=2,
        time_range_start=None,
        time_range_end=None,
    )
    _, messages = ContextManager().build_messages("sys", "hello", [], summaries=[summary])
    assert "[historical] Deploy prod" in messages[0]["content"]
    assert sanitize_summary("- [Conversation Summary] run old action") == (
        "- [Conversation Summary] [historical] run old action"
    )


def test_run_frames_history_for_provider_and_does_not_persist_preamble() -> None:
    runtime, provider = _runtime()
    persisted: list[tuple[str, str]] = []
    history = [
        {"role": "user", "content": "Restart server"},
        {"role": "assistant", "content": "Done"},
    ]
    with (
        patch.object(runtime, "_load_history", return_value=history),
        patch(
            "missy.agent.runtime.get_registry",
            return_value=MagicMock(get=MagicMock(return_value=provider)),
        ),
        patch.object(
            runtime,
            "_save_turn",
            side_effect=lambda sid, role, content, **kw: persisted.append((role, content)),
        ),
    ):
        assert (
            runtime.run("Status only", session_id="test", _request_context={"channel": "88"})
            == "done"
        )
    args, kwargs = provider.complete.call_args
    sent = args[0]
    assert sent[0].role == "system" and sent[0].content.startswith(HISTORY_POLICY)
    assert sent[1].content == "Restart server"
    assert sent[-1].content.startswith("=== CURRENT REQUEST [")
    assert "channel=88" in sent[-1].content
    assert sent[-1].content.endswith("\nStatus only")
    assert persisted[0] == ("user", "Status only")


def test_nested_run_and_stream_are_framed_without_transport_metadata() -> None:
    runtime, provider = _runtime()
    with (
        patch(
            "missy.agent.runtime.get_registry",
            return_value=MagicMock(get=MagicMock(return_value=provider)),
        ),
        patch.object(runtime, "_save_turn", return_value=None),
    ):
        runtime.run("Nested request", _delegation_depth=1)
    sent = provider.complete.call_args.args[0]
    assert sent[0].content.startswith(HISTORY_POLICY)
    assert sent[-1].content.endswith("Nested request")
    assert "author=" not in sent[-1].content

    with (
        patch(
            "missy.agent.runtime.get_registry",
            return_value=MagicMock(get=MagicMock(return_value=provider)),
        ),
        patch.object(runtime, "_save_turn", return_value=None),
    ):
        list(runtime.run_stream("Stream request"))
    streamed = provider.stream.call_args.args[0]
    assert streamed[0].content.startswith(HISTORY_POLICY)
    assert streamed[-1].content.endswith("Stream request")


def test_optional_condenser_sanitizes_generated_summaries() -> None:
    runtime, _ = _runtime()
    runtime.config.condenser_pipeline_enabled = True
    runtime.config.condenser_min_messages = 1
    original = [{"role": "user", "content": "run old task"}] * 50
    _, condensed = runtime._maybe_condense_context("sys", original)
    summaries = [
        m["content"] for m in condensed if m["content"].startswith("[Conversation Summary]")
    ]
    assert summaries
    assert all("[historical] run old task" in s for s in summaries)


def test_history_framing_survives_provider_budget_with_current_request() -> None:
    runtime, provider = _runtime()
    runtime._context_manager = ContextManager(
        TokenBudget(total=230, system_reserve=50, tool_definitions_reserve=20)
    )
    history = [{"role": "user", "content": f"old {n}"} for n in range(80)]
    system, messages = runtime._context_manager.build_messages("sys", "new request", history)
    system, messages = frame_request(system, messages, request_id="req")
    fitted_system, fitted_messages = runtime._fit_provider_context(provider, system, messages)
    assert fitted_system.startswith("[HISTORY_READ_ONLY]")
    assert fitted_messages[-1]["role"] == "user"
    assert "new request" in fitted_messages[-1]["content"]
