"""FX-B regression: real Discord conversation turns must land in
:class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

The 2026-07-09/10 validation-harness run found only 3 rows in
``~/.missy/memory.db`` despite 937 ``agent.run.start`` events from real
Discord traffic. Root cause: ``AgentRuntime._make_memory_store()``
constructed the JSON-file-backed ``missy.memory.store.MemoryStore``
(``~/.missy/memory.json``) while every read path (``memory_search``,
``memory_describe``, ``memory_expand``, compaction, hatching, vision
memory) assumes the SQLite backend at ``~/.missy/memory.db``. Discord's
channel handler drives the agent through the exact same
:meth:`AgentRuntime.run` entry point exercised here (see
``missy/cli/main.py``'s ``_process_channel`` -> ``_discord_agent.run``),
so this is the closest production-equivalent bridge without standing up
a real Discord gateway connection.

These tests use a real on-disk :class:`SQLiteMemoryStore` (no mocking of
the memory layer itself) and a fake in-process provider (no network/LLM
calls) so the whole write -> read round trip through actual SQL is
exercised.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.core.exceptions import ProviderError
from missy.memory.sqlite_store import SQLiteMemoryStore
from missy.providers.base import CompletionResponse


def _make_config(**overrides) -> AgentConfig:
    defaults = {
        "provider": "fake",
        "system_prompt": "You are Missy.",
        "max_iterations": 1,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _fake_provider(response_text: str = "Hello from Missy!", *, raises: bool = False) -> MagicMock:
    provider = MagicMock()
    provider.name = "fake"
    provider.is_available.return_value = True
    if raises:
        provider.complete.side_effect = ProviderError("delegate crashed")
    else:
        provider.complete.return_value = CompletionResponse(
            content=response_text,
            model="fake-model",
            provider="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
            tool_calls=[],
            finish_reason="stop",
        )
    return provider


def _fake_registry(provider) -> MagicMock:
    registry = MagicMock()
    registry.get.return_value = provider
    registry.get_available.return_value = [provider]
    return registry


def _run_via_discord_equivalent_path(rt: AgentRuntime, user_input: str, session_id: str, provider):
    """Drive the agent exactly the way Discord's channel handler does.

    ``missy/cli/main.py``'s Discord message loop calls
    ``_discord_agent.run(enriched_prompt, session_id)`` in a thread
    executor. This helper reproduces that call with the same provider/
    tool-registry mocking used elsewhere in the suite, but without
    mocking the memory store -- ``rt._memory_store`` is a real
    ``SQLiteMemoryStore`` supplied by the caller.
    """
    registry = _fake_registry(provider)
    with ExitStack() as stack:
        stack.enter_context(patch("missy.agent.runtime.get_registry", return_value=registry))
        stack.enter_context(
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError)
        )
        stack.enter_context(patch("missy.agent.runtime.censor_response", side_effect=lambda x: x))
        return rt.run(user_input, session_id=session_id)


@pytest.fixture
def sqlite_store(tmp_path):
    return SQLiteMemoryStore(db_path=str(tmp_path / "memory.db"))


def _resolve_sid(rt: AgentRuntime, raw_session_id: str) -> str:
    """Mirror AgentRuntime._resolve_session()'s stable-ID derivation.

    ``run(session_id=...)`` does not use the caller's raw string as the
    stored session key directly -- it derives a deterministic UUID5 via
    ``SessionManager.create_session_with_id()`` so that pooled
    thread-executor calls (e.g. Discord's ``run_in_executor``) sharing the
    same logical session still hit the same history. Tests that filter by
    exact session_id must resolve it the same way.
    """
    return str(rt._session_mgr.create_session_with_id(raw_session_id).id)


class TestBasicPersistence:
    def test_user_and_assistant_turns_land_in_sqlite(self, sqlite_store):
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store
        provider = _fake_provider("The capital of France is Paris.")

        response = _run_via_discord_equivalent_path(
            rt, "What is the capital of France?", "discord-session-1", provider
        )

        assert response == "The capital of France is Paris."
        recent = sqlite_store.get_recent_turns(limit=10)
        contents = [t.content for t in recent]
        assert "What is the capital of France?" in contents
        assert "The capital of France is Paris." in contents

    def test_turns_retrievable_via_get_session_turns(self, sqlite_store):
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store
        provider = _fake_provider("42")

        _run_via_discord_equivalent_path(rt, "What is 6*7?", "discord-session-2", provider)

        turns = sqlite_store.get_session_turns(_resolve_sid(rt, "discord-session-2"), limit=10)
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "What is 6*7?"
        assert turns[1].role == "assistant"
        assert turns[1].content == "42"

    def test_turns_retrievable_via_authorized_memory_search(self, sqlite_store):
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store
        provider = _fake_provider("Kubernetes is a container orchestration platform.")

        _run_via_discord_equivalent_path(
            rt, "Tell me about kubernetes", "discord-session-3", provider
        )

        results = sqlite_store.search("kubernetes", limit=10)
        assert any("Kubernetes" in t.content for t in results)


class TestSessionRestartAndResume:
    def test_turns_persist_across_new_runtime_instance(self, tmp_path):
        db_path = str(tmp_path / "memory.db")
        store1 = SQLiteMemoryStore(db_path=db_path)
        rt1 = AgentRuntime(_make_config())
        rt1._memory_store = store1
        provider = _fake_provider("First session reply")
        _run_via_discord_equivalent_path(rt1, "First message", "resume-session", provider)

        # Simulate a process restart: brand-new runtime + store pointed at
        # the same on-disk database.
        store2 = SQLiteMemoryStore(db_path=db_path)
        rt2 = AgentRuntime(_make_config())
        rt2._memory_store = store2
        history = rt2._load_history(_resolve_sid(rt2, "resume-session"))

        assert any(h["content"] == "First message" for h in history)
        assert any(h["content"] == "First session reply" for h in history)


class TestConcurrentUsersAndChannels:
    def test_different_sessions_do_not_leak_into_each_other(self, sqlite_store):
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store

        _run_via_discord_equivalent_path(
            rt, "Alice's private question", "channel-alice", _fake_provider("Alice's answer")
        )
        _run_via_discord_equivalent_path(
            rt, "Bob's private question", "channel-bob", _fake_provider("Bob's answer")
        )

        alice_turns = sqlite_store.get_session_turns(_resolve_sid(rt, "channel-alice"), limit=10)
        bob_turns = sqlite_store.get_session_turns(_resolve_sid(rt, "channel-bob"), limit=10)

        alice_contents = [t.content for t in alice_turns]
        bob_contents = [t.content for t in bob_turns]

        assert "Alice's private question" in alice_contents
        assert "Bob's private question" not in alice_contents
        assert "Bob's private question" in bob_contents
        assert "Alice's private question" not in bob_contents


class TestFailedProviderCalls:
    def test_user_turn_still_persisted_when_provider_fails(self, sqlite_store):
        # FX-B: a crashing/timing-out delegate must not erase evidence of
        # what the user actually asked. The user turn is saved before the
        # provider call is attempted; only the (nonexistent) assistant
        # turn is absent.
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store
        provider = _fake_provider(raises=True)

        with pytest.raises(ProviderError):
            _run_via_discord_equivalent_path(rt, "This will fail", "failing-session", provider)

        turns = sqlite_store.get_session_turns(_resolve_sid(rt, "failing-session"), limit=10)
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "This will fail"

    def test_persistence_failure_emits_audit_event_not_silent_swallow(self):
        rt = AgentRuntime(_make_config())
        broken_store = MagicMock()
        broken_store.add_turn.side_effect = RuntimeError("disk full")
        rt._memory_store = broken_store
        provider = _fake_provider("reply anyway")

        with patch.object(rt, "_emit_event", wraps=rt._emit_event) as spy_emit:
            response = _run_via_discord_equivalent_path(
                rt, "hello", "broken-store-session", provider
            )

        # The run itself must not fail just because persistence failed.
        assert response == "reply anyway"
        persist_failures = [
            call
            for call in spy_emit.call_args_list
            if call.kwargs.get("event_type") == "memory.persist_failed"
        ]
        assert len(persist_failures) >= 1


class TestRedactionAndPrivacyScope:
    def test_session_id_is_recorded_correctly_for_multi_account_isolation(self, sqlite_store):
        # Discord sessions are keyed distinctly per account/guild/channel
        # upstream; verify the runtime faithfully records whatever session
        # id it is given rather than collapsing distinct scopes.
        rt = AgentRuntime(_make_config())
        rt._memory_store = sqlite_store

        _run_via_discord_equivalent_path(
            rt,
            "message in guild A channel 1",
            "discord:guildA:chan1",
            _fake_provider("reply A"),
        )
        _run_via_discord_equivalent_path(
            rt,
            "message in guild A channel 2",
            "discord:guildA:chan2",
            _fake_provider("reply B"),
        )

        sid1 = _resolve_sid(rt, "discord:guildA:chan1")
        sid2 = _resolve_sid(rt, "discord:guildA:chan2")
        assert sid1 != sid2

        chan1 = sqlite_store.get_session_turns(sid1, limit=10)
        chan2 = sqlite_store.get_session_turns(sid2, limit=10)
        assert len(chan1) == 2
        assert len(chan2) == 2
        assert all(t.session_id == sid1 for t in chan1)
        assert all(t.session_id == sid2 for t in chan2)
        assert any(t.content == "message in guild A channel 1" for t in chan1)
        assert any(t.content == "message in guild A channel 2" for t in chan2)
