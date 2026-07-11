"""SR-3.3 regression tests: memory_search/memory_describe/memory_expand
must actually work when dispatched through AgentRuntime._execute_tool(),
the real production code path.

Before this fix, two independent bugs stacked to make these tools
completely non-functional in production despite AgentRuntime explicitly
telling the model to use them after every large-tool-output interception:

1. None of the three tools declared ``permissions: ToolPermissions``
   (BaseTool/ToolRegistry._check_permissions() requires it) -- dispatch
   through the real ToolRegistry crashed with AttributeError before the
   tool's own logic ever ran.
2. Even with (1) fixed, AgentRuntime._execute_tool() never injected the
   ``_memory_store``/``_session_id`` private kwargs these tools rely on
   to function -- every call returned "Memory store is not available."

All prior tests for these tools called ``tool.execute(...)`` directly
with ``_memory_store`` manually supplied, which exercises neither bug:
that bypasses both the registry's permission check and the runtime's
kwarg injection entirely.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.memory.sqlite_store import ConversationTurn, LargeContentRecord, SQLiteMemoryStore
from missy.providers.base import ToolCall
from missy.tools import registry as registry_module
from missy.tools.builtin import register_builtin_tools
from missy.tools.registry import init_tool_registry


@pytest.fixture(autouse=True)
def _restore_tool_registry_singleton():
    """init_tool_registry() replaces the process-level singleton; restore
    it afterward so this file doesn't leak state into other test modules
    that run in the same process.
    """
    original = registry_module._registry
    yield
    registry_module._registry = original


@pytest.fixture
def runtime_with_real_registry():
    """A real AgentRuntime wired to a real ToolRegistry with the real
    built-in tools registered, and a real temp-file-backed SQLiteMemoryStore
    -- as close to production wiring as a unit test can get.
    """
    registry = init_tool_registry()
    register_builtin_tools(registry)

    cfg = AgentConfig()
    runtime = AgentRuntime(cfg)
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    runtime._memory_store = SQLiteMemoryStore(db)
    return runtime


class TestMemoryToolsWorkThroughRealDispatch:
    def test_memory_expand_retrieves_large_content(self, runtime_with_real_registry):
        runtime = runtime_with_real_registry
        record = LargeContentRecord.new(
            session_id="sess-A", tool_name="shell_exec", content="y" * 9000, summary="big output"
        )
        content_id = runtime._memory_store.store_large_content(record)

        tc = ToolCall(id="tc1", name="memory_expand", arguments={"item_id": content_id})
        result = runtime._execute_tool(tc, session_id="sess-A", task_id="task1")

        assert not result.is_error
        assert content_id in result.content
        assert "y" * 100 in result.content

    def test_memory_describe_retrieves_metadata(self, runtime_with_real_registry):
        runtime = runtime_with_real_registry
        record = LargeContentRecord.new(
            session_id="sess-A", tool_name="shell_exec", content="z" * 9000, summary="big output"
        )
        content_id = runtime._memory_store.store_large_content(record)

        tc = ToolCall(id="tc1", name="memory_describe", arguments={"item_id": content_id})
        result = runtime._execute_tool(tc, session_id="sess-A", task_id="task1")

        assert not result.is_error
        assert "sess-A" in result.content

    def test_memory_search_finds_stored_turns(self, runtime_with_real_registry):
        runtime = runtime_with_real_registry
        runtime._memory_store.add_turn(
            ConversationTurn.new("sess-A", "user", "project zeta rollout plan")
        )

        tc = ToolCall(id="tc1", name="memory_search", arguments={"query": "zeta"})
        result = runtime._execute_tool(tc, session_id="sess-A", task_id="task1")

        assert not result.is_error
        assert "zeta" in result.content.lower()

    def test_memory_search_defaults_to_calling_session_not_all_sessions(
        self, runtime_with_real_registry
    ):
        """When the model omits session_id, results must be scoped to the
        current session, not leak content from unrelated sessions.
        """
        runtime = runtime_with_real_registry
        runtime._memory_store.add_turn(
            ConversationTurn.new("sess-A", "user", "shared-keyword in session A")
        )
        runtime._memory_store.add_turn(
            ConversationTurn.new("sess-B", "user", "shared-keyword in session B")
        )

        tc = ToolCall(id="tc1", name="memory_search", arguments={"query": "shared-keyword"})
        result = runtime._execute_tool(tc, session_id="sess-A", task_id="task1")

        assert not result.is_error
        assert "session A" in result.content
        assert "session B" not in result.content

    def test_memory_search_explicit_session_id_still_overridable(self, runtime_with_real_registry):
        """The model may still explicitly request a different session's
        history -- this is documented, intentional behavior for a
        single-user local assistant retrieving related earlier context,
        not a leak (the tool schema documents session_id as an explicit
        opt-in override, and it is not the default).

        This test previously asserted only
        `"unique-marker-xyz" in result.content.lower()`, which is a
        false-pass: the tool's own "No results found for
        'unique-marker-xyz'." failure message echoes the query term, so
        the assertion was trivially satisfied even while the override was
        silently discarded and no session-B content was actually
        retrieved (`_execute_tool()` stripped `session_id` from the tool
        call's arguments -- to avoid colliding with the `session_id=`
        kwarg passed to `registry.execute()` -- before the memory-tool
        injection block ever saw it, so `_session_id` was always set to
        the *current* session regardless of what the model requested).
        Assert on the actual match content and the absence of the
        no-results message so a regression is caught for real.
        """
        runtime = runtime_with_real_registry
        runtime._memory_store.add_turn(
            ConversationTurn.new("sess-B", "user", "unique-marker-xyz in session B")
        )

        tc = ToolCall(
            id="tc1",
            name="memory_search",
            arguments={"query": "unique-marker-xyz", "session_id": "sess-B"},
        )
        result = runtime._execute_tool(tc, session_id="sess-A", task_id="task1")

        assert not result.is_error
        assert "No results found" not in result.content
        assert "session B" in result.content


class TestMemoryStoreInjectionScopedToMemoryTools:
    """The _memory_store/_session_id injection must not leak into unrelated
    tool calls -- only the three memory-retrieval tools should receive it.
    """

    def test_non_memory_tool_args_unmodified(self, runtime_with_real_registry):
        from missy.agent.runtime import _MEMORY_RETRIEVAL_TOOL_NAMES

        assert "shell_exec" not in _MEMORY_RETRIEVAL_TOOL_NAMES
        assert {"memory_search", "memory_describe", "memory_expand"} == set(
            _MEMORY_RETRIEVAL_TOOL_NAMES
        )
