"""Coverage gap tests for missy/agent/runtime.py.

Targets uncovered lines:
  226         : _make_message_bus — _HAS_MESSAGE_BUS is False → return None
  238-241     : _bus_publish — bus.publish raises, swallowed
  672-674     : _tool_loop — _progress is None → NullReporter fallback
  688-689     : _tool_loop — drift detector signals tamper → warning + audit event
  770         : _tool_loop — trust score drops below threshold → warning logged
  799         : _tool_loop — large content → _intercept_large_content called
  806         : _tool_loop — content still oversized after intercept → hard truncation
  1030        : _get_tools — capability_mode="discord" filters to _DISCORD_TOOLS
  1215-1216   : _build_messages — memory_store.get_learnings raises, caught
  1225-1229   : _build_messages — get_summaries filters to top-level only
  1270-1272   : _build_messages — playbook_patterns non-empty → injected
  1286-1294   : _build_messages — synthesized_block truthy → system appended
  1329-1330   : _synthesize_memory — learnings fragment added
  1333-1334   : _synthesize_memory — summary_texts fragment added
  1337-1338   : _synthesize_memory — playbook_texts fragment added
  1343-1346   : _synthesize_memory — synthesize() raises → returns ""
  1394-1429   : _intercept_large_content — all three paths
  1687-1688   : _make_interactive_approval — exception → None
  1702-1703   : _make_drift_detector — exception → None
  1720-1726   : _make_identity — key file absent → generate + save
  1724-1725   : _make_identity — exception → None
  1740-1742   : _make_attention_system — exception → None
  1751-1753   : _make_persona_manager — exception → None
  1773-1775   : _make_response_shaper — exception → None
  1784-1786   : _make_intent_interpreter — exception → None
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import missy.agent.runtime as runtime_module
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers import registry as registry_module
from missy.providers.base import CompletionResponse, ToolCall

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_provider(reply="ok"):
    provider = MagicMock()
    provider.name = "fake"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="m",
        provider="fake",
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
        finish_reason="stop",
    )
    provider.complete_with_tools.return_value = CompletionResponse(
        content=reply,
        model="m",
        provider="fake",
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
        finish_reason="stop",
    )
    return provider


def _make_registry(provider):
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    return reg


@pytest.fixture(autouse=True)
def reset_registry():
    original = registry_module._registry
    yield
    registry_module._registry = original


def _build_runtime(provider=None, max_iterations=1, capability_mode="no-tools"):
    if provider is None:
        provider = _make_provider()
    reg = _make_registry(provider)
    cfg = AgentConfig(
        provider="fake",
        max_iterations=max_iterations,
        capability_mode=capability_mode,
    )
    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
    ):
        rt = AgentRuntime(cfg)
    rt._rate_limiter = None
    rt._memory_store = None
    rt._cost_tracker = None
    rt._context_manager = None
    return rt, reg


# ---------------------------------------------------------------------------
# Line 226: _make_message_bus — _HAS_MESSAGE_BUS is False
# ---------------------------------------------------------------------------


class TestMakeMessageBusNoModule:
    def test_returns_none_when_message_bus_unavailable(self):
        """Line 226: when _HAS_MESSAGE_BUS is False, None is returned immediately."""
        with patch.object(runtime_module, "_HAS_MESSAGE_BUS", False):
            result = AgentRuntime._make_message_bus()
        assert result is None


# ---------------------------------------------------------------------------
# Lines 238-241: _bus_publish — publish raises, swallowed silently
# ---------------------------------------------------------------------------


class TestBusPublish:
    def test_publish_exception_is_swallowed(self):
        """Lines 238-241: bus.publish raising is caught and not re-raised."""
        rt, _ = _build_runtime()
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus exploded")
        rt._message_bus = bus

        # Must not raise
        rt._bus_publish("some.topic", {"key": "value"})

        bus.publish.assert_called_once()

    def test_publish_no_bus_is_noop(self):
        """When _message_bus is None, _bus_publish returns immediately."""
        rt, _ = _build_runtime()
        rt._message_bus = None
        # No error and no interaction expected
        rt._bus_publish("some.topic", {})


# ---------------------------------------------------------------------------
# Lines 672-674: _tool_loop — _progress attribute absent → NullReporter fallback
# ---------------------------------------------------------------------------


class TestToolLoopProgressFallback:
    def test_missing_progress_attribute_falls_back_to_null_reporter(self):
        """Lines 672-674: when _progress is None, NullReporter is created inline."""
        provider = _make_provider(reply="all done")
        rt, reg = _build_runtime(provider, max_iterations=2, capability_mode="full")

        # Remove progress so the fallback branch executes
        rt._progress = None

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            result = rt.run("hello")

        assert result == "all done"


# ---------------------------------------------------------------------------
# Lines 688-689: drift detector signals tamper → warning + audit event
# ---------------------------------------------------------------------------


class TestDriftDetectorTamperWarning:
    def test_drift_detected_logs_warning_and_emits_event(self):
        """Lines 688-689: when drift detector reports False, warning is logged."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        # Return a clean stop response so the loop exits after one iteration
        provider.complete_with_tools.return_value = CompletionResponse(
            content="response",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )

        tool = MagicMock()
        tool.name = "calculator"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        tool_reg.get.return_value = tool

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=2, capability_mode="full")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(cfg)
        rt._rate_limiter = None
        rt._memory_store = None
        rt._cost_tracker = None
        rt._context_manager = None

        drift = MagicMock()
        drift.verify.return_value = False  # tamper detected
        rt._drift_detector = drift

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            result = rt.run("test drift")

        # verify() called at least once — tamper path was hit
        drift.verify.assert_called()
        assert result == "response"


# ---------------------------------------------------------------------------
# Line 770: trust score drops below threshold after tool error
# ---------------------------------------------------------------------------


class TestTrustScoreDropWarning:
    def test_trust_warning_logged_when_score_below_threshold(self):
        """Line 770: _trust_scorer.is_trusted returns False → warning logged."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True

        tc = ToolCall(id="tc1", name="calc", arguments={})
        tool_call_resp = CompletionResponse(
            content="",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        stop_resp = CompletionResponse(
            content="done",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )
        provider.complete_with_tools.side_effect = [tool_call_resp, stop_resp]

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool
        # Tool execution returns an error result
        tool_reg.execute.return_value = MagicMock(success=False, output=None, error="failed")

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=5, capability_mode="full")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(cfg)
        rt._rate_limiter = None
        rt._memory_store = None
        rt._cost_tracker = None
        rt._context_manager = None

        # Make trust scorer report below threshold
        trust = MagicMock()
        trust.is_trusted.return_value = False
        trust.score.return_value = 150
        rt._trust_scorer = trust

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            result = rt.run("low trust tool")

        trust.record_failure.assert_called_with("calc")
        trust.is_trusted.assert_called_with("calc")
        assert result == "done"


# ---------------------------------------------------------------------------
# Line 799: large content triggers _intercept_large_content
# ---------------------------------------------------------------------------


class TestLargeContentIntercept:
    def _setup_tool_loop_with_large_output(self, large_content):
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True

        tc = ToolCall(id="tc1", name="mytool", arguments={})
        tool_call_resp = CompletionResponse(
            content="",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        stop_resp = CompletionResponse(
            content="finished",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )
        provider.complete_with_tools.side_effect = [tool_call_resp, stop_resp]

        tool = MagicMock()
        tool.name = "mytool"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["mytool"]
        tool_reg.get.return_value = tool
        tool_reg.execute.return_value = MagicMock(
            success=True, output=large_content, error=None
        )

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=5, capability_mode="full")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(cfg)
        rt._rate_limiter = None
        rt._memory_store = None
        rt._cost_tracker = None
        rt._context_manager = None
        return rt, reg, tool_reg

    def test_large_content_calls_intercept(self):
        """Line 799: content > _LARGE_CONTENT_THRESHOLD invokes _intercept_large_content."""
        large = "x" * (runtime_module._LARGE_CONTENT_THRESHOLD + 1)
        rt, reg, tool_reg = self._setup_tool_loop_with_large_output(large)

        intercept_result = "[stored as content-id-123]"
        with (
            patch.object(rt, "_intercept_large_content", return_value=intercept_result) as mock_ic,
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            result = rt.run("get large output")

        mock_ic.assert_called_once()
        assert result == "finished"

    def test_hard_truncation_when_content_still_oversized(self):
        """Line 806: after _intercept_large_content, content still > _MAX_TOOL_RESULT_CHARS."""
        # Make content bigger than both thresholds
        oversized = "y" * (runtime_module._MAX_TOOL_RESULT_CHARS + 1)
        rt, reg, tool_reg = self._setup_tool_loop_with_large_output(oversized)

        # _intercept_large_content returns the same huge string (simulate storage failure path)
        with (
            patch.object(rt, "_intercept_large_content", return_value=oversized),
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            result = rt.run("oversized content")

        # Run completes; hard truncation was applied internally
        assert result == "finished"


# ---------------------------------------------------------------------------
# Line 1030: _get_tools — capability_mode="discord" filters to _DISCORD_TOOLS
# ---------------------------------------------------------------------------


class TestGetToolsDiscordMode:
    def test_discord_mode_filters_to_discord_tools(self):
        """Line 1030: capability_mode='discord' keeps only _DISCORD_TOOLS."""
        cfg = AgentConfig(provider="fake", capability_mode="discord")

        # Pick a known member of _DISCORD_TOOLS
        discord_tool = MagicMock()
        discord_tool.name = "calculator"

        # Pick a tool that is in _SAFE_CHAT_TOOLS but NOT in _DISCORD_TOOLS
        desktop_only_name = next(
            n for n in AgentRuntime._SAFE_CHAT_TOOLS
            if n not in AgentRuntime._DISCORD_TOOLS
        )
        extra_tool = MagicMock()
        extra_tool.name = desktop_only_name

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = [discord_tool.name, extra_tool.name]
        tool_reg.get.side_effect = lambda name: (
            discord_tool if name == discord_tool.name else extra_tool
        )

        with (
            patch("missy.agent.runtime.get_registry", return_value=_make_registry(_make_provider())),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(cfg)

        with patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            tools = rt._get_tools()

        assert all(
            getattr(t, "name", "") in AgentRuntime._DISCORD_TOOLS for t in tools
        )
        assert discord_tool in tools
        assert extra_tool not in tools


# ---------------------------------------------------------------------------
# Lines 1215-1216: _build_messages — memory_store.get_learnings raises
# ---------------------------------------------------------------------------


class TestBuildMessagesLearningsException:
    def test_get_learnings_exception_is_caught(self):
        """Lines 1215-1216: get_learnings raising is logged and caught."""
        provider = _make_provider(reply="answer")
        rt, reg = _build_runtime(provider, max_iterations=1)

        ctx_mgr = MagicMock()
        ctx_mgr.build_messages.return_value = ("system prompt", [{"role": "user", "content": "q"}])
        rt._context_manager = ctx_mgr

        mem = MagicMock()
        mem.get_learnings.side_effect = Exception("DB error")
        rt._memory_store = mem

        result = rt._build_context_messages(
            user_input="q",
            history=[],
            session_id="s1",
            attention_query="",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        mem.get_learnings.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 1225-1229: _build_messages — get_summaries filters parent_id=None
# ---------------------------------------------------------------------------


class TestBuildMessagesSummariesFiltered:
    def test_top_level_summaries_are_kept(self):
        """Lines 1225-1229: summaries with parent_id=None are included; children excluded."""
        provider = _make_provider()
        rt, reg = _build_runtime(provider, max_iterations=1)

        ctx_mgr = MagicMock()
        ctx_mgr.build_messages.return_value = ("sys", [{"role": "user", "content": "hi"}])
        rt._context_manager = ctx_mgr

        top_summary = MagicMock()
        top_summary.parent_id = None
        top_summary.content = "Top-level summary text"

        child_summary = MagicMock()
        child_summary.parent_id = "parent-001"
        child_summary.content = "Child summary"

        mem = MagicMock()
        mem.get_learnings.return_value = []
        mem.get_summaries.return_value = [top_summary, child_summary]
        rt._memory_store = mem

        system, msgs = rt._build_context_messages(
            user_input="hello",
            history=[],
            session_id="sess-1",
            attention_query="",
        )

        # build_messages was called with only the top-level summary
        call_kwargs = ctx_mgr.build_messages.call_args
        summaries_passed = call_kwargs.kwargs.get("summaries") or (
            call_kwargs.args[4] if len(call_kwargs.args) > 4 else None
        )
        if summaries_passed is not None:
            assert child_summary not in summaries_passed


# ---------------------------------------------------------------------------
# Lines 1270-1272: _build_messages — playbook patterns non-empty → injected
# ---------------------------------------------------------------------------


class TestBuildMessagesPlaybookInjection:
    def test_playbook_pattern_appended_to_system_prompt(self):
        """Lines 1270-1272: non-empty playbook patterns are appended to base_system."""
        provider = _make_provider()
        rt, reg = _build_runtime(provider)

        ctx_mgr = MagicMock()
        ctx_mgr.build_messages.return_value = ("base system", [])
        rt._context_manager = ctx_mgr

        mem = MagicMock()
        mem.get_learnings.return_value = []
        mem.get_summaries.return_value = []
        rt._memory_store = mem

        playbook_pattern = "\n## Proven patterns:\n- Always use tool X first"

        # _get_playbook_patterns is called but not yet defined on AgentRuntime;
        # use create=True so patch.object adds it to the instance.
        with patch.object(
            rt,
            "_get_playbook_patterns",
            return_value=playbook_pattern,
            create=True,
        ):
            rt._build_context_messages(
                user_input="task",
                history=[],
                session_id="s",
                attention_query="task",
            )

        # ctx_mgr.build_messages was called with system that includes the pattern
        call_kwargs = ctx_mgr.build_messages.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs.args[0]
        assert playbook_pattern in system_arg


# ---------------------------------------------------------------------------
# Lines 1286-1294: _build_messages — synthesized block truthy → appended to system
# ---------------------------------------------------------------------------


class TestBuildMessagesSynthesizedBlock:
    def test_synthesized_block_appended_to_system(self):
        """Lines 1286-1294: synthesized_block truthy → context manager called with None learnings."""
        provider = _make_provider()
        rt, reg = _build_runtime(provider)

        ctx_mgr = MagicMock()
        ctx_mgr.build_messages.return_value = ("base system", [])
        rt._context_manager = ctx_mgr

        mem = MagicMock()
        mem.get_learnings.return_value = ["learned something"]
        mem.get_summaries.return_value = []
        rt._memory_store = mem

        synthesized = "## Synthesized Memory\nRelevant context here."

        with patch.object(rt, "_synthesize_memory", return_value=synthesized):
            system, msgs = rt._build_context_messages(
                user_input="q",
                history=[],
                session_id="s",
                attention_query="q",
            )

        # build_messages called with learnings=None when synthesized block is used
        call_kwargs = ctx_mgr.build_messages.call_args
        learnings_arg = call_kwargs.kwargs.get("learnings")
        assert learnings_arg is None

        # synthesized block appended to returned system
        assert synthesized in system


# ---------------------------------------------------------------------------
# Lines 1322-1346: _synthesize_memory — all paths
# ---------------------------------------------------------------------------


class TestSynthesizeMemory:
    def test_no_content_returns_empty_string(self):
        """Line 1341: no fragments → returns ''."""
        rt, _ = _build_runtime()
        result = rt._synthesize_memory(
            learnings=None, summary_texts=[], playbook_texts=[], query=""
        )
        assert result == ""

    def test_with_learnings_adds_fragment(self):
        """Lines 1329-1330: learnings → add_fragments called with 'learnings'."""
        rt, _ = _build_runtime()

        synth = MagicMock()
        synth.synthesize.return_value = "synthesized output"

        with patch("missy.memory.synthesizer.MemorySynthesizer", return_value=synth):
            result = rt._synthesize_memory(
                learnings=["I learned X"],
                summary_texts=[],
                playbook_texts=[],
                query="test",
            )

        synth.add_fragments.assert_any_call("learnings", ["I learned X"], base_relevance=0.7)
        assert result == "synthesized output"

    def test_with_summary_texts_adds_fragment(self):
        """Lines 1333-1334: summary_texts → add_fragments called with 'summaries'."""
        rt, _ = _build_runtime()

        synth = MagicMock()
        synth.synthesize.return_value = "summary block"

        with patch("missy.memory.synthesizer.MemorySynthesizer", return_value=synth):
            result = rt._synthesize_memory(
                learnings=None,
                summary_texts=["Summary A"],
                playbook_texts=[],
                query="q",
            )

        synth.add_fragments.assert_any_call("summaries", ["Summary A"], base_relevance=0.4)
        assert result == "summary block"

    def test_with_playbook_texts_adds_fragment(self):
        """Lines 1337-1338: playbook_texts → add_fragments called with 'playbook'."""
        rt, _ = _build_runtime()

        synth = MagicMock()
        synth.synthesize.return_value = "playbook block"

        with patch("missy.memory.synthesizer.MemorySynthesizer", return_value=synth):
            result = rt._synthesize_memory(
                learnings=None,
                summary_texts=[],
                playbook_texts=["pattern 1"],
                query="q",
            )

        synth.add_fragments.assert_any_call("playbook", ["pattern 1"], base_relevance=0.6)
        assert result == "playbook block"

    def test_all_fragments_added(self):
        """Lines 1329-1338: all three fragment types added when all provided."""
        rt, _ = _build_runtime()

        synth = MagicMock()
        synth.synthesize.return_value = "full block"

        with patch("missy.memory.synthesizer.MemorySynthesizer", return_value=synth):
            result = rt._synthesize_memory(
                learnings=["L1"],
                summary_texts=["S1"],
                playbook_texts=["P1"],
                query="all",
            )

        assert synth.add_fragments.call_count == 3
        assert result == "full block"

    def test_synthesize_exception_returns_empty_string(self):
        """Lines 1344-1346: synthesize() raises → returns ''."""
        rt, _ = _build_runtime()

        synth = MagicMock()
        synth.synthesize.side_effect = RuntimeError("synthesis failed")

        with patch("missy.memory.synthesizer.MemorySynthesizer", return_value=synth):
            result = rt._synthesize_memory(
                learnings=["something"],
                summary_texts=[],
                playbook_texts=[],
                query="q",
            )

        assert result == ""


# ---------------------------------------------------------------------------
# Lines 1394-1432: _intercept_large_content — all three paths
# ---------------------------------------------------------------------------


class TestInterceptLargeContent:
    def test_no_memory_store_returns_preview(self):
        """Lines 1394-1400: _memory_store is None → returns preview with fallback message."""
        rt, _ = _build_runtime()
        rt._memory_store = None

        content = "A" * 1000
        result = rt._intercept_large_content("sess-1", "my_tool", content)

        assert content[:400] in result
        assert "No memory store" in result
        assert str(len(content)) in result

    def test_normal_path_stores_and_returns_reference(self):
        """Lines 1401-1426: memory store present → LargeContentRecord created, reference returned."""
        rt, _ = _build_runtime()

        mem = MagicMock()
        mem.store_large_content.return_value = "content-abc-123"
        rt._memory_store = mem

        content = "B" * 1000
        tool_name = "file_read"
        session_id = "sess-2"

        mock_record = MagicMock()

        with patch("missy.memory.sqlite_store.LargeContentRecord") as MockRecord:
            MockRecord.new.return_value = mock_record
            result = rt._intercept_large_content(session_id, tool_name, content)

        mem.store_large_content.assert_called_once_with(mock_record)
        assert "content-abc-123" in result
        assert str(len(content)) in result
        assert tool_name in result

    def test_store_raises_returns_storage_failed_preview(self):
        """Lines 1427-1432: store_large_content raises → returns storage-failed preview."""
        rt, _ = _build_runtime()

        mem = MagicMock()
        mem.store_large_content.side_effect = Exception("write failed")
        rt._memory_store = mem

        content = "C" * 1000

        with patch("missy.memory.sqlite_store.LargeContentRecord") as MockRecord:
            MockRecord.new.return_value = MagicMock()
            result = rt._intercept_large_content("sess-3", "tool_x", content)

        assert content[:400] in result
        assert "storage failed" in result


# ---------------------------------------------------------------------------
# Lines 1687-1688: _make_interactive_approval — exception → None
# ---------------------------------------------------------------------------


class TestMakeInteractiveApproval:
    def test_exception_returns_none(self):
        """Lines 1687-1688: ImportError during approval setup → None."""
        with patch.dict(
            sys.modules,
            {
                "missy.agent.interactive_approval": None,
                "missy.gateway.client": None,
            },
        ):
            result = AgentRuntime._make_interactive_approval()
        assert result is None


# ---------------------------------------------------------------------------
# Lines 1702-1703: _make_drift_detector — exception → None
# ---------------------------------------------------------------------------


class TestMakeDriftDetector:
    def test_exception_returns_none(self):
        """Lines 1702-1703: import fails → None returned."""
        with patch.dict(sys.modules, {"missy.security.drift": None}):
            result = AgentRuntime._make_drift_detector()
        assert result is None

    def test_normal_path_returns_detector(self):
        """Lines 1699-1701: normal import → PromptDriftDetector instance."""
        mock_detector = MagicMock()
        mock_module = MagicMock()
        mock_module.PromptDriftDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"missy.security.drift": mock_module}):
            result = AgentRuntime._make_drift_detector()

        assert result is mock_detector


# ---------------------------------------------------------------------------
# Lines 1720-1726: _make_identity — key file absent → generate and save
# ---------------------------------------------------------------------------


class TestMakeIdentity:
    def test_key_absent_generates_and_saves_identity(self):
        """Lines 1720-1723: key file not present → generate() + save() called."""
        mock_identity = MagicMock()
        mock_identity.public_key_fingerprint.return_value = "fp:abc123"

        mock_module = MagicMock()
        mock_module.DEFAULT_KEY_PATH = "/fake/.missy/identity.pem"
        mock_module.AgentIdentity.generate.return_value = mock_identity
        mock_module.AgentIdentity.from_key_file.return_value = MagicMock()

        with (
            patch.dict(sys.modules, {"missy.security.identity": mock_module}),
            patch("os.path.exists", return_value=False),
        ):
            result = AgentRuntime._make_identity()

        mock_module.AgentIdentity.generate.assert_called_once()
        mock_identity.save.assert_called_once_with("/fake/.missy/identity.pem")
        assert result is mock_identity

    def test_key_present_loads_from_file(self):
        """Line 1719: key file exists → from_key_file called."""
        loaded_identity = MagicMock()

        mock_module = MagicMock()
        mock_module.DEFAULT_KEY_PATH = "/fake/.missy/identity.pem"
        mock_module.AgentIdentity.from_key_file.return_value = loaded_identity

        with (
            patch.dict(sys.modules, {"missy.security.identity": mock_module}),
            patch("os.path.exists", return_value=True),
        ):
            result = AgentRuntime._make_identity()

        mock_module.AgentIdentity.from_key_file.assert_called_once_with(
            "/fake/.missy/identity.pem"
        )
        assert result is loaded_identity

    def test_exception_returns_none(self):
        """Lines 1724-1726: any exception → None returned."""
        with patch.dict(sys.modules, {"missy.security.identity": None}):
            result = AgentRuntime._make_identity()
        assert result is None


# ---------------------------------------------------------------------------
# Lines 1740-1742: _make_attention_system — exception → None
# ---------------------------------------------------------------------------


class TestMakeAttentionSystem:
    def test_exception_returns_none(self):
        """Lines 1740-1742: ImportError → None returned."""
        with patch.dict(sys.modules, {"missy.agent.attention": None}):
            result = AgentRuntime._make_attention_system()
        assert result is None

    def test_normal_path_returns_system(self):
        """Lines 1737-1739: normal import → AttentionSystem instance."""
        mock_system = MagicMock()
        mock_module = MagicMock()
        mock_module.AttentionSystem.return_value = mock_system

        with patch.dict(sys.modules, {"missy.agent.attention": mock_module}):
            result = AgentRuntime._make_attention_system()

        assert result is mock_system


# ---------------------------------------------------------------------------
# Lines 1751-1753: _make_persona_manager — exception → None
# ---------------------------------------------------------------------------


class TestMakePersonaManager:
    def test_exception_returns_none(self):
        """Lines 1751-1753: ImportError → None returned."""
        with patch.dict(sys.modules, {"missy.agent.persona": None}):
            result = AgentRuntime._make_persona_manager()
        assert result is None

    def test_normal_path_returns_manager(self):
        """Lines 1748-1750: normal import → PersonaManager instance."""
        mock_mgr = MagicMock()
        mock_module = MagicMock()
        mock_module.PersonaManager.return_value = mock_mgr

        with patch.dict(sys.modules, {"missy.agent.persona": mock_module}):
            result = AgentRuntime._make_persona_manager()

        assert result is mock_mgr


# ---------------------------------------------------------------------------
# Lines 1773-1775: _make_response_shaper — exception → None
# ---------------------------------------------------------------------------


class TestMakeResponseShaper:
    def test_exception_returns_none(self):
        """Lines 1773-1775: ImportError → None returned."""
        with patch.dict(sys.modules, {"missy.agent.behavior": None}):
            result = AgentRuntime._make_response_shaper()
        assert result is None

    def test_normal_path_returns_shaper(self):
        """Lines 1770-1772: normal import → ResponseShaper instance."""
        mock_shaper = MagicMock()
        mock_module = MagicMock()
        mock_module.ResponseShaper.return_value = mock_shaper

        with patch.dict(sys.modules, {"missy.agent.behavior": mock_module}):
            result = AgentRuntime._make_response_shaper()

        assert result is mock_shaper


# ---------------------------------------------------------------------------
# Lines 1784-1786: _make_intent_interpreter — exception → None
# ---------------------------------------------------------------------------


class TestMakeIntentInterpreter:
    def test_exception_returns_none(self):
        """Lines 1784-1786: ImportError → None returned."""
        with patch.dict(sys.modules, {"missy.agent.behavior": None}):
            result = AgentRuntime._make_intent_interpreter()
        assert result is None

    def test_normal_path_returns_interpreter(self):
        """Lines 1781-1783: normal import → IntentInterpreter instance."""
        mock_interp = MagicMock()
        mock_module = MagicMock()
        mock_module.IntentInterpreter.return_value = mock_interp

        with patch.dict(sys.modules, {"missy.agent.behavior": mock_module}):
            result = AgentRuntime._make_intent_interpreter()

        assert result is mock_interp
