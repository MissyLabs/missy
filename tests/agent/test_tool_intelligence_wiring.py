"""Tests for tool-intelligence wiring in AgentRuntime: auto candidate
generation from RequestTracker patterns, and provider-benchmark gating of
_get_tools().
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from missy.agent.runtime import AgentConfig, AgentRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(tool_intelligence=None, **config_kwargs) -> AgentRuntime:
    """Build an AgentRuntime with heavy subsystems mocked out."""
    cfg = AgentConfig(tool_intelligence=tool_intelligence, **config_kwargs)
    with (
        patch("missy.agent.runtime.get_registry", side_effect=RuntimeError("no providers")),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(cfg)
    runtime._rate_limiter = None
    runtime._memory_store = None
    runtime._cost_tracker = None
    runtime._context_manager = None
    runtime._drift_detector = None
    return runtime


# ---------------------------------------------------------------------------
# _make_candidate_generator
# ---------------------------------------------------------------------------


class TestMakeCandidateGenerator:
    def test_disabled_by_default(self) -> None:
        runtime = _make_runtime()
        assert runtime._candidate_generator is None

    def test_none_tool_intelligence_disables(self) -> None:
        runtime = _make_runtime(tool_intelligence=None)
        assert runtime._candidate_generator is None

    def test_enabled_builds_generator(self) -> None:
        intel = SimpleNamespace(
            candidate_generation_enabled=True,
            allow_shell=False,
            min_pattern_count=3,
            check_every_n_requests=5,
        )
        runtime = _make_runtime(tool_intelligence=intel)
        assert runtime._candidate_generator is not None

    def test_import_failure_degrades_gracefully(self) -> None:
        intel = SimpleNamespace(candidate_generation_enabled=True, allow_shell=False)
        with patch(
            "missy.tools.intelligence.CandidateGenerator",
            side_effect=ImportError("boom"),
        ):
            runtime = _make_runtime(tool_intelligence=intel)
        assert runtime._candidate_generator is None


# ---------------------------------------------------------------------------
# _maybe_synthesize_candidates
# ---------------------------------------------------------------------------


class TestMaybeSynthesizeCandidates:
    def _enabled_runtime(self, check_every_n_requests: int = 1) -> AgentRuntime:
        intel = SimpleNamespace(
            candidate_generation_enabled=True,
            allow_shell=False,
            min_pattern_count=3,
            check_every_n_requests=check_every_n_requests,
        )
        return _make_runtime(tool_intelligence=intel)

    def test_noop_when_generator_missing(self) -> None:
        runtime = _make_runtime()  # generator disabled
        runtime._request_tracker = MagicMock()
        runtime._tracked_request_count = 1
        runtime._maybe_synthesize_candidates()
        runtime._request_tracker.get_frequent_patterns.assert_not_called()

    def test_noop_when_not_at_throttle_boundary(self) -> None:
        runtime = self._enabled_runtime(check_every_n_requests=5)
        runtime._request_tracker = MagicMock()
        runtime._tracked_request_count = 3  # not a multiple of 5
        runtime._maybe_synthesize_candidates()
        runtime._request_tracker.get_frequent_patterns.assert_not_called()

    def test_generates_new_candidate_for_fresh_pattern(self) -> None:
        runtime = self._enabled_runtime()
        runtime._tracked_request_count = 1

        pattern = SimpleNamespace(pattern_key="abc123", representative="read file x")
        runtime._request_tracker = MagicMock()
        runtime._request_tracker.get_frequent_patterns.return_value = [pattern]

        mock_store = MagicMock()
        mock_store.get_by_pattern_key.return_value = None  # not yet proposed

        mock_candidate = MagicMock()
        runtime._candidate_generator = MagicMock()
        runtime._candidate_generator.generate_from_pattern.return_value = SimpleNamespace(
            ok=True, candidate=mock_candidate
        )

        with patch("missy.tools.intelligence.get_candidate_store", return_value=mock_store):
            runtime._maybe_synthesize_candidates()

        runtime._candidate_generator.generate_from_pattern.assert_called_once_with(pattern)
        mock_store.add.assert_called_once_with(mock_candidate)

    def test_skips_pattern_with_existing_candidate(self) -> None:
        runtime = self._enabled_runtime()
        runtime._tracked_request_count = 1

        pattern = SimpleNamespace(pattern_key="abc123", representative="read file x")
        runtime._request_tracker = MagicMock()
        runtime._request_tracker.get_frequent_patterns.return_value = [pattern]

        mock_store = MagicMock()
        mock_store.get_by_pattern_key.return_value = MagicMock()  # already proposed

        runtime._candidate_generator = MagicMock()

        with patch("missy.tools.intelligence.get_candidate_store", return_value=mock_store):
            runtime._maybe_synthesize_candidates()

        runtime._candidate_generator.generate_from_pattern.assert_not_called()
        mock_store.add.assert_not_called()

    def test_generation_failure_is_swallowed(self) -> None:
        runtime = self._enabled_runtime()
        runtime._tracked_request_count = 1
        runtime._request_tracker = MagicMock()
        runtime._request_tracker.get_frequent_patterns.side_effect = RuntimeError("db gone")

        # Should not raise.
        runtime._maybe_synthesize_candidates()

    def test_track_request_triggers_synthesis(self) -> None:
        runtime = self._enabled_runtime(check_every_n_requests=1)
        runtime._request_tracker = MagicMock()
        runtime._request_tracker.get_frequent_patterns.return_value = []

        runtime._track_request("do a thing", "sess-1", [], "anthropic")

        runtime._request_tracker.record.assert_called_once()
        runtime._request_tracker.get_frequent_patterns.assert_called_once()


# ---------------------------------------------------------------------------
# Provider gate wiring in _get_tools
# ---------------------------------------------------------------------------


class TestApplyProviderGate:
    def test_noop_when_gating_disabled(self) -> None:
        runtime = _make_runtime(provider="anthropic")
        result = runtime._apply_provider_gate(["calculator", "shell_exec"])
        assert result == ["calculator", "shell_exec"]

    def test_filters_denied_tools_when_enabled(self) -> None:
        intel = SimpleNamespace(
            provider_gating_enabled=True,
            provider_gating_min_samples=3,
            provider_gating_min_composite=0.4,
        )
        runtime = _make_runtime(provider="ollama", tool_intelligence=intel)

        mock_gate = MagicMock()
        mock_gate.filter_tools.return_value = (["calculator"], {"shell_exec": "weak provider"})

        with patch("missy.tools.intelligence.ToolProviderGate", return_value=mock_gate):
            result = runtime._apply_provider_gate(["calculator", "shell_exec"])

        assert result == ["calculator"]
        mock_gate.filter_tools.assert_called_once_with(["calculator", "shell_exec"], "ollama")

    def test_gate_construction_failure_falls_back_ungated(self) -> None:
        intel = SimpleNamespace(
            provider_gating_enabled=True,
            provider_gating_min_samples=3,
            provider_gating_min_composite=0.4,
        )
        runtime = _make_runtime(provider="ollama", tool_intelligence=intel)

        with patch(
            "missy.tools.intelligence.ToolProviderGate", side_effect=RuntimeError("no db")
        ):
            result = runtime._apply_provider_gate(["calculator", "shell_exec"])

        assert result == ["calculator", "shell_exec"]

    def test_gate_is_cached_across_calls(self) -> None:
        intel = SimpleNamespace(
            provider_gating_enabled=True,
            provider_gating_min_samples=3,
            provider_gating_min_composite=0.4,
        )
        runtime = _make_runtime(provider="ollama", tool_intelligence=intel)

        mock_gate = MagicMock()
        mock_gate.filter_tools.return_value = (["calculator"], {})

        with patch(
            "missy.tools.intelligence.ToolProviderGate", return_value=mock_gate
        ) as mock_cls:
            runtime._apply_provider_gate(["calculator"])
            runtime._apply_provider_gate(["calculator"])

        mock_cls.assert_called_once()


class TestGetToolsWithProviderGate:
    def test_get_tools_applies_gate_end_to_end(self) -> None:
        t_calc = MagicMock()
        t_calc.name = "calculator"
        t_shell = MagicMock()
        t_shell.name = "shell_exec"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator", "shell_exec"]
        tool_reg.is_enabled.return_value = True
        tool_reg.get.side_effect = {"calculator": t_calc, "shell_exec": t_shell}.get

        intel = SimpleNamespace(
            provider_gating_enabled=True,
            provider_gating_min_samples=3,
            provider_gating_min_composite=0.4,
            candidate_generation_enabled=False,
        )
        runtime = _make_runtime(provider="ollama", tool_intelligence=intel)

        mock_gate = MagicMock()
        mock_gate.filter_tools.return_value = (["calculator"], {"shell_exec": "weak"})

        with (
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
            patch("missy.tools.intelligence.ToolProviderGate", return_value=mock_gate),
        ):
            tools = runtime._get_tools()

        assert [t.name for t in tools] == ["calculator"]
