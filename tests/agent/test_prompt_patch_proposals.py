"""Tests for F13 — organic ERROR_AVOIDANCE prompt-patch proposals.

Drives AgentRuntime._maybe_propose_error_patch directly, with a temp-backed
PromptPatchManager so nothing touches ~/.missy.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.prompt_patches import PatchStatus, PatchType, PromptPatchManager
from missy.agent.runtime import AgentConfig, AgentRuntime


def _make_provider():
    from missy.providers.base import CompletionResponse

    p = MagicMock()
    p.name = "fake"
    p.is_available.return_value = True
    p.complete.return_value = CompletionResponse(
        content="ok", model="m", provider="fake", usage={}, raw={}, finish_reason="stop"
    )
    return p


def _make_runtime(**config_kwargs) -> AgentRuntime:
    provider = _make_provider()
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    cfg = AgentConfig(provider="fake", **config_kwargs)
    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(cfg)
    runtime._memory_store = None
    runtime._cost_tracking_enabled = False
    return runtime


@pytest.fixture
def patch_mgr(tmp_path: Path) -> PromptPatchManager:
    return PromptPatchManager(store_path=str(tmp_path / "patches.json"))


class TestConfigDefault:
    def test_flag_defaults_false(self) -> None:
        assert AgentConfig().prompt_patch_proposals_enabled is False


class TestDisabledByDefault:
    def test_no_proposal_when_disabled(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime()  # flag defaults False
        runtime._patch_manager = patch_mgr
        result = runtime._maybe_propose_error_patch("shell_exec", "permission denied")
        assert result is None
        assert patch_mgr.list_all() == []
        assert patch_mgr.list_proposed() == []


class TestEnabled:
    def test_proposes_error_avoidance_patch(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        runtime._patch_manager = patch_mgr
        patch_obj = runtime._maybe_propose_error_patch("file_read", "no such file: /x")
        assert patch_obj is not None
        assert patch_obj.patch_type == PatchType.ERROR_AVOIDANCE
        # ERROR_AVOIDANCE is not auto-approved -> stays PROPOSED for review.
        assert patch_obj.status == PatchStatus.PROPOSED
        assert "file_read" in patch_obj.content
        assert patch_obj in patch_mgr.list_proposed()

    def test_dedup_same_tool_and_error(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        runtime._patch_manager = patch_mgr
        first = runtime._maybe_propose_error_patch("file_read", "no such file: /x")
        second = runtime._maybe_propose_error_patch("file_read", "no such file: /x")
        assert first is not None
        assert second is None  # deduped
        assert len(patch_mgr.list_proposed()) == 1

    def test_distinct_errors_each_propose(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        runtime._patch_manager = patch_mgr
        a = runtime._maybe_propose_error_patch("shell_exec", "permission denied")
        b = runtime._maybe_propose_error_patch("shell_exec", "command not found")
        assert a is not None and b is not None
        assert len(patch_mgr.list_proposed()) == 2

    def test_distinct_tools_each_propose(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        runtime._patch_manager = patch_mgr
        a = runtime._maybe_propose_error_patch("file_read", "boom")
        b = runtime._maybe_propose_error_patch("web_fetch", "boom")
        assert a is not None and b is not None
        assert len(patch_mgr.list_proposed()) == 2

    def test_never_raises_on_internal_error(self) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        broken = MagicMock()
        broken.propose.side_effect = RuntimeError("store exploded")
        runtime._patch_manager = broken
        # Must swallow and return None, not propagate.
        assert runtime._maybe_propose_error_patch("t", "err") is None

    def test_multiline_error_uses_first_line(self, patch_mgr: PromptPatchManager) -> None:
        runtime = _make_runtime(prompt_patch_proposals_enabled=True)
        runtime._patch_manager = patch_mgr
        patch_obj = runtime._maybe_propose_error_patch(
            "shell_exec", "line one is the summary\nstack frame\nmore noise"
        )
        assert patch_obj is not None
        assert "line one is the summary" in patch_obj.content
        assert "stack frame" not in patch_obj.content
