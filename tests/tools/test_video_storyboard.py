"""Tests for F16 — VideoStoryboardTool orchestration (mocked sub-tools)."""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.tools.base import ToolResult
from missy.tools.builtin.video_storyboard import VideoStoryboardTool


def _gen_ok(path_prefix="clip"):
    gen = MagicMock()
    counter = {"n": 0}

    def _exec(**kwargs):
        counter["n"] += 1
        return ToolResult(success=True, output={"path": f"/v/{path_prefix}{counter['n']}.mp4"})

    gen.execute.side_effect = _exec
    return gen


def _edit_ok():
    edit = MagicMock()

    def _exec(*, operation, **kwargs):
        return ToolResult(success=True, output={"path": f"/v/{operation}_out.mp4"})

    edit.execute.side_effect = _exec
    return edit


class TestValidation:
    def test_empty_scenes_rejected(self) -> None:
        r = VideoStoryboardTool(_gen_ok(), _edit_ok()).execute(scenes=[])
        assert r.success is False
        assert "non-empty list" in r.error

    def test_scene_without_prompt_rejected(self) -> None:
        r = VideoStoryboardTool(_gen_ok(), _edit_ok()).execute(scenes=[{"duration": 3}])
        assert r.success is False
        assert "prompt" in r.error

    def test_too_many_scenes(self) -> None:
        scenes = [{"prompt": f"s{i}"} for i in range(20)]
        r = VideoStoryboardTool(_gen_ok(), _edit_ok()).execute(scenes=scenes)
        assert r.success is False
        assert "too many scenes" in r.error


class TestOrchestration:
    def test_single_scene_no_concat(self) -> None:
        gen, edit = _gen_ok(), _edit_ok()
        r = VideoStoryboardTool(gen, edit).execute(scenes=[{"prompt": "a sunrise"}])
        assert r.success is True
        assert r.output["scene_count"] == 1
        assert r.output["path"] == "/v/clip1.mp4"
        # No concat for a single clip.
        assert not any(c.kwargs.get("operation") == "concat" for c in edit.execute.call_args_list)

    def test_multi_scene_concats(self) -> None:
        gen, edit = _gen_ok(), _edit_ok()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "sunrise"}, {"prompt": "starry night"}]
        )
        assert r.success is True
        assert r.output["scene_count"] == 2
        ops = [c.kwargs.get("operation") for c in edit.execute.call_args_list]
        assert "concat" in ops
        assert r.output["path"] == "/v/concat_out.mp4"

    def test_duration_triggers_trim(self) -> None:
        gen, edit = _gen_ok(), _edit_ok()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "clip", "duration": 3.0}, {"prompt": "clip2", "duration": 2.0}]
        )
        trims = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "trim"]
        assert len(trims) == 2
        assert trims[0].kwargs["duration"] == 3.0

    def test_title_adds_text_overlay(self) -> None:
        gen, edit = _gen_ok(), _edit_ok()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}], title="Day and Night"
        )
        text_calls = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "text"]
        assert len(text_calls) == 1
        assert text_calls[0].kwargs["text"] == "Day and Night"
        assert r.output["path"] == "/v/text_out.mp4"

    def test_forwards_shared_kwargs_to_generate(self) -> None:
        gen, edit = _gen_ok(), _edit_ok()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "x"}], backend="wan14b", width=832, seed=42
        )
        _, kwargs = gen.execute.call_args
        assert kwargs["backend"] == "wan14b"
        assert kwargs["width"] == 832
        assert kwargs["seed"] == 42


class TestFailures:
    def test_generation_failure_aborts(self) -> None:
        gen = MagicMock()
        gen.execute.return_value = ToolResult(success=False, output=None, error="GPU offline")
        r = VideoStoryboardTool(gen, _edit_ok()).execute(scenes=[{"prompt": "x"}])
        assert r.success is False
        assert "generation failed" in r.error
        assert "GPU offline" in r.error

    def test_concat_failure_reported(self) -> None:
        gen = _gen_ok()
        edit = MagicMock()

        def _exec(*, operation, **kwargs):
            if operation == "concat":
                return ToolResult(success=False, output=None, error="ffmpeg died")
            return ToolResult(success=True, output={"path": "/v/x.mp4"})

        edit.execute.side_effect = _exec
        r = VideoStoryboardTool(gen, edit).execute(scenes=[{"prompt": "a"}, {"prompt": "b"}])
        assert r.success is False
        assert "concat failed" in r.error


class TestMetadataAndSchema:
    def test_permissions_union(self) -> None:
        t = VideoStoryboardTool()
        assert t.permissions.network is True
        assert t.permissions.shell is True
        assert t.permissions.filesystem_write is True

    def test_schema_requires_scenes(self) -> None:
        schema = VideoStoryboardTool().get_schema()
        assert schema["name"] == "video_storyboard"
        assert "scenes" in schema["parameters"]["required"]

    def test_registered_in_builtins(self) -> None:
        from missy.tools.builtin import _ALL_TOOL_CLASSES

        assert VideoStoryboardTool in _ALL_TOOL_CLASSES
