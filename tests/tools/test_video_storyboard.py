"""Tests for F16 — VideoStoryboardTool orchestration (mocked sub-tools)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Part III: captions, per-scene transitions, overrides, continuity, soundtrack
# ---------------------------------------------------------------------------


def _gen_seeded(path_prefix="clip"):
    """Generator mock that echoes a distinct seed per clip, like the real tool."""
    gen = MagicMock()
    counter = {"n": 0}

    def _exec(**kwargs):
        counter["n"] += 1
        return ToolResult(
            success=True,
            output={"path": f"/v/{path_prefix}{counter['n']}.mp4", "seed": 1000 + counter["n"]},
        )

    gen.execute.side_effect = _exec
    return gen


def _edit_tracking():
    """Edit mock whose outputs are distinguishable per operation + call order."""
    edit = MagicMock()
    counter = {"n": 0}

    def _exec(*, operation, **kwargs):
        counter["n"] += 1
        return ToolResult(success=True, output={"path": f"/v/{operation}{counter['n']}.mp4"})

    edit.execute.side_effect = _exec
    return edit


class TestCaptions:
    def test_caption_overlays_each_captioned_scene(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[
                {"prompt": "a", "caption": "Dawn"},
                {"prompt": "b"},
                {"prompt": "c", "caption": "Dusk"},
            ]
        )
        assert r.success, r.error
        texts = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "text"]
        assert [c.kwargs["text"] for c in texts] == ["Dawn", "Dusk"]
        assert all(c.kwargs["position"] == "bottom" for c in texts)

    def test_caption_failure_degrades_to_uncaptioned_clip(self) -> None:
        gen = _gen_seeded()
        edit = MagicMock()

        def _exec(*, operation, **kwargs):
            if operation == "text":
                return ToolResult(success=False, output=None, error="font missing")
            return ToolResult(success=True, output={"path": f"/v/{operation}.mp4"})

        edit.execute.side_effect = _exec
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a", "caption": "Dawn"}, {"prompt": "b"}]
        )
        assert r.success, r.error  # storyboard survives; clip used uncaptioned
        assert r.output["scenes"][0]["path"] == "/v/clip1.mp4"


class TestTransitions:
    def test_uniform_transitions_single_concat(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}], transition="none"
        )
        assert r.success, r.error
        concats = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "concat"]
        assert len(concats) == 1
        assert concats[0].kwargs["transition"] == "none"
        assert len(concats[0].kwargs["inputs"]) == 3

    def test_mixed_per_scene_transitions_fold_pairwise(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[
                {"prompt": "a"},
                {"prompt": "b", "transition": "none"},
                {"prompt": "c", "transition": "crossfade"},
            ],
        )
        assert r.success, r.error
        concats = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "concat"]
        assert [c.kwargs["transition"] for c in concats] == ["none", "crossfade"]
        # Each fold joins the running result with the next clip.
        assert all(len(c.kwargs["inputs"]) == 2 for c in concats)

    def test_non_last_scene_transition_now_honored(self) -> None:
        # Regression for the S2 bug: only scenes[-1] used to be consulted.
        gen, edit = _gen_seeded(), _edit_tracking()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b", "transition": "none"}, {"prompt": "c"}],
            transition="none",
        )
        concats = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "concat"]
        assert len(concats) == 1  # uniform: none/none
        assert concats[0].kwargs["transition"] == "none"

    def test_invalid_transition_rejected(self) -> None:
        r = VideoStoryboardTool(_gen_seeded(), _edit_tracking()).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b", "transition": "wipe"}]
        )
        assert not r.success
        assert "wipe" in r.error


class TestPerSceneOverrides:
    def test_whitelisted_overrides_forwarded_and_win_over_shared(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[
                {"prompt": "a", "seed": 7, "negative_prompt": "rain", "steps": 30},
                {"prompt": "b"},
            ],
            seed=42,
        )
        first = gen.execute.call_args_list[0].kwargs
        second = gen.execute.call_args_list[1].kwargs
        assert first["seed"] == 7 and first["negative_prompt"] == "rain" and first["steps"] == 30
        assert second["seed"] == 42  # shared kwarg still applies where not overridden

    def test_storyboard_vocabulary_keys_not_forwarded(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a", "caption": "hi", "duration": 2.0, "transition": "none"}]
        )
        kwargs = gen.execute.call_args.kwargs
        assert "caption" not in kwargs
        assert "duration" not in kwargs
        assert "transition" not in kwargs

    def test_per_scene_seed_echoed_in_output(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(scenes=[{"prompt": "a"}, {"prompt": "b"}])
        assert [s["seed"] for s in r.output["scenes"]] == [1001, 1002]
        assert [s["prompt"] for s in r.output["scenes"]] == ["a", "b"]


class TestContinuity:
    def test_chains_last_frame_into_next_scene(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}],
            continuity=True,
            backend="wan",
        )
        assert r.success, r.error
        frames = [
            c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "extract_frame"
        ]
        assert len(frames) == 2  # after every scene but the last
        assert all(c.kwargs["at"] == -1.0 for c in frames)
        gen_calls = gen.execute.call_args_list
        assert "image_path" not in gen_calls[0].kwargs  # scene 0 is t2v
        assert gen_calls[1].kwargs["image_path"].startswith("/v/extract_frame")
        assert gen_calls[2].kwargs["image_path"].startswith("/v/extract_frame")
        assert r.output["continuity"] is True

    def test_scene_own_image_path_wins_over_chain(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b", "image_path": "/img/own.png"}],
            continuity=True,
        )
        assert gen.execute.call_args_list[1].kwargs["image_path"] == "/img/own.png"

    def test_rejected_for_text_only_backends(self) -> None:
        for backend in ("animatediff", "wan14b"):
            r = VideoStoryboardTool(_gen_seeded(), _edit_tracking()).execute(
                scenes=[{"prompt": "a"}, {"prompt": "b"}], continuity=True, backend=backend
            )
            assert not r.success
            assert "continuity" in r.error

    def test_svd_needs_scene0_image(self) -> None:
        r = VideoStoryboardTool(_gen_seeded(), _edit_tracking()).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}], continuity=True, backend="svd"
        )
        assert not r.success
        assert "scene 0" in r.error

    def test_extraction_failure_is_hard_error(self) -> None:
        gen = _gen_seeded()
        edit = MagicMock()

        def _exec(*, operation, **kwargs):
            if operation == "extract_frame":
                return ToolResult(success=False, output=None, error="no frame")
            return ToolResult(success=True, output={"path": f"/v/{operation}.mp4"})

        edit.execute.side_effect = _exec
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}], continuity=True
        )
        assert not r.success
        assert "frame extraction" in r.error


class TestSoundtrack:
    def test_audio_path_muxed_over_final_assembly(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}, {"prompt": "b"}],
            audio_path="/a/score.mp3",
            audio_mode="mix",
            audio_loop=True,
        )
        assert r.success, r.error
        audio_calls = [
            c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "audio"
        ]
        assert len(audio_calls) == 1
        assert audio_calls[0].kwargs["audio_file"] == "/a/score.mp3"
        assert audio_calls[0].kwargs["audio_mode"] == "mix"
        assert audio_calls[0].kwargs["loop"] is True
        # The mux is the last edit and its output is the final path.
        assert edit.execute.call_args_list[-1] is audio_calls[0]
        assert r.output["path"].startswith("/v/audio")
        assert r.output["soundtrack"] == "/a/score.mp3"

    def test_soundtrack_failure_is_hard_error(self) -> None:
        gen = _gen_seeded()
        edit = MagicMock()

        def _exec(*, operation, **kwargs):
            if operation == "audio":
                return ToolResult(success=False, output=None, error="bad track")
            return ToolResult(success=True, output={"path": f"/v/{operation}.mp4"})

        edit.execute.side_effect = _exec
        r = VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}], audio_path="/a/score.mp3"
        )
        assert not r.success
        assert "soundtrack" in r.error


class TestPartIIIValidation:
    def test_title_seconds_clamped(self) -> None:
        gen, edit = _gen_seeded(), _edit_tracking()
        VideoStoryboardTool(gen, edit).execute(
            scenes=[{"prompt": "a"}], title="T", title_seconds=500.0
        )
        text_call = [c for c in edit.execute.call_args_list if c.kwargs.get("operation") == "text"][
            0
        ]
        assert text_call.kwargs["end"] == 30.0

    def test_schema_documents_new_params(self) -> None:
        props = VideoStoryboardTool().get_schema()["parameters"]["properties"]
        for key in ("continuity", "audio_path", "audio_mode", "audio_loop"):
            assert key in props


def test_tdeep_047_production_children_dispatch_through_registry_reference_monitor():
    registry = MagicMock()
    registry.execute.return_value = ToolResult(
        success=False,
        output=None,
        error="policy denied child",
        policy_denied=True,
    )
    tool = VideoStoryboardTool()
    with patch("missy.tools.registry.get_tool_registry", return_value=registry):
        result = tool.execute(scenes=[{"prompt": "safe scene"}])
    registry.execute.assert_called_once_with("video_generate", backend="wan", prompt="safe scene")
    assert not result.success
    assert "policy denied child" in result.error
