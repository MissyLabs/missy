"""Tests for the video_edit tool (ffmpeg-backed splice/trim/text/speed/resize)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from missy.tools.builtin.video_edit import (
    VideoEditTool,
    _atempo_chain,
    _build_concat_command,
    _build_concat_filter,
    _build_resize_command,
    _build_speed_command,
    _build_text_command,
    _build_text_filter,
    _build_trim_command,
    _encode_args,
    _probe,
    _unique_dest,
)


def _info(
    path: str = "/v/in.mp4",
    *,
    width: int = 640,
    height: int = 480,
    fps: float = 24.0,
    duration: float = 4.0,
    has_audio: bool = True,
) -> dict[str, Any]:
    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "has_audio": has_audio,
        "size_bytes": 1000,
    }


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _probe_json(
    *, width: int = 640, height: int = 480, has_audio: bool = True, duration: float = 4.0
) -> str:
    streams = [
        {
            "codec_type": "video",
            "width": width,
            "height": height,
            "avg_frame_rate": "24/1",
            "duration": str(duration),
        }
    ]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return json.dumps({"streams": streams, "format": {"duration": str(duration), "size": "2048"}})


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


class TestEncodeArgs:
    def test_h264_default(self) -> None:
        args = _encode_args("h264-mp4", 17, has_audio=True)
        assert args[:2] == ["-c:v", "libx264"]
        assert "17" in args
        assert "aac" in args
        assert "+faststart" in args

    def test_h265(self) -> None:
        args = _encode_args("h265-mp4", 20, has_audio=False)
        assert "libx265" in args
        assert "aac" not in args

    def test_nvenc_uses_cq_not_crf(self) -> None:
        args = _encode_args("nvenc_h264-mp4", 17, has_audio=False)
        assert "h264_nvenc" in args
        assert "-cq" in args
        assert "-crf" not in args

    def test_crf_clamped(self) -> None:
        args = _encode_args("h264-mp4", 99, has_audio=False)
        assert "51" in args


class TestConcatFilter:
    def test_plain_concat_with_audio(self) -> None:
        infos = [_info("/a.mp4"), _info("/b.mp4")]
        fc, vlabel, alabel = _build_concat_filter(
            infos, width=640, height=480, fps=24, transition="none", transition_duration=0.5
        )
        assert vlabel == "[vout]" and alabel == "[aout]"
        assert "concat=n=2:v=1:a=1" in fc
        assert "scale=640:480:force_original_aspect_ratio=decrease" in fc
        assert "fps=24" in fc
        assert "anullsrc" not in fc

    def test_silent_track_synthesized_for_mixed_audio(self) -> None:
        infos = [_info("/a.mp4", has_audio=True), _info("/b.mp4", has_audio=False, duration=3.0)]
        fc, _, alabel = _build_concat_filter(
            infos, width=640, height=480, fps=24, transition="none", transition_duration=0.5
        )
        assert alabel == "[aout]"
        assert "anullsrc=channel_layout=stereo" in fc
        assert "atrim=duration=3.000" in fc

    def test_video_only_when_no_input_has_audio(self) -> None:
        infos = [_info("/a.mp4", has_audio=False), _info("/b.mp4", has_audio=False)]
        fc, vlabel, alabel = _build_concat_filter(
            infos, width=640, height=480, fps=24, transition="none", transition_duration=0.5
        )
        assert alabel is None
        assert "concat=n=2:v=1:a=0" in fc
        assert "anullsrc" not in fc

    def test_crossfade_offsets_accumulate(self) -> None:
        infos = [
            _info("/a.mp4", duration=4.0),
            _info("/b.mp4", duration=3.0),
            _info("/c.mp4", duration=2.0),
        ]
        fc, vlabel, alabel = _build_concat_filter(
            infos, width=640, height=480, fps=24, transition="crossfade", transition_duration=0.5
        )
        assert vlabel == "[vout]" and alabel == "[aout]"
        # First fade at 4.0-0.5=3.5, second at 3.5+(3.0-0.5)=6.0.
        assert "xfade=transition=fade:duration=0.500:offset=3.500" in fc
        assert "xfade=transition=fade:duration=0.500:offset=6.000" in fc
        assert fc.count("acrossfade=d=0.500") == 2


class TestConcatCommand:
    def test_canvas_from_first_input_fps_is_max(self) -> None:
        infos = [
            _info("/a.mp4", width=1024, height=576, fps=6.0),
            _info("/b.mp4", width=832, height=480, fps=24.0),
        ]
        cmd = _build_concat_command(
            infos,
            "/out.mp4",
            transition="none",
            transition_duration=0.5,
            video_format="h264-mp4",
            crf=17,
        )
        assert cmd.count("-i") == 2
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "scale=1024:576" in fc
        assert "fps=24.0" in fc
        assert cmd[0] == "ffmpeg"
        assert "-nostdin" in cmd
        assert cmd[-1] == "/out.mp4"


class TestTrimCommand:
    def test_start_end_and_reencode(self) -> None:
        cmd = _build_trim_command(
            "/in.mp4",
            "/out.mp4",
            start=1.5,
            end=3.0,
            has_audio=True,
            video_format="h264-mp4",
            crf=17,
        )
        assert cmd[cmd.index("-ss") : cmd.index("-ss") + 2] == ["-ss", "1.500"]
        assert cmd[cmd.index("-to") : cmd.index("-to") + 2] == ["-to", "3.000"]
        assert "-c" not in cmd or "copy" not in cmd  # frame-accurate re-encode
        assert "libx264" in cmd


class TestTextFilter:
    def test_uses_textfile_never_inline_text(self) -> None:
        f = _build_text_filter(
            textfile="/tmp/t.txt",
            font_file="/f.ttf",
            font_size=32,
            font_color="white",
            x="20",
            y="20",
            box=True,
            box_color="black",
            box_opacity=0.5,
            start=0.0,
            end=0.0,
        )
        assert "textfile=/tmp/t.txt" in f
        assert ":text=" not in f  # only textfile=, never an inline text= option
        # Literal rendering: without expansion=none a stray '%' in the text
        # makes drawtext silently render nothing (ffmpeg still exits 0).
        assert "expansion=none" in f
        assert "fontfile=/f.ttf" in f
        assert "box=1" in f
        assert "boxcolor=black@0.50" in f
        assert "enable=" not in f

    def test_timed_overlay_uses_between(self) -> None:
        f = _build_text_filter(
            textfile="/tmp/t.txt",
            font_file=None,
            font_size=32,
            font_color="white",
            x="0",
            y="0",
            box=False,
            box_color="black",
            box_opacity=0.5,
            start=1.0,
            end=2.5,
        )
        assert "enable=between(t\\,1.000\\,2.500)" in f
        assert "box=1" not in f

    def test_start_only_uses_gte(self) -> None:
        f = _build_text_filter(
            textfile="/tmp/t.txt",
            font_file=None,
            font_size=32,
            font_color="white",
            x="0",
            y="0",
            box=False,
            box_color="black",
            box_opacity=0.5,
            start=2.0,
            end=0.0,
        )
        assert "enable=gte(t\\,2.000)" in f

    def test_text_command_copies_audio(self) -> None:
        cmd = _build_text_command(
            "/in.mp4",
            "/out.mp4",
            text_filter="drawtext=textfile=/t.txt",
            has_audio=True,
            video_format="h264-mp4",
            crf=17,
        )
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == "copy"
        assert "aac" not in cmd


class TestSpeed:
    def test_atempo_chain_within_range_single_stage(self) -> None:
        assert _atempo_chain(1.5) == "atempo=1.5"

    def test_atempo_chain_fast(self) -> None:
        assert _atempo_chain(4.0) == "atempo=2,atempo=2"

    def test_atempo_chain_slow(self) -> None:
        assert _atempo_chain(0.25) == "atempo=0.5,atempo=0.5"

    def test_speed_command_maps_video_and_audio(self) -> None:
        cmd = _build_speed_command(
            "/in.mp4",
            "/out.mp4",
            factor=2.0,
            has_audio=True,
            video_format="h264-mp4",
            crf=17,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "setpts=PTS/2" in fc
        assert "atempo=2" in fc
        assert "[vout]" in cmd and "[aout]" in cmd

    def test_speed_command_video_only(self) -> None:
        cmd = _build_speed_command(
            "/in.mp4",
            "/out.mp4",
            factor=0.5,
            has_audio=False,
            video_format="h264-mp4",
            crf=17,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "atempo" not in fc
        assert "[aout]" not in cmd


class TestResizeCommand:
    def test_explicit_dims(self) -> None:
        cmd = _build_resize_command(
            "/in.mp4",
            "/out.mp4",
            width=1280,
            height=720,
            has_audio=False,
            video_format="h264-mp4",
            crf=17,
        )
        assert "scale=1280:720" in cmd

    def test_zero_dim_derives_from_aspect(self) -> None:
        cmd = _build_resize_command(
            "/in.mp4",
            "/out.mp4",
            width=1280,
            height=0,
            has_audio=False,
            video_format="h264-mp4",
            crf=17,
        )
        assert "scale=1280:-2" in cmd


class TestUniqueDest:
    def test_collision_appends_suffix(self, tmp_path: Path) -> None:
        target = tmp_path / "out.mp4"
        target.write_bytes(b"x")
        dest = _unique_dest(str(target))
        assert dest == tmp_path / "out_1.mp4"

    def test_default_dir_timestamped(self, tmp_path: Path) -> None:
        with patch("missy.tools.builtin.video_edit._DEFAULT_OUTPUT_DIR", str(tmp_path)):
            dest = _unique_dest("")
        assert dest.parent == tmp_path
        assert dest.name.startswith("edit_")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class TestProbe:
    def test_parses_streams(self) -> None:
        with patch("missy.tools.builtin.video_edit.subprocess.run") as run:
            run.return_value = _FakeProc(stdout=_probe_json(width=832, height=480))
            info = _probe("/v.mp4")
        assert info["width"] == 832
        assert info["fps"] == 24.0
        assert info["has_audio"] is True
        assert info["duration"] == 4.0

    def test_no_video_stream_rejected(self) -> None:
        payload = json.dumps({"streams": [{"codec_type": "audio"}], "format": {}})
        with patch("missy.tools.builtin.video_edit.subprocess.run") as run:
            run.return_value = _FakeProc(stdout=payload)
            with pytest.raises(ValueError, match="no video stream"):
                _probe("/a.mp3")

    def test_probe_failure_actionable(self) -> None:
        with patch("missy.tools.builtin.video_edit.subprocess.run") as run:
            run.return_value = _FakeProc(returncode=1, stderr="Invalid data")
            with pytest.raises(ValueError, match="not a readable video file"):
                _probe("/garbage.bin")


# ---------------------------------------------------------------------------
# execute() validation
# ---------------------------------------------------------------------------


class TestExecuteValidation:
    def test_unknown_operation(self) -> None:
        result = VideoEditTool().execute(operation="explode")
        assert not result.success
        assert "Unknown operation" in (result.error or "")

    def test_unknown_video_format(self) -> None:
        result = VideoEditTool().execute(operation="trim", input="/x.mp4", video_format="avi")
        assert not result.success
        assert "video_format" in (result.error or "")

    def test_concat_needs_two_inputs(self) -> None:
        result = VideoEditTool().execute(operation="concat", inputs=["/only.mp4"])
        assert not result.success
        assert "at least 2" in (result.error or "")

    def test_single_input_ops_need_input(self) -> None:
        for op in ("trim", "text", "speed", "resize"):
            result = VideoEditTool().execute(operation=op)
            assert not result.success
            assert "'input'" in (result.error or "")

    def test_missing_file_rejected(self) -> None:
        result = VideoEditTool().execute(operation="trim", input="/nope/missing.mp4", end=2)
        assert not result.success
        assert "not found" in (result.error or "")

    def _tool_with_probe(self, tmp_path: Path, **probe_kw: Any):
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        return src

    def test_trim_needs_valid_range(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
            result = VideoEditTool().execute(
                operation="trim",
                input=str(src),
                start=3.0,
                end=1.0,
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "end > start" in (result.error or "")

    def test_text_needs_text(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
            result = VideoEditTool().execute(
                operation="text",
                input=str(src),
                text="   ",
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "needs 'text'" in (result.error or "")

    def test_text_unknown_position(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
            result = VideoEditTool().execute(
                operation="text",
                input=str(src),
                text="hi",
                position="under",
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "position" in (result.error or "")

    def test_text_rejects_filter_injection_in_xy_and_colors(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        attempts = [
            {"x": "0,movie=/etc/passwd", "y": "0"},
            {"x": "0", "y": "0:fontfile=/etc/shadow"},
            {"font_color": "white:box=0"},
            {"box_color": "black,negate"},
        ]
        for extra in attempts:
            with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
                result = VideoEditTool().execute(
                    operation="text",
                    input=str(src),
                    text="hi",
                    save_path=str(tmp_path / "o.mp4"),
                    **extra,
                )
            assert not result.success, extra
            assert "not allowed" in (result.error or "")

    def test_speed_factor_one_rejected(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=1.0,
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success

    def test_resize_needs_a_dimension(self, tmp_path: Path) -> None:
        src = self._tool_with_probe(tmp_path)
        with patch("missy.tools.builtin.video_edit._probe", return_value=_info(str(src))):
            result = VideoEditTool().execute(
                operation="resize",
                input=str(src),
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success

    def test_unknown_transition(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        with patch(
            "missy.tools.builtin.video_edit._probe", side_effect=[_info(str(a)), _info(str(b))]
        ):
            result = VideoEditTool().execute(
                operation="concat",
                inputs=[str(a), str(b)],
                transition="wipe",
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "transition" in (result.error or "")

    def test_crossfade_rejects_too_short_clip(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        infos = [_info(str(a), duration=4.0), _info(str(b), duration=0.3)]
        with patch("missy.tools.builtin.video_edit._probe", side_effect=infos):
            result = VideoEditTool().execute(
                operation="concat",
                inputs=[str(a), str(b)],
                transition="crossfade",
                transition_duration=0.5,
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "too short" in (result.error or "")


# ---------------------------------------------------------------------------
# execute() happy paths (ffmpeg mocked)
# ---------------------------------------------------------------------------


def _fake_run_factory(dest_holder: dict[str, Any]):
    """Return a subprocess.run replacement that fabricates ffmpeg output and
    answers ffprobe with a canned JSON payload."""

    def fake_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
        if cmd[0] == "ffprobe":
            return _FakeProc(stdout=_probe_json(**dest_holder.get("probe_kw", {})))
        # ffmpeg: write the destination file (last argv entry).
        dest_holder["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"\x00" * 128)
        return _FakeProc()

    return fake_run


class TestExecuteHappyPaths:
    def test_trim_flow(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="trim",
                input=str(src),
                start=1.0,
                duration=2.0,
                save_path=str(tmp_path / "out.mp4"),
            )
        assert result.success, result.error
        assert result.output["operation"] == "trim"
        assert result.output["path"] == str(tmp_path / "out.mp4")
        assert result.output["duration_seconds"] == 4.0  # probed, not requested
        assert "-to" in holder["cmd"]
        assert holder["cmd"][holder["cmd"].index("-to") + 1] == "3.000"

    def test_concat_flow_output_probed(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="concat",
                inputs=[str(a), str(b)],
                save_path=str(tmp_path / "joined.mp4"),
            )
        assert result.success, result.error
        assert result.output["inputs"] == 2
        assert result.output["encoder"] == "libx264"
        assert "-filter_complex" in holder["cmd"]

    def test_text_flow_writes_and_cleans_textfile(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="text",
                input=str(src),
                text="Hello: it's 100% 'safe', even with , and ;",
                save_path=str(tmp_path / "out.mp4"),
            )
        assert result.success, result.error
        vf = holder["cmd"][holder["cmd"].index("-vf") + 1]
        assert "textfile=" in vf
        assert "Hello" not in vf  # content never enters the filter graph
        textfile = vf.split("textfile=")[1].split(":")[0]
        assert not Path(textfile).exists()  # cleaned up

    def test_speed_flow(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=2.0,
                save_path=str(tmp_path / "out.mp4"),
            )
        assert result.success, result.error
        assert "setpts=PTS/2" in holder["cmd"][holder["cmd"].index("-filter_complex") + 1]

    def test_resize_flow_nvenc(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="resize",
                input=str(src),
                width=1280,
                height=720,
                video_format="nvenc_h264-mp4",
                save_path=str(tmp_path / "out.mp4"),
            )
        assert result.success, result.error
        assert result.output["encoder"] == "h264_nvenc"
        assert "h264_nvenc" in holder["cmd"]

    def test_save_collision_appends_suffix(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        existing = tmp_path / "out.mp4"
        existing.write_bytes(b"keep me")
        holder: dict[str, Any] = {}
        with patch(
            "missy.tools.builtin.video_edit.subprocess.run",
            side_effect=_fake_run_factory(holder),
        ):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=2.0,
                save_path=str(existing),
            )
        assert result.success, result.error
        assert result.output["path"] == str(tmp_path / "out_1.mp4")
        assert existing.read_bytes() == b"keep me"


# ---------------------------------------------------------------------------
# execute() errors
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    def test_ffmpeg_failure_surfaces_stderr_tail(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")

        def fake_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
            if cmd[0] == "ffprobe":
                return _FakeProc(stdout=_probe_json())
            return _FakeProc(returncode=1, stderr="line1\nEncoder blew up")

        with patch("missy.tools.builtin.video_edit.subprocess.run", side_effect=fake_run):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=2.0,
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "Encoder blew up" in (result.error or "")

    def test_ffmpeg_timeout_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")

        def fake_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
            if cmd[0] == "ffprobe":
                return _FakeProc(stdout=_probe_json())
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

        with patch("missy.tools.builtin.video_edit.subprocess.run", side_effect=fake_run):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=2.0,
                save_path=str(tmp_path / "o.mp4"),
                timeout=7,
            )
        assert not result.success
        assert "timed out" in (result.error or "")

    def test_empty_output_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")

        def fake_run(cmd: list[str], **kwargs: Any) -> _FakeProc:
            if cmd[0] == "ffprobe":
                return _FakeProc(stdout=_probe_json())
            return _FakeProc()  # "succeeds" but writes nothing

        with patch("missy.tools.builtin.video_edit.subprocess.run", side_effect=fake_run):
            result = VideoEditTool().execute(
                operation="speed",
                input=str(src),
                factor=2.0,
                save_path=str(tmp_path / "o.mp4"),
            )
        assert not result.success
        assert "no output file" in (result.error or "")


# ---------------------------------------------------------------------------
# Policy resolvers + schema
# ---------------------------------------------------------------------------


class TestResolversAndSchema:
    def test_resolve_shell_command_names_both_binaries(self) -> None:
        assert VideoEditTool().resolve_shell_command({}) == "ffmpeg && ffprobe"

    def test_resolve_filesystem_targets_concat(self) -> None:
        reads, writes = VideoEditTool().resolve_filesystem_targets(
            {"inputs": ["/a.mp4", "/b.mp4"], "save_path": "/out/x.mp4"}
        )
        assert reads == ["/a.mp4", "/b.mp4"]
        assert writes == ["/out/x.mp4"]

    def test_resolve_filesystem_targets_single_input_defaults(self) -> None:
        reads, writes = VideoEditTool().resolve_filesystem_targets(
            {"input": "/v.mp4", "font_file": "/f.ttf"}
        )
        assert reads == ["/v.mp4", "/f.ttf"]
        assert len(writes) == 1 and writes[0].endswith("videos")

    def test_permissions(self) -> None:
        perms = VideoEditTool.permissions
        assert perms.shell and perms.filesystem_read and perms.filesystem_write
        assert not perms.network

    def test_schema_operations_enumerated(self) -> None:
        schema = VideoEditTool().get_schema()
        assert schema["name"] == "video_edit"
        props = schema["parameters"]["properties"]
        assert set(props["operation"]["enum"]) == {"concat", "trim", "text", "speed", "resize"}
        assert "operation" in schema["parameters"]["required"]

    def test_registered_as_builtin(self) -> None:
        from missy.tools.builtin import _ALL_TOOL_CLASSES
        from missy.tools.builtin import VideoEditTool as Exported

        assert Exported in _ALL_TOOL_CLASSES
