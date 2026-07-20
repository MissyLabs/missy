"""Built-in tool: edit videos with ffmpeg (splice, trim, text overlay, ...).

Complements :mod:`missy.tools.builtin.video_generate`: that tool *creates*
clips via ComfyUI; this one edits existing files using the external
``ffmpeg``/``ffprobe`` binaries (design record: ``video.md`` Part II at the
repo root). Deterministic cut/join/overlay work needs no diffusion model,
so it runs as a direct subprocess instead of a ComfyUI graph.

Operations (selected via ``operation``):

* ``"concat"`` -- splice two or more videos into one. Inputs are
  normalized first (letterbox scale/pad to a common canvas, common fps,
  silent audio synthesized for inputs that lack a track when any input
  has one) so clips of mixed resolution/frame-rate/audio-presence join
  cleanly. Optional pairwise crossfade (``transition="crossfade"``).
* ``"trim"`` -- frame-accurate cut (``start`` + ``end`` or ``duration``).
* ``"text"`` -- overlay text via drawtext. The text is passed through a
  temp *textfile*, never interpolated into the filter graph, so arbitrary
  user text cannot break or inject into it. Position presets, timed
  captions (``start``/``end``), styled box background.
* ``"speed"`` -- change playback speed 0.25x-4x (audio pitch preserved
  via chained ``atempo``).
* ``"resize"`` -- rescale (0 for one dimension derives it from aspect).
* ``"extract_frame"`` -- export one frame as a PNG/JPEG still
  (``at=-1`` = last frame). Enables reviewing a clip with the vision
  tools and last-frame -> ``image_path`` scene chaining (see
  ``video_storyboard``'s continuity mode).
* ``"audio"`` -- lay an audio file onto an existing video
  (``audio_mode="replace"``/``"mix"``, optional ``loop``). The video
  stream is stream-copied, never re-encoded; output duration always
  equals the video's.

All operations re-encode through the same quality surface as
``video_generate``: ``video_format`` = ``h264-mp4`` | ``h265-mp4`` |
``nvenc_h264-mp4`` (NVENC GPU encode), ``crf`` (default 17), yuv420p,
``+faststart``, AAC 192k audio. Output lands in ``~/.missy/videos/``
(never overwriting -- numeric suffix on collision) unless ``save_path``
is given, and the result is ffprobe'd after encoding so the returned
dimensions/duration describe the actual file.

Complex edits compose by calling the tool repeatedly (trim -> text ->
concat), exactly like ``video_generate``'s "iteration is just calling
the tool again" contract.

Example::

    from missy.tools.builtin.video_edit import VideoEditTool

    tool = VideoEditTool()
    result = tool.execute(
        operation="concat",
        inputs=["/home/u/.missy/videos/a.mp4", "/home/u/.missy/videos/b.mp4"],
        transition="crossfade",
    )
    assert result.success
    print(result.output["path"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"
_DEFAULT_OUTPUT_DIR = str(Path.home() / ".missy" / "videos")
_DEFAULT_TIMEOUT_SECONDS = 600

_VALID_OPERATIONS = frozenset(
    {"concat", "trim", "text", "speed", "resize", "extract_frame", "audio"}
)
_VALID_TRANSITIONS = frozenset({"none", "crossfade"})
_VALID_AUDIO_MODES = frozenset({"replace", "mix"})

# extract_frame output containers ffmpeg's image2 muxer handles by extension.
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

# Tool-facing format name -> (video encoder argv, container ext). Mirrors
# video_generate's _VIDEO_FORMATS surface.
_VIDEO_FORMATS = frozenset({"h264-mp4", "h265-mp4", "nvenc_h264-mp4"})

# drawtext position presets -> (x, y) filter expressions.
_TEXT_POSITIONS = {
    "top-left": ("20", "20"),
    "top": ("(w-text_w)/2", "20"),
    "top-right": ("w-text_w-20", "20"),
    "center-left": ("20", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "center-right": ("w-text_w-20", "(h-text_h)/2"),
    "bottom-left": ("20", "h-text_h-20"),
    "bottom": ("(w-text_w)/2", "h-text_h-20"),
    "bottom-right": ("w-text_w-20", "h-text_h-20"),
}

# Candidate default fonts, first match wins (font_file param overrides).
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

#: Environment variables safe to pass to ffmpeg/ffprobe subprocesses.
#: Prevents API key leakage, same precedent as tts_speak's _SAFE_TTS_ENV_VARS.
_SAFE_FFMPEG_ENV_VARS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        # NVENC needs the CUDA driver to be discoverable.
        "CUDA_VISIBLE_DEVICES",
    }
)


def _ffmpeg_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _SAFE_FFMPEG_ENV_VARS}


# The only user-supplied strings that enter the filter graph directly are the
# x/y position expressions and color names -- everything else is either a
# number, an enum, or (for the text itself) routed through a textfile. These
# whitelists keep a crafted value from smuggling extra filter options or a
# whole new filter (e.g. a `movie=` source reading files the filesystem
# policy never saw) into the graph: no `:` (option separator), `,`/`;`
# (filter separators), `=`, quotes, or backslashes.
_EXPR_SAFE = re.compile(r"^[0-9A-Za-z_+\-*/(). ]+$")
_COLOR_SAFE = re.compile(r"^[0-9A-Za-z#@.]+$")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _find_font() -> str | None:
    """Locate a default TTF font for drawtext."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _auto_font_size(text: str, font_file: str | None, *, width: int, height: int) -> int:
    """Choose a readable default size that keeps every line inside the frame.

    The old ``height / 12`` default worked for short titles but let longer
    captions extend beyond both sides of the image. Pillow and ffmpeg both
    use FreeType, so when Pillow is available its measurement is an accurate
    preflight for drawtext. Pillow is optional; the fallback deliberately
    uses a conservative glyph-width estimate so video editing still works in
    a minimal installation.

    An explicitly requested ``font_size`` is never changed -- this only
    defines the behavior of the documented automatic (zero) value.
    """
    desired = max(16, int(height) // 12)
    # Position presets leave 20 px at each edge; a boxed caption adds a
    # 10 px border. Keep a little extra breathing room as well.
    available = max(1, int(width) - 48)
    lines = text.splitlines() or [text]

    if font_file:
        try:
            from PIL import ImageFont

            def widest_at(size: int) -> float:
                font = ImageFont.truetype(font_file, size=size)
                return max(float(font.getlength(line)) for line in lines)

            widest = widest_at(desired)
            if widest <= available:
                return desired
            fitted = max(8, int(desired * available / widest))
            # Rounding and hinting can move the exact FreeType measurement by
            # a pixel. Walk down until the measured text genuinely fits.
            while fitted > 8 and widest_at(fitted) > available:
                fitted -= 1
            return fitted
        except (ImportError, OSError, ValueError):
            logger.debug("Unable to measure drawtext font; using conservative fallback")

    longest = max((len(line) for line in lines), default=0)
    if longest:
        # A typical sans-serif glyph averages ~0.6 em. 0.75 is intentionally
        # conservative for the no-Pillow path while avoiding unreadably tiny
        # text for ordinary prose.
        desired = min(desired, max(8, int(available / (longest * 0.75))))
    return desired


def _unique_dest(save_path: str, ext: str = ".mp4", prefix: str = "edit_") -> Path:
    """Resolve the output path; never overwrite (numeric suffix on collision).

    Same contract as video_generate's _retrieve_video destination logic.
    """
    if save_path:
        dest = Path(save_path).expanduser()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path(_DEFAULT_OUTPUT_DIR) / f"{prefix}{ts}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = dest
    counter = 1
    while dest.exists():
        dest = base.with_name(f"{base.stem}_{counter}{base.suffix}")
        counter += 1
    return dest


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------


def _probe(path: str, *, timeout: int = 30, expect: str = "video") -> dict[str, Any]:
    """Probe a media file; returns a compact info dict.

    ``expect="video"`` (default) requires a video stream, matching every
    Part II operation. ``expect="audio"`` requires an audio stream instead
    (video optional), for the ``audio`` mux operation's soundtrack input.

    Raises ``ValueError`` with an actionable message when the file is not
    probeable media.
    """
    cmd = [
        _FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=_ffmpeg_env(), timeout=timeout
        )
    except FileNotFoundError:
        raise ValueError("ffprobe not found. Install ffmpeg: sudo apt install ffmpeg") from None
    except subprocess.TimeoutExpired:
        raise ValueError(f"ffprobe timed out on {path}") from None
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        raise ValueError(
            f"not a readable {expect} file: {path} ({err[-1] if err else 'probe failed'})"
        )

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if expect == "audio":
        if audio is None:
            raise ValueError(f"no audio stream in {path}")
    elif video is None:
        raise ValueError(f"no video stream in {path}")

    # avg_frame_rate is "num/den"; guard against "0/0" on odd containers.
    fps = 0.0
    raw_rate = (video or {}).get("avg_frame_rate") or (video or {}).get("r_frame_rate") or "0/1"
    try:
        num, _, den = raw_rate.partition("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration = 0.0
    duration_sources = (
        (video or {}).get("duration"),
        (audio or {}).get("duration"),
        data.get("format", {}).get("duration"),
    )
    for source in duration_sources:
        try:
            duration = float(source)
            break
        except (TypeError, ValueError):
            continue

    return {
        "path": path,
        "width": int((video or {}).get("width", 0)),
        "height": int((video or {}).get("height", 0)),
        "fps": round(fps, 3),
        "duration": duration,
        "has_audio": audio is not None,
        "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Command builders (pure functions -- unit-tested without ffmpeg).
# ---------------------------------------------------------------------------


def _encode_args(video_format: str, crf: int, *, has_audio: bool) -> list[str]:
    """Common output-encoding argv tail (before the output path)."""
    crf = int(_clamp(crf, 0, 51))
    if video_format == "h265-mp4":
        args = ["-c:v", "libx265", "-crf", str(crf), "-preset", "medium"]
    elif video_format == "nvenc_h264-mp4":
        # NVENC has no -crf; -cq is its closest constant-quality knob.
        args = ["-c:v", "h264_nvenc", "-cq", str(crf), "-preset", "p5"]
    else:  # h264-mp4
        args = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"]
    args += ["-pix_fmt", "yuv420p"]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", "192k"]
    args += ["-movflags", "+faststart"]
    return args


def _build_concat_filter(
    infos: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
    transition: str,
    transition_duration: float,
) -> tuple[str, str, str | None]:
    """Build the concat filter_complex.

    Returns ``(filter_complex, video_label, audio_label_or_None)``.
    Audio is included when *any* input has an audio stream; inputs
    without one get a synthesized silent track so concat's stream pairs
    always match up.
    """
    any_audio = any(i["has_audio"] for i in infos)
    n = len(infos)
    parts: list[str] = []

    # Normalize every input to the common canvas/rate.
    for idx, info in enumerate(infos):
        parts.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"format=yuv420p[v{idx}]"
        )
        if any_audio:
            if info["has_audio"]:
                parts.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo[a{idx}]")
            else:
                dur = max(info["duration"], 0.1)
                parts.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                    f"atrim=duration={dur:.3f}[a{idx}]"
                )

    if transition == "crossfade" and n >= 2:
        # Chain pairwise xfade/acrossfade; each fade consumes
        # transition_duration seconds of overlap, so offsets accumulate.
        td = transition_duration
        vprev, aprev = "v0", "a0"
        offset = 0.0
        for idx in range(1, n):
            offset += max(infos[idx - 1]["duration"] - td, 0.1)
            vout = f"vx{idx}" if idx < n - 1 else "vout"
            parts.append(
                f"[{vprev}][v{idx}]xfade=transition=fade:duration={td:.3f}:"
                f"offset={offset:.3f}[{vout}]"
            )
            vprev = vout
            if any_audio:
                aout = f"ax{idx}" if idx < n - 1 else "aout"
                parts.append(f"[{aprev}][a{idx}]acrossfade=d={td:.3f}[{aout}]")
                aprev = aout
        return (
            ";".join(parts),
            "[vout]",
            "[aout]" if any_audio else None,
        )

    # Plain concat.
    inter = "".join(f"[v{idx}][a{idx}]" if any_audio else f"[v{idx}]" for idx in range(n))
    parts.append(
        f"{inter}concat=n={n}:v=1:a={1 if any_audio else 0}"
        + ("[vout][aout]" if any_audio else "[vout]")
    )
    return (";".join(parts), "[vout]", "[aout]" if any_audio else None)


def _build_concat_command(
    infos: list[dict[str, Any]],
    dest: str,
    *,
    transition: str,
    transition_duration: float,
    video_format: str,
    crf: int,
) -> list[str]:
    """Full ffmpeg argv for the concat operation."""
    # Target canvas: first input's resolution, fastest frame rate of the set.
    width = infos[0]["width"]
    height = infos[0]["height"]
    fps = max((i["fps"] for i in infos if i["fps"] > 0), default=24.0)
    filter_complex, vlabel, alabel = _build_concat_filter(
        infos,
        width=width,
        height=height,
        fps=fps,
        transition=transition,
        transition_duration=transition_duration,
    )
    cmd = [_FFMPEG, "-nostdin", "-hide_banner", "-y"]
    for info in infos:
        cmd += ["-i", info["path"]]
    cmd += ["-filter_complex", filter_complex, "-map", vlabel]
    if alabel:
        cmd += ["-map", alabel]
    cmd += _encode_args(video_format, crf, has_audio=alabel is not None)
    cmd.append(dest)
    return cmd


def _build_trim_command(
    input_path: str,
    dest: str,
    *,
    start: float,
    end: float,
    has_audio: bool,
    video_format: str,
    crf: int,
) -> list[str]:
    """Full ffmpeg argv for a frame-accurate trim (re-encode, not -c copy)."""
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        input_path,
    ]
    cmd += _encode_args(video_format, crf, has_audio=has_audio)
    cmd.append(dest)
    return cmd


def _build_text_filter(
    *,
    textfile: str,
    font_file: str | None,
    font_size: int,
    font_color: str,
    x: str,
    y: str,
    box: bool,
    box_color: str,
    box_opacity: float,
    start: float,
    end: float,
) -> str:
    """drawtext filter string. The text itself lives in ``textfile`` so no
    user-controlled characters ever enter the filter graph.

    ``expansion=none`` renders the text literally: without it, drawtext's
    default expansion mode treats ``%`` as the start of a ``%{...}``
    function and a stray one (e.g. "100% done") makes it render *nothing*
    while ffmpeg still exits 0.
    """
    opts = [f"textfile={textfile}", "expansion=none"]
    if font_file:
        opts.append(f"fontfile={font_file}")
    opts += [
        f"fontsize={font_size}",
        f"fontcolor={font_color}",
        f"x={x}",
        f"y={y}",
    ]
    if box:
        opts.append("box=1")
        opts.append(f"boxcolor={box_color}@{box_opacity:.2f}")
        opts.append("boxborderw=10")
    if end > 0:
        opts.append(f"enable=between(t\\,{start:.3f}\\,{end:.3f})")
    elif start > 0:
        opts.append(f"enable=gte(t\\,{start:.3f})")
    return "drawtext=" + ":".join(opts)


def _build_text_command(
    input_path: str,
    dest: str,
    *,
    text_filter: str,
    has_audio: bool,
    video_format: str,
    crf: int,
) -> list[str]:
    """Full ffmpeg argv for the text-overlay operation."""
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        input_path,
        "-vf",
        text_filter,
    ]
    if has_audio:
        cmd += ["-c:a", "copy"]  # overlay only touches video; keep audio bit-exact
    cmd += _encode_args(video_format, crf, has_audio=False)  # audio handled above
    cmd.append(dest)
    return cmd


def _atempo_chain(factor: float) -> str:
    """Chain atempo stages so each stays inside its stable 0.5-2.0 range."""
    stages: list[float] = []
    remaining = factor
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.6g}" for s in stages)


def _build_speed_command(
    input_path: str,
    dest: str,
    *,
    factor: float,
    has_audio: bool,
    video_format: str,
    crf: int,
) -> list[str]:
    """Full ffmpeg argv for the speed operation."""
    cmd = [_FFMPEG, "-nostdin", "-hide_banner", "-y", "-i", input_path]
    vfilter = f"[0:v]setpts=PTS/{factor:.6g}[vout]"
    if has_audio:
        filter_complex = f"{vfilter};[0:a]{_atempo_chain(factor)}[aout]"
        cmd += ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]"]
    else:
        cmd += ["-filter_complex", vfilter, "-map", "[vout]"]
    cmd += _encode_args(video_format, crf, has_audio=has_audio)
    cmd.append(dest)
    return cmd


def _build_resize_command(
    input_path: str,
    dest: str,
    *,
    width: int,
    height: int,
    has_audio: bool,
    video_format: str,
    crf: int,
) -> list[str]:
    """Full ffmpeg argv for the resize operation. Zero width/height derives
    that dimension from the input aspect ratio (even-snapped)."""
    w = width if width > 0 else -2
    h = height if height > 0 else -2
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale={w}:{h}",
    ]
    if has_audio:
        cmd += ["-c:a", "copy"]
    cmd += _encode_args(video_format, crf, has_audio=False)
    cmd.append(dest)
    return cmd


def _build_extract_frame_command(input_path: str, dest: str, *, at: float) -> list[str]:
    """Full ffmpeg argv to export the frame at ``at`` seconds as an image.

    Input-side ``-ss`` seeks to the nearest keyframe then decodes forward,
    so the first delivered frame is the one at/after ``at`` -- exact for
    this purpose and fast even on long inputs. ``-update 1`` writes a
    single image instead of expecting a sequence pattern in the filename.
    """
    cmd = [
        _FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-y",
        "-ss",
        f"{at:.3f}",
        "-i",
        input_path,
        "-frames:v",
        "1",
        "-update",
        "1",
    ]
    if Path(dest).suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", "2"]
    cmd.append(dest)
    return cmd


def _build_audio_mux_command(
    input_path: str,
    audio_path: str,
    dest: str,
    *,
    mode: str,
    loop: bool,
    video_has_audio: bool,
) -> list[str]:
    """Full ffmpeg argv to lay an audio file onto a video.

    The video stream is stream-copied (``-c:v copy``): remuxing an audio
    track is not an edit of the video stream, so a lossless copy is
    strictly better than a re-encode here. The output duration always
    equals the video's: the new track is ``apad``-ed (or looped with
    ``-stream_loop -1``) and ``-shortest`` cuts it at the video's end.

    ``mode="mix"`` blends the new track with the video's existing audio
    (``amix`` with ``normalize=0`` so levels don't jump, ``duration=first``
    pinning the mix to the original track's length); with no existing
    audio it degrades to a plain replace.
    """
    cmd = [_FFMPEG, "-nostdin", "-hide_banner", "-y", "-i", input_path]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", audio_path]
    if mode == "mix" and video_has_audio:
        cmd += [
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
        ]
    else:
        # apad + -shortest: pad a short track with silence out to the
        # video's end, and cut a long (or looped) one exactly there.
        cmd += ["-map", "0:v", "-map", "1:a", "-af", "apad", "-shortest"]
    cmd += [
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        dest,
    ]
    return cmd


def _run_ffmpeg(cmd: list[str], *, timeout: int) -> str | None:
    """Run an ffmpeg argv. Returns None on success, error string on failure."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=_ffmpeg_env(), timeout=timeout
        )
    except FileNotFoundError:
        return "ffmpeg not found. Install ffmpeg: sudo apt install ffmpeg"
    except subprocess.TimeoutExpired:
        return f"ffmpeg timed out after {timeout}s"
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        return f"ffmpeg failed (exit {proc.returncode}): {tail}"
    return None


class VideoEditTool(BaseTool):
    """Edit videos with ffmpeg: splice, trim, overlay text, speed, resize.

    Attributes:
        name: ``"video_edit"``
        description: One-line description for function-calling schemas.
        permissions: ``shell=True``, ``filesystem_read=True``,
            ``filesystem_write=True``.
    """

    name = "video_edit"
    description = (
        "Edit existing video files with ffmpeg. operation='concat' splices "
        "2+ videos into one (inputs list; mixed resolutions/frame-rates/"
        "audio are normalized automatically; transition='crossfade' for "
        "fades). operation='trim' cuts a segment (start + end/duration). "
        "operation='text' overlays styled text (captions, titles, "
        "watermarks) with position presets and optional timing. "
        "operation='speed' changes playback speed 0.25x-4x (audio pitch "
        "preserved). operation='resize' rescales. operation='extract_frame' "
        "exports one frame as a PNG/JPEG image (at=-1 for the last frame -- "
        "useful for reviewing a clip or seeding the next scene of an "
        "image-to-video generation). operation='audio' lays an audio file "
        "onto a video (audio_mode 'replace' or 'mix', optional loop; video "
        "stream copied losslessly). Returns the local path to the produced "
        "file plus its actual probed dimensions/duration. Chain complex "
        "edits by calling this tool repeatedly (e.g. trim each clip, then "
        "concat, then add a title)."
    )
    permissions = ToolPermissions(shell=True, filesystem_read=True, filesystem_write=True)

    parameters: dict[str, Any] = {
        "operation": {
            "type": "string",
            "enum": sorted(_VALID_OPERATIONS),
            "description": (
                "The edit to perform: concat | trim | text | speed | resize | "
                "extract_frame | audio."
            ),
            "required": True,
        },
        "inputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "concat only: 2+ local video paths, spliced in order.",
        },
        "input": {
            "type": "string",
            "description": "All operations except concat: the local video path to edit.",
        },
        "at": {
            "type": "number",
            "description": (
                "extract_frame: timestamp in seconds to grab the frame at. "
                "-1 (default) = the last frame of the video."
            ),
        },
        "audio_file": {
            "type": "string",
            "description": "audio operation: local audio file to lay onto the video.",
        },
        "audio_mode": {
            "type": "string",
            "enum": sorted(_VALID_AUDIO_MODES),
            "description": (
                "audio operation: 'replace' (default) swaps the video's audio for "
                "the new track; 'mix' blends the two."
            ),
        },
        "loop": {
            "type": "boolean",
            "description": (
                "audio operation: loop a short track until the video ends "
                "(default false: a short track is padded with silence instead)."
            ),
        },
        "transition": {
            "type": "string",
            "enum": sorted(_VALID_TRANSITIONS),
            "description": "concat only: 'none' (default) or 'crossfade'.",
        },
        "transition_duration": {
            "type": "number",
            "description": "concat crossfade length in seconds (default 0.5).",
        },
        "start": {
            "type": "number",
            "description": (
                "trim: segment start seconds (default 0). "
                "text: overlay start seconds (default 0 = from the beginning)."
            ),
        },
        "end": {
            "type": "number",
            "description": (
                "trim: segment end seconds (or use duration). "
                "text: overlay end seconds (default 0 = until the end)."
            ),
        },
        "duration": {
            "type": "number",
            "description": "trim: segment length in seconds (alternative to end).",
        },
        "text": {
            "type": "string",
            "description": "text operation: the text to overlay (any characters are safe).",
        },
        "position": {
            "type": "string",
            "enum": sorted(_TEXT_POSITIONS),
            "description": "text: placement preset (default 'bottom'). Or pass x/y instead.",
        },
        "x": {
            "type": "string",
            "description": "text: explicit drawtext x expression (overrides position).",
        },
        "y": {
            "type": "string",
            "description": "text: explicit drawtext y expression (overrides position).",
        },
        "font_size": {
            "type": "integer",
            "description": (
                "text: font size in px (default 0 = auto based on video height, "
                "reduced as needed so every line fits inside the frame)."
            ),
        },
        "font_color": {
            "type": "string",
            "description": "text: font color name or 0xRRGGBB (default 'white').",
        },
        "font_file": {
            "type": "string",
            "description": "text: path to a .ttf font (default: system DejaVu Sans Bold).",
        },
        "box": {
            "type": "boolean",
            "description": "text: draw a background box behind the text (default true).",
        },
        "box_color": {
            "type": "string",
            "description": "text: box color (default 'black').",
        },
        "box_opacity": {
            "type": "number",
            "description": "text: box opacity 0.0-1.0 (default 0.5).",
        },
        "factor": {
            "type": "number",
            "description": "speed: playback speed multiplier, 0.25-4.0 (2.0 = twice as fast).",
        },
        "width": {
            "type": "integer",
            "description": "resize: target width (0 = derive from aspect ratio).",
        },
        "height": {
            "type": "integer",
            "description": "resize: target height (0 = derive from aspect ratio).",
        },
        "video_format": {
            "type": "string",
            "enum": sorted(_VIDEO_FORMATS),
            "description": (
                "Output encoding: 'h264-mp4' (default), 'h265-mp4', or "
                "'nvenc_h264-mp4' (NVENC GPU encode)."
            ),
        },
        "crf": {
            "type": "integer",
            "description": "Encoder quality, lower = better (default 17).",
        },
        "save_path": {
            "type": "string",
            "description": (
                "Destination path (default: timestamped file under "
                "~/.missy/videos/). Never overwrites; a numeric suffix is "
                "appended on collision."
            ),
        },
        "timeout": {
            "type": "integer",
            "description": "Max seconds for the ffmpeg run (default 600).",
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        # SR-1.5 convention (see tts_speak.py): no `command` kwarg exists,
        # so declare the real host binaries every invocation runs -- both
        # must be allow-listed in ShellPolicy.
        return f"{_FFMPEG} && {_FFPROBE}"

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Reads the input video(s) and optional font file; writes the
        output to ``save_path`` or the default videos directory."""
        read_paths = [p for p in (kwargs.get("inputs") or []) if p]
        for key in ("input", "font_file", "audio_file"):
            if kwargs.get(key):
                read_paths.append(kwargs[key])
        save_path = kwargs.get("save_path") or ""
        write_paths = [save_path] if save_path else [_DEFAULT_OUTPUT_DIR]
        return (read_paths, write_paths)

    def execute(
        self,
        *,
        operation: str = "",
        inputs: list[str] | None = None,
        input: str = "",  # noqa: A002 - tool-facing parameter name
        transition: str = "none",
        transition_duration: float = 0.5,
        start: float = 0.0,
        end: float = 0.0,
        duration: float = 0.0,
        text: str = "",
        position: str = "bottom",
        x: str = "",
        y: str = "",
        font_size: int = 0,
        font_color: str = "white",
        font_file: str = "",
        box: bool = True,
        box_color: str = "black",
        box_opacity: float = 0.5,
        factor: float = 1.0,
        width: int = 0,
        height: int = 0,
        at: float = -1.0,
        audio_file: str = "",
        audio_mode: str = "replace",
        loop: bool = False,
        video_format: str = "h264-mp4",
        crf: int = 17,
        save_path: str = "",
        timeout: int = 0,
        **_kwargs: Any,
    ) -> ToolResult:
        """Run one edit operation and return the local path to the result.

        Args:
            operation: ``"concat"``, ``"trim"``, ``"text"``, ``"speed"``,
                ``"resize"``, ``"extract_frame"``, or ``"audio"``.
            inputs: concat only -- 2+ local video paths, joined in order.
            input: The video to edit (all operations except concat).
            transition: concat -- ``"none"`` (default) or ``"crossfade"``.
            transition_duration: concat crossfade seconds (default 0.5,
                clamped 0.1-5.0).
            start: trim segment start / text overlay start (seconds).
            end: trim segment end / text overlay end (seconds).
            duration: trim -- segment length (alternative to ``end``).
            text: text operation -- the string to overlay. Passed via a
                temp textfile, so any characters are safe.
            position: text placement preset (default ``"bottom"``).
            x: Explicit drawtext x expression (overrides ``position``).
            y: Explicit drawtext y expression (overrides ``position``).
            font_size: Text size in px (0 = auto: video height / 12,
                reduced as needed so every line fits inside the frame).
            font_color: drawtext color (default ``"white"``).
            font_file: Path to a ``.ttf`` (default: system DejaVu bold).
            box: Draw a background box behind the text (default True).
            box_color: Box color (default ``"black"``).
            box_opacity: Box opacity 0-1 (default 0.5).
            factor: speed multiplier, clamped 0.25-4.0.
            width: resize target width (0 = derive from aspect).
            height: resize target height (0 = derive from aspect).
            at: extract_frame -- timestamp in seconds; ``-1`` (default) =
                the last frame (derived from the probed duration/fps).
            audio_file: audio operation -- the audio file to lay on.
            audio_mode: audio operation -- ``"replace"`` (default) or
                ``"mix"`` (blend with the video's existing track).
            loop: audio operation -- loop a short track to the video's
                length (default False: padded with silence instead).
            video_format: ``"h264-mp4"`` (default), ``"h265-mp4"``, or
                ``"nvenc_h264-mp4"`` (GPU encode). Ignored by
                ``extract_frame`` (image output) and ``audio`` (video
                stream copied bit-exact, never re-encoded).
            crf: Encoder quality, lower = better (default 17).
            save_path: Optional destination (collision-safe). Defaults to
                a timestamped file under ``~/.missy/videos/``.
            timeout: Max seconds for the ffmpeg run (0 = 600).

        Returns:
            :class:`~missy.tools.base.ToolResult` with ``output`` set to a
            dict with ``path``, ``operation``, ``width``, ``height``,
            ``fps``, ``duration_seconds``, ``has_audio``, ``size_bytes``,
            ``inputs``, ``encoder``, and ``elapsed_seconds`` on success.
        """
        started = time.monotonic()
        operation = (operation or "").strip().lower()
        if operation not in _VALID_OPERATIONS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation {operation!r}. Valid: {sorted(_VALID_OPERATIONS)}.",
            )
        if video_format not in _VIDEO_FORMATS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown video_format {video_format!r}. Valid: {sorted(_VIDEO_FORMATS)}.",
            )
        timeout = int(timeout) or _DEFAULT_TIMEOUT_SECONDS

        # --- resolve + probe inputs ------------------------------------
        if operation == "concat":
            paths = [str(Path(p).expanduser()) for p in (inputs or []) if p]
            if len(paths) < 2:
                return ToolResult(
                    success=False,
                    output=None,
                    error="concat needs an 'inputs' list with at least 2 video paths.",
                )
        else:
            if not input:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"{operation} needs an 'input' video path.",
                )
            paths = [str(Path(input).expanduser())]

        missing = [p for p in paths if not Path(p).is_file()]
        if missing:
            return ToolResult(
                success=False, output=None, error=f"Input file not found: {missing[0]}"
            )
        try:
            infos = [_probe(p) for p in paths]
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        if operation == "extract_frame":
            if save_path and Path(save_path).suffix.lower() not in _IMAGE_EXTENSIONS:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        f"extract_frame save_path must end in one of "
                        f"{sorted(_IMAGE_EXTENSIONS)}, got: {save_path}"
                    ),
                )
            dest = _unique_dest(save_path, ext=".png", prefix="frame_")
        else:
            dest = _unique_dest(save_path)
        textfile_path: str | None = None
        try:
            # --- build the command per operation ------------------------
            if operation == "concat":
                transition = (transition or "none").strip().lower()
                if transition not in _VALID_TRANSITIONS:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Unknown transition {transition!r}. Valid: {sorted(_VALID_TRANSITIONS)}.",
                    )
                transition_duration = _clamp(float(transition_duration), 0.1, 5.0)
                if transition == "crossfade":
                    too_short = [i["path"] for i in infos if i["duration"] <= transition_duration]
                    if too_short:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=(
                                f"crossfade of {transition_duration}s needs every clip to be "
                                f"longer than the fade; too short: {too_short[0]}"
                            ),
                        )
                cmd = _build_concat_command(
                    infos,
                    str(dest),
                    transition=transition,
                    transition_duration=transition_duration,
                    video_format=video_format,
                    crf=crf,
                )

            elif operation == "trim":
                src = infos[0]
                start = max(0.0, float(start))
                if duration > 0:
                    end = start + float(duration)
                if end <= 0 and src["duration"] > 0:
                    end = src["duration"]
                if end <= start:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"trim needs end > start (got start={start}, end={end}).",
                    )
                cmd = _build_trim_command(
                    paths[0],
                    str(dest),
                    start=start,
                    end=end,
                    has_audio=src["has_audio"],
                    video_format=video_format,
                    crf=crf,
                )

            elif operation == "text":
                if not text.strip():
                    return ToolResult(
                        success=False, output=None, error="text operation needs 'text'."
                    )
                src = infos[0]
                resolved_font = font_file or _find_font()
                if font_file and not Path(font_file).is_file():
                    return ToolResult(
                        success=False, output=None, error=f"font_file not found: {font_file}"
                    )
                if position not in _TEXT_POSITIONS and not (x and y):
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"Unknown position {position!r}. Valid: "
                            f"{sorted(_TEXT_POSITIONS)} (or pass x and y)."
                        ),
                    )
                px, py = (x, y) if (x and y) else _TEXT_POSITIONS[position]
                for label, value, pattern in (
                    ("x", px, _EXPR_SAFE),
                    ("y", py, _EXPR_SAFE),
                    ("font_color", font_color, _COLOR_SAFE),
                    ("box_color", box_color, _COLOR_SAFE),
                ):
                    if not pattern.match(value):
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"{label} contains characters not allowed in a filter expression: {value!r}",
                        )
                size = (
                    int(font_size)
                    if font_size > 0
                    else _auto_font_size(
                        text,
                        resolved_font,
                        width=src["width"],
                        height=src["height"],
                    )
                )
                # Text goes through a temp file: no filter-graph escaping of
                # user content, ever.
                fd, textfile_path = tempfile.mkstemp(suffix=".txt", prefix="missy_drawtext_")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                text_filter = _build_text_filter(
                    textfile=textfile_path,
                    font_file=resolved_font,
                    font_size=size,
                    font_color=font_color,
                    x=px,
                    y=py,
                    box=bool(box),
                    box_color=box_color,
                    box_opacity=_clamp(float(box_opacity), 0.0, 1.0),
                    start=max(0.0, float(start)),
                    end=max(0.0, float(end)),
                )
                cmd = _build_text_command(
                    paths[0],
                    str(dest),
                    text_filter=text_filter,
                    has_audio=src["has_audio"],
                    video_format=video_format,
                    crf=crf,
                )

            elif operation == "speed":
                src = infos[0]
                factor = _clamp(float(factor), 0.25, 4.0)
                if abs(factor - 1.0) < 1e-9:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="speed needs a 'factor' other than 1.0 (0.25-4.0).",
                    )
                cmd = _build_speed_command(
                    paths[0],
                    str(dest),
                    factor=factor,
                    has_audio=src["has_audio"],
                    video_format=video_format,
                    crf=crf,
                )

            elif operation == "extract_frame":
                src = infos[0]
                at = float(at)
                if at < 0:
                    # Last frame: back off from the end by at least one
                    # (and a bit) frame period so the seek always lands on
                    # a real frame even for low-fps clips.
                    fps_period = 1.5 / src["fps"] if src["fps"] > 0 else 0.1
                    at = max(0.0, src["duration"] - max(0.1, fps_period))
                elif src["duration"] > 0 and at >= src["duration"]:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"at={at} is past the end of the video "
                            f"({src['duration']:.3f}s); use at=-1 for the last frame."
                        ),
                    )
                cmd = _build_extract_frame_command(paths[0], str(dest), at=at)

            elif operation == "audio":
                src = infos[0]
                if not audio_file:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="audio operation needs 'audio_file' (the track to lay on).",
                    )
                audio_mode = (audio_mode or "replace").strip().lower()
                if audio_mode not in _VALID_AUDIO_MODES:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"Unknown audio_mode {audio_mode!r}. "
                            f"Valid: {sorted(_VALID_AUDIO_MODES)}."
                        ),
                    )
                audio_path = str(Path(audio_file).expanduser())
                if not Path(audio_path).is_file():
                    return ToolResult(
                        success=False, output=None, error=f"audio_file not found: {audio_file}"
                    )
                try:
                    _probe(audio_path, expect="audio")
                except ValueError as exc:
                    return ToolResult(success=False, output=None, error=str(exc))
                cmd = _build_audio_mux_command(
                    paths[0],
                    audio_path,
                    str(dest),
                    mode=audio_mode,
                    loop=bool(loop),
                    video_has_audio=src["has_audio"],
                )

            else:  # resize
                src = infos[0]
                width = int(width)
                height = int(height)
                if width <= 0 and height <= 0:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="resize needs width and/or height (> 0).",
                    )
                # Snap to even (yuv420p requires even dimensions).
                width, height = width // 2 * 2, height // 2 * 2
                cmd = _build_resize_command(
                    paths[0],
                    str(dest),
                    width=width,
                    height=height,
                    has_audio=src["has_audio"],
                    video_format=video_format,
                    crf=crf,
                )

            # --- run ----------------------------------------------------
            logger.debug("video_edit %s: %s", operation, " ".join(cmd))
            err = _run_ffmpeg(cmd, timeout=timeout)
            if err is not None:
                return ToolResult(success=False, output=None, error=err)
            if not dest.is_file() or dest.stat().st_size == 0:
                return ToolResult(
                    success=False,
                    output=None,
                    error="ffmpeg reported success but produced no output file.",
                )

            # Probe the *result* so the reported numbers describe the
            # actual file, not the request.
            try:
                out_info = _probe(str(dest))
            except ValueError as exc:
                return ToolResult(
                    success=False, output=None, error=f"output verification failed: {exc}"
                )

            if operation == "extract_frame":
                encoder = dest.suffix.lstrip(".").lower()
            elif operation == "audio":
                encoder = "copy"  # video stream is never re-encoded by this op
            else:
                encoder = {
                    "h264-mp4": "libx264",
                    "h265-mp4": "libx265",
                    "nvenc_h264-mp4": "h264_nvenc",
                }[video_format]
            output: dict[str, Any] = {
                "path": str(dest),
                "operation": operation,
                "width": out_info["width"],
                "height": out_info["height"],
                "fps": out_info["fps"],
                "duration_seconds": round(out_info["duration"], 3),
                "has_audio": out_info["has_audio"],
                "size_bytes": out_info["size_bytes"],
                "inputs": len(paths),
                "encoder": encoder,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            if operation == "extract_frame":
                output["at"] = round(at, 3)
            return ToolResult(success=True, output=output)
        finally:
            if textfile_path:
                import contextlib

                with contextlib.suppress(OSError):
                    os.unlink(textfile_path)
            # Never leave a zero-byte/partial artifact behind on failure.
            if dest.exists() and dest.stat().st_size == 0:
                import contextlib

                with contextlib.suppress(OSError):
                    dest.unlink()
