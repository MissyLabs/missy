"""Built-in tool: generate a short video via a local ComfyUI server.

Talks to a running `ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_
instance's HTTP API to render a short video clip using one of two backends:

* ``"svd"`` -- Stable Video Diffusion image-to-video. Animates a single
  input image into a 14-25 frame clip. Uses ``ImageOnlyCheckpointLoader``
  + ``SVD_img2vid_Conditioning`` + ``VideoLinearCFGGuidance``, the same
  node graph as ComfyUI's own official SVD example.
* ``"animatediff"`` -- AnimateDiff Evolved text-to-video. Generates new
  frames from a text prompt using a motion module on top of a standard
  SD1.x checkpoint. Uses ``ADE_LoadAnimateDiffModel`` +
  ``ADE_ApplyAnimateDiffModelSimple`` + ``ADE_UseEvolvedSampling`` (the
  `AnimateDiff-Evolved <https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved>`_
  custom node pack) with a standard KSampler/CLIPTextEncode graph.

Both backends finish with `VideoHelperSuite
<https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite>`_'s
``VHS_VideoCombine`` node to mux the generated frames into an actual MP4
file, which this tool then copies to a local Missy-managed directory
(``~/.missy/videos/`` by default) and returns the path to.

To "improve a video based on feedback": re-invoke this tool with the same
prompt/image but adjusted parameters (a different ``prompt`` wording,
``motion_bucket_id``/``augmentation_level`` for more or less motion,
``context_length`` for AnimateDiff coherence, a different ``seed``, etc.)
-- there is no separate "revise" API; iteration is just calling this tool
again with different arguments, the same way a human would in the
ComfyUI UI. Use ``vision_capture``/``vision_analyze`` (or ask the user)
to look at the result before deciding what to adjust.

Requires:

* A running ComfyUI server reachable at ``comfyui_host``:``comfyui_port``
  (default ``127.0.0.1:8199``) -- this tool does not start ComfyUI itself.
* The ``ComfyUI-AnimateDiff-Evolved`` and ``ComfyUI-VideoHelperSuite``
  custom node packs installed in ComfyUI's ``custom_nodes/`` (for the
  ``animatediff`` backend and for video muxing on both backends).
* For ``svd``: ``svd.safetensors`` or ``svd_xt.safetensors`` in
  ComfyUI's ``models/checkpoints/``.
* For ``animatediff``: a base SD1.x checkpoint (e.g.
  ``v1-5-pruned-emaonly.safetensors``) plus a motion module (e.g.
  ``mm_sd_v15_v2.ckpt``) in ``models/animatediff_models/`` or
  ``custom_nodes/ComfyUI-AnimateDiff-Evolved/models/``.

Example::

    from missy.tools.builtin.video_generate import VideoGenerateTool

    tool = VideoGenerateTool()
    result = tool.execute(backend="svd", image_path="/path/to/photo.png")
    assert result.success
    print(result.output["path"])  # -> ~/.missy/videos/video_....mp4
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8199
_DEFAULT_TIMEOUT = 600  # video generation is much slower than a single image
_POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_OUTPUT_DIR = str(Path.home() / ".missy" / "videos")
_MAX_RESPONSE_BYTES = 300 * 1024 * 1024  # 300 MB, for the /view download fallback

_SVD_CHECKPOINTS = {"svd": "svd.safetensors", "svd_xt": "svd_xt.safetensors"}
_DEFAULT_ANIMATEDIFF_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
_DEFAULT_MOTION_MODULE = "mm_sd_v15_v2.ckpt"
_DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, watermark, distorted, deformed, jpeg artifacts"

_VALID_BACKENDS = frozenset({"svd", "animatediff"})


def _build_svd_workflow(
    *,
    ckpt_name: str,
    image_name: str,
    width: int,
    height: int,
    video_frames: int,
    motion_bucket_id: int,
    fps: int,
    augmentation_level: float,
    steps: int,
    cfg: float,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Build the ComfyUI API-format workflow graph for SVD image-to-video.

    Mirrors ComfyUI's own official SVD example workflow: an
    ``ImageOnlyCheckpointLoader`` feeds ``SVD_img2vid_Conditioning``
    (which also builds the empty video latent), a
    ``VideoLinearCFGGuidance``-patched model runs through a standard
    ``KSampler``, and the decoded frames are combined into a video by
    ``VHS_VideoCombine``.
    """
    return {
        "1": {
            "class_type": "ImageOnlyCheckpointLoader",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "3": {
            "class_type": "SVD_img2vid_Conditioning",
            "inputs": {
                "clip_vision": ["1", 1],
                "init_image": ["2", 0],
                "vae": ["1", 2],
                "width": width,
                "height": height,
                "video_frames": video_frames,
                "motion_bucket_id": motion_bucket_id,
                "fps": fps,
                "augmentation_level": augmentation_level,
            },
        },
        "4": {
            "class_type": "VideoLinearCFGGuidance",
            "inputs": {"model": ["1", 0], "min_cfg": 1.0},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["3", 0],
                "negative": ["3", 1],
                "latent_image": ["3", 2],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["6", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def _build_animatediff_workflow(
    *,
    ckpt_name: str,
    motion_module: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    video_frames: int,
    fps: int,
    context_length: int,
    context_overlap: int,
    steps: int,
    cfg: float,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Build the ComfyUI API-format workflow graph for AnimateDiff text-to-video.

    Loads a base SD1.x checkpoint, applies the AnimateDiff-Evolved motion
    module via ``ADE_LoadAnimateDiffModel`` + ``ADE_ApplyAnimateDiffModelSimple``,
    patches sampling via ``ADE_UseEvolvedSampling`` (with a sliding-context
    window so ``video_frames`` can exceed the motion module's native
    training window), runs a standard ``KSampler``/``CLIPTextEncode``
    graph, and combines the result into a video via ``VHS_VideoCombine``.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": video_frames},
        },
        "5": {
            "class_type": "ADE_LoadAnimateDiffModel",
            "inputs": {"model_name": motion_module},
        },
        "6": {
            "class_type": "ADE_ApplyAnimateDiffModelSimple",
            "inputs": {"motion_model": ["5", 0]},
        },
        "7": {
            "class_type": "ADE_StandardUniformContextOptions",
            "inputs": {
                "context_length": context_length,
                "context_stride": 1,
                "context_overlap": context_overlap,
            },
        },
        "8": {
            "class_type": "ADE_UseEvolvedSampling",
            "inputs": {
                "model": ["1", 0],
                "beta_schedule": "autoselect",
                "m_models": ["6", 0],
                "context_options": ["7", 0],
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["8", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def _extract_video_output(history_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Find the ``VHS_VideoCombine`` output within a ``/history`` entry.

    Scans every node's outputs for a ``gifs`` key (VideoHelperSuite's
    output field name regardless of the final container format) rather
    than assuming a fixed node id, since the workflow graphs built above
    could change node ids independently of this extraction logic.

    Returns:
        The first video descriptor dict (``filename``/``subfolder``/
        ``type``/``frame_rate``/``fullpath``), or ``None`` if no node
        produced one.
    """
    outputs = history_entry.get("outputs", {})
    for node_output in outputs.values():
        gifs = node_output.get("gifs")
        if gifs:
            return gifs[0]
    return None


class VideoGenerateTool(BaseTool):
    """Generate a short video via a local ComfyUI server.

    Attributes:
        name: ``"video_generate"``
        description: One-line description for function-calling schemas.
        permissions: ``network=True``, ``filesystem_read=True``,
            ``filesystem_write=True``.
    """

    name = "video_generate"
    description = (
        "Generate a short video via a local ComfyUI server. backend='svd' "
        "animates a single input image (image_path) into motion using "
        "Stable Video Diffusion. backend='animatediff' generates new video "
        "frames from a text prompt using AnimateDiff. Returns the local "
        "path to the produced .mp4. To improve a result based on feedback, "
        "call this again with adjusted parameters (prompt wording, "
        "motion_bucket_id/augmentation_level for more or less motion, a "
        "different seed, etc.) -- there is no separate revise step."
    )
    permissions = ToolPermissions(network=True, filesystem_read=True, filesystem_write=True)

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        """SR-1.4-class: this tool's real network target is the configured
        ComfyUI host, not something the registry's static-only
        ``allowed_hosts`` heuristic would otherwise see."""
        host = kwargs.get("comfyui_host") or _DEFAULT_HOST
        port = kwargs.get("comfyui_port") or _DEFAULT_PORT
        return [f"{host}:{port}"]

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        """This tool reads ``image_path`` (svd backend) and writes the
        final video to ``save_path`` (or the default videos directory) --
        neither matches the registry's generic path-kwarg heuristic."""
        image_path = kwargs.get("image_path") or ""
        save_path = kwargs.get("save_path") or ""
        read_paths = [image_path] if image_path else []
        write_paths = [save_path] if save_path else [_DEFAULT_OUTPUT_DIR]
        return (read_paths, write_paths)

    def execute(
        self,
        *,
        backend: str = "svd",
        image_path: str = "",
        prompt: str = "",
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        checkpoint: str = "",
        video_frames: int = 0,
        fps: int = 0,
        motion_bucket_id: int = 127,
        augmentation_level: float = 0.0,
        width: int = 0,
        height: int = 0,
        steps: int = 20,
        cfg: float = 0.0,
        seed: int = 0,
        context_length: int = 16,
        context_overlap: int = 4,
        save_path: str = "",
        comfyui_host: str = _DEFAULT_HOST,
        comfyui_port: int = _DEFAULT_PORT,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        """Generate a video and return the local path to the result.

        Args:
            backend: ``"svd"`` (image-to-video) or ``"animatediff"``
                (text-to-video).
            image_path: Local path to the source image. Required for
                ``backend="svd"``.
            prompt: Text prompt describing the desired video. Required
                for ``backend="animatediff"``.
            negative_prompt: What to avoid (AnimateDiff only).
            checkpoint: Override the default checkpoint filename for the
                chosen backend (must already exist in ComfyUI's
                ``models/checkpoints/``).
            video_frames: Number of frames to generate. Defaults to 25
                for svd, 16 for animatediff.
            fps: Output frame rate. Defaults to 6 for svd, 8 for animatediff.
            motion_bucket_id: SVD-only, 1-1023. Higher = more motion.
            augmentation_level: SVD-only, 0.0-10.0. Noise added to the
                conditioning image; higher values diverge more from it.
            width: Output width. Defaults to 1024 (svd) / 512 (animatediff).
            height: Output height. Defaults to 576 (svd) / 512 (animatediff).
            steps: Sampling steps (default 20).
            cfg: Classifier-free guidance scale. Defaults to 2.5 (svd) /
                7.5 (animatediff).
            seed: Sampling seed. ``0`` picks a random seed each call.
            context_length: AnimateDiff-only sliding-context window size.
            context_overlap: AnimateDiff-only sliding-context overlap.
            save_path: Optional destination path for the final video.
                Defaults to a timestamped file under ``~/.missy/videos/``.
            comfyui_host: ComfyUI server host (default ``127.0.0.1``).
            comfyui_port: ComfyUI server port (default ``8199``).
            timeout: Maximum seconds to wait for generation to complete.

        Returns:
            :class:`~missy.tools.base.ToolResult` with ``output`` set to
            a dict with ``path``, ``backend``, ``frames``, ``fps``, and
            ``prompt_id`` on success.
        """
        backend = (backend or "svd").strip().lower()
        if backend not in _VALID_BACKENDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown backend {backend!r}; must be one of {sorted(_VALID_BACKENDS)}.",
            )

        base_url = f"http://{comfyui_host}:{comfyui_port}"
        filename_prefix = f"missy_{backend}_{uuid.uuid4().hex[:8]}"

        try:
            from missy.gateway.client import PolicyHTTPClient
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"HTTP client unavailable: {exc}")

        try:
            # PolicyHTTPClient directly (not the create_client() factory,
            # which doesn't expose max_response_bytes) so the /view
            # download fallback isn't capped at the default 50 MB --
            # longer video generations can exceed that.
            with PolicyHTTPClient(
                session_id="video_generate_tool",
                task_id="video_generate",
                timeout=timeout,
                category="tool",
                max_response_bytes=_MAX_RESPONSE_BYTES,
            ) as http:
                if backend == "svd":
                    if not image_path:
                        return ToolResult(
                            success=False, output=None, error="backend='svd' requires image_path."
                        )
                    img_file = Path(image_path).expanduser()
                    if not img_file.is_file():
                        return ToolResult(
                            success=False, output=None, error=f"image_path not found: {image_path}"
                        )

                    with open(img_file, "rb") as f:
                        upload_resp = http.post(
                            f"{base_url}/upload/image",
                            files={"image": (img_file.name, f, "image/png")},
                        )
                    if upload_resp.status_code != 200:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"ComfyUI image upload failed: HTTP {upload_resp.status_code}",
                        )
                    uploaded_name = upload_resp.json().get("name")
                    if not uploaded_name:
                        return ToolResult(
                            success=False,
                            output=None,
                            error="ComfyUI image upload response missing filename.",
                        )

                    workflow = _build_svd_workflow(
                        ckpt_name=checkpoint or _SVD_CHECKPOINTS["svd_xt"],
                        image_name=uploaded_name,
                        width=width or 1024,
                        height=height or 576,
                        video_frames=video_frames or 25,
                        motion_bucket_id=motion_bucket_id,
                        fps=fps or 6,
                        augmentation_level=augmentation_level,
                        steps=steps,
                        cfg=cfg or 2.5,
                        seed=seed or random.randint(0, 2**32 - 1),
                        filename_prefix=filename_prefix,
                    )
                    effective_frames = video_frames or 25
                    effective_fps = fps or 6
                else:
                    if not prompt:
                        return ToolResult(
                            success=False,
                            output=None,
                            error="backend='animatediff' requires prompt.",
                        )
                    workflow = _build_animatediff_workflow(
                        ckpt_name=checkpoint or _DEFAULT_ANIMATEDIFF_CHECKPOINT,
                        motion_module=_DEFAULT_MOTION_MODULE,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width or 512,
                        height=height or 512,
                        video_frames=video_frames or 16,
                        fps=fps or 8,
                        context_length=context_length,
                        context_overlap=context_overlap,
                        steps=steps,
                        cfg=cfg or 7.5,
                        seed=seed or random.randint(0, 2**32 - 1),
                        filename_prefix=filename_prefix,
                    )
                    effective_frames = video_frames or 16
                    effective_fps = fps or 8

                client_id = f"missy-{uuid.uuid4().hex[:12]}"
                submit_resp = http.post(
                    f"{base_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                if submit_resp.status_code != 200:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"ComfyUI rejected the workflow: HTTP "
                            f"{submit_resp.status_code}: {submit_resp.text[:500]}"
                        ),
                    )

                submit_data = submit_resp.json()
                node_errors = submit_data.get("node_errors") or {}
                if node_errors:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"ComfyUI workflow validation errors: {json.dumps(node_errors)[:1000]}",
                    )
                prompt_id = submit_data.get("prompt_id")
                if not prompt_id:
                    return ToolResult(
                        success=False, output=None, error="ComfyUI did not return a prompt_id."
                    )

                history_entry = self._wait_for_completion(http, base_url, prompt_id, timeout)
                if isinstance(history_entry, str):
                    # An error message was returned instead of an entry.
                    return ToolResult(success=False, output=None, error=history_entry)

                video_info = _extract_video_output(history_entry)
                if video_info is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            "ComfyUI completed but produced no video output "
                            "(no VHS_VideoCombine 'gifs' output found)."
                        ),
                    )

                final_path = self._retrieve_video(http, base_url, video_info, save_path)

            return ToolResult(
                success=True,
                output={
                    "path": final_path,
                    "backend": backend,
                    "frames": effective_frames,
                    "fps": video_info.get("frame_rate", effective_fps),
                    "prompt_id": prompt_id,
                },
            )
        except Exception as exc:
            logger.exception("video_generate failed")
            return ToolResult(success=False, output=None, error=str(exc))

    @staticmethod
    def _wait_for_completion(
        http: Any, base_url: str, prompt_id: str, timeout: int
    ) -> dict[str, Any] | str:
        """Poll ``/history/{prompt_id}`` until completion, error, or timeout.

        Returns:
            The history entry dict on success, or a human-readable error
            string on failure/timeout (the caller distinguishes the two
            by type since a dict is never a valid error message here).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            hist_resp = http.get(f"{base_url}/history/{prompt_id}")
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                entry = hist.get(prompt_id)
                if entry is not None:
                    status = entry.get("status", {})
                    if status.get("completed"):
                        return entry
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        return f"ComfyUI generation failed: {json.dumps(messages)[:1000]}"
            time.sleep(_POLL_INTERVAL_SECONDS)
        return f"Timed out after {timeout}s waiting for ComfyUI to finish generating."

    @staticmethod
    def _retrieve_video(
        http: Any, base_url: str, video_info: dict[str, Any], save_path: str
    ) -> str:
        """Copy the generated video to a local Missy-managed path.

        Prefers a direct filesystem copy via ``video_info["fullpath"]``
        (ComfyUI and Missy share a filesystem in the common local-server
        deployment this tool targets), falling back to an HTTP download
        via ``/view`` if that path doesn't exist -- e.g. ComfyUI running
        on a different host.
        """
        if save_path:
            dest = Path(save_path).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(video_info.get("filename", "video.mp4")).suffix or ".mp4"
            dest = Path(_DEFAULT_OUTPUT_DIR) / f"video_{ts}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        fullpath = video_info.get("fullpath")
        if fullpath and Path(fullpath).is_file():
            shutil.copy2(fullpath, dest)
            return str(dest)

        resp = http.get(
            f"{base_url}/view",
            params={
                "filename": video_info.get("filename", ""),
                "subfolder": video_info.get("subfolder", ""),
                "type": video_info.get("type", "output"),
            },
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return str(dest)

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": sorted(_VALID_BACKENDS),
                        "description": (
                            "'svd' animates an input image (image_path). "
                            "'animatediff' generates video from a text prompt."
                        ),
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Local path to the source image. Required for backend='svd'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the video. Required for backend='animatediff'.",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What to avoid (animatediff only).",
                    },
                    "checkpoint": {
                        "type": "string",
                        "description": (
                            "Override the default checkpoint filename for the chosen "
                            "backend (must already exist in ComfyUI's models/checkpoints/)."
                        ),
                    },
                    "video_frames": {
                        "type": "integer",
                        "description": "Number of frames (default: 25 for svd, 16 for animatediff).",
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Output frame rate (default: 6 for svd, 8 for animatediff).",
                    },
                    "motion_bucket_id": {
                        "type": "integer",
                        "description": "svd only, 1-1023. Higher = more motion (default: 127).",
                    },
                    "augmentation_level": {
                        "type": "number",
                        "description": (
                            "svd only, 0.0-10.0. Noise added to the conditioning image; "
                            "higher values diverge more from it (default: 0.0)."
                        ),
                    },
                    "width": {
                        "type": "integer",
                        "description": "Output width (default: 1024 for svd, 512 for animatediff).",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Output height (default: 576 for svd, 512 for animatediff).",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Sampling steps (default: 20).",
                    },
                    "cfg": {
                        "type": "number",
                        "description": "Classifier-free guidance scale (default: 2.5 for svd, 7.5 for animatediff).",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Sampling seed. 0 (default) picks a random seed each call.",
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "animatediff only: sliding-context window size (default: 16).",
                    },
                    "context_overlap": {
                        "type": "integer",
                        "description": "animatediff only: sliding-context overlap (default: 4).",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional destination path for the final video (default: a timestamped file under ~/.missy/videos/).",
                    },
                    "comfyui_host": {
                        "type": "string",
                        "description": "ComfyUI server host (default: 127.0.0.1).",
                    },
                    "comfyui_port": {
                        "type": "integer",
                        "description": "ComfyUI server port (default: 8199).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to wait for generation to complete (default: 600).",
                    },
                },
                "required": ["backend"],
            },
        }
