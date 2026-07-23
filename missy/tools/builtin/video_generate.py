"""Built-in tool: generate a short video (optionally with audio) via ComfyUI.

Talks to a running `ComfyUI <https://github.com/comfyanonymous/ComfyUI>`_
instance's HTTP API to render a short video clip using one of three
backends:

* ``"wan"`` -- Wan 2.2 TI2V 5B, the recommended/highest-quality backend.
  Text-to-video by default; also does image-to-video when ``image_path``
  is provided. 24 fps native. Uses ``UNETLoader`` + ``ModelSamplingSD3``
  + ``Wan22ImageToVideoLatent``, matching ComfyUI's official Wan 2.2
  example graph.
* ``"svd"`` -- Stable Video Diffusion image-to-video. Animates a single
  input image into a 25-frame clip. Uses ``ImageOnlyCheckpointLoader``
  + ``SVD_img2vid_Conditioning`` + ``VideoLinearCFGGuidance``.
* ``"animatediff"`` -- AnimateDiff Evolved text-to-video on SD1.5, with a
  ``FreeU_V2`` quality patch. Uses the AnimateDiff-Evolved custom node
  pack (``ADE_*`` nodes).

Quality post-processing (all backends, all inside the same workflow
graph, GPU-accelerated):

* **Frame interpolation** -- core ``FrameInterpolate`` (RIFE) multiplies
  the frame rate; enabled automatically for the low-fps svd/animatediff
  backends so output plays smoothly instead of as a slide show.
* **Upscaling** -- optional 2x RealESRGAN pass (``upscale=True``).

Audio (all backends): ``audio_prompt`` generates a soundtrack in the same
graph -- with Stable Audio 3.0 medium base by default (``audio_model``,
open-weight, May 2026), or Stable Audio Open 1.0 as a legacy fallback --
sized to the final clip duration, and muxes it into the MP4 via
``VHS_VideoCombine``'s ``audio`` input. Alternatively ``audio_path`` muxes an
existing local audio file.

The tool preflights the server before submitting: it verifies a GPU
(CUDA/ROCm/MPS/XPU) device is actually present (``allow_cpu=True`` to
override) and that every model file the requested workflow needs exists,
returning actionable download instructions when one is missing. On
timeout the queued/running job is cancelled server-side rather than left
hogging the GPU. The effective seed is always echoed back in the output
so a good result can be reproduced or refined.

To "improve a video based on feedback": re-invoke this tool with
adjusted parameters (different ``prompt`` wording, the echoed ``seed``
with more ``steps``, ``motion_bucket_id`` for more/less motion, etc.) --
iteration is just calling the tool again, the same way a human would in
the ComfyUI UI. Use ``vision_capture``/``vision_analyze`` (or ask the
user) to look at the result before deciding what to adjust.

Model files expected on the ComfyUI side (see ``video.md`` at the repo
root for sources):

* wan: ``wan2.2_ti2v_5B_fp16.safetensors`` (``models/diffusion_models/``),
  ``umt5_xxl_fp8_e4m3fn_scaled.safetensors`` (``models/text_encoders/``),
  ``wan2.2_vae.safetensors`` (``models/vae/``)
* svd: ``svd_xt.safetensors`` (``models/checkpoints/``)
* animatediff: ``v1-5-pruned-emaonly.safetensors`` (``models/checkpoints/``)
  plus ``mm_sd_v15_v2.ckpt`` (``models/animatediff_models/``)
* audio (default, stable-audio-3): ``stable_audio_3_medium_base.safetensors``
  (``models/checkpoints/``) plus ``t5gemma_b_b_ul2.safetensors``
  (``models/text_encoders/``); requires a ComfyUI build with SA3 model support
* audio (legacy, stable-audio-open-1.0): ``stable-audio-open-1.0.safetensors``
  (``models/checkpoints/``) plus ``t5_base.safetensors`` (``models/text_encoders/``)
* interpolation: ``rife_v4.26.safetensors`` (``models/frame_interpolation/``)
* upscale: ``RealESRGAN_x2plus.pth`` (``models/upscale_models/``)

Example::

    from missy.tools.builtin.video_generate import VideoGenerateTool

    tool = VideoGenerateTool()
    result = tool.execute(
        backend="wan",
        prompt="a red fox trotting through fresh snow, golden hour",
        audio_prompt="soft wind, crunching snow footsteps",
    )
    assert result.success
    print(result.output["path"])  # -> ~/.missy/videos/video_....mp4
    print(result.output["seed"])  # reuse to refine the same clip
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import random
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

# ComfyUI server selection. A deployment points video_generate at a remote
# ComfyUI (e.g. a stronger GPU box on the LAN) via MISSY_COMFYUI_HOST, which may
# be a single host or a comma-separated ordered list, each optionally
# "host:port". The local server (127.0.0.1:8199) is always appended as a final
# fallback, so generation degrades to the local box when a configured remote is
# down. MISSY_COMFYUI_PORT sets the default port for entries without one. An
# explicit comfyui_host kwarg overrides all of this with a single host (no
# fallback), preserving explicit-target semantics for tests/policy checks.
_LOCAL_FALLBACK = ("127.0.0.1", 8199)


def _default_port() -> int:
    raw = os.environ.get("MISSY_COMFYUI_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    return 8199


def _comfyui_candidates_from_env() -> list[tuple[str, int]]:
    """Ordered ``(host, port)`` ComfyUI candidates from ``MISSY_COMFYUI_HOST``.

    Parses the (possibly comma-separated, optionally ``host:port``) env value
    and always appends the local fallback ``127.0.0.1:8199`` last unless a local
    host is already present.
    """
    default_port = _default_port()
    cands: list[tuple[str, int]] = []
    for entry in os.environ.get("MISSY_COMFYUI_HOST", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            host, _, port = entry.partition(":")
            cands.append((host.strip(), int(port) if port.strip().isdigit() else default_port))
        else:
            cands.append((entry, default_port))
    if not cands:
        cands.append(("127.0.0.1", default_port))
    if not any(host in ("127.0.0.1", "localhost") for host, _ in cands):
        cands.append(_LOCAL_FALLBACK)
    return cands


# First candidate's host/port, for the execute() signature default and display.
_DEFAULT_HOST = _comfyui_candidates_from_env()[0][0]
_DEFAULT_PORT = _comfyui_candidates_from_env()[0][1]
_POLL_INTERVAL_SECONDS = 2.0
_DEFAULT_OUTPUT_DIR = str(Path.home() / ".missy" / "videos")
_MAX_RESPONSE_BYTES = 300 * 1024 * 1024  # 300 MB, for the /view download fallback

_VALID_BACKENDS = frozenset({"wan", "wan14b", "svd", "animatediff"})

# Tool-facing format name -> VHS_VideoCombine format value.
_VIDEO_FORMATS = {
    "h264-mp4": "video/h264-mp4",
    "h265-mp4": "video/h265-mp4",
    "nvenc_h264-mp4": "video/nvenc_h264-mp4",  # NVENC: GPU-accelerated encode
}

# Model filenames (as they appear in ComfyUI's model folders).
_SVD_CHECKPOINT = "svd_xt.safetensors"
_ANIMATEDIFF_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
_ANIMATEDIFF_MOTION_MODULE = "mm_sd_v15_v2.ckpt"
_WAN_DIFFUSION_MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
_WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_WAN_VAE = "wan2.2_vae.safetensors"
# Wan 2.2 A14B MoE (the "14B" model): two experts (high/low noise) for the
# text-to-video path, plus the wan 2.1 VAE (the A14B uses 2.1's VAE, not 2.2's).
# Shares the umt5 text encoder with the 5B. fp8_scaled weights (~14 GB each);
# the experts run sequentially so peak VRAM is ~one expert.
_WAN14B_T2V_HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
_WAN14B_T2V_LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
_WAN21_VAE = "wan_2.1_vae.safetensors"
# Audio backends. Stable Audio 3.0 (medium base, open-weight, May 2026) is the
# default/recommended text-to-audio model; Stable Audio Open 1.0 (2024) is kept
# as a legacy fallback for ComfyUI installs that predate SA3 core-model support.
_AUDIO3_CHECKPOINT = "stable_audio_3_medium_base.safetensors"
_AUDIO3_TEXT_ENCODER = "t5gemma_b_b_ul2.safetensors"
_AUDIO_CHECKPOINT = "stable-audio-open-1.0.safetensors"
_AUDIO_TEXT_ENCODER = "t5_base.safetensors"
_VALID_AUDIO_MODELS = frozenset({"stable-audio-3", "stable-audio-open-1.0"})
_DEFAULT_AUDIO_MODEL = "stable-audio-3"
_INTERPOLATION_MODEL = "rife_v4.26.safetensors"
_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"

# Where a missing model can be obtained, for actionable preflight errors.
_MODEL_SOURCES = {
    _WAN_DIFFUSION_MODEL: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    _WAN_TEXT_ENCODER: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    _WAN_VAE: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    _WAN14B_T2V_HIGH: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged (split_files/diffusion_models/)",
    _WAN14B_T2V_LOW: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged (split_files/diffusion_models/)",
    _WAN21_VAE: "huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged (split_files/vae/)",
    _AUDIO3_CHECKPOINT: "huggingface.co/Comfy-Org/stable-audio-3 (checkpoints/)",
    _AUDIO3_TEXT_ENCODER: "huggingface.co/Comfy-Org/stable-audio-3 (text_encoders/)",
    _AUDIO_CHECKPOINT: "huggingface.co/Comfy-Org/stable-audio-open-1.0_repackaged",
    _AUDIO_TEXT_ENCODER: "huggingface.co/google-t5/t5-base (model.safetensors, renamed)",
    _INTERPOLATION_MODEL: "huggingface.co/Comfy-Org/frame_interpolation",
    _UPSCALE_MODEL: "github.com/xinntao/Real-ESRGAN/releases",
    _SVD_CHECKPOINT: "huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
}

_SD_NEGATIVE_PROMPT = "blurry, low quality, watermark, distorted, deformed, jpeg artifacts"
# Wan's standard negative prompt (from the official Wan 2.2 ComfyUI template).
_WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_AUDIO_NEGATIVE_PROMPT = "noise, distortion, low quality"

# Per-backend defaults: (width, height, frames, fps, steps, cfg, sampler,
# scheduler, auto-interpolation multiplier, auto timeout seconds).
_BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    # Wan 2.2 5B is 24 fps native; defaults are sized down from the model's
    # 1280x704x121 native config so a single clip stays in single-digit
    # minutes on an 8 GB card (ComfyUI offloads the fp16 weights to RAM).
    "wan": {
        "width": 832,
        "height": 480,
        "frames": 81,
        "fps": 24,
        "steps": 20,
        "cfg": 5.0,
        "sampler": "uni_pc",
        "scheduler": "simple",
        "interpolate": 1,
        "timeout": 3600,
        "dim_step": 32,
        "dim_max": 1280,
        "frames_min": 5,
        "frames_max": 121,
    },
    # Wan 2.2 A14B MoE, text-to-video. Higher quality than the 5B but ~14 GB
    # per expert; the two experts run sequentially (high-noise then low-noise)
    # so peak VRAM is ~one expert -- a tight but workable fit on 16 GB, slower
    # on 8 GB (heavy offload). Sampler/steps/cfg from ComfyUI's official 14B
    # example (euler/simple, 20 steps, cfg 3.5, mid-step high/low boundary).
    "wan14b": {
        "width": 832,
        "height": 480,
        "frames": 81,
        "fps": 24,
        "steps": 20,
        "cfg": 3.5,
        "sampler": "euler",
        "scheduler": "simple",
        "interpolate": 1,
        "timeout": 3600,
        "dim_step": 32,
        "dim_max": 1280,
        "frames_min": 5,
        "frames_max": 121,
    },
    "svd": {
        "width": 1024,
        "height": 576,
        "frames": 25,
        "fps": 6,
        "steps": 20,
        "cfg": 2.5,
        "sampler": "euler",
        "scheduler": "karras",
        # 25 frames @ 6 fps is a slide show; 4x RIFE makes it 24 fps.
        "interpolate": 4,
        "timeout": 1200,
        "dim_step": 8,
        "dim_max": 1280,
        # SVD-XT is trained for 25 frames; more degrades badly.
        "frames_min": 5,
        "frames_max": 25,
    },
    "animatediff": {
        "width": 512,
        "height": 512,
        "frames": 16,
        "fps": 8,
        "steps": 25,
        "cfg": 7.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "interpolate": 2,
        "timeout": 1200,
        "dim_step": 8,
        "dim_max": 1024,
        "frames_min": 8,
        "frames_max": 128,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _snap_dim(value: int, step: int, lo: int, hi: int) -> int:
    """Round ``value`` to the nearest multiple of ``step`` within [lo, hi]."""
    snapped = int(round(value / step)) * step
    return int(_clamp(snapped, lo, hi))


# ---------------------------------------------------------------------------
# Workflow graph builders.
#
# Each backend builder returns ``(graph, image_ref)`` where ``image_ref`` is
# the ``[node_id, output_index]`` reference to the decoded IMAGE batch. The
# ``_append_*`` helpers then compose post-processing/audio/mux stages onto
# the same graph using prefixed node ids that cannot collide with the
# builders' numeric ids.
# ---------------------------------------------------------------------------


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
    sampler: str,
    scheduler: str,
    seed: int,
) -> tuple[dict[str, Any], list[Any]]:
    """SVD image-to-video graph, mirroring ComfyUI's official SVD example."""
    graph = {
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
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
    }
    return graph, ["6", 0]


def _build_animatediff_workflow(
    *,
    ckpt_name: str,
    motion_module: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    video_frames: int,
    context_length: int,
    context_overlap: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    freeu: bool = True,
) -> tuple[dict[str, Any], list[Any]]:
    """AnimateDiff-Evolved text-to-video graph with an optional FreeU_V2
    quality patch on the base SD1.5 model."""
    graph: dict[str, Any] = {
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
    }
    base_model_ref: list[Any] = ["1", 0]
    if freeu:
        graph["12"] = {
            "class_type": "FreeU_V2",
            "inputs": {"model": base_model_ref, "b1": 1.3, "b2": 1.4, "s1": 0.9, "s2": 0.2},
        }
        base_model_ref = ["12", 0]
    graph.update(
        {
            "8": {
                "class_type": "ADE_UseEvolvedSampling",
                "inputs": {
                    "model": base_model_ref,
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
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            },
            "10": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
            },
        }
    )
    return graph, ["10", 0]


def _build_wan_workflow(
    *,
    diffusion_model: str,
    text_encoder: str,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    image_name: str,
    width: int,
    height: int,
    video_frames: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    weight_dtype: str = "fp8_e4m3fn",
) -> tuple[dict[str, Any], list[Any]]:
    """Wan 2.2 TI2V 5B graph (text-to-video, or image-to-video when
    ``image_name`` is set), mirroring ComfyUI's official Wan 2.2 example:
    ``ModelSamplingSD3(shift=8)`` on the UNET, umt5 text encoder, and
    ``Wan22ImageToVideoLatent`` for the (optionally image-seeded) latent.

    The UNET is loaded in fp8 by default: the fp16 weights alone are
    ~10 GB, and on the 8 GB cards this deployment targets the sampler
    OOMs at the default resolution unless the weights are halved (this
    was hit live on the RTX 3070; fp8 quantization of the 5B model costs
    little visible quality)."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": weight_dtype},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["1", 0], "shift": 8.0},
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": video_frames,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
    }
    if image_name:
        graph["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        graph["7"]["inputs"]["start_image"] = ["10", 0]
    return graph, ["9", 0]


def _build_wan22_14b_workflow(
    *,
    diffusion_high: str,
    diffusion_low: str,
    text_encoder: str,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    video_frames: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    weight_dtype: str = "default",
) -> tuple[dict[str, Any], list[Any]]:
    """Wan 2.2 A14B MoE text-to-video graph, mirroring ComfyUI's official
    Wan 2.2 14B example: two expert UNETs -- the high-noise expert handles the
    first half of sampling, the low-noise expert the second -- each wrapped in
    ``ModelSamplingSD3(shift=8)`` and chained via two ``KSamplerAdvanced`` nodes
    across a mid-step boundary. The high stage adds noise and returns leftover
    noise; the low stage picks up from there without re-adding noise. Uses the
    umt5 text encoder and the wan 2.1 VAE (the A14B uses the 2.1 VAE, not the
    5B's 2.2 VAE) and ``EmptyHunyuanLatentVideo`` for the latent.

    Text-to-video only -- A14B image-to-video is a separate (i2v-expert) graph.
    """
    boundary = max(1, steps // 2)
    graph: dict[str, Any] = {
        "hi": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_high, "weight_dtype": weight_dtype},
        },
        "lo": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_low, "weight_dtype": weight_dtype},
        },
        "hims": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["hi", 0], "shift": 8.0}},
        "loms": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lo", 0], "shift": 8.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "wan"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["clip", 0]},
        },
        "lat": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": video_frames,
                "batch_size": 1,
            },
        },
        "khigh": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["hims", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["lat", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": boundary,
                "return_with_leftover_noise": "enable",
            },
        },
        "klow": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["loms", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["khigh", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "start_at_step": boundary,
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
            },
        },
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["klow", 0], "vae": ["vae", 0]}},
    }
    return graph, ["dec", 0]


def _append_upscale(
    graph: dict[str, Any], image_ref: list[Any], model_name: str = _UPSCALE_MODEL
) -> list[Any]:
    """Add a RealESRGAN upscale pass; returns the new IMAGE ref."""
    graph["up1"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}}
    graph["up2"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["up1", 0], "image": image_ref},
    }
    return ["up2", 0]


def _append_interpolation(
    graph: dict[str, Any],
    image_ref: list[Any],
    multiplier: int,
    model_name: str = _INTERPOLATION_MODEL,
) -> list[Any]:
    """Add a RIFE frame-interpolation pass; returns the new IMAGE ref."""
    graph["fi1"] = {
        "class_type": "FrameInterpolationModelLoader",
        "inputs": {"model_name": model_name},
    }
    graph["fi2"] = {
        "class_type": "FrameInterpolate",
        "inputs": {"interp_model": ["fi1", 0], "images": image_ref, "multiplier": multiplier},
    }
    return ["fi2", 0]


def _append_audio_generation(
    graph: dict[str, Any],
    *,
    audio_prompt: str,
    audio_negative_prompt: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    ckpt_name: str = _AUDIO_CHECKPOINT,
    text_encoder: str = _AUDIO_TEXT_ENCODER,
) -> list[Any]:
    """Add a Stable Audio Open text-to-audio branch; returns the AUDIO ref.

    Follows ComfyUI's official stable-audio example: the checkpoint
    provides the diffusion model and audio VAE, while the T5-base text
    encoder is loaded separately via ``CLIPLoader`` (the checkpoint does
    not embed one).
    """
    graph["au0"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": text_encoder, "type": "stable_audio"},
    }
    graph["au1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}}
    graph["au2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": audio_prompt, "clip": ["au0", 0]},
    }
    graph["au3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": audio_negative_prompt, "clip": ["au0", 0]},
    }
    graph["au4"] = {
        "class_type": "ConditioningStableAudio",
        "inputs": {
            "positive": ["au2", 0],
            "negative": ["au3", 0],
            "seconds_start": 0.0,
            "seconds_total": seconds,
        },
    }
    graph["au5"] = {
        "class_type": "EmptyLatentAudio",
        "inputs": {"seconds": seconds, "batch_size": 1},
    }
    graph["au6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["au1", 0],
            "positive": ["au4", 0],
            "negative": ["au4", 1],
            "latent_image": ["au5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "dpmpp_3m_sde_gpu",
            "scheduler": "exponential",
            "denoise": 1.0,
        },
    }
    graph["au7"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["au6", 0], "vae": ["au1", 2]},
    }
    return ["au7", 0]


def _append_audio_generation_sa3(
    graph: dict[str, Any],
    *,
    audio_prompt: str,
    audio_negative_prompt: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    ckpt_name: str = _AUDIO3_CHECKPOINT,
    text_encoder: str = _AUDIO3_TEXT_ENCODER,
) -> list[Any]:
    """Add a Stable Audio 3.0 (medium base) text-to-audio branch; returns AUDIO.

    Follows ComfyUI's official ``audio_stable_audio_3_medium_base`` template's
    direct (no-reprompt) path: the checkpoint supplies the diffusion model and
    audio VAE, the T5-Gemma encoder is loaded via ``CLIPLoader(type="stable_audio")``
    (same loader type string as SA-Open 1.0, only the encoder file differs), and
    the positive/negative ``CLIPTextEncode`` conditionings feed ``KSampler``
    directly -- SA3 drops the ``ConditioningStableAudio`` node, so clip length is
    controlled solely by ``EmptyLatentAudio``'s ``seconds``. Sampler/scheduler
    are ``lcm``/``simple`` per the template. Every node class_type here already
    exists in current ComfyUI; running SA3 additionally requires a ComfyUI build
    whose model loaders recognise the SA3 checkpoint architecture plus the two
    SA3 model files (preflight surfaces either if missing).
    """
    graph["au0"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": text_encoder, "type": "stable_audio"},
    }
    graph["au1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}}
    graph["au2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": audio_prompt, "clip": ["au0", 0]},
    }
    graph["au3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": audio_negative_prompt, "clip": ["au0", 0]},
    }
    graph["au5"] = {
        "class_type": "EmptyLatentAudio",
        "inputs": {"seconds": seconds, "batch_size": 1},
    }
    graph["au6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["au1", 0],
            "positive": ["au2", 0],
            "negative": ["au3", 0],
            "latent_image": ["au5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "lcm",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    graph["au7"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["au6", 0], "vae": ["au1", 2]},
    }
    return ["au7", 0]


def _append_audio_file(graph: dict[str, Any], uploaded_name: str) -> list[Any]:
    """Add a LoadAudio node for a file already uploaded to ComfyUI's input
    directory; returns the AUDIO ref."""
    graph["au1"] = {"class_type": "LoadAudio", "inputs": {"audio": uploaded_name}}
    return ["au1", 0]


def _append_video_combine(
    graph: dict[str, Any],
    image_ref: list[Any],
    *,
    audio_ref: list[Any] | None,
    frame_rate: int,
    filename_prefix: str,
    video_format: str,
    crf: int,
) -> None:
    """Terminate the graph with a ``VHS_VideoCombine`` mux node.

    Format-specific encoder options differ per VHS format (nvenc takes a
    bitrate instead of crf), so only the inputs that format declares are
    passed.
    """
    inputs: dict[str, Any] = {
        "images": image_ref,
        "frame_rate": frame_rate,
        "loop_count": 0,
        "filename_prefix": filename_prefix,
        "format": _VIDEO_FORMATS[video_format],
        "save_metadata": True,
        "pingpong": False,
        "save_output": True,
    }
    if video_format == "h264-mp4":
        inputs.update({"pix_fmt": "yuv420p", "crf": crf})
    elif video_format == "h265-mp4":
        inputs.update({"pix_fmt": "yuv420p10le", "crf": crf})
    else:  # nvenc_h264-mp4 (GPU encode)
        inputs.update({"pix_fmt": "yuv420p", "bitrate": 20, "megabit": True})
    if audio_ref is not None:
        inputs["audio"] = audio_ref
    graph["out"] = {"class_type": "VHS_VideoCombine", "inputs": inputs}


def _extract_video_output(history_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Find the ``VHS_VideoCombine`` output within a ``/history`` entry.

    Scans every node's outputs for a ``gifs`` key (VideoHelperSuite's
    output field name regardless of the final container format) rather
    than assuming a fixed node id.

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
    """Generate a short video (optionally with audio) via a local ComfyUI
    server.

    Attributes:
        name: ``"video_generate"``
        description: One-line description for function-calling schemas.
        permissions: ``network=True``, ``filesystem_read=True``,
            ``filesystem_write=True``.
    """

    name = "video_generate"
    description = (
        "Generate a short video via a local GPU-backed ComfyUI server. "
        "backend='wan' (recommended default, Wan 2.2 5B) generates 24fps video "
        "from a text prompt, or animates an image if image_path is also given. "
        "backend='wan14b' is the higher-quality Wan 2.2 A14B text-to-video model "
        "(slower, needs more VRAM; text-to-video only). "
        "backend='svd' animates a single input image (image_path). "
        "backend='animatediff' generates video from a text prompt (legacy). "
        "Pass audio_prompt to generate a matching soundtrack (music/sfx/"
        "ambience) muxed into the video, or audio_path to mux an existing "
        "audio file. Returns the local path to the produced .mp4 plus the "
        "seed used. To improve a result based on feedback, call this again "
        "with adjusted parameters (prompt wording, the returned seed with "
        "more steps, motion_bucket_id for more/less motion, etc.) -- there "
        "is no separate revise step."
    )
    permissions = ToolPermissions(network=True, filesystem_read=True, filesystem_write=True)

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        """SR-1.4-class: this tool's real network target is the configured
        ComfyUI host(s), not something the registry's static-only
        ``allowed_hosts`` heuristic would otherwise see. Returns every candidate
        (primary + fallbacks) so the policy engine allows the ones actually
        tried."""
        host = (kwargs.get("comfyui_host") or "").strip()
        if host:
            port = int(kwargs.get("comfyui_port") or 8199)
            return [f"{host}:{port}"]
        return [f"{h}:{p}" for h, p in _comfyui_candidates_from_env()]

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        """This tool reads ``image_path``/``audio_path`` and writes the
        final video to ``save_path`` (or the default videos directory) --
        neither matches the registry's generic path-kwarg heuristic."""
        read_paths = [p for p in (kwargs.get("image_path"), kwargs.get("audio_path")) if p]
        save_path = kwargs.get("save_path") or ""
        write_paths = [save_path] if save_path else [_DEFAULT_OUTPUT_DIR]
        return (read_paths, write_paths)

    def execute(
        self,
        *,
        backend: str = "wan",
        prompt: str = "",
        image_path: str = "",
        negative_prompt: str = "",
        audio_prompt: str = "",
        audio_negative_prompt: str = _AUDIO_NEGATIVE_PROMPT,
        audio_path: str = "",
        audio_steps: int = 50,
        audio_cfg: float = 5.0,
        audio_model: str = _DEFAULT_AUDIO_MODEL,
        checkpoint: str = "",
        motion_module: str = "",
        video_frames: int = 0,
        fps: int = 0,
        motion_bucket_id: int = 127,
        augmentation_level: float = 0.0,
        width: int = 0,
        height: int = 0,
        steps: int = 0,
        cfg: float = 0.0,
        sampler: str = "",
        scheduler: str = "",
        seed: int = 0,
        context_length: int = 16,
        context_overlap: int = 4,
        interpolate: int = 0,
        upscale: bool = False,
        video_format: str = "h264-mp4",
        crf: int = 17,
        allow_cpu: bool = False,
        save_path: str = "",
        comfyui_host: str = "",
        comfyui_port: int = 0,
        timeout: int = 0,
        **_kwargs: Any,
    ) -> ToolResult:
        """Generate a video and return the local path to the result.

        Args:
            backend: ``"wan"`` (text- or image-to-video, recommended),
                ``"svd"`` (image-to-video), or ``"animatediff"``
                (text-to-video).
            prompt: Text prompt describing the desired video. Required for
                ``wan`` and ``animatediff``.
            image_path: Local path to a source image. Required for ``svd``;
                optional for ``wan`` (switches it to image-to-video).
            negative_prompt: What to avoid. Defaults to a per-backend
                standard negative prompt.
            audio_prompt: Description of a soundtrack to generate (via
                ``audio_model``) and mux into the video. Works with every
                backend. Mutually exclusive with ``audio_path``.
            audio_negative_prompt: What the soundtrack should avoid.
            audio_path: Local audio file to mux into the video instead of
                generating one.
            audio_model: Text-to-audio model -- ``"stable-audio-3"`` (default,
                recommended) or ``"stable-audio-open-1.0"`` (legacy fallback).
            audio_steps: Sampling steps for audio generation (default 50).
            audio_cfg: CFG scale for audio generation (default 5.0).
            checkpoint: Override the backend's default model filename
                (checkpoint for svd/animatediff, diffusion model for wan).
            motion_module: animatediff only -- override the motion module
                filename (default ``mm_sd_v15_v2.ckpt``).
            video_frames: Frames to generate before interpolation.
                Defaults: 81 wan / 25 svd / 16 animatediff. Clamped to each
                model's sane range.
            fps: Base frame rate before interpolation. Defaults: 24 wan /
                6 svd / 8 animatediff.
            motion_bucket_id: svd only, 1-1023. Higher = more motion.
            augmentation_level: svd only, 0.0-10.0. Noise added to the
                conditioning image; higher diverges more from it.
            width: Output width. Defaults: 832 wan / 1024 svd / 512
                animatediff. Snapped to the backend's required multiple.
            height: Output height. Defaults: 480 wan / 576 svd / 512
                animatediff.
            steps: Sampling steps. Defaults: 20 wan / 20 svd / 25
                animatediff.
            cfg: Guidance scale. Defaults: 5.0 wan / 2.5 svd / 7.5
                animatediff.
            sampler: Sampler name override (defaults per backend).
            scheduler: Scheduler name override (defaults per backend).
            seed: Sampling seed. ``0`` picks a random seed; the effective
                seed is always echoed back in the output.
            context_length: animatediff sliding-context window size.
            context_overlap: animatediff sliding-context overlap.
            interpolate: RIFE frame-interpolation multiplier. ``0`` = auto
                (4x svd, 2x animatediff, off for wan); ``1`` = off; 2-8 =
                multiply frame count/rate by that factor.
            upscale: Apply a 2x RealESRGAN upscale before interpolation.
            video_format: ``"h264-mp4"`` (default), ``"h265-mp4"``, or
                ``"nvenc_h264-mp4"`` (GPU encode).
            crf: Encoder quality, lower = better (default 17; ignored for
                nvenc).
            allow_cpu: Permit generation on a ComfyUI server with no GPU
                (refused by default -- ~50x slower).
            save_path: Optional destination path for the final video.
                Defaults to a timestamped file under ``~/.missy/videos/``.
                Never overwrites an existing file (a numeric suffix is
                appended instead).
            comfyui_host: Explicit single ComfyUI host override (no fallback).
                When empty (the normal case), the server is selected from
                ``MISSY_COMFYUI_HOST`` (a single host or comma-separated ordered
                list), with ``127.0.0.1:8199`` always tried last as a fallback if
                a configured remote is down.
            comfyui_port: Port for an explicit ``comfyui_host`` (default 8199).
            timeout: Max seconds to wait. ``0`` = auto (3600 wan, 1200
                others). On timeout the job is cancelled server-side.

        Returns:
            :class:`~missy.tools.base.ToolResult` with ``output`` set to a
            dict with ``path``, ``backend``, ``frames``, ``frames_generated``,
            ``fps``, ``duration_seconds``, ``width``, ``height``, ``seed``,
            ``steps``, ``audio``, ``gpu``, ``size_bytes``, ``prompt_id``,
            and ``elapsed_seconds`` on success.
        """
        started = time.monotonic()
        backend = (backend or "wan").strip().lower()
        if backend not in _VALID_BACKENDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown backend {backend!r}; must be one of {sorted(_VALID_BACKENDS)}.",
            )
        audio_model = (audio_model or _DEFAULT_AUDIO_MODEL).strip().lower()
        if audio_model not in _VALID_AUDIO_MODELS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown audio_model {audio_model!r}; must be one of {sorted(_VALID_AUDIO_MODELS)}.",
            )
        if video_format not in _VIDEO_FORMATS:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Unknown video_format {video_format!r}; must be one of "
                    f"{sorted(_VIDEO_FORMATS)}."
                ),
            )
        if audio_prompt and audio_path:
            return ToolResult(
                success=False,
                output=None,
                error="audio_prompt and audio_path are mutually exclusive; pass only one.",
            )
        if backend == "svd" and not image_path:
            return ToolResult(
                success=False, output=None, error="backend='svd' requires image_path."
            )
        if backend in ("wan", "wan14b", "animatediff") and not prompt:
            return ToolResult(
                success=False, output=None, error=f"backend='{backend}' requires prompt."
            )
        if backend == "animatediff" and image_path:
            return ToolResult(
                success=False,
                output=None,
                error="backend='animatediff' is text-to-video and does not take image_path; "
                "use backend='wan' or 'svd' to animate an image.",
            )
        if backend == "wan14b" and image_path:
            return ToolResult(
                success=False,
                output=None,
                error="backend='wan14b' is text-to-video (A14B) only; use backend='wan' (the 5B "
                "TI2V model) or 'svd' to animate an image.",
            )
        for label, path_str in (("image_path", image_path), ("audio_path", audio_path)):
            if path_str and not Path(path_str).expanduser().is_file():
                return ToolResult(
                    success=False, output=None, error=f"{label} not found: {path_str}"
                )

        d = _BACKEND_DEFAULTS[backend]
        width = _snap_dim(width or d["width"], d["dim_step"], 256, d["dim_max"])
        height = _snap_dim(height or d["height"], d["dim_step"], 256, d["dim_max"])
        video_frames = int(_clamp(video_frames or d["frames"], d["frames_min"], d["frames_max"]))
        if backend in ("wan", "wan14b"):
            # Wan latent lengths must be 4k+1 (node input step is 4 from a
            # default of 49); round to the nearest valid length.
            video_frames = int(round((video_frames - 1) / 4)) * 4 + 1
        fps = int(_clamp(fps or d["fps"], 1, 60))
        steps = int(_clamp(steps or d["steps"], 1, 150))
        cfg = float(_clamp(cfg or d["cfg"], 0.0, 30.0))
        sampler = sampler or d["sampler"]
        scheduler = scheduler or d["scheduler"]
        motion_bucket_id = int(_clamp(motion_bucket_id, 1, 1023))
        augmentation_level = float(_clamp(augmentation_level, 0.0, 10.0))
        crf = int(_clamp(crf, 0, 51))
        interpolate = int(_clamp(interpolate, 0, 8)) or d["interpolate"]
        timeout = timeout or d["timeout"]
        seed = seed or random.randint(1, 2**32 - 1)
        negative_prompt = negative_prompt or (
            _WAN_NEGATIVE_PROMPT if backend in ("wan", "wan14b") else _SD_NEGATIVE_PROMPT
        )

        final_frames = (video_frames - 1) * interpolate + 1 if interpolate > 1 else video_frames
        final_fps = fps * interpolate if interpolate > 1 else fps
        duration_seconds = round(final_frames / final_fps, 2)

        # Ordered ComfyUI candidates. An explicit comfyui_host kwarg is a single
        # target (no fallback); otherwise use the env-configured list, which
        # already ends in the local fallback.
        if comfyui_host.strip():
            comfyui_candidates = [(comfyui_host.strip(), int(comfyui_port or 8199))]
        else:
            comfyui_candidates = _comfyui_candidates_from_env()
        filename_prefix = f"missy_{backend}_{uuid.uuid4().hex[:8]}"

        try:
            from missy.gateway.client import PolicyHTTPClient
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"HTTP client unavailable: {exc}")

        try:
            # PolicyHTTPClient directly (not the create_client() factory,
            # which doesn't expose max_response_bytes) so the /view
            # download fallback isn't capped at the default 50 MB.
            with PolicyHTTPClient(
                session_id="video_generate_tool",
                task_id="video_generate",
                timeout=timeout,
                category="tool",
                max_response_bytes=_MAX_RESPONSE_BYTES,
            ) as http:
                # Pick the first reachable, GPU-capable candidate, falling back
                # to the local server when a configured remote is down.
                base_url = ""
                gpu: dict[str, Any] = {}
                preflight_errors: list[str] = []
                for idx, (cand_host, cand_port) in enumerate(comfyui_candidates):
                    cand_url = f"http://{cand_host}:{cand_port}"
                    probe = self._preflight_gpu(http, cand_url, allow_cpu)
                    if isinstance(probe, dict):
                        base_url, gpu = cand_url, probe
                        if idx > 0:
                            logger.warning(
                                "video_generate: ComfyUI fell back to %s (earlier "
                                "candidate(s) unavailable: %s)",
                                cand_url,
                                "; ".join(preflight_errors),
                            )
                            gpu["fallback_from"] = [f"{h}:{p}" for h, p in comfyui_candidates[:idx]]
                        break
                    preflight_errors.append(f"{cand_url}: {probe}")
                if not base_url:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No usable ComfyUI server. " + " | ".join(preflight_errors),
                    )

                required_models = self._required_models(
                    backend=backend,
                    checkpoint=checkpoint,
                    motion_module=motion_module,
                    audio_generation=bool(audio_prompt),
                    audio_model=audio_model,
                    interpolate=interpolate,
                    upscale=upscale,
                )
                models_error = self._check_models(http, base_url, required_models)
                if models_error:
                    if audio_prompt and audio_model == "stable-audio-3":
                        models_error += (
                            '; older ComfyUI installs can use '
                            'audio_model="stable-audio-open-1.0" as a legacy fallback.'
                        )
                    return ToolResult(success=False, output=None, error=models_error)

                uploaded_image = ""
                if image_path:
                    uploaded_image = self._upload_file(http, base_url, image_path)
                    if uploaded_image.startswith("ERROR:"):
                        return ToolResult(
                            success=False, output=None, error=uploaded_image[len("ERROR:") :]
                        )
                uploaded_audio = ""
                if audio_path:
                    uploaded_audio = self._upload_file(http, base_url, audio_path)
                    if uploaded_audio.startswith("ERROR:"):
                        return ToolResult(
                            success=False, output=None, error=uploaded_audio[len("ERROR:") :]
                        )

                if backend == "svd":
                    graph, image_ref = _build_svd_workflow(
                        ckpt_name=checkpoint or _SVD_CHECKPOINT,
                        image_name=uploaded_image,
                        width=width,
                        height=height,
                        video_frames=video_frames,
                        motion_bucket_id=motion_bucket_id,
                        fps=fps,
                        augmentation_level=augmentation_level,
                        steps=steps,
                        cfg=cfg,
                        sampler=sampler,
                        scheduler=scheduler,
                        seed=seed,
                    )
                elif backend == "animatediff":
                    graph, image_ref = _build_animatediff_workflow(
                        ckpt_name=checkpoint or _ANIMATEDIFF_CHECKPOINT,
                        motion_module=motion_module or _ANIMATEDIFF_MOTION_MODULE,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        video_frames=video_frames,
                        context_length=context_length,
                        context_overlap=context_overlap,
                        steps=steps,
                        cfg=cfg,
                        sampler=sampler,
                        scheduler=scheduler,
                        seed=seed,
                    )
                elif backend == "wan14b":
                    graph, image_ref = _build_wan22_14b_workflow(
                        diffusion_high=_WAN14B_T2V_HIGH,
                        diffusion_low=_WAN14B_T2V_LOW,
                        text_encoder=_WAN_TEXT_ENCODER,
                        vae_name=_WAN21_VAE,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        video_frames=video_frames,
                        steps=steps,
                        cfg=cfg,
                        sampler=sampler,
                        scheduler=scheduler,
                        seed=seed,
                    )
                else:
                    graph, image_ref = _build_wan_workflow(
                        diffusion_model=checkpoint or _WAN_DIFFUSION_MODEL,
                        text_encoder=_WAN_TEXT_ENCODER,
                        vae_name=_WAN_VAE,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image_name=uploaded_image,
                        width=width,
                        height=height,
                        video_frames=video_frames,
                        steps=steps,
                        cfg=cfg,
                        sampler=sampler,
                        scheduler=scheduler,
                        seed=seed,
                    )

                if upscale:
                    image_ref = _append_upscale(graph, image_ref)
                if interpolate > 1:
                    image_ref = _append_interpolation(graph, image_ref, interpolate)

                audio_ref = None
                audio_info: dict[str, Any] | None = None
                if audio_prompt:
                    audio_builder = (
                        _append_audio_generation
                        if audio_model == "stable-audio-open-1.0"
                        else _append_audio_generation_sa3
                    )
                    audio_ref = audio_builder(
                        graph,
                        audio_prompt=audio_prompt,
                        audio_negative_prompt=audio_negative_prompt,
                        seconds=max(1.0, duration_seconds),
                        steps=int(_clamp(audio_steps, 1, 200)),
                        cfg=float(_clamp(audio_cfg, 0.0, 30.0)),
                        seed=seed,
                    )
                    audio_info = {
                        "source": "generated",
                        "prompt": audio_prompt,
                        "model": audio_model,
                    }
                elif audio_path:
                    audio_ref = _append_audio_file(graph, uploaded_audio)
                    audio_info = {"source": "file", "path": audio_path}

                _append_video_combine(
                    graph,
                    image_ref,
                    audio_ref=audio_ref,
                    frame_rate=final_fps,
                    filename_prefix=filename_prefix,
                    video_format=video_format,
                    crf=crf,
                )

                client_id = f"missy-{uuid.uuid4().hex[:12]}"
                submit_resp = http.post(
                    f"{base_url}/prompt",
                    json={"prompt": graph, "client_id": client_id},
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

            out_file = Path(final_path)
            return ToolResult(
                success=True,
                output={
                    "path": final_path,
                    "backend": backend,
                    "frames": final_frames,
                    "frames_generated": video_frames,
                    "fps": video_info.get("frame_rate", final_fps),
                    "duration_seconds": duration_seconds,
                    "width": width * (2 if upscale else 1),
                    "height": height * (2 if upscale else 1),
                    "seed": seed,
                    "steps": steps,
                    "motion_bucket_id": motion_bucket_id if backend == "svd" else None,
                    "augmentation_level": augmentation_level if backend == "svd" else None,
                    "interpolate": interpolate,
                    "upscale": upscale,
                    "video_format": video_format,
                    "audio": audio_info,
                    "gpu": gpu,
                    "comfyui_host": base_url,
                    "size_bytes": out_file.stat().st_size if out_file.is_file() else 0,
                    "prompt_id": prompt_id,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
            )
        except Exception as exc:
            logger.exception("video_generate failed")
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------
    # Preflight helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preflight_gpu(
        http: Any, base_url: str, allow_cpu: bool, probe_timeout: float = 8.0
    ) -> dict[str, Any] | str:
        """Verify the ComfyUI server is reachable and GPU-accelerated.

        Uses a short ``probe_timeout`` (not the full generation timeout) so an
        unreachable candidate fails fast, letting the caller fall back to the
        next ComfyUI without hanging.

        Returns:
            A gpu-info dict on success, or a human-readable error string
            (the caller distinguishes the two by type).
        """
        try:
            resp = http.get(f"{base_url}/system_stats", timeout=probe_timeout)
        except Exception as exc:
            return f"ComfyUI server unreachable at {base_url}: {exc}"
        if resp.status_code != 200:
            return f"ComfyUI /system_stats returned HTTP {resp.status_code}."
        devices = resp.json().get("devices", [])
        accel = [dev for dev in devices if dev.get("type") in ("cuda", "rocm", "mps", "xpu")]
        if not accel and not allow_cpu:
            return (
                "ComfyUI reports no GPU device (CPU-only inference would be "
                "~50x slower). Fix the server's CUDA setup, or pass "
                "allow_cpu=True to proceed anyway."
            )
        dev = accel[0] if accel else (devices[0] if devices else {})
        return {
            "name": dev.get("name", "unknown"),
            "type": dev.get("type", "cpu"),
            "vram_total_gb": round(dev.get("vram_total", 0) / 2**30, 1),
        }

    @staticmethod
    def _required_models(
        *,
        backend: str,
        checkpoint: str,
        motion_module: str,
        audio_generation: bool,
        interpolate: int,
        upscale: bool,
        audio_model: str = _DEFAULT_AUDIO_MODEL,
    ) -> list[tuple[str, str]]:
        """List the ``(model_folder, filename)`` pairs this run needs."""
        required: list[tuple[str, str]] = []
        if backend == "svd":
            required.append(("checkpoints", checkpoint or _SVD_CHECKPOINT))
        elif backend == "animatediff":
            required.append(("checkpoints", checkpoint or _ANIMATEDIFF_CHECKPOINT))
            required.append(("animatediff_models", motion_module or _ANIMATEDIFF_MOTION_MODULE))
        elif backend == "wan14b":
            required.append(("diffusion_models", _WAN14B_T2V_HIGH))
            required.append(("diffusion_models", _WAN14B_T2V_LOW))
            required.append(("text_encoders", _WAN_TEXT_ENCODER))
            required.append(("vae", _WAN21_VAE))
        else:
            required.append(("diffusion_models", checkpoint or _WAN_DIFFUSION_MODEL))
            required.append(("text_encoders", _WAN_TEXT_ENCODER))
            required.append(("vae", _WAN_VAE))
        if audio_generation:
            if audio_model == "stable-audio-open-1.0":
                required.append(("checkpoints", _AUDIO_CHECKPOINT))
                required.append(("text_encoders", _AUDIO_TEXT_ENCODER))
            else:
                required.append(("checkpoints", _AUDIO3_CHECKPOINT))
                required.append(("text_encoders", _AUDIO3_TEXT_ENCODER))
        if interpolate > 1:
            required.append(("frame_interpolation", _INTERPOLATION_MODEL))
        if upscale:
            required.append(("upscale_models", _UPSCALE_MODEL))
        return required

    @staticmethod
    def _check_models(http: Any, base_url: str, required: list[tuple[str, str]]) -> str | None:
        """Return an actionable error if a required model file is missing.

        Best-effort: if the ``/models/{folder}`` endpoint is unavailable
        the check is skipped and ComfyUI's own workflow validation is
        relied on instead.
        """
        missing: list[str] = []
        cache: dict[str, list[str] | None] = {}
        for folder, filename in required:
            if folder not in cache:
                try:
                    resp = http.get(f"{base_url}/models/{folder}")
                    cache[folder] = resp.json() if resp.status_code == 200 else None
                except Exception:
                    cache[folder] = None
            listing = cache[folder]
            if listing is None:
                continue
            if filename not in listing:
                source = _MODEL_SOURCES.get(filename)
                hint = f" (download from {source})" if source else ""
                missing.append(f"models/{folder}/{filename}{hint}")
        if missing:
            return "ComfyUI is missing required model file(s): " + "; ".join(missing)
        return None

    @staticmethod
    def _upload_file(http: Any, base_url: str, local_path: str) -> str:
        """Upload a local file into ComfyUI's input directory.

        Returns:
            The server-side filename, or an ``"ERROR:"``-prefixed message.
        """
        src = Path(local_path).expanduser()
        mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        with open(src, "rb") as f:
            resp = http.post(
                f"{base_url}/upload/image",
                files={"image": (src.name, f, mime)},
            )
        if resp.status_code != 200:
            return f"ERROR:ComfyUI upload of {src.name} failed: HTTP {resp.status_code}"
        name = resp.json().get("name")
        if not name:
            return "ERROR:ComfyUI upload response missing filename."
        return name

    # ------------------------------------------------------------------
    # Submission lifecycle helpers
    # ------------------------------------------------------------------

    @classmethod
    def _wait_for_completion(
        cls, http: Any, base_url: str, prompt_id: str, timeout: int
    ) -> dict[str, Any] | str:
        """Poll ``/history/{prompt_id}`` until completion, error, or timeout.

        On timeout the job is cancelled server-side (dequeued if still
        pending, interrupted if currently running) so it doesn't keep
        occupying the GPU after this tool has given up on it.

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
        queue_note = cls._cancel(http, base_url, prompt_id)
        return (
            f"Timed out after {timeout}s waiting for ComfyUI to finish "
            f"generating; the job was {queue_note}."
        )

    @staticmethod
    def _cancel(http: Any, base_url: str, prompt_id: str) -> str:
        """Cancel a queued or running prompt so it stops using the GPU.

        Deletes the prompt from the pending queue, and interrupts the
        server's current execution only when the running item is actually
        ours (``/interrupt`` is global, so blindly calling it could kill
        someone else's job).

        Returns:
            A short description of what was done, for the error message.
        """
        try:
            queue_resp = http.get(f"{base_url}/queue")
            queue = queue_resp.json() if queue_resp.status_code == 200 else {}
            running_ids = {item[1] for item in queue.get("queue_running", []) if len(item) > 1}
            http.post(f"{base_url}/queue", json={"delete": [prompt_id]})
            if prompt_id in running_ids:
                http.post(f"{base_url}/interrupt", json={})
                return "interrupted mid-run and cancelled"
            return "removed from the queue"
        except Exception as exc:  # cancellation is best-effort
            logger.warning("failed to cancel ComfyUI prompt %s: %s", prompt_id, exc)
            return f"left running (cancel failed: {exc})"

    @staticmethod
    def _retrieve_video(
        http: Any, base_url: str, video_info: dict[str, Any], save_path: str
    ) -> str:
        """Copy the generated video to a local Missy-managed path.

        Prefers a direct filesystem copy via ``video_info["fullpath"]``
        (ComfyUI and Missy share a filesystem in the common local-server
        deployment this tool targets), falling back to an HTTP download
        via ``/view`` if that path doesn't exist -- e.g. ComfyUI running
        on a different host. Never overwrites an existing file: a numeric
        suffix is appended on collision.
        """
        if save_path:
            dest = Path(save_path).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(video_info.get("filename", "video.mp4")).suffix or ".mp4"
            dest = Path(_DEFAULT_OUTPUT_DIR) / f"video_{ts}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        base = dest
        counter = 1
        while dest.exists():
            dest = base.with_name(f"{base.stem}_{counter}{base.suffix}")
            counter += 1

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
                            "'wan' (recommended): 24fps text-to-video, or image-to-video "
                            "when image_path is given. 'svd': animates an input image. "
                            "'animatediff': legacy text-to-video."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Text prompt describing the video. Required for backend "
                            "'wan' and 'animatediff'."
                        ),
                    },
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Local path to a source image. Required for 'svd'; optional "
                            "for 'wan' (switches it to image-to-video)."
                        ),
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "What to avoid (defaults to a per-backend standard).",
                    },
                    "audio_prompt": {
                        "type": "string",
                        "description": (
                            "Describe a soundtrack (music/sfx/ambience) to generate and "
                            "mux into the video. Works with every backend."
                        ),
                    },
                    "audio_negative_prompt": {
                        "type": "string",
                        "description": "What the generated soundtrack should avoid.",
                    },
                    "audio_path": {
                        "type": "string",
                        "description": (
                            "Local audio file to mux into the video instead of "
                            "generating one (mutually exclusive with audio_prompt)."
                        ),
                    },
                    "audio_steps": {
                        "type": "integer",
                        "description": "Audio generation sampling steps (default: 50).",
                    },
                    "audio_cfg": {
                        "type": "number",
                        "description": "Audio generation CFG scale (default: 5.0).",
                    },
                    "audio_model": {
                        "type": "string",
                        "enum": sorted(_VALID_AUDIO_MODELS),
                        "description": (
                            "Text-to-audio model for audio_prompt. 'stable-audio-3' "
                            "(default, recommended) or 'stable-audio-open-1.0' (legacy "
                            "fallback for ComfyUI installs without SA3 support)."
                        ),
                    },
                    "checkpoint": {
                        "type": "string",
                        "description": (
                            "Override the backend's default model filename (must already "
                            "exist in the matching ComfyUI models folder)."
                        ),
                    },
                    "motion_module": {
                        "type": "string",
                        "description": (
                            "animatediff only: motion module filename (default: mm_sd_v15_v2.ckpt)."
                        ),
                    },
                    "video_frames": {
                        "type": "integer",
                        "description": (
                            "Frames to generate before interpolation (default: 81 wan, "
                            "25 svd, 16 animatediff)."
                        ),
                    },
                    "fps": {
                        "type": "integer",
                        "description": (
                            "Base frame rate before interpolation (default: 24 wan, "
                            "6 svd, 8 animatediff)."
                        ),
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
                        "description": "Output width (default: 832 wan, 1024 svd, 512 animatediff).",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Output height (default: 480 wan, 576 svd, 512 animatediff).",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Sampling steps (default: 20 wan/svd, 25 animatediff).",
                    },
                    "cfg": {
                        "type": "number",
                        "description": (
                            "Classifier-free guidance scale (default: 5.0 wan, 2.5 svd, "
                            "7.5 animatediff)."
                        ),
                    },
                    "sampler": {
                        "type": "string",
                        "description": (
                            "Sampler override (default: uni_pc wan, euler svd, "
                            "dpmpp_2m animatediff)."
                        ),
                    },
                    "scheduler": {
                        "type": "string",
                        "description": (
                            "Scheduler override (default: simple wan, karras svd/animatediff)."
                        ),
                    },
                    "seed": {
                        "type": "integer",
                        "description": (
                            "Sampling seed. 0 (default) picks a random seed; the "
                            "effective seed is echoed back in the result for reproducing "
                            "or refining a clip."
                        ),
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "animatediff only: sliding-context window size (default: 16).",
                    },
                    "context_overlap": {
                        "type": "integer",
                        "description": "animatediff only: sliding-context overlap (default: 4).",
                    },
                    "interpolate": {
                        "type": "integer",
                        "description": (
                            "RIFE frame-interpolation multiplier: 0 = auto (4x svd -> "
                            "24fps, 2x animatediff, off for wan), 1 = off, 2-8 = "
                            "multiply frame rate."
                        ),
                    },
                    "upscale": {
                        "type": "boolean",
                        "description": "Apply a 2x RealESRGAN upscale pass (default: false).",
                    },
                    "video_format": {
                        "type": "string",
                        "enum": sorted(_VIDEO_FORMATS),
                        "description": (
                            "Output encoding: h264-mp4 (default), h265-mp4, or "
                            "nvenc_h264-mp4 (GPU encode)."
                        ),
                    },
                    "crf": {
                        "type": "integer",
                        "description": "Encoder quality, lower = better (default: 17; ignored for nvenc).",
                    },
                    "allow_cpu": {
                        "type": "boolean",
                        "description": (
                            "Permit generation on a ComfyUI server without a GPU "
                            "(default: false -- refused as ~50x slower)."
                        ),
                    },
                    "save_path": {
                        "type": "string",
                        "description": (
                            "Optional destination path for the final video (default: a "
                            "timestamped file under ~/.missy/videos/). Existing files "
                            "are never overwritten."
                        ),
                    },
                    # comfyui_host / comfyui_port are deliberately NOT advertised
                    # to the model: which ComfyUI server to use is a deployment
                    # concern set via MISSY_COMFYUI_HOST/PORT (or the execute()
                    # kwargs for tests), not something the model should choose.
                    # Exposing them invited the model to pass "127.0.0.1",
                    # overriding a configured remote host.
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Max seconds to wait for generation (default 0 = auto: "
                            "3600 wan, 1200 others). The job is cancelled server-side "
                            "on timeout."
                        ),
                    },
                },
                "required": ["backend"],
            },
        }
