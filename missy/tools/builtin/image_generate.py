"""Built-in tool: generate still images with the configured ComfyUI GPU.

The video generator cannot satisfy a request for a still image: it is slower,
returns the wrong media type, and requires the caller to invent an extra frame
extraction step.  This tool submits a native Stable Diffusion image workflow,
waits for ComfyUI, downloads the PNG into Missy's managed image directory, and
returns the real path and reproducibility parameters.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin.video_generate import (
    VideoGenerateTool,
    _comfyui_candidates_from_env,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
_DEFAULT_OUTPUT_DIR = str(Path.home() / ".missy" / "images")
_DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted anatomy, extra limbs, duplicate, watermark, "
    "signature, text, jpeg artifacts"
)
_MAX_RESPONSE_BYTES = 50 * 1024 * 1024


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _snap_dimension(value: int) -> int:
    return int(round(_clamp(value, 256, 1024) / 64)) * 64


def _build_image_workflow(
    *,
    checkpoint: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    batch_size: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Return a core-node ComfyUI text-to-image workflow."""
    return {
        "load_checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["load_checkpoint", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["load_checkpoint", 1]},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["load_checkpoint", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["load_checkpoint", 2]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
        },
    }


def _extract_image_outputs(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all ComfyUI image descriptors produced by the workflow."""
    images: list[dict[str, Any]] = []
    for node_output in history_entry.get("outputs", {}).values():
        for image in node_output.get("images", []):
            if isinstance(image, dict) and image.get("filename"):
                images.append(image)
    return images


class ImageGenerateTool(BaseTool):
    """Generate one or more still PNG images with ComfyUI."""

    name = "image_generate"
    description = (
        "Generate a still image from a text prompt using the GPU-backed ComfyUI "
        "Stable Diffusion workflow. Use this for requests to create, draw, render, "
        "or generate an image, picture, illustration, artwork, poster, or wallpaper; "
        "do not use video_generate for a still image. Returns actual local PNG path(s), "
        "the seed, dimensions, checkpoint, and generation metadata. On Discord, call "
        "discord_upload_file with the returned path to deliver the image. Unless the "
        "user explicitly requests different generation settings, use the published "
        "defaults exactly: 512x512, 28 steps, cfg 7.0, dpmpp_2m/karras, one image, "
        "and a random seed."
    )
    permissions = ToolPermissions(network=True, filesystem_write=True)

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = str(kwargs.get("comfyui_host") or "").strip()
        if host:
            return [f"{host}:{int(kwargs.get('comfyui_port') or 8199)}"]
        return [f"{host}:{port}" for host, port in _comfyui_candidates_from_env()]

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        save_path = str(kwargs.get("save_path") or "").strip()
        return ([], [save_path] if save_path else [_DEFAULT_OUTPUT_DIR])

    def execute(
        self,
        *,
        prompt: str,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        width: int = 512,
        height: int = 512,
        steps: int = 28,
        cfg: float = 7.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        seed: int = 0,
        batch_size: int = 1,
        allow_cpu: bool = False,
        save_path: str = "",
        comfyui_host: str = "",
        comfyui_port: int = 0,
        timeout: int = 600,
        **_kwargs: Any,
    ) -> ToolResult:
        """Generate still image files and return their real local paths."""
        started = time.monotonic()
        prompt = str(prompt or "").strip()
        if not prompt:
            return ToolResult(
                success=False, output=None, error="prompt must be a non-empty string."
            )
        checkpoint = str(checkpoint or _DEFAULT_CHECKPOINT).strip()
        width = _snap_dimension(int(width))
        height = _snap_dimension(int(height))
        steps = int(_clamp(int(steps), 1, 100))
        cfg = float(_clamp(float(cfg), 0.0, 30.0))
        batch_size = int(_clamp(int(batch_size), 1, 4))
        timeout = int(_clamp(int(timeout), 10, 3600))
        seed = int(seed) or random.randint(1, 2**32 - 1)

        candidates = (
            [(comfyui_host.strip(), int(comfyui_port or 8199))]
            if comfyui_host.strip()
            else _comfyui_candidates_from_env()
        )
        filename_prefix = f"missy_image_{uuid.uuid4().hex[:8]}"

        try:
            from missy.gateway.client import PolicyHTTPClient
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"HTTP client unavailable: {exc}")

        try:
            with PolicyHTTPClient(
                session_id="image_generate_tool",
                task_id="image_generate",
                timeout=timeout,
                category="tool",
                max_response_bytes=_MAX_RESPONSE_BYTES,
            ) as http:
                base_url = ""
                gpu: dict[str, Any] = {}
                preflight_errors: list[str] = []
                required_models = [("checkpoints", checkpoint)]
                for index, (host, port) in enumerate(candidates):
                    candidate_url = f"http://{host}:{port}"
                    probe = VideoGenerateTool._preflight_gpu(http, candidate_url, allow_cpu)
                    if not isinstance(probe, dict):
                        preflight_errors.append(f"{candidate_url}: {probe}")
                        continue

                    model_error = VideoGenerateTool._check_models(
                        http, candidate_url, required_models
                    )
                    if model_error:
                        preflight_errors.append(f"{candidate_url}: {model_error}")
                        continue

                    base_url, gpu = candidate_url, probe
                    if index:
                        logger.warning(
                            "image_generate: ComfyUI fell back to %s (earlier "
                            "candidate(s) unusable: %s)",
                            candidate_url,
                            "; ".join(preflight_errors),
                        )
                        gpu["fallback_from"] = [f"{h}:{p}" for h, p in candidates[:index]]
                    break
                if not base_url:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No usable ComfyUI server. " + " | ".join(preflight_errors),
                    )

                graph = _build_image_workflow(
                    checkpoint=checkpoint,
                    prompt=prompt,
                    negative_prompt=negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    sampler=sampler,
                    scheduler=scheduler,
                    seed=seed,
                    batch_size=batch_size,
                    filename_prefix=filename_prefix,
                )
                submit_response = http.post(
                    f"{base_url}/prompt",
                    json={"prompt": graph, "client_id": f"missy-{uuid.uuid4().hex[:12]}"},
                )
                if submit_response.status_code != 200:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"ComfyUI rejected the image workflow: HTTP "
                            f"{submit_response.status_code}: {submit_response.text[:500]}"
                        ),
                    )
                submit_data = submit_response.json()
                node_errors = submit_data.get("node_errors") or {}
                if node_errors:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            "ComfyUI image workflow validation errors: "
                            f"{json.dumps(node_errors)[:1000]}"
                        ),
                    )
                prompt_id = submit_data.get("prompt_id")
                if not prompt_id:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="ComfyUI did not return a prompt_id for the image workflow.",
                    )

                history_entry = VideoGenerateTool._wait_for_completion(
                    http, base_url, prompt_id, timeout
                )
                if isinstance(history_entry, str):
                    return ToolResult(success=False, output=None, error=history_entry)
                image_outputs = _extract_image_outputs(history_entry)
                if not image_outputs:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="ComfyUI completed but produced no image output.",
                    )
                paths = self._retrieve_images(http, base_url, image_outputs, save_path)

            return ToolResult(
                success=True,
                output={
                    "path": paths[0],
                    "paths": paths,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler": sampler,
                    "scheduler": scheduler,
                    "checkpoint": checkpoint,
                    "gpu": gpu,
                    "comfyui_host": base_url,
                    "prompt_id": prompt_id,
                    "size_bytes": sum(Path(path).stat().st_size for path in paths),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
            )
        except Exception as exc:
            logger.exception("image_generate failed")
            return ToolResult(success=False, output=None, error=str(exc))

    @staticmethod
    def _retrieve_images(
        http: Any,
        base_url: str,
        image_outputs: list[dict[str, Any]],
        save_path: str,
    ) -> list[str]:
        paths: list[str] = []
        requested = Path(save_path).expanduser() if save_path else None
        for index, image_info in enumerate(image_outputs):
            if requested is not None:
                destination = requested
                if index:
                    destination = requested.with_name(
                        f"{requested.stem}_{index + 1}{requested.suffix or '.png'}"
                    )
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix = "" if index == 0 else f"_{index + 1}"
                destination = Path(_DEFAULT_OUTPUT_DIR) / f"image_{timestamp}{suffix}.png"
            if not destination.suffix:
                destination = destination.with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            base_destination = destination
            collision = 1
            while destination.exists():
                destination = base_destination.with_name(
                    f"{base_destination.stem}_{collision}{base_destination.suffix}"
                )
                collision += 1

            response = http.get(
                f"{base_url}/view",
                params={
                    "filename": image_info.get("filename", ""),
                    "subfolder": image_info.get("subfolder", ""),
                    "type": image_info.get("type", "output"),
                },
            )
            response.raise_for_status()
            destination.write_bytes(response.content)
            paths.append(str(destination))
        return paths

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Describe the still image."},
                    "negative_prompt": {
                        "type": "string",
                        "description": "Visual qualities or objects to avoid; omit to use the built-in default.",
                        "default": _DEFAULT_NEGATIVE_PROMPT,
                    },
                    "checkpoint": {
                        "type": "string",
                        "description": f"ComfyUI checkpoint filename (default {_DEFAULT_CHECKPOINT}).",
                        "default": _DEFAULT_CHECKPOINT,
                    },
                    "width": {
                        "type": "integer",
                        "minimum": 256,
                        "maximum": 1024,
                        "description": (
                            "Output width in pixels. Use 512 unless the user explicitly "
                            "requests a different width."
                        ),
                        "default": 512,
                    },
                    "height": {
                        "type": "integer",
                        "minimum": 256,
                        "maximum": 1024,
                        "description": (
                            "Output height in pixels. Use 512 unless the user explicitly "
                            "requests a different height."
                        ),
                        "default": 512,
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Sampling steps; omit to use 28.",
                        "default": 28,
                    },
                    "cfg": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 30,
                        "description": "Classifier-free guidance scale; omit to use 7.0.",
                        "default": 7.0,
                    },
                    "sampler": {
                        "type": "string",
                        "description": "ComfyUI sampler; omit to use dpmpp_2m.",
                        "default": "dpmpp_2m",
                    },
                    "scheduler": {
                        "type": "string",
                        "description": "ComfyUI scheduler; omit to use karras.",
                        "default": "karras",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Use 0 for a random seed; reuse a returned seed to reproduce.",
                        "default": 0,
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "description": "Number of image variations; omit to generate one.",
                        "default": 1,
                    },
                    "allow_cpu": {
                        "type": "boolean",
                        "description": "Allow CPU-only generation; disabled by default.",
                        "default": False,
                    },
                    "save_path": {"type": "string"},
                    "timeout": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 3600,
                        "description": "Generation timeout in seconds; omit to use 600.",
                        "default": 600,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        }
