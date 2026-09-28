"""Built-in tool: generate sung music with ACE-Step 1.5 via ComfyUI.

This tool deliberately sits beside :mod:`video_generate` instead of adding a
``lyrics`` switch to Stable Audio.  Text-to-audio and singing synthesis have
different input contracts and should never silently substitute for one
another.  The initial backend is ComfyUI's native ACE-Step 1.5 text-to-music
workflow, which accepts lyrics plus musical controls and produces a complete
song mix.

ACE-Step's text-to-music workflow does not produce isolated stems or phoneme
timings.  The result therefore reports those artifacts as unavailable rather
than pretending that the full mix is safe lip-sync input.  A later stem or
score-conditioned backend can fill the same explicit output slots.
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
from missy.tools.builtin.video_generate import (
    VideoGenerateTool,
    _comfyui_candidates_from_env,
)

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = str(Path.home() / ".missy" / "audio")
_MAX_RESPONSE_BYTES = 300 * 1024 * 1024

_ACE_MODEL = "acestep_v1.5_turbo.safetensors"
_ACE_TEXT_ENCODER_SMALL = "qwen_0.6b_ace15.safetensors"
_ACE_TEXT_ENCODER_LARGE = "qwen_4b_ace15.safetensors"
_ACE_VAE = "ace_1.5_vae.safetensors"

_MODEL_SOURCE = "huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files"
_REQUIRED_MODELS = (
    ("diffusion_models", _ACE_MODEL),
    ("text_encoders", _ACE_TEXT_ENCODER_SMALL),
    ("text_encoders", _ACE_TEXT_ENCODER_LARGE),
    ("vae", _ACE_VAE),
)
_REQUIRED_NODES = frozenset(
    {
        "ConditioningZeroOut",
        "DualCLIPLoader",
        "EmptyAceStep1.5LatentAudio",
        "KSampler",
        "ModelSamplingAuraFlow",
        "SaveAudio",
        "TextEncodeAceStepAudio1.5",
        "UNETLoader",
        "VAEDecodeAudio",
        "VAELoader",
    }
)

_TIME_SIGNATURES = frozenset({"2", "3", "4", "6"})
_LANGUAGES = frozenset(
    {
        "ar",
        "az",
        "bg",
        "bn",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fa",
        "fi",
        "fr",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "id",
        "is",
        "it",
        "ja",
        "ko",
        "la",
        "lt",
        "ms",
        "ne",
        "nl",
        "no",
        "pa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sa",
        "sk",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "th",
        "tl",
        "tr",
        "uk",
        "unknown",
        "ur",
        "vi",
        "yue",
        "zh",
    }
)
_KEY_SCALES = frozenset(
    f"{note} {mode}"
    for note in (
        "C",
        "C#",
        "Db",
        "D",
        "D#",
        "Eb",
        "E",
        "F",
        "F#",
        "Gb",
        "G",
        "G#",
        "Ab",
        "A",
        "A#",
        "Bb",
        "B",
    )
    for mode in ("major", "minor")
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _build_ace_step_workflow(
    *,
    lyrics: str,
    tags: str,
    duration: float,
    bpm: int,
    keyscale: str,
    time_signature: str,
    language: str,
    seed: int,
    steps: int,
    sampler_cfg: float,
    lyric_cfg: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    generate_audio_codes: bool,
    filename_prefix: str,
    diffusion_model: str = _ACE_MODEL,
    text_encoder_small: str = _ACE_TEXT_ENCODER_SMALL,
    text_encoder_large: str = _ACE_TEXT_ENCODER_LARGE,
    vae_name: str = _ACE_VAE,
) -> dict[str, Any]:
    """Build ComfyUI's native ACE-Step 1.5 text-to-music graph."""
    return {
        "model": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": diffusion_model, "weight_dtype": "default"},
        },
        # Keeping the language models on CPU leaves the 8 GB GPU available for
        # the diffusion model and VAE.  This is slower but avoids predictable
        # OOM failures on Missy's RTX 3070.
        "clip": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": text_encoder_small,
                "clip_name2": text_encoder_large,
                "type": "ace",
                "device": "cpu",
            },
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "condition": {
            "class_type": "TextEncodeAceStepAudio1.5",
            "inputs": {
                "clip": ["clip", 0],
                "tags": tags,
                "lyrics": lyrics,
                "seed": seed,
                "bpm": bpm,
                "duration": duration,
                "timesignature": time_signature,
                "language": language,
                "keyscale": keyscale,
                "generate_audio_codes": generate_audio_codes,
                "cfg_scale": lyric_cfg,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
            },
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["condition", 0]},
        },
        "sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["model", 0], "shift": 3.0},
        },
        "latent": {
            "class_type": "EmptyAceStep1.5LatentAudio",
            "inputs": {"seconds": duration, "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["sampling", 0],
                "positive": ["condition", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "seed": seed,
                "steps": steps,
                "cfg": sampler_cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "decode": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
        },
        "output": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["decode", 0], "filename_prefix": filename_prefix},
        },
    }


def _extract_audio_output(history_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first ComfyUI audio descriptor in a history entry."""
    for node_output in history_entry.get("outputs", {}).values():
        audio = node_output.get("audio")
        if isinstance(audio, list) and audio:
            return audio[0]
        if isinstance(audio, dict):
            return audio
    return None


class SingingGenerateTool(BaseTool):
    """Generate a complete sung music mix with ACE-Step 1.5."""

    name = "singing_generate"
    description = (
        "Generate an actual sung song from lyrics and musical direction using "
        "ACE-Step 1.5. Supports style tags, duration, BPM, key, time signature, "
        "language, and reproducible seeds. This is for singing; do not use TTS "
        "or video_generate.audio_prompt as a substitute. The ACE text-to-music "
        "backend returns a complete mix, not isolated vocal/instrumental stems, "
        "so its mix must not be used as lip-sync input. Set validate_only=true "
        "to check GPU, ComfyUI node, and model readiness without rendering."
    )
    permissions = ToolPermissions(network=True, filesystem_write=True)

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = str(kwargs.get("comfyui_host") or "").strip()
        if host:
            return [f"{host}:{int(kwargs.get('comfyui_port') or 8199)}"]
        return [f"{host}:{port}" for host, port in _comfyui_candidates_from_env()]

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        save_path = str(kwargs.get("save_path") or "")
        return ([], [save_path] if save_path else [_DEFAULT_OUTPUT_DIR])

    @staticmethod
    def _check_nodes(http: Any, base_url: str) -> str | None:
        try:
            response = http.get(f"{base_url}/object_info")
            if response.status_code != 200:
                return f"ComfyUI /object_info returned HTTP {response.status_code}."
            available = set(response.json())
        except Exception as exc:
            return f"Could not inspect ComfyUI ACE-Step support: {exc}"
        missing = sorted(_REQUIRED_NODES - available)
        if missing:
            return (
                "ComfyUI is missing ACE-Step 1.5 node support: "
                + ", ".join(missing)
                + ". Update ComfyUI to a release with native ACE-Step 1.5 support."
            )
        return None

    @staticmethod
    def _check_models(
        http: Any, base_url: str, required: tuple[tuple[str, str], ...]
    ) -> str | None:
        missing: list[str] = []
        unavailable: list[str] = []
        listings: dict[str, list[str] | None] = {}
        for folder, filename in required:
            if folder not in listings:
                try:
                    response = http.get(f"{base_url}/models/{folder}")
                    listings[folder] = response.json() if response.status_code == 200 else None
                except Exception:
                    listings[folder] = None
                if listings[folder] is None:
                    unavailable.append(folder)
            listing = listings[folder]
            if listing is not None and filename not in listing:
                missing.append(f"models/{folder}/{filename}")
        if unavailable:
            return (
                "Could not verify ACE-Step model inventory for ComfyUI folder(s): "
                + ", ".join(sorted(set(unavailable)))
                + ". Update or repair the ComfyUI /models API before rendering."
            )
        if not missing:
            return None
        return (
            "ComfyUI is missing ACE-Step 1.5 model file(s): "
            + "; ".join(missing)
            + f". Download the official ComfyUI files from {_MODEL_SOURCE}."
        )

    @staticmethod
    def _destination(save_path: str, audio_info: dict[str, Any]) -> Path:
        if save_path:
            dest = Path(save_path).expanduser()
        else:
            suffix = Path(str(audio_info.get("filename") or "song.flac")).suffix or ".flac"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = Path(_DEFAULT_OUTPUT_DIR) / f"song_{stamp}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        original = dest
        index = 1
        while dest.exists():
            dest = original.with_name(f"{original.stem}_{index}{original.suffix}")
            index += 1
        return dest

    @classmethod
    def _retrieve_audio(
        cls, http: Any, base_url: str, audio_info: dict[str, Any], save_path: str
    ) -> str:
        dest = cls._destination(save_path, audio_info)
        fullpath = audio_info.get("fullpath")
        if fullpath and Path(str(fullpath)).is_file():
            shutil.copy2(str(fullpath), dest)
            return str(dest)
        response = http.get(
            f"{base_url}/view",
            params={
                "filename": audio_info.get("filename", ""),
                "subfolder": audio_info.get("subfolder", ""),
                "type": audio_info.get("type", "output"),
            },
        )
        response.raise_for_status()
        dest.write_bytes(response.content)
        return str(dest)

    def execute(
        self,
        *,
        lyrics: str = "",
        tags: str = "",
        duration: float = 30.0,
        bpm: int = 120,
        keyscale: str = "C major",
        time_signature: str = "4",
        language: str = "en",
        seed: int = 0,
        steps: int = 8,
        sampler_cfg: float = 1.0,
        lyric_cfg: float = 2.0,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        generate_audio_codes: bool = True,
        validate_only: bool = False,
        allow_cpu: bool = False,
        save_path: str = "",
        comfyui_host: str = "",
        comfyui_port: int = 0,
        timeout: int = 0,
        **_kwargs: Any,
    ) -> ToolResult:
        """Validate ACE-Step readiness or generate a song mix."""
        started = time.monotonic()
        lyrics = lyrics.strip()
        tags = tags.strip()
        keyscale = keyscale.strip()
        time_signature = str(time_signature).strip()
        language = language.strip().lower()

        if not validate_only and not lyrics:
            return ToolResult(success=False, output=None, error="lyrics must not be empty.")
        if not validate_only and not tags:
            return ToolResult(
                success=False,
                output=None,
                error="tags must describe the song's genre, instrumentation, and vocal style.",
            )
        if keyscale not in _KEY_SCALES:
            return ToolResult(
                success=False, output=None, error=f"Unsupported keyscale: {keyscale!r}."
            )
        if time_signature not in _TIME_SIGNATURES:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unsupported time_signature {time_signature!r}; use 2, 3, 4, or 6.",
            )
        if language not in _LANGUAGES:
            return ToolResult(
                success=False, output=None, error=f"Unsupported language: {language!r}."
            )

        duration = round(_clamp(float(duration), 5.0, 180.0), 2)
        bpm = int(_clamp(int(bpm), 10, 300))
        steps = int(_clamp(int(steps), 1, 50))
        sampler_cfg = float(_clamp(float(sampler_cfg), 0.0, 30.0))
        lyric_cfg = float(_clamp(float(lyric_cfg), 0.0, 30.0))
        temperature = float(_clamp(float(temperature), 0.0, 2.0))
        top_p = float(_clamp(float(top_p), 0.0, 1.0))
        top_k = int(_clamp(int(top_k), 0, 100))
        min_p = float(_clamp(float(min_p), 0.0, 1.0))
        timeout = int(timeout or 3600)
        seed = int(seed or random.randint(1, 2**32 - 1))

        candidates = (
            [(comfyui_host.strip(), int(comfyui_port or 8199))]
            if comfyui_host.strip()
            else _comfyui_candidates_from_env()
        )
        try:
            from missy.gateway.client import PolicyHTTPClient
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"HTTP client unavailable: {exc}")

        try:
            with PolicyHTTPClient(
                session_id="singing_generate_tool",
                task_id="singing_generate",
                timeout=timeout,
                category="tool",
                max_response_bytes=_MAX_RESPONSE_BYTES,
            ) as http:
                base_url = ""
                gpu: dict[str, Any] = {}
                errors: list[str] = []
                for index, (host, port) in enumerate(candidates):
                    candidate_url = f"http://{host}:{port}"
                    probe = VideoGenerateTool._preflight_gpu(http, candidate_url, allow_cpu)
                    if isinstance(probe, dict):
                        base_url, gpu = candidate_url, probe
                        if index:
                            gpu["fallback_from"] = [f"{h}:{p}" for h, p in candidates[:index]]
                        break
                    errors.append(f"{candidate_url}: {probe}")
                if not base_url:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No usable ComfyUI server. " + " | ".join(errors),
                    )

                nodes_error = self._check_nodes(http, base_url)
                if nodes_error:
                    return ToolResult(success=False, output=None, error=nodes_error)
                models_error = self._check_models(http, base_url, _REQUIRED_MODELS)
                if models_error:
                    return ToolResult(success=False, output=None, error=models_error)

                readiness = {
                    "ready": True,
                    "backend": "ace-step-1.5",
                    "gpu": gpu,
                    "comfyui_host": base_url,
                    "models": [filename for _, filename in _REQUIRED_MODELS],
                }
                if validate_only:
                    return ToolResult(success=True, output=readiness)

                filename_prefix = f"missy_singing_{uuid.uuid4().hex[:8]}"
                graph = _build_ace_step_workflow(
                    lyrics=lyrics,
                    tags=tags,
                    duration=duration,
                    bpm=bpm,
                    keyscale=keyscale,
                    time_signature=time_signature,
                    language=language,
                    seed=seed,
                    steps=steps,
                    sampler_cfg=sampler_cfg,
                    lyric_cfg=lyric_cfg,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    generate_audio_codes=bool(generate_audio_codes),
                    filename_prefix=filename_prefix,
                )
                response = http.post(
                    f"{base_url}/prompt",
                    json={"prompt": graph, "client_id": f"missy-{uuid.uuid4().hex[:12]}"},
                )
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"ComfyUI rejected the workflow: HTTP {response.status_code}: {response.text[:500]}",
                    )
                submitted = response.json()
                node_errors = submitted.get("node_errors") or {}
                if node_errors:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"ComfyUI workflow validation errors: {json.dumps(node_errors)[:1000]}",
                    )
                prompt_id = submitted.get("prompt_id")
                if not prompt_id:
                    return ToolResult(
                        success=False, output=None, error="ComfyUI did not return a prompt_id."
                    )

                history = VideoGenerateTool._wait_for_completion(
                    http, base_url, str(prompt_id), timeout
                )
                if isinstance(history, str):
                    return ToolResult(success=False, output=None, error=history)
                audio_info = _extract_audio_output(history)
                if audio_info is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="ComfyUI completed but produced no SaveAudio output.",
                    )
                mix_path = self._retrieve_audio(http, base_url, audio_info, save_path)

            manifest = {
                "backend": "ace-step-1.5",
                "mix_path": mix_path,
                "lyrics": lyrics,
                "tags": tags,
                "duration_seconds": duration,
                "bpm": bpm,
                "keyscale": keyscale,
                "time_signature": time_signature,
                "language": language,
                "seed": seed,
                "steps": steps,
                "sampler_cfg": sampler_cfg,
                "lyric_cfg": lyric_cfg,
                "generate_audio_codes": bool(generate_audio_codes),
                "models": readiness["models"],
                "prompt_id": prompt_id,
            }
            manifest_path = Path(mix_path).with_suffix(Path(mix_path).suffix + ".manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            output_file = Path(mix_path)
            return ToolResult(
                success=True,
                output={
                    **readiness,
                    "path": mix_path,
                    "mix_path": mix_path,
                    "vocal_path": None,
                    "instrumental_path": None,
                    "timing_path": None,
                    "lip_sync_audio_path": None,
                    "stem_status": "not_available_from_ace_text_to_music",
                    "lip_sync_ready": False,
                    "manifest_path": str(manifest_path),
                    "duration_seconds": duration,
                    "bpm": bpm,
                    "keyscale": keyscale,
                    "time_signature": time_signature,
                    "language": language,
                    "seed": seed,
                    "steps": steps,
                    "prompt_id": prompt_id,
                    "size_bytes": output_file.stat().st_size if output_file.is_file() else 0,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
            )
        except Exception as exc:
            logger.exception("singing_generate failed")
            return ToolResult(success=False, output=None, error=str(exc))

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "lyrics": {
                        "type": "string",
                        "description": "Lyrics to sing. Use section labels such as [Verse] and [Chorus].",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Genre, instrumentation, mood, tempo feel, and vocal style.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Song length in seconds, clamped to 5-180 (default 30).",
                    },
                    "bpm": {"type": "integer", "description": "Tempo, 10-300 BPM."},
                    "keyscale": {
                        "type": "string",
                        "enum": sorted(_KEY_SCALES),
                        "description": "Musical key and major/minor scale.",
                    },
                    "time_signature": {
                        "type": "string",
                        "enum": sorted(_TIME_SIGNATURES),
                        "description": "Beat count used by ACE-Step: 2, 3, 4, or 6.",
                    },
                    "language": {
                        "type": "string",
                        "enum": sorted(_LANGUAGES),
                        "description": "Lyrics language code (default en).",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Reproducible seed; 0 selects a random seed.",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Diffusion steps (ACE turbo default 8).",
                    },
                    "generate_audio_codes": {
                        "type": "boolean",
                        "description": "Use ACE's language model planning for higher quality (default true).",
                    },
                    "validate_only": {
                        "type": "boolean",
                        "description": "Only verify GPU, nodes, and model files; do not render.",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional destination; defaults under ~/.missy/audio/.",
                    },
                    "comfyui_host": {
                        "type": "string",
                        "description": "Explicit ComfyUI host; normally use configured fallback selection.",
                    },
                    "comfyui_port": {"type": "integer", "description": "Explicit host port."},
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum render seconds; default 3600.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }
