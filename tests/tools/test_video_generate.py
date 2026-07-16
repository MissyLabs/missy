"""Tests for missy.tools.builtin.video_generate.VideoGenerateTool.

All tests mock missy.gateway.client.PolicyHTTPClient to avoid real network
calls, a real ComfyUI server, or policy-engine initialisation. The
workflow-graph builders and history/video extraction helpers are also
tested directly against representative ComfyUI API response shapes
(captured from a real, live ComfyUI instance during development).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.tools.builtin.video_generate import (
    VideoGenerateTool,
    _append_audio_file,
    _append_audio_generation,
    _append_audio_generation_sa3,
    _append_interpolation,
    _append_upscale,
    _append_video_combine,
    _build_animatediff_workflow,
    _build_svd_workflow,
    _build_wan_workflow,
    _extract_video_output,
)

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _assert_wiring_consistent(graph: dict) -> None:
    """Every [node_id, output_index] reference must point at a real node."""
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph, f"dangling reference to node {value[0]!r}"


def _class_types(graph: dict) -> set[str]:
    return {node["class_type"] for node in graph.values()}


# ---------------------------------------------------------------------------
# Workflow graph builders
# ---------------------------------------------------------------------------


class TestBuildSvdWorkflow:
    def _build(self, **overrides):
        kwargs = {
            "ckpt_name": "svd_xt.safetensors",
            "image_name": "uploaded.png",
            "width": 1024,
            "height": 576,
            "video_frames": 25,
            "motion_bucket_id": 127,
            "fps": 6,
            "augmentation_level": 0.0,
            "steps": 20,
            "cfg": 2.5,
            "sampler": "euler",
            "scheduler": "karras",
            "seed": 42,
        }
        kwargs.update(overrides)
        return _build_svd_workflow(**kwargs)

    def test_produces_expected_node_types(self) -> None:
        graph, image_ref = self._build()
        assert _class_types(graph) == {
            "ImageOnlyCheckpointLoader",
            "LoadImage",
            "SVD_img2vid_Conditioning",
            "VideoLinearCFGGuidance",
            "KSampler",
            "VAEDecode",
        }
        assert graph[image_ref[0]]["class_type"] == "VAEDecode"
        _assert_wiring_consistent(graph)

    def test_parameters_threaded_through(self) -> None:
        graph, _ = self._build(
            ckpt_name="svd.safetensors",
            image_name="my_photo.png",
            video_frames=14,
            motion_bucket_id=200,
            fps=8,
            augmentation_level=0.5,
            seed=999,
        )
        loader = next(n for n in graph.values() if n["class_type"] == "ImageOnlyCheckpointLoader")
        assert loader["inputs"]["ckpt_name"] == "svd.safetensors"
        load_image = next(n for n in graph.values() if n["class_type"] == "LoadImage")
        assert load_image["inputs"]["image"] == "my_photo.png"
        cond = next(n for n in graph.values() if n["class_type"] == "SVD_img2vid_Conditioning")
        assert cond["inputs"]["video_frames"] == 14
        assert cond["inputs"]["motion_bucket_id"] == 200
        assert cond["inputs"]["fps"] == 8
        assert cond["inputs"]["augmentation_level"] == 0.5
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        assert sampler["inputs"]["seed"] == 999


class TestBuildAnimatediffWorkflow:
    def _build(self, **overrides):
        kwargs = {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors",
            "motion_module": "mm_sd_v15_v2.ckpt",
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "video_frames": 16,
            "context_length": 16,
            "context_overlap": 4,
            "steps": 25,
            "cfg": 7.5,
            "sampler": "dpmpp_2m",
            "scheduler": "karras",
            "seed": 1,
        }
        kwargs.update(overrides)
        return _build_animatediff_workflow(**kwargs)

    def test_produces_expected_node_types_with_freeu(self) -> None:
        graph, image_ref = self._build()
        assert _class_types(graph) == {
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "EmptyLatentImage",
            "ADE_LoadAnimateDiffModel",
            "ADE_ApplyAnimateDiffModelSimple",
            "ADE_StandardUniformContextOptions",
            "ADE_UseEvolvedSampling",
            "FreeU_V2",
            "KSampler",
            "VAEDecode",
        }
        assert graph[image_ref[0]]["class_type"] == "VAEDecode"
        _assert_wiring_consistent(graph)
        # FreeU sits between the checkpoint and the evolved-sampling patch.
        evolved = next(n for n in graph.values() if n["class_type"] == "ADE_UseEvolvedSampling")
        assert graph[evolved["inputs"]["model"][0]]["class_type"] == "FreeU_V2"

    def test_freeu_can_be_disabled(self) -> None:
        graph, _ = self._build(freeu=False)
        assert "FreeU_V2" not in _class_types(graph)
        evolved = next(n for n in graph.values() if n["class_type"] == "ADE_UseEvolvedSampling")
        assert graph[evolved["inputs"]["model"][0]]["class_type"] == "CheckpointLoaderSimple"
        _assert_wiring_consistent(graph)

    def test_prompt_and_batch_size_threaded_through(self) -> None:
        graph, _ = self._build(
            prompt="a majestic eagle",
            negative_prompt="ugly",
            width=768,
            height=768,
            video_frames=24,
            context_length=8,
            context_overlap=2,
        )
        texts = [n["inputs"]["text"] for n in graph.values() if n["class_type"] == "CLIPTextEncode"]
        assert "a majestic eagle" in texts
        assert "ugly" in texts
        latent = next(n for n in graph.values() if n["class_type"] == "EmptyLatentImage")
        assert latent["inputs"]["batch_size"] == 24
        assert latent["inputs"]["width"] == 768
        ctx = next(
            n for n in graph.values() if n["class_type"] == "ADE_StandardUniformContextOptions"
        )
        assert ctx["inputs"]["context_length"] == 8
        assert ctx["inputs"]["context_overlap"] == 2


class TestBuildWanWorkflow:
    def _build(self, **overrides):
        kwargs = {
            "diffusion_model": "wan2.2_ti2v_5B_fp16.safetensors",
            "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae_name": "wan2.2_vae.safetensors",
            "prompt": "a fox in snow",
            "negative_prompt": "static",
            "image_name": "",
            "width": 832,
            "height": 480,
            "video_frames": 81,
            "steps": 20,
            "cfg": 5.0,
            "sampler": "uni_pc",
            "scheduler": "simple",
            "seed": 7,
        }
        kwargs.update(overrides)
        return _build_wan_workflow(**kwargs)

    def test_text_to_video_node_types(self) -> None:
        graph, image_ref = self._build()
        assert _class_types(graph) == {
            "UNETLoader",
            "CLIPLoader",
            "VAELoader",
            "CLIPTextEncode",
            "ModelSamplingSD3",
            "Wan22ImageToVideoLatent",
            "KSampler",
            "VAEDecode",
        }
        assert graph[image_ref[0]]["class_type"] == "VAEDecode"
        _assert_wiring_consistent(graph)
        # No LoadImage and no start_image in pure text-to-video mode.
        latent = next(n for n in graph.values() if n["class_type"] == "Wan22ImageToVideoLatent")
        assert "start_image" not in latent["inputs"]

    def test_image_to_video_adds_start_image(self) -> None:
        graph, _ = self._build(image_name="uploaded.png")
        load = next(n for n in graph.values() if n["class_type"] == "LoadImage")
        assert load["inputs"]["image"] == "uploaded.png"
        latent = next(n for n in graph.values() if n["class_type"] == "Wan22ImageToVideoLatent")
        assert latent["inputs"]["start_image"] is not None
        _assert_wiring_consistent(graph)

    def test_clip_loader_uses_wan_type_and_shift_is_8(self) -> None:
        graph, _ = self._build()
        clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        assert clip["inputs"]["type"] == "wan"
        sampling = next(n for n in graph.values() if n["class_type"] == "ModelSamplingSD3")
        assert sampling["inputs"]["shift"] == 8.0


# ---------------------------------------------------------------------------
# Post-processing / audio / mux composition
# ---------------------------------------------------------------------------


class TestGraphComposition:
    def _base_graph(self):
        return _build_svd_workflow(
            ckpt_name="svd_xt.safetensors",
            image_name="img.png",
            width=1024,
            height=576,
            video_frames=25,
            motion_bucket_id=127,
            fps=6,
            augmentation_level=0.0,
            steps=20,
            cfg=2.5,
            sampler="euler",
            scheduler="karras",
            seed=1,
        )

    def test_upscale_then_interpolate_then_combine(self) -> None:
        graph, image_ref = self._base_graph()
        image_ref = _append_upscale(graph, image_ref)
        assert graph[image_ref[0]]["class_type"] == "ImageUpscaleWithModel"
        image_ref = _append_interpolation(graph, image_ref, 4)
        assert graph[image_ref[0]]["class_type"] == "FrameInterpolate"
        assert graph[image_ref[0]]["inputs"]["multiplier"] == 4
        # Interpolation consumes the upscaled frames.
        assert graph[graph[image_ref[0]]["inputs"]["images"][0]]["class_type"] == (
            "ImageUpscaleWithModel"
        )
        _append_video_combine(
            graph,
            image_ref,
            audio_ref=None,
            frame_rate=24,
            filename_prefix="pfx",
            video_format="h264-mp4",
            crf=17,
        )
        combine = next(n for n in graph.values() if n["class_type"] == "VHS_VideoCombine")
        assert combine["inputs"]["frame_rate"] == 24
        assert combine["inputs"]["crf"] == 17
        assert combine["inputs"]["format"] == "video/h264-mp4"
        assert "audio" not in combine["inputs"]
        _assert_wiring_consistent(graph)

    def test_audio_generation_branch_wired_into_combine(self) -> None:
        graph, image_ref = self._base_graph()
        audio_ref = _append_audio_generation(
            graph,
            audio_prompt="gentle rain",
            audio_negative_prompt="noise",
            seconds=4.0,
            steps=50,
            cfg=5.0,
            seed=9,
        )
        assert graph[audio_ref[0]]["class_type"] == "VAEDecodeAudio"
        # The T5 text encoder is loaded separately (the checkpoint has none).
        clip_loader = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        assert clip_loader["inputs"]["type"] == "stable_audio"
        cond = next(n for n in graph.values() if n["class_type"] == "ConditioningStableAudio")
        assert cond["inputs"]["seconds_total"] == 4.0
        latent = next(n for n in graph.values() if n["class_type"] == "EmptyLatentAudio")
        assert latent["inputs"]["seconds"] == 4.0
        _append_video_combine(
            graph,
            image_ref,
            audio_ref=audio_ref,
            frame_rate=24,
            filename_prefix="pfx",
            video_format="h264-mp4",
            crf=17,
        )
        combine = next(n for n in graph.values() if n["class_type"] == "VHS_VideoCombine")
        assert combine["inputs"]["audio"] == audio_ref
        _assert_wiring_consistent(graph)

    def test_audio_generation_sa3_branch(self) -> None:
        graph, image_ref = self._base_graph()
        audio_ref = _append_audio_generation_sa3(
            graph,
            audio_prompt="gentle waves and seagulls",
            audio_negative_prompt="noise",
            seconds=5.0,
            steps=50,
            cfg=7.0,
            seed=11,
        )
        assert graph[audio_ref[0]]["class_type"] == "VAEDecodeAudio"
        # SA3 uses the T5-Gemma encoder via the same stable_audio CLIP type.
        clip_loader = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        assert clip_loader["inputs"]["type"] == "stable_audio"
        assert clip_loader["inputs"]["clip_name"] == "t5gemma_b_b_ul2.safetensors"
        ckpt = next(n for n in graph.values() if n["class_type"] == "CheckpointLoaderSimple")
        assert ckpt["inputs"]["ckpt_name"] == "stable_audio_3_medium_base.safetensors"
        # SA3 drops ConditioningStableAudio; conditioning feeds the sampler directly
        # and duration comes solely from EmptyLatentAudio. (The audio sampler is the
        # SA3 builder's fixed "au6" node -- the base graph has its own video KSampler.)
        assert not any(n["class_type"] == "ConditioningStableAudio" for n in graph.values())
        sampler = graph["au6"]
        assert sampler["class_type"] == "KSampler"
        assert sampler["inputs"]["sampler_name"] == "lcm"
        assert sampler["inputs"]["scheduler"] == "simple"
        assert graph[sampler["inputs"]["positive"][0]]["class_type"] == "CLIPTextEncode"
        assert graph["au5"]["inputs"]["seconds"] == 5.0
        _append_video_combine(
            graph,
            image_ref,
            audio_ref=audio_ref,
            frame_rate=24,
            filename_prefix="pfx",
            video_format="h264-mp4",
            crf=17,
        )
        _assert_wiring_consistent(graph)

    def test_audio_file_branch(self) -> None:
        graph, image_ref = self._base_graph()
        audio_ref = _append_audio_file(graph, "song.mp3")
        assert graph[audio_ref[0]]["class_type"] == "LoadAudio"
        assert graph[audio_ref[0]]["inputs"]["audio"] == "song.mp3"
        _append_video_combine(
            graph,
            image_ref,
            audio_ref=audio_ref,
            frame_rate=6,
            filename_prefix="pfx",
            video_format="h264-mp4",
            crf=17,
        )
        _assert_wiring_consistent(graph)

    def test_nvenc_format_uses_bitrate_not_crf(self) -> None:
        graph, image_ref = self._base_graph()
        _append_video_combine(
            graph,
            image_ref,
            audio_ref=None,
            frame_rate=24,
            filename_prefix="pfx",
            video_format="nvenc_h264-mp4",
            crf=17,
        )
        combine = next(n for n in graph.values() if n["class_type"] == "VHS_VideoCombine")
        assert combine["inputs"]["format"] == "video/nvenc_h264-mp4"
        assert "crf" not in combine["inputs"]
        assert combine["inputs"]["bitrate"] == 20
        assert combine["inputs"]["megabit"] is True


# ---------------------------------------------------------------------------
# _extract_video_output
# ---------------------------------------------------------------------------


class TestExtractVideoOutput:
    def test_finds_gifs_key_regardless_of_node_id(self) -> None:
        history_entry = {
            "outputs": {
                "3": {"images": [{"filename": "preview.png"}]},
                "out": {
                    "gifs": [
                        {
                            "filename": "missy_svd_test_00001.mp4",
                            "subfolder": "",
                            "type": "output",
                            "format": "video/h264-mp4",
                            "frame_rate": 24.0,
                            "fullpath": "/tmp/comfyui/output/missy_svd_test_00001.mp4",
                        }
                    ]
                },
            }
        }
        result = _extract_video_output(history_entry)
        assert result is not None
        assert result["filename"] == "missy_svd_test_00001.mp4"

    def test_returns_none_when_no_gifs_output(self) -> None:
        history_entry = {"outputs": {"3": {"images": [{"filename": "preview.png"}]}}}
        assert _extract_video_output(history_entry) is None

    def test_returns_none_for_empty_outputs(self) -> None:
        assert _extract_video_output({"outputs": {}}) is None
        assert _extract_video_output({}) is None


# ---------------------------------------------------------------------------
# Mock HTTP plumbing
# ---------------------------------------------------------------------------

_GPU_STATS = {
    "devices": [
        {"name": "cuda:0 NVIDIA GeForce RTX 3070", "type": "cuda", "vram_total": 8222998528}
    ]
}
_CPU_STATS = {"devices": [{"name": "cpu", "type": "cpu", "vram_total": 0}]}

_MODEL_LISTINGS = {
    "checkpoints": [
        "svd_xt.safetensors",
        "v1-5-pruned-emaonly.safetensors",
        "stable_audio_3_medium_base.safetensors",
        "stable-audio-open-1.0.safetensors",
    ],
    "animatediff_models": ["mm_sd_v15_v2.ckpt"],
    "diffusion_models": ["wan2.2_ti2v_5B_fp16.safetensors"],
    "text_encoders": [
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "t5gemma_b_b_ul2.safetensors",
        "t5_base.safetensors",
    ],
    "vae": ["wan2.2_vae.safetensors"],
    "frame_interpolation": ["rife_v4.26.safetensors", "film_net_fp16.safetensors"],
    "upscale_models": ["RealESRGAN_x2plus.pth"],
}


def _make_response(
    *, status_code: int = 200, json_data=None, text: str = "", content: bytes = b""
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_client(
    *,
    history: dict | None = None,
    stats: dict | None = None,
    model_listings: dict | None = None,
    post_side_effect=None,
    view_bytes: bytes = b"",
) -> MagicMock:
    """A PolicyHTTPClient double that routes GETs by URL."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    listings = model_listings if model_listings is not None else _MODEL_LISTINGS

    def _get(url, **kwargs):
        if "/system_stats" in url:
            return _make_response(json_data=stats if stats is not None else _GPU_STATS)
        if "/models/" in url:
            folder = url.rsplit("/", 1)[-1]
            listing = listings.get(folder)
            if listing is None:
                return _make_response(status_code=404)
            return _make_response(json_data=listing)
        if "/history/" in url:
            return _make_response(json_data=history if history is not None else {})
        if url.endswith("/queue"):
            return _make_response(json_data={"queue_running": [], "queue_pending": []})
        if "/view" in url:
            return _make_response(content=view_bytes)
        raise AssertionError(f"unexpected GET {url}")

    client.get.side_effect = _get
    if post_side_effect is not None:
        client.post.side_effect = post_side_effect
    return client


def _history_for(prompt_id: str, video_path: Path, frame_rate: float = 24.0) -> dict:
    return {
        prompt_id: {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "out": {
                    "gifs": [
                        {
                            "filename": video_path.name,
                            "subfolder": "",
                            "type": "output",
                            "format": "video/h264-mp4",
                            "frame_rate": frame_rate,
                            "fullpath": str(video_path),
                        }
                    ]
                }
            },
        }
    }


# ---------------------------------------------------------------------------
# VideoGenerateTool.execute() — validation and error paths
# ---------------------------------------------------------------------------


class TestVideoGenerateExecuteValidation:
    def test_unknown_backend_rejected(self) -> None:
        result = VideoGenerateTool().execute(backend="not-a-real-backend")
        assert result.success is False
        assert "backend" in result.error.lower()

    def test_unknown_video_format_rejected(self) -> None:
        result = VideoGenerateTool().execute(backend="wan", prompt="x", video_format="avi")
        assert result.success is False
        assert "video_format" in result.error

    def test_svd_requires_image_path(self) -> None:
        result = VideoGenerateTool().execute(backend="svd", image_path="")
        assert result.success is False
        assert "image_path" in result.error

    def test_svd_rejects_missing_image_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.png"
        result = VideoGenerateTool().execute(backend="svd", image_path=str(missing))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_wan_and_animatediff_require_prompt(self) -> None:
        for backend in ("wan", "animatediff"):
            result = VideoGenerateTool().execute(backend=backend, prompt="")
            assert result.success is False
            assert "prompt" in result.error.lower()

    def test_animatediff_rejects_image_path(self, tmp_path: Path) -> None:
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        result = VideoGenerateTool().execute(
            backend="animatediff", prompt="a cat", image_path=str(img)
        )
        assert result.success is False
        assert "animatediff" in result.error

    def test_audio_prompt_and_audio_path_mutually_exclusive(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"x")
        result = VideoGenerateTool().execute(
            backend="wan", prompt="x", audio_prompt="rain", audio_path=str(audio)
        )
        assert result.success is False
        assert "mutually exclusive" in result.error

    def test_missing_audio_file_rejected(self, tmp_path: Path) -> None:
        result = VideoGenerateTool().execute(
            backend="wan", prompt="x", audio_path=str(tmp_path / "nope.mp3")
        )
        assert result.success is False
        assert "audio_path" in result.error


class TestPreflight:
    def test_cpu_only_server_refused(self) -> None:
        mock_client = _make_mock_client(stats=_CPU_STATS)
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="wan", prompt="a fox")
        assert result.success is False
        assert "no gpu" in result.error.lower()
        # Nothing was submitted.
        assert mock_client.post.call_count == 0

    def test_cpu_only_server_allowed_with_allow_cpu(self, tmp_path: Path) -> None:
        video_out = tmp_path / "out.mp4"
        video_out.write_bytes(b"video")
        mock_client = _make_mock_client(
            stats=_CPU_STATS,
            history=_history_for("pid-cpu", video_out),
            post_side_effect=[_make_response(json_data={"prompt_id": "pid-cpu"})],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan",
                prompt="a fox",
                allow_cpu=True,
                save_path=str(tmp_path / "saved.mp4"),
            )
        assert result.success is True, result.error
        assert result.output["gpu"]["type"] == "cpu"

    def test_missing_model_reported_with_source(self) -> None:
        listings = dict(_MODEL_LISTINGS, diffusion_models=[])
        mock_client = _make_mock_client(model_listings=listings)
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="wan", prompt="a fox")
        assert result.success is False
        assert "wan2.2_ti2v_5B_fp16.safetensors" in result.error
        assert "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" in result.error

    def test_model_listing_endpoint_unavailable_is_tolerated(self, tmp_path: Path) -> None:
        video_out = tmp_path / "out.mp4"
        video_out.write_bytes(b"video")
        mock_client = _make_mock_client(
            model_listings={},  # every /models/* returns 404
            history=_history_for("pid-x", video_out),
            post_side_effect=[_make_response(json_data={"prompt_id": "pid-x"})],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan", prompt="a fox", save_path=str(tmp_path / "s.mp4")
            )
        assert result.success is True, result.error


# ---------------------------------------------------------------------------
# VideoGenerateTool.execute() — full happy paths, mocked HTTP client
# ---------------------------------------------------------------------------


class TestVideoGenerateExecuteHappyPaths:
    def test_full_svd_flow_with_generated_audio(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        video_out = tmp_path / "comfyui_output" / "missy_svd_abc123_00001.mp4"
        video_out.parent.mkdir(parents=True)
        video_out.write_bytes(b"fake video bytes")
        save_path = tmp_path / "saved" / "out.mp4"

        mock_client = _make_mock_client(
            history=_history_for("pid-1", video_out),
            post_side_effect=[
                _make_response(json_data={"name": "uploaded.png", "type": "input"}),
                _make_response(json_data={"prompt_id": "pid-1", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="svd",
                image_path=str(image_path),
                audio_prompt="gentle rain on leaves",
                save_path=str(save_path),
                seed=1234,
            )

        assert result.success is True, result.error
        assert result.output["path"] == str(save_path)
        assert result.output["backend"] == "svd"
        # svd default: 25 frames, auto-interpolated 4x -> 97 frames @ 24 fps.
        assert result.output["frames_generated"] == 25
        assert result.output["frames"] == 97
        assert result.output["duration_seconds"] > 3.5
        assert result.output["seed"] == 1234
        assert result.output["audio"] == {
            "source": "generated",
            "prompt": "gentle rain on leaves",
            "model": "stable-audio-3",
        }
        assert result.output["gpu"]["type"] == "cuda"
        assert save_path.read_bytes() == b"fake video bytes"

        upload_call = mock_client.post.call_args_list[0]
        assert "/upload/image" in upload_call.args[0]
        prompt_call = mock_client.post.call_args_list[1]
        assert "/prompt" in prompt_call.args[0]
        graph = prompt_call.kwargs["json"]["prompt"]
        types = {n["class_type"] for n in graph.values()}
        # Audio branch and interpolation made it into the submitted graph.
        assert "VAEDecodeAudio" in types
        assert "FrameInterpolate" in types
        combine = next(n for n in graph.values() if n["class_type"] == "VHS_VideoCombine")
        assert combine["inputs"]["audio"] is not None
        assert combine["inputs"]["frame_rate"] == 24

    def test_full_wan_text_to_video_flow(self, tmp_path: Path) -> None:
        video_out = tmp_path / "missy_wan_abc_00001.mp4"
        video_out.write_bytes(b"wan video")
        save_path = tmp_path / "wan.mp4"

        mock_client = _make_mock_client(
            history=_history_for("pid-wan", video_out),
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-wan", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan",
                prompt="a red fox trotting through snow",
                save_path=str(save_path),
            )

        assert result.success is True, result.error
        assert result.output["backend"] == "wan"
        # wan defaults: 81 frames @ 24 fps, no interpolation.
        assert result.output["frames"] == 81
        assert result.output["frames_generated"] == 81
        assert result.output["seed"] > 0  # random seed echoed
        assert result.output["audio"] is None
        # Only one POST: no upload for text-to-video.
        assert mock_client.post.call_count == 1
        graph = mock_client.post.call_args.kwargs["json"]["prompt"]
        types = {n["class_type"] for n in graph.values()}
        assert "UNETLoader" in types
        assert "FrameInterpolate" not in types

    def test_full_animatediff_flow(self, tmp_path: Path) -> None:
        video_out = tmp_path / "missy_ad_abc_00001.mp4"
        video_out.write_bytes(b"fake ad video")
        save_path = tmp_path / "ad_out.mp4"

        mock_client = _make_mock_client(
            history=_history_for("pid-ad", video_out, frame_rate=16.0),
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-ad", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="animatediff",
                prompt="a cat playing piano",
                save_path=str(save_path),
            )

        assert result.success is True, result.error
        assert result.output["backend"] == "animatediff"
        # animatediff defaults: 16 frames, auto-interpolated 2x -> 31 @ 16fps.
        assert result.output["frames_generated"] == 16
        assert result.output["frames"] == 31
        graph = mock_client.post.call_args.kwargs["json"]["prompt"]
        types = {n["class_type"] for n in graph.values()}
        assert "FreeU_V2" in types
        assert "FrameInterpolate" in types

    def test_audio_file_muxed(self, tmp_path: Path) -> None:
        audio_path = tmp_path / "song.mp3"
        audio_path.write_bytes(b"mp3 bytes")
        video_out = tmp_path / "out.mp4"
        video_out.write_bytes(b"video")

        mock_client = _make_mock_client(
            history=_history_for("pid-a", video_out),
            post_side_effect=[
                _make_response(json_data={"name": "song.mp3", "type": "input"}),
                _make_response(json_data={"prompt_id": "pid-a", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan",
                prompt="dancing robot",
                audio_path=str(audio_path),
                save_path=str(tmp_path / "muxed.mp4"),
            )

        assert result.success is True, result.error
        assert result.output["audio"] == {"source": "file", "path": str(audio_path)}
        graph = mock_client.post.call_args_list[1].kwargs["json"]["prompt"]
        load_audio = next(n for n in graph.values() if n["class_type"] == "LoadAudio")
        assert load_audio["inputs"]["audio"] == "song.mp3"
        # Upload used the audio MIME type, not image/png.
        upload_call = mock_client.post.call_args_list[0]
        assert upload_call.kwargs["files"]["image"][2] == "audio/mpeg"

    def test_upscale_doubles_reported_dimensions(self, tmp_path: Path) -> None:
        video_out = tmp_path / "out.mp4"
        video_out.write_bytes(b"video")
        mock_client = _make_mock_client(
            history=_history_for("pid-u", video_out),
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-u", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan",
                prompt="x",
                upscale=True,
                save_path=str(tmp_path / "u.mp4"),
            )
        assert result.success is True, result.error
        assert result.output["width"] == 832 * 2
        assert result.output["height"] == 480 * 2
        graph = mock_client.post.call_args.kwargs["json"]["prompt"]
        assert any(n["class_type"] == "ImageUpscaleWithModel" for n in graph.values())

    def test_save_path_collision_appends_suffix(self, tmp_path: Path) -> None:
        video_out = tmp_path / "out.mp4"
        video_out.write_bytes(b"new video")
        save_path = tmp_path / "existing.mp4"
        save_path.write_bytes(b"precious old video")

        mock_client = _make_mock_client(
            history=_history_for("pid-c", video_out),
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-c", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan", prompt="x", save_path=str(save_path)
            )
        assert result.success is True, result.error
        assert result.output["path"] == str(tmp_path / "existing_1.mp4")
        assert save_path.read_bytes() == b"precious old video"

    def test_view_fallback_used_when_fullpath_missing(self, tmp_path: Path) -> None:
        save_path = tmp_path / "downloaded.mp4"
        history = {
            "pid-4": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {
                    "out": {
                        "gifs": [
                            {
                                "filename": "remote_video.mp4",
                                "subfolder": "",
                                "type": "output",
                                "frame_rate": 24.0,
                                "fullpath": "/nonexistent/remote/path/remote_video.mp4",
                            }
                        ]
                    }
                },
            }
        }
        mock_client = _make_mock_client(
            history=history,
            view_bytes=b"downloaded video bytes",
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-4", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="wan", prompt="x", save_path=str(save_path)
            )
        assert result.success is True, result.error
        assert save_path.read_bytes() == b"downloaded video bytes"


# ---------------------------------------------------------------------------
# Error paths and timeout cancellation
# ---------------------------------------------------------------------------


class TestVideoGenerateErrors:
    def test_upload_failure_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")
        mock_client = _make_mock_client(
            post_side_effect=[_make_response(status_code=500, text="server error")],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="svd", image_path=str(image_path))
        assert result.success is False
        assert "upload" in result.error.lower()

    def test_node_errors_reported(self) -> None:
        mock_client = _make_mock_client(
            post_side_effect=[
                _make_response(
                    json_data={
                        "prompt_id": None,
                        "node_errors": {"5": {"errors": [{"message": "bad checkpoint"}]}},
                    }
                ),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="wan", prompt="x")
        assert result.success is False
        assert "validation" in result.error.lower()

    def test_generation_error_status_reported(self) -> None:
        history = {
            "pid-err": {
                "status": {
                    "completed": False,
                    "status_str": "error",
                    "messages": [["execution_error", {"node_id": "5"}]],
                },
                "outputs": {},
            }
        }
        mock_client = _make_mock_client(
            history=history,
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-err", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="wan", prompt="x", timeout=5)
        assert result.success is False
        assert "failed" in result.error.lower()

    def test_no_video_output_reported(self) -> None:
        history = {
            "pid-2": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"3": {"images": [{"filename": "no_video.png"}]}},
            }
        }
        mock_client = _make_mock_client(
            history=history,
            post_side_effect=[
                _make_response(json_data={"prompt_id": "pid-2", "node_errors": {}}),
            ],
        )
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="wan", prompt="x")
        assert result.success is False
        assert "no video output" in result.error.lower()

    def test_timeout_cancels_job_server_side(self) -> None:
        posts: list[tuple] = []

        def _post(url, **kwargs):
            posts.append((url, kwargs))
            if "/prompt" in url:
                return _make_response(json_data={"prompt_id": "pid-3", "node_errors": {}})
            return _make_response(json_data={})

        mock_client = _make_mock_client(history={}, post_side_effect=_post)
        with (
            patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client),
            patch("missy.tools.builtin.video_generate.time.sleep"),
        ):
            result = VideoGenerateTool().execute(backend="wan", prompt="x", timeout=1)

        assert result.success is False
        assert "timed out" in result.error.lower()
        # The pending job was deleted from the server queue.
        queue_deletes = [
            kwargs for url, kwargs in posts if url.endswith("/queue") and "json" in kwargs
        ]
        assert queue_deletes and queue_deletes[0]["json"] == {"delete": ["pid-3"]}


# ---------------------------------------------------------------------------
# resolve_network_hosts / resolve_filesystem_targets / get_schema
# ---------------------------------------------------------------------------


class TestVideoGenerateToolResolvers:
    def test_resolve_network_hosts_default(self) -> None:
        tool = VideoGenerateTool()
        assert tool.resolve_network_hosts({}) == ["127.0.0.1:8199"]

    def test_resolve_network_hosts_custom(self) -> None:
        tool = VideoGenerateTool()
        hosts = tool.resolve_network_hosts({"comfyui_host": "10.0.0.5", "comfyui_port": 9000})
        assert hosts == ["10.0.0.5:9000"]

    def test_resolve_filesystem_targets_with_all_paths(self) -> None:
        tool = VideoGenerateTool()
        read_paths, write_paths = tool.resolve_filesystem_targets(
            {"image_path": "/a/b.png", "audio_path": "/a/c.mp3", "save_path": "/c/d.mp4"}
        )
        assert read_paths == ["/a/b.png", "/a/c.mp3"]
        assert write_paths == ["/c/d.mp4"]

    def test_resolve_filesystem_targets_defaults(self) -> None:
        tool = VideoGenerateTool()
        read_paths, write_paths = tool.resolve_filesystem_targets({})
        assert read_paths == []
        assert len(write_paths) == 1
        assert "videos" in write_paths[0]

    def test_get_schema_has_backend_enum_and_required(self) -> None:
        schema = VideoGenerateTool().get_schema()
        assert schema["name"] == "video_generate"
        props = schema["parameters"]["properties"]
        assert set(props["backend"]["enum"]) == {"wan", "svd", "animatediff"}
        assert set(props["video_format"]["enum"]) == {"h264-mp4", "h265-mp4", "nvenc_h264-mp4"}
        assert "audio_prompt" in props
        assert "interpolate" in props
        assert schema["parameters"]["required"] == ["backend"]

    def test_permissions_declared(self) -> None:
        tool = VideoGenerateTool()
        assert tool.permissions.network is True
        assert tool.permissions.filesystem_read is True
        assert tool.permissions.filesystem_write is True
