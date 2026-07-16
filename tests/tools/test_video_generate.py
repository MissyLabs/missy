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
    _build_animatediff_workflow,
    _build_svd_workflow,
    _extract_video_output,
)

# ---------------------------------------------------------------------------
# Workflow graph builders
# ---------------------------------------------------------------------------


class TestBuildSvdWorkflow:
    def test_produces_expected_node_types(self) -> None:
        wf = _build_svd_workflow(
            ckpt_name="svd_xt.safetensors",
            image_name="uploaded.png",
            width=1024,
            height=576,
            video_frames=25,
            motion_bucket_id=127,
            fps=6,
            augmentation_level=0.0,
            steps=20,
            cfg=2.5,
            seed=42,
            filename_prefix="missy_svd_test",
        )
        class_types = {node["class_type"] for node in wf.values()}
        assert class_types == {
            "ImageOnlyCheckpointLoader",
            "LoadImage",
            "SVD_img2vid_Conditioning",
            "VideoLinearCFGGuidance",
            "KSampler",
            "VAEDecode",
            "VHS_VideoCombine",
        }

    def test_wiring_references_are_internally_consistent(self) -> None:
        wf = _build_svd_workflow(
            ckpt_name="svd_xt.safetensors",
            image_name="uploaded.png",
            width=1024,
            height=576,
            video_frames=25,
            motion_bucket_id=127,
            fps=6,
            augmentation_level=0.0,
            steps=20,
            cfg=2.5,
            seed=42,
            filename_prefix="missy_svd_test",
        )
        # Every [node_id, output_index] reference must point at a node
        # that actually exists in the graph.
        for node in wf.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in wf, f"dangling reference to node {value[0]!r}"

    def test_checkpoint_and_image_name_threaded_through(self) -> None:
        wf = _build_svd_workflow(
            ckpt_name="svd.safetensors",
            image_name="my_photo.png",
            width=800,
            height=450,
            video_frames=14,
            motion_bucket_id=200,
            fps=8,
            augmentation_level=0.5,
            steps=15,
            cfg=3.0,
            seed=999,
            filename_prefix="pfx",
        )
        loader = next(n for n in wf.values() if n["class_type"] == "ImageOnlyCheckpointLoader")
        assert loader["inputs"]["ckpt_name"] == "svd.safetensors"
        load_image = next(n for n in wf.values() if n["class_type"] == "LoadImage")
        assert load_image["inputs"]["image"] == "my_photo.png"
        conditioning = next(n for n in wf.values() if n["class_type"] == "SVD_img2vid_Conditioning")
        assert conditioning["inputs"]["video_frames"] == 14
        assert conditioning["inputs"]["motion_bucket_id"] == 200
        assert conditioning["inputs"]["fps"] == 8
        assert conditioning["inputs"]["augmentation_level"] == 0.5
        combine = next(n for n in wf.values() if n["class_type"] == "VHS_VideoCombine")
        assert combine["inputs"]["filename_prefix"] == "pfx"
        assert combine["inputs"]["frame_rate"] == 8


class TestBuildAnimatediffWorkflow:
    def test_produces_expected_node_types(self) -> None:
        wf = _build_animatediff_workflow(
            ckpt_name="v1-5-pruned-emaonly.safetensors",
            motion_module="mm_sd_v15_v2.ckpt",
            prompt="a cat",
            negative_prompt="blurry",
            width=512,
            height=512,
            video_frames=16,
            fps=8,
            context_length=16,
            context_overlap=4,
            steps=20,
            cfg=7.5,
            seed=1,
            filename_prefix="missy_ad_test",
        )
        class_types = {node["class_type"] for node in wf.values()}
        assert class_types == {
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "EmptyLatentImage",
            "ADE_LoadAnimateDiffModel",
            "ADE_ApplyAnimateDiffModelSimple",
            "ADE_StandardUniformContextOptions",
            "ADE_UseEvolvedSampling",
            "KSampler",
            "VAEDecode",
            "VHS_VideoCombine",
        }

    def test_wiring_references_are_internally_consistent(self) -> None:
        wf = _build_animatediff_workflow(
            ckpt_name="v1-5-pruned-emaonly.safetensors",
            motion_module="mm_sd_v15_v2.ckpt",
            prompt="a cat",
            negative_prompt="blurry",
            width=512,
            height=512,
            video_frames=16,
            fps=8,
            context_length=16,
            context_overlap=4,
            steps=20,
            cfg=7.5,
            seed=1,
            filename_prefix="missy_ad_test",
        )
        for node in wf.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in wf, f"dangling reference to node {value[0]!r}"

    def test_prompt_and_batch_size_threaded_through(self) -> None:
        wf = _build_animatediff_workflow(
            ckpt_name="v1-5-pruned-emaonly.safetensors",
            motion_module="mm_sd_v15_v2.ckpt",
            prompt="a majestic eagle",
            negative_prompt="ugly",
            width=768,
            height=768,
            video_frames=24,
            fps=12,
            context_length=8,
            context_overlap=2,
            steps=25,
            cfg=8.0,
            seed=7,
            filename_prefix="pfx",
        )
        positive = wf["2"]
        assert positive["inputs"]["text"] == "a majestic eagle"
        negative = wf["3"]
        assert negative["inputs"]["text"] == "ugly"
        latent = wf["4"]
        assert latent["inputs"]["batch_size"] == 24
        assert latent["inputs"]["width"] == 768
        context_opts = wf["7"]
        assert context_opts["inputs"]["context_length"] == 8
        assert context_opts["inputs"]["context_overlap"] == 2
        combine = wf["11"]
        assert combine["inputs"]["frame_rate"] == 12
        assert combine["inputs"]["filename_prefix"] == "pfx"


# ---------------------------------------------------------------------------
# _extract_video_output
# ---------------------------------------------------------------------------


class TestExtractVideoOutput:
    def test_finds_gifs_key_regardless_of_node_id(self) -> None:
        history_entry = {
            "outputs": {
                "3": {"images": [{"filename": "preview.png"}]},
                "9": {
                    "gifs": [
                        {
                            "filename": "missy_svd_test_00001.mp4",
                            "subfolder": "",
                            "type": "output",
                            "format": "video/h264-mp4",
                            "frame_rate": 6.0,
                            "fullpath": "/tmp/comfyui/output/missy_svd_test_00001.mp4",
                        }
                    ]
                },
            }
        }
        result = _extract_video_output(history_entry)
        assert result is not None
        assert result["filename"] == "missy_svd_test_00001.mp4"
        assert result["fullpath"] == "/tmp/comfyui/output/missy_svd_test_00001.mp4"

    def test_returns_none_when_no_gifs_output(self) -> None:
        history_entry = {"outputs": {"3": {"images": [{"filename": "preview.png"}]}}}
        assert _extract_video_output(history_entry) is None

    def test_returns_none_for_empty_outputs(self) -> None:
        assert _extract_video_output({"outputs": {}}) is None
        assert _extract_video_output({}) is None


# ---------------------------------------------------------------------------
# VideoGenerateTool.execute() — validation and error paths
# ---------------------------------------------------------------------------


class TestVideoGenerateExecuteValidation:
    def test_unknown_backend_rejected(self) -> None:
        result = VideoGenerateTool().execute(backend="not-a-real-backend")
        assert result.success is False
        assert "backend" in result.error.lower()

    def test_svd_requires_image_path(self) -> None:
        result = VideoGenerateTool().execute(backend="svd", image_path="")
        assert result.success is False
        assert "image_path" in result.error

    def test_svd_rejects_missing_image_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.png"
        result = VideoGenerateTool().execute(backend="svd", image_path=str(missing))
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_animatediff_requires_prompt(self) -> None:
        result = VideoGenerateTool().execute(backend="animatediff", prompt="")
        assert result.success is False
        assert "prompt" in result.error.lower()


# ---------------------------------------------------------------------------
# VideoGenerateTool.execute() — full happy path, mocked HTTP client
# ---------------------------------------------------------------------------


def _make_response(
    *, status_code: int = 200, json_data: dict | None = None, text: str = "", content: bytes = b""
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


class TestVideoGenerateExecuteSvdHappyPath:
    def test_full_svd_flow_mocked(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        video_out = tmp_path / "comfyui_output" / "missy_svd_abc123_00001.mp4"
        video_out.parent.mkdir(parents=True)
        video_out.write_bytes(b"fake video bytes")
        save_path = tmp_path / "saved" / "out.mp4"

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png", "subfolder": "", "type": "input"}),
            _make_response(json_data={"prompt_id": "pid-1", "number": 1, "node_errors": {}}),
        ]
        mock_client.get.return_value = _make_response(
            json_data={
                "pid-1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "7": {
                            "gifs": [
                                {
                                    "filename": video_out.name,
                                    "subfolder": "",
                                    "type": "output",
                                    "format": "video/h264-mp4",
                                    "frame_rate": 6.0,
                                    "fullpath": str(video_out),
                                }
                            ]
                        }
                    },
                }
            }
        )

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="svd",
                image_path=str(image_path),
                save_path=str(save_path),
                video_frames=25,
                fps=6,
            )

        assert result.success is True, result.error
        assert result.output["path"] == str(save_path)
        assert result.output["backend"] == "svd"
        assert result.output["frames"] == 25
        assert result.output["prompt_id"] == "pid-1"
        assert save_path.read_bytes() == b"fake video bytes"

        # Upload used the real image bytes, and /prompt got a real workflow.
        upload_call = mock_client.post.call_args_list[0]
        assert "/upload/image" in upload_call.args[0]
        prompt_call = mock_client.post.call_args_list[1]
        assert "/prompt" in prompt_call.args[0]
        assert prompt_call.kwargs["json"]["prompt"]  # non-empty workflow graph

    def test_upload_failure_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = _make_response(status_code=500, text="server error")

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="svd", image_path=str(image_path))

        assert result.success is False
        assert "upload" in result.error.lower()

    def test_node_errors_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png"}),
            _make_response(
                json_data={
                    "prompt_id": None,
                    "node_errors": {"5": {"errors": [{"message": "bad checkpoint"}]}},
                }
            ),
        ]

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="svd", image_path=str(image_path))

        assert result.success is False
        assert "validation" in result.error.lower()

    def test_generation_error_status_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png"}),
            _make_response(json_data={"prompt_id": "pid-err", "node_errors": {}}),
        ]
        mock_client.get.return_value = _make_response(
            json_data={
                "pid-err": {
                    "status": {
                        "completed": False,
                        "status_str": "error",
                        "messages": [["execution_error", {"node_id": "5"}]],
                    },
                    "outputs": {},
                }
            }
        )

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="svd", image_path=str(image_path), timeout=5
            )

        assert result.success is False
        assert "failed" in result.error.lower()

    def test_no_video_output_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png"}),
            _make_response(json_data={"prompt_id": "pid-2", "node_errors": {}}),
        ]
        mock_client.get.return_value = _make_response(
            json_data={
                "pid-2": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"3": {"images": [{"filename": "no_video.png"}]}},
                }
            }
        )

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(backend="svd", image_path=str(image_path))

        assert result.success is False
        assert "no video output" in result.error.lower()

    def test_timeout_reported(self, tmp_path: Path) -> None:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png"}),
            _make_response(json_data={"prompt_id": "pid-3", "node_errors": {}}),
        ]
        # Never completes.
        mock_client.get.return_value = _make_response(json_data={})

        with (
            patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client),
            patch("missy.tools.builtin.video_generate.time.sleep"),
        ):
            result = VideoGenerateTool().execute(
                backend="svd", image_path=str(image_path), timeout=1
            )

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_view_fallback_used_when_fullpath_missing(self, tmp_path: Path) -> None:
        """If ComfyUI's fullpath isn't reachable (e.g. different host), fall
        back to downloading the bytes via /view."""
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake")
        save_path = tmp_path / "downloaded.mp4"

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = [
            _make_response(json_data={"name": "uploaded.png"}),
            _make_response(json_data={"prompt_id": "pid-4", "node_errors": {}}),
        ]

        def _get_side_effect(url, **kwargs):
            if "/history/" in url:
                return _make_response(
                    json_data={
                        "pid-4": {
                            "status": {"completed": True, "status_str": "success"},
                            "outputs": {
                                "7": {
                                    "gifs": [
                                        {
                                            "filename": "remote_video.mp4",
                                            "subfolder": "",
                                            "type": "output",
                                            "frame_rate": 6.0,
                                            "fullpath": "/nonexistent/remote/path/remote_video.mp4",
                                        }
                                    ]
                                }
                            },
                        }
                    }
                )
            assert "/view" in url
            return _make_response(content=b"downloaded video bytes")

        mock_client.get.side_effect = _get_side_effect

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="svd", image_path=str(image_path), save_path=str(save_path)
            )

        assert result.success is True, result.error
        assert save_path.read_bytes() == b"downloaded video bytes"


class TestVideoGenerateExecuteAnimatediffHappyPath:
    def test_full_animatediff_flow_mocked(self, tmp_path: Path) -> None:
        video_out = tmp_path / "comfyui_output" / "missy_ad_abc_00001.mp4"
        video_out.parent.mkdir(parents=True)
        video_out.write_bytes(b"fake ad video")
        save_path = tmp_path / "ad_out.mp4"

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = _make_response(
            json_data={"prompt_id": "pid-ad", "node_errors": {}}
        )
        mock_client.get.return_value = _make_response(
            json_data={
                "pid-ad": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "11": {
                            "gifs": [
                                {
                                    "filename": video_out.name,
                                    "subfolder": "",
                                    "type": "output",
                                    "frame_rate": 8.0,
                                    "fullpath": str(video_out),
                                }
                            ]
                        }
                    },
                }
            }
        )

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=mock_client):
            result = VideoGenerateTool().execute(
                backend="animatediff",
                prompt="a cat playing piano",
                save_path=str(save_path),
                video_frames=16,
                fps=8,
            )

        assert result.success is True, result.error
        assert result.output["backend"] == "animatediff"
        assert result.output["frames"] == 16
        # Only one POST (the /prompt submission) -- no image upload for animatediff.
        assert mock_client.post.call_count == 1
        assert "/prompt" in mock_client.post.call_args.args[0]


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

    def test_resolve_filesystem_targets_with_image_and_save_path(self) -> None:
        tool = VideoGenerateTool()
        read_paths, write_paths = tool.resolve_filesystem_targets(
            {"image_path": "/a/b.png", "save_path": "/c/d.mp4"}
        )
        assert read_paths == ["/a/b.png"]
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
        assert set(props["backend"]["enum"]) == {"svd", "animatediff"}
        assert schema["parameters"]["required"] == ["backend"]

    def test_permissions_declared(self) -> None:
        tool = VideoGenerateTool()
        assert tool.permissions.network is True
        assert tool.permissions.filesystem_read is True
        assert tool.permissions.filesystem_write is True
