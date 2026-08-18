"""Tests for the native ComfyUI still-image generation tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.tools.builtin.image_generate import (
    ImageGenerateTool,
    _build_image_workflow,
    _extract_image_outputs,
)


def _response(*, status: int = 200, data=None, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = data
    response.text = "response text"
    response.content = content
    response.raise_for_status.side_effect = None
    return response


def _client(*, checkpoint_present: bool = True, image_bytes: bytes = b"png-data") -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    def get(url, **_kwargs):
        if url.endswith("/system_stats"):
            return _response(
                data={
                    "devices": [
                        {
                            "name": "cuda:0 RTX",
                            "type": "cuda",
                            "vram_total": 8 * 2**30,
                        }
                    ]
                }
            )
        if url.endswith("/models/checkpoints"):
            checkpoints = ["v1-5-pruned-emaonly.safetensors"] if checkpoint_present else []
            return _response(data=checkpoints)
        if "/history/pid-image" in url:
            return _response(
                data={
                    "pid-image": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "save": {
                                "images": [
                                    {
                                        "filename": "missy_image_00001_.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
        if url.endswith("/view"):
            return _response(content=image_bytes)
        raise AssertionError(f"unexpected GET {url}")

    client.get.side_effect = get
    client.post.return_value = _response(data={"prompt_id": "pid-image", "node_errors": {}})
    return client


class TestImageWorkflow:
    def test_uses_native_still_image_nodes(self):
        graph = _build_image_workflow(
            checkpoint="model.safetensors",
            prompt="a tabby cat",
            negative_prompt="blurry",
            width=512,
            height=640,
            steps=25,
            cfg=7.0,
            sampler="dpmpp_2m",
            scheduler="karras",
            seed=123,
            batch_size=1,
            filename_prefix="missy_test",
        )

        assert graph["load_checkpoint"]["class_type"] == "CheckpointLoaderSimple"
        assert graph["sample"]["class_type"] == "KSampler"
        assert graph["save"]["class_type"] == "SaveImage"
        assert graph["sample"]["inputs"]["seed"] == 123
        assert not any("Video" in node["class_type"] for node in graph.values())

    def test_extracts_all_image_descriptors(self):
        entry = {
            "outputs": {
                "save": {
                    "images": [
                        {"filename": "one.png"},
                        {"filename": "two.png"},
                    ]
                }
            }
        }
        assert [item["filename"] for item in _extract_image_outputs(entry)] == [
            "one.png",
            "two.png",
        ]


class TestImageGenerateExecute:
    def test_empty_prompt_is_rejected_before_network(self):
        result = ImageGenerateTool().execute(prompt="   ")
        assert not result.success
        assert "prompt" in result.error

    def test_generates_and_downloads_png_with_reproducibility_metadata(self, tmp_path: Path):
        client = _client(image_bytes=b"real-png-bytes")
        output_path = tmp_path / "cat.png"

        with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
            result = ImageGenerateTool().execute(
                prompt="a comedic cartoon tabby cat",
                seed=4242,
                width=530,
                height=620,
                save_path=str(output_path),
            )

        assert result.success, result.error
        assert result.output["path"] == str(output_path)
        assert result.output["paths"] == [str(output_path)]
        assert result.output["seed"] == 4242
        assert result.output["width"] == 512
        assert result.output["height"] == 640
        assert result.output["gpu"]["type"] == "cuda"
        assert output_path.read_bytes() == b"real-png-bytes"
        submitted = client.post.call_args.kwargs["json"]["prompt"]
        assert submitted["positive"]["inputs"]["text"] == "a comedic cartoon tabby cat"

    def test_missing_checkpoint_is_reported_without_submission(self):
        client = _client(checkpoint_present=False)
        with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
            result = ImageGenerateTool().execute(prompt="a cat")

        assert not result.success
        assert "v1-5-pruned-emaonly.safetensors" in result.error
        client.post.assert_not_called()

    def test_schema_and_policy_resolvers_cover_real_targets(self, tmp_path: Path):
        tool = ImageGenerateTool()
        schema = tool.get_schema()
        assert schema["name"] == "image_generate"
        assert schema["parameters"]["required"] == ["prompt"]
        assert schema["parameters"]["additionalProperties"] is False
        assert tool.resolve_network_hosts({"comfyui_host": "gpu-box", "comfyui_port": 9000}) == [
            "gpu-box:9000"
        ]
        assert tool.resolve_filesystem_targets({"save_path": str(tmp_path / "x.png")}) == (
            [],
            [str(tmp_path / "x.png")],
        )
        assert tool.permissions.network
        assert tool.permissions.filesystem_write
