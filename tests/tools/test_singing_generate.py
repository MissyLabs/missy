"""Tests for the ACE-Step singing generation tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.tools.builtin.singing_generate import (
    SingingGenerateTool,
    _build_ace_step_workflow,
    _extract_audio_output,
)

_MODELS = {
    "diffusion_models": ["acestep_v1.5_turbo.safetensors"],
    "text_encoders": ["qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors"],
    "vae": ["ace_1.5_vae.safetensors"],
}
_NODES = {
    name: {}
    for name in {
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
}
_GPU = {
    "devices": [
        {
            "name": "NVIDIA GeForce RTX 3070",
            "type": "cuda",
            "vram_total": 8 * 2**30,
        }
    ]
}


def _response(*, status: int = 200, data=None, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {} if data is None else data
    response.text = ""
    response.content = content
    response.raise_for_status = MagicMock()
    return response


def _client(*, history=None, models=None, nodes=None, posts=None) -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    model_data = _MODELS if models is None else models

    def get(url, **_kwargs):
        if url.endswith("/system_stats"):
            return _response(data=_GPU)
        if url.endswith("/object_info"):
            return _response(data=_NODES if nodes is None else nodes)
        if "/models/" in url:
            folder = url.rsplit("/", 1)[-1]
            return _response(data=model_data.get(folder, []))
        if "/history/" in url:
            return _response(data=history or {})
        if "/view" in url:
            return _response(content=b"remote flac")
        if url.endswith("/queue"):
            return _response(data={"queue_running": [], "queue_pending": []})
        raise AssertionError(f"unexpected GET: {url}")

    client.get.side_effect = get
    if posts is not None:
        client.post.side_effect = posts
    return client


def _workflow(**overrides):
    kwargs = {
        "lyrics": "[Verse]\nTiny cat, mighty song",
        "tags": "playful synth-pop, expressive female vocal",
        "duration": 20.0,
        "bpm": 120,
        "keyscale": "C major",
        "time_signature": "4",
        "language": "en",
        "seed": 42,
        "steps": 8,
        "sampler_cfg": 1.0,
        "lyric_cfg": 2.0,
        "temperature": 0.85,
        "top_p": 0.9,
        "top_k": 0,
        "min_p": 0.0,
        "generate_audio_codes": True,
        "filename_prefix": "missy_test",
    }
    kwargs.update(overrides)
    return _build_ace_step_workflow(**kwargs)


def test_workflow_uses_native_ace_nodes_and_cpu_text_encoders() -> None:
    graph = _workflow()
    types = {node["class_type"] for node in graph.values()}
    assert types == {
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
    assert graph["clip"]["inputs"]["device"] == "cpu"
    assert graph["condition"]["inputs"]["lyrics"].startswith("[Verse]")
    assert graph["condition"]["inputs"]["bpm"] == 120
    assert graph["latent"]["inputs"]["seconds"] == 20.0
    assert graph["sampler"]["inputs"]["steps"] == 8
    assert graph["output"]["inputs"]["audio"] == ["decode", 0]


def test_every_workflow_reference_points_to_a_node() -> None:
    graph = _workflow()
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph


def test_extract_audio_output() -> None:
    descriptor = {"filename": "song.flac", "type": "output"}
    assert _extract_audio_output({"outputs": {"output": {"audio": [descriptor]}}}) == descriptor
    assert _extract_audio_output({"outputs": {"output": {"audio": descriptor}}}) == descriptor
    assert _extract_audio_output({"outputs": {}}) is None


def test_requires_real_lyrics_and_style() -> None:
    tool = SingingGenerateTool()
    result = tool.execute(tags="pop")
    assert result.success is False
    assert "lyrics" in result.error
    result = tool.execute(lyrics="sing this")
    assert result.success is False
    assert "tags" in result.error


def test_rejects_invalid_music_controls() -> None:
    tool = SingingGenerateTool()
    assert not tool.execute(lyrics="x", tags="pop", keyscale="H major").success
    assert not tool.execute(lyrics="x", tags="pop", time_signature="5").success
    assert not tool.execute(lyrics="x", tags="pop", language="klingon").success


def test_validate_only_reports_ready_without_submitting() -> None:
    client = _client()
    with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
        result = SingingGenerateTool().execute(validate_only=True)
    assert result.success is True, result.error
    assert result.output["ready"] is True
    assert result.output["backend"] == "ace-step-1.5"
    assert result.output["gpu"]["type"] == "cuda"
    assert client.post.call_count == 0


def test_missing_nodes_are_actionable() -> None:
    client = _client(nodes={})
    with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
        result = SingingGenerateTool().execute(validate_only=True)
    assert result.success is False
    assert "ACE-Step 1.5 node support" in result.error
    assert "Update ComfyUI" in result.error


def test_node_inventory_http_failure_is_actionable() -> None:
    client = _client()
    client.get.side_effect = lambda _url, **_kwargs: _response(status=503)
    error = SingingGenerateTool._check_nodes(client, "http://comfy")
    assert "HTTP 503" in error


def test_node_inventory_exception_is_actionable() -> None:
    client = MagicMock()
    client.get.side_effect = RuntimeError("connection lost")
    error = SingingGenerateTool._check_nodes(client, "http://comfy")
    assert "connection lost" in error


def test_missing_models_are_actionable() -> None:
    client = _client(models={})
    with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
        result = SingingGenerateTool().execute(validate_only=True)
    assert result.success is False
    assert "acestep_v1.5_turbo.safetensors" in result.error
    assert "qwen_0.6b_ace15.safetensors" in result.error
    assert "Comfy-Org/ace_step_1.5_ComfyUI_files" in result.error


def test_unavailable_model_inventory_fails_closed() -> None:
    client = MagicMock()
    client.get.return_value = _response(status=404)
    error = SingingGenerateTool._check_models(
        client, "http://comfy", (("diffusion_models", "ace.safetensors"),)
    )
    assert "Could not verify" in error
    assert "diffusion_models" in error


def test_remote_audio_download_and_collision_safe_destination(tmp_path: Path) -> None:
    existing = tmp_path / "song.flac"
    existing.write_bytes(b"keep me")
    client = MagicMock()
    client.get.return_value = _response(content=b"downloaded flac")
    path = SingingGenerateTool._retrieve_audio(
        client,
        "http://comfy",
        {"filename": "remote.flac", "subfolder": "audio", "type": "output"},
        str(existing),
    )
    assert Path(path).name == "song_1.flac"
    assert Path(path).read_bytes() == b"downloaded flac"
    assert existing.read_bytes() == b"keep me"
    client.get.assert_called_once()


def test_full_generation_writes_mix_and_manifest(tmp_path: Path) -> None:
    generated = tmp_path / "comfy" / "song.flac"
    generated.parent.mkdir()
    generated.write_bytes(b"generated flac")
    prompt_id = "ace-prompt"
    history = {
        prompt_id: {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "output": {
                    "audio": [
                        {
                            "filename": generated.name,
                            "subfolder": "",
                            "type": "output",
                            "fullpath": str(generated),
                        }
                    ]
                }
            },
        }
    }
    client = _client(
        history=history,
        posts=[_response(data={"prompt_id": prompt_id, "node_errors": {}})],
    )
    destination = tmp_path / "result.flac"
    with patch("missy.gateway.client.PolicyHTTPClient", return_value=client):
        result = SingingGenerateTool().execute(
            lyrics="[Chorus]\nNo more autotune crimes",
            tags="playful electropop, feline lead vocal",
            duration=2,  # clamped to the safe minimum
            bpm=500,
            seed=99,
            save_path=str(destination),
        )

    assert result.success is True, result.error
    assert destination.read_bytes() == b"generated flac"
    assert result.output["mix_path"] == str(destination)
    assert result.output["duration_seconds"] == 5.0
    assert result.output["bpm"] == 300
    assert result.output["seed"] == 99
    assert result.output["vocal_path"] is None
    assert result.output["lip_sync_audio_path"] is None
    assert result.output["lip_sync_ready"] is False
    manifest = Path(result.output["manifest_path"])
    assert manifest.is_file()
    assert '"backend": "ace-step-1.5"' in manifest.read_text()
    graph = client.post.call_args.kwargs["json"]["prompt"]
    assert graph["condition"]["inputs"]["lyrics"] == "[Chorus]\nNo more autotune crimes"


def test_schema_and_policy_resolvers() -> None:
    tool = SingingGenerateTool()
    schema = tool.get_schema()
    assert schema["name"] == "singing_generate"
    assert "validate_only" in schema["parameters"]["properties"]
    assert "C major" in schema["parameters"]["properties"]["keyscale"]["enum"]
    assert tool.resolve_network_hosts({"comfyui_host": "10.0.0.5", "comfyui_port": 9000}) == [
        "10.0.0.5:9000"
    ]
    reads, writes = tool.resolve_filesystem_targets({"save_path": "/tmp/song.flac"})
    assert reads == []
    assert writes == ["/tmp/song.flac"]
    assert tool.permissions.network is True
    assert tool.permissions.filesystem_write is True


def test_tool_is_registered() -> None:
    from missy.tools.builtin import _ALL_TOOL_CLASSES

    assert SingingGenerateTool in _ALL_TOOL_CLASSES
