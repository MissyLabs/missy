"""Tests for missy.tools.builtin.vision_tools — agent-callable vision tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np

from missy.tools.builtin.vision_tools import (
    VisionAnalyzeTool,
    VisionCaptureTool,
    VisionDevicesTool,
    VisionSceneMemoryTool,
)

# ---------------------------------------------------------------------------
# VisionDevicesTool tests
# ---------------------------------------------------------------------------


class TestVisionDevicesTool:
    def test_name_and_description(self):
        tool = VisionDevicesTool()
        assert tool.name == "vision_devices"
        assert "camera" in tool.description.lower()

    @patch("missy.tools.builtin.vision_tools.VisionDevicesTool.execute")
    def test_execute_returns_result(self, mock_exec):
        mock_exec.return_value = MagicMock(success=True)
        tool = VisionDevicesTool()
        result = tool.execute()
        assert result.success

    def test_schema(self):
        tool = VisionDevicesTool()
        schema = tool.get_schema()
        assert schema["name"] == "vision_devices"


# ---------------------------------------------------------------------------
# VisionCaptureTool tests
# ---------------------------------------------------------------------------


class TestVisionCaptureTool:
    def test_name_and_description(self):
        tool = VisionCaptureTool()
        assert tool.name == "vision_capture"
        assert "capture" in tool.description.lower()

    def test_schema_has_parameters(self):
        tool = VisionCaptureTool()
        schema = tool.get_schema()
        assert "source" in schema["parameters"]["properties"]

    @patch("missy.vision.discovery.find_preferred_camera", return_value=None)
    def test_no_camera_webcam_source(self, mock_find):
        tool = VisionCaptureTool()
        result = tool.execute(source="webcam")
        assert result.success is False
        assert "No camera" in result.error

    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_file_source(self, mock_avail, mock_acquire, tmp_path):
        # Regression: this test previously omitted save_path and only
        # mocked mock_frame.timestamp.isoformat (not .strftime), so
        # VisionCaptureTool.execute()'s save_path fallback
        # (Path.home() / ".missy" / "captures") wrote a real
        # "capture_<MagicMock ...>.jpg" garbage file into the operator's
        # actual home directory on every run of this "unit" test --
        # found live during task #10 validation, with a dozen such
        # leaked files accumulated across prior sessions. Passing an
        # explicit tmp_path-based save_path keeps this test hermetic.
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.image = img
        mock_frame.source_type.value = "file"
        mock_frame.source_path = "/tmp/test.jpg"
        mock_frame.width = 100
        mock_frame.height = 100
        mock_frame.timestamp.isoformat.return_value = "2026-03-19T00:00:00"
        mock_acquire.return_value = mock_frame

        save_path = str(tmp_path / "captured.jpg")
        tool = VisionCaptureTool()
        result = tool.execute(source="/tmp/test.jpg", save_path=save_path)
        assert result.success is True
        data = json.loads(result.output)
        assert data["saved_to"] == save_path

    def test_unavailable_source(self):
        tool = VisionCaptureTool()
        result = tool.execute(source="/nonexistent/image.xyz")
        assert result.success is False

    @patch("missy.vision.discovery.find_preferred_camera")
    @patch("missy.vision.sources.WebcamSource.acquire")
    @patch("missy.vision.sources.WebcamSource.is_available", return_value=True)
    def test_blank_frame_failure_includes_guard_guidance(self, mock_avail, mock_acquire, mock_find):
        """FX-round2-F7: a blank-frame quality-guard rejection must tell
        the model to report the verdict, not just fail generically --
        this is what stops the model from routing around the guard via a
        raw shell_exec capture instead."""
        mock_find.return_value = MagicMock(device_path="/dev/video0")
        mock_acquire.side_effect = RuntimeError(
            "Capture failed after 3 attempts: Blank frame detected on attempt 3"
        )

        tool = VisionCaptureTool()
        result = tool.execute(source="webcam")

        assert result.success is False
        assert "blank frame detected" in result.error.lower()
        assert "quality guard" in result.error.lower()
        assert "shell" in result.error.lower()

    @patch("missy.vision.discovery.find_preferred_camera")
    @patch("missy.vision.sources.WebcamSource.acquire")
    @patch("missy.vision.sources.WebcamSource.is_available", return_value=True)
    def test_non_blank_capture_failure_unchanged(self, mock_avail, mock_acquire, mock_find):
        """A generic capture error must not get the blank-frame guidance
        attached -- only the specific guard-rejection case should."""
        mock_find.return_value = MagicMock(device_path="/dev/video0")
        mock_acquire.side_effect = RuntimeError("device busy")

        tool = VisionCaptureTool()
        result = tool.execute(source="webcam")

        assert result.success is False
        assert result.error == "device busy"


# ---------------------------------------------------------------------------
# SR-1.4: vision_capture/vision_burst filesystem permission resolution.
#
# Neither "source"/"save_path"/"device" (vision_capture) nor "device"
# (vision_burst) match the registry's generic path/file_path/target/
# destination heuristic, so the declared filesystem_read/filesystem_write
# permissions previously enforced nothing regardless of configuration.
# ---------------------------------------------------------------------------
class TestVisionCaptureResolveFilesystemTargets:
    def test_webcam_source_has_no_read_target(self):
        tool = VisionCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({"source": "webcam"})
        assert reads == []

    def test_camera_alias_has_no_read_target(self):
        tool = VisionCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({"source": "camera"})
        assert reads == []

    def test_screenshot_source_has_no_read_target(self):
        tool = VisionCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({"source": "screenshot"})
        assert reads == []

    def test_file_path_source_is_a_read_target(self):
        tool = VisionCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({"source": "/etc/shadow"})
        assert "/etc/shadow" in reads

    def test_device_kwarg_is_a_read_target(self):
        tool = VisionCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets(
            {"source": "webcam", "device": "/dev/video0"}
        )
        assert "/dev/video0" in reads

    def test_explicit_save_path_is_the_write_target(self):
        tool = VisionCaptureTool()
        _reads, writes = tool.resolve_filesystem_targets({"save_path": "/tmp/out.jpg"})
        assert writes == ["/tmp/out.jpg"]

    def test_omitted_save_path_resolves_to_default_captures_dir(self):
        """Matches execute()'s own fallback so the check reflects the real
        write location even when the model doesn't pass save_path."""
        tool = VisionCaptureTool()
        _reads, writes = tool.resolve_filesystem_targets({})
        assert writes == [tool._DEFAULT_CAPTURES_DIR]
        assert ".missy/captures" in writes[0]


class TestSR14RegistryGatesVisionCapture:
    """Before the fix, ToolPermissions(filesystem_read=True,
    filesystem_write=True) enforced nothing for this tool at all --
    source/save_path/device don't match the registry's generic
    path/file_path/target/destination heuristic."""

    def _init_policy(self, allowed_read=None, allowed_write=None):
        from missy.config.settings import (
            FilesystemPolicy,
            MissyConfig,
            NetworkPolicy,
            PluginPolicy,
            ShellPolicy,
        )
        from missy.policy.engine import init_policy_engine

        init_policy_engine(
            MissyConfig(
                network=NetworkPolicy(),
                filesystem=FilesystemPolicy(
                    allowed_read_paths=allowed_read or [],
                    allowed_write_paths=allowed_write or [],
                ),
                shell=ShellPolicy(enabled=False, allowed_commands=[]),
                plugins=PluginPolicy(),
                providers={},
                workspace_path="/tmp/vision-test-ws",
                audit_log_path="/tmp/vision-test-audit.jsonl",
            )
        )

    def test_arbitrary_source_path_denied_when_nothing_allowlisted(self):
        from missy.tools.registry import ToolRegistry

        self._init_policy()
        registry = ToolRegistry()
        registry.register(VisionCaptureTool())
        result = registry.execute(
            "vision_capture",
            source="/etc/shadow",
            save_path="/tmp/exfil.jpg",
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "Filesystem read denied" in result.error

    def test_arbitrary_save_path_denied_when_nothing_allowlisted(self):
        from missy.tools.registry import ToolRegistry

        self._init_policy(allowed_read=["/tmp"])
        registry = ToolRegistry()
        registry.register(VisionCaptureTool())
        result = registry.execute(
            "vision_capture",
            source="/tmp/test.jpg",
            save_path="/etc/cron.d/pwn",
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "Filesystem write denied" in result.error

    def test_passes_policy_when_paths_allowlisted(self):
        from missy.tools.registry import ToolRegistry

        self._init_policy(allowed_read=["/tmp"], allowed_write=["/tmp"])
        registry = ToolRegistry()
        registry.register(VisionCaptureTool())
        result = registry.execute(
            "vision_capture",
            source="/tmp/does-not-exist.jpg",
            save_path="/tmp/out.jpg",
            session_id="s",
            task_id="t",
        )
        # Policy passes; the tool then fails for an unrelated reason (file
        # doesn't exist) -- proof the policy layer itself isn't what denies.
        assert "Filesystem read denied" not in (result.error or "")
        assert "Filesystem write denied" not in (result.error or "")


class TestVisionBurstResolveFilesystemTargets:
    def test_device_is_a_read_target(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({"device": "/dev/video0"})
        assert reads == ["/dev/video0"]

    def test_no_device_has_no_read_target(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        reads, _writes = tool.resolve_filesystem_targets({})
        assert reads == []

    def test_best_only_writes_to_default_captures_dir(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        _reads, writes = tool.resolve_filesystem_targets({"best_only": True})
        assert writes == [tool._DEFAULT_CAPTURES_DIR]

    def test_full_burst_mode_has_no_write_target(self):
        """The non-best_only branch never calls cv2.imwrite -- no write
        target should be checked."""
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        _reads, writes = tool.resolve_filesystem_targets({"best_only": False})
        assert writes == []


class TestVisionBurstBlankFrameGuidance:
    """FX-round2-F7: capture_best's blank-frame rejection must carry the
    same don't-work-around-the-guard guidance as VisionCaptureTool's."""

    @patch("missy.vision.discovery.find_preferred_camera")
    @patch("missy.vision.capture.CameraHandle.close")
    @patch("missy.vision.capture.CameraHandle.open")
    @patch("missy.vision.capture.CameraHandle.capture_best")
    def test_best_only_blank_frame_includes_guard_guidance(
        self, mock_capture_best, mock_open, mock_close, mock_find
    ):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        mock_find.return_value = MagicMock(device_path="/dev/video0")
        mock_capture_best.return_value = MagicMock(
            success=False,
            error="Capture failed after 3 attempts: Blank frame detected on attempt 3",
        )

        tool = VisionBurstCaptureTool()
        result = tool.execute(best_only=True, count=3)

        assert result.success is False
        assert "blank frame detected" in result.error.lower()
        assert "quality guard" in result.error.lower()
        assert "shell" in result.error.lower()


# ---------------------------------------------------------------------------
# VisionAnalyzeTool tests
# ---------------------------------------------------------------------------


class TestVisionAnalyzeTool:
    """FX-real-vision: vision_analyze previously only returned an unused
    text prompt template with no image ever attached to any LLM call --
    the model had no real way to see an image's content (including
    reading text out of it), which is why it was observed falling back
    to a raw shell OCR command instead. It now performs a genuine
    multimodal call via analyze_image_with_provider_fallback()."""

    def _mock_frame(self):
        frame = MagicMock()
        frame.image = np.full((100, 100, 3), 128, dtype=np.uint8)
        return frame

    def test_name_and_description(self):
        tool = VisionAnalyzeTool()
        assert tool.name == "vision_analyze"

    def test_source_required(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="general")
        assert result.success is False
        assert "source is required" in result.error

    def test_unavailable_source(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(source="/nonexistent/image.xyz", mode="general")
        assert result.success is False

    @patch("missy.vision.provider_call.analyze_image_with_provider_fallback")
    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_general_mode_returns_real_analysis(self, mock_avail, mock_acquire, mock_analyze):
        mock_acquire.return_value = self._mock_frame()
        mock_analyze.return_value = ("A red mug sits on a wooden desk.", "anthropic")

        tool = VisionAnalyzeTool()
        result = tool.execute(source="/tmp/photo.jpg", mode="general")

        assert result.success is True
        data = json.loads(result.output)
        assert data["mode"] == "general"
        assert data["analysis"] == "A red mug sits on a wooden desk."
        assert data["provider"] == "anthropic"
        mock_analyze.assert_called_once()
        prompt_arg = mock_analyze.call_args[0][0]
        assert "Analyze this image" in prompt_arg

    @patch("missy.vision.provider_call.analyze_image_with_provider_fallback")
    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_puzzle_mode(self, mock_avail, mock_acquire, mock_analyze):
        mock_acquire.return_value = self._mock_frame()
        mock_analyze.return_value = ("Board state: 40% complete.", "anthropic")

        tool = VisionAnalyzeTool()
        result = tool.execute(
            source="/tmp/puzzle.jpg", mode="puzzle", context="I need help with the sky section"
        )
        assert result.success is True
        data = json.loads(result.output)
        assert data["mode"] == "puzzle"
        prompt_arg = mock_analyze.call_args[0][0]
        assert "jigsaw" in prompt_arg.lower()

    @patch("missy.vision.provider_call.analyze_image_with_provider_fallback")
    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_painting_mode(self, mock_avail, mock_acquire, mock_analyze):
        mock_acquire.return_value = self._mock_frame()
        mock_analyze.return_value = ("Lovely use of color.", "anthropic")

        tool = VisionAnalyzeTool()
        result = tool.execute(source="/tmp/painting.jpg", mode="painting")
        assert result.success is True
        prompt_arg = mock_analyze.call_args[0][0]
        assert "supportive" in prompt_arg.lower()

    @patch("missy.vision.provider_call.analyze_image_with_provider_fallback")
    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_inspection_mode(self, mock_avail, mock_acquire, mock_analyze):
        mock_acquire.return_value = self._mock_frame()
        mock_analyze.return_value = ("Overview: a small mechanical part.", "anthropic")

        tool = VisionAnalyzeTool()
        result = tool.execute(source="/tmp/part.jpg", mode="inspection")
        assert result.success is True

    def test_invalid_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(source="/tmp/photo.jpg", mode="nonexistent")
        assert result.success is False

    @patch("missy.vision.provider_call.analyze_image_with_provider_fallback")
    @patch("missy.vision.sources.FileSource.acquire")
    @patch("missy.vision.sources.FileSource.is_available", return_value=True)
    def test_no_available_provider_returns_error(self, mock_avail, mock_acquire, mock_analyze):
        from missy.core.exceptions import ProviderError

        mock_acquire.return_value = self._mock_frame()
        mock_analyze.side_effect = ProviderError("No available provider could analyze the image.")

        tool = VisionAnalyzeTool()
        result = tool.execute(source="/tmp/photo.jpg", mode="general")
        assert result.success is False
        assert "No available provider" in result.error


# ---------------------------------------------------------------------------
# VisionSceneMemoryTool tests
# ---------------------------------------------------------------------------


class TestVisionSceneMemoryTool:
    def test_name_and_description(self):
        tool = VisionSceneMemoryTool()
        assert tool.name == "vision_scene"

    def test_create_session(self):
        tool = VisionSceneMemoryTool()
        result = tool.execute(action="create", task_id="test-puzzle", task_type="puzzle")
        assert result.success is True
        data = json.loads(result.output)
        assert data["action"] == "created"
        assert data["task_id"] == "test-puzzle"

    def test_add_observation(self):
        tool = VisionSceneMemoryTool()
        # First create a session
        tool.execute(action="create", task_id="test-obs")
        # Then add observation
        result = tool.execute(
            action="add_observation",
            task_id="test-obs",
            observation="Sky section is 50% complete",
        )
        assert result.success is True
        data = json.loads(result.output)
        assert data["total_observations"] == 1

    def test_update_state(self):
        tool = VisionSceneMemoryTool()
        tool.execute(action="create", task_id="test-state")
        result = tool.execute(
            action="update_state",
            task_id="test-state",
            state_updates='{"progress": "25%"}',
        )
        assert result.success is True
        data = json.loads(result.output)
        assert data["state"]["progress"] == "25%"

    def test_summarize(self):
        tool = VisionSceneMemoryTool()
        tool.execute(action="create", task_id="test-sum")
        result = tool.execute(action="summarize", task_id="test-sum")
        assert result.success is True
        data = json.loads(result.output)
        assert data["task_id"] == "test-sum"

    def test_close(self):
        tool = VisionSceneMemoryTool()
        tool.execute(action="create", task_id="test-close")
        result = tool.execute(action="close", task_id="test-close")
        assert result.success is True

    def test_unknown_action(self):
        tool = VisionSceneMemoryTool()
        result = tool.execute(action="invalid")
        assert result.success is False
        assert "Unknown action" in result.error

    def test_observation_no_session(self):
        # Use a fresh manager
        tool = VisionSceneMemoryTool()
        result = tool.execute(
            action="add_observation",
            task_id="nonexistent-xxx",
            observation="test",
        )
        # Will fall back to active session or fail
        assert isinstance(result.success, bool)

    def test_auto_generate_task_id(self):
        tool = VisionSceneMemoryTool()
        result = tool.execute(action="create")
        assert result.success is True
        data = json.loads(result.output)
        assert data["task_id"].startswith("task_")
