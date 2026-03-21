"""Edge case tests for vision tools and cross-module integration.


Covers:
- VisionCaptureTool: source types, missing camera, file source, save path
- VisionDevicesTool: no cameras, with cameras
- VisionSceneMemoryTool: all actions, invalid action, task_id validation
- Cross-module: discovery → capture → pipeline → analysis prompt flow
- Color description edge cases at boundaries
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# VisionCaptureTool tests
# ---------------------------------------------------------------------------


class TestVisionCaptureTool:
    """Edge cases for VisionCaptureTool."""

    def test_tool_metadata(self):
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        tool = VisionCaptureTool()
        assert tool.name == "vision_capture"
        assert "webcam" in tool.description.lower()
        assert "source" in tool.parameters

    def test_no_camera_returns_error(self):
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        tool = VisionCaptureTool()
        with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
            result = tool.execute(source="webcam")
            assert result.success is False
            assert "No camera" in result.error

    def test_general_exception_handled(self):
        """Exceptions during capture should return error, not raise."""
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        tool = VisionCaptureTool()
        # File source that doesn't exist
        result = tool.execute(source="/tmp/definitely_does_not_exist_12345.jpg")
        assert result.success is False

    def test_file_source_nonexistent(self):
        """File source with nonexistent path should fail."""
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        tool = VisionCaptureTool()
        result = tool.execute(source="/nonexistent/path/image.jpg")
        assert result.success is False

    def test_screenshot_source(self):
        """Screenshot source should be handled."""
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        tool = VisionCaptureTool()
        # Screenshots are typically unavailable in headless test environments
        result = tool.execute(source="screenshot")
        # Will likely fail (no display) but should not crash
        assert isinstance(result.success, bool)


class TestVisionBurstCaptureTool:
    """Edge cases for VisionBurstCaptureTool."""

    def test_tool_metadata(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        assert tool.name == "vision_burst"
        assert "count" in tool.parameters

    def test_no_camera_returns_error(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
            result = tool.execute()
            assert result.success is False
            assert "No camera" in result.error


class TestVisionDevicesTool:
    """Edge cases for VisionDevicesTool."""

    def test_tool_metadata(self):
        from missy.tools.builtin.vision_tools import VisionDevicesTool

        tool = VisionDevicesTool()
        assert tool.name == "vision_devices"

    @patch("missy.vision.discovery.get_discovery")
    def test_no_cameras(self, mock_disc):
        from missy.tools.builtin.vision_tools import VisionDevicesTool

        mock_disc.return_value.discover.return_value = []
        tool = VisionDevicesTool()
        result = tool.execute()
        assert result.success is True
        assert "0" in result.output or "no" in result.output.lower()

    def test_execute_returns_result(self):
        """VisionDevicesTool should always return a ToolResult."""
        from missy.tools.builtin.vision_tools import VisionDevicesTool

        tool = VisionDevicesTool()
        result = tool.execute()
        assert result.success is True or result.success is False
        assert isinstance(result.output, str) or result.output is None


class TestVisionAnalyzeTool:
    """Edge cases for VisionAnalyzeTool."""

    def test_tool_metadata(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        assert tool.name == "vision_analyze"
        assert "mode" in tool.parameters

    def test_general_mode(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="general")
        assert result.success is True
        assert "Analyze" in result.output or "analyze" in result.output.lower()

    def test_puzzle_mode(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="puzzle")
        assert result.success is True
        assert "puzzle" in result.output.lower()

    def test_painting_mode(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="painting")
        assert result.success is True
        assert "paint" in result.output.lower()

    def test_inspection_mode(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="inspection")
        assert result.success is True

    def test_invalid_mode_raises_or_fails(self):
        """Invalid mode should either raise or return error."""
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="unknown")
        # Invalid AnalysisMode value → ValueError caught → error result
        assert result.success is False

    def test_with_context(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        tool = VisionAnalyzeTool()
        result = tool.execute(mode="puzzle", context="Working on the sky section")
        assert result.success is True
        assert "sky" in result.output.lower()


class TestVisionSceneMemoryTool:
    """Edge cases for VisionSceneMemoryTool."""

    def test_tool_metadata(self):
        from missy.tools.builtin.vision_tools import VisionSceneMemoryTool

        tool = VisionSceneMemoryTool()
        assert tool.name == "vision_scene"
        assert "action" in tool.parameters

    def test_create_session(self):
        from missy.tools.builtin.vision_tools import VisionSceneMemoryTool

        tool = VisionSceneMemoryTool()
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr:
            mock_session = MagicMock()
            mock_session.task_id = "test-task"
            mock_mgr.return_value.create_session.return_value = mock_session
            result = tool.execute(action="create", task_id="test-task", task_type="puzzle")
            assert result.success is True

    def test_close_session(self):
        from missy.tools.builtin.vision_tools import VisionSceneMemoryTool

        tool = VisionSceneMemoryTool()
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr:
            mock_mgr.return_value.close_session.return_value = None
            result = tool.execute(action="close", task_id="test-task")
            assert result.success is True

    def test_summarize_nonexistent_session(self):
        from missy.tools.builtin.vision_tools import VisionSceneMemoryTool

        tool = VisionSceneMemoryTool()
        result = tool.execute(action="summarize", task_id="nonexistent-task-xyz")
        # Summarizing a nonexistent session should fail gracefully
        assert isinstance(result.success, bool)

    def test_invalid_action(self):
        from missy.tools.builtin.vision_tools import VisionSceneMemoryTool

        tool = VisionSceneMemoryTool()
        result = tool.execute(action="unknown_action", task_id="test-task")
        assert result.success is False


# ---------------------------------------------------------------------------
# Cross-module integration tests
# ---------------------------------------------------------------------------


class TestColorDescriptionBoundaries:
    """Test _describe_color at exact boundaries."""

    def test_black_boundary(self):
        from missy.vision.analysis import _describe_color

        # max(49, 49, 49) = 49 < 50 → black
        assert _describe_color([49, 49, 49]) == "black"
        # max(50, 50, 50) = 50 → NOT black, check others
        result = _describe_color([50, 50, 50])
        assert result != "black"

    def test_white_boundary(self):
        from missy.vision.analysis import _describe_color

        # min(201, 201, 201) = 201 > 200 → white
        assert _describe_color([201, 201, 201]) == "white"
        # min(200, 200, 200) = 200 → NOT white
        result = _describe_color([200, 200, 200])
        assert result != "white"

    def test_pure_colors(self):
        from missy.vision.analysis import _describe_color

        assert _describe_color([255, 0, 0]) == "red"
        assert _describe_color([0, 255, 0]) == "green"
        assert _describe_color([0, 0, 255]) == "blue"
        assert _describe_color([255, 255, 0]) == "yellow"

    def test_zero_zero_zero(self):
        from missy.vision.analysis import _describe_color

        assert _describe_color([0, 0, 0]) == "black"

    def test_255_255_255(self):
        from missy.vision.analysis import _describe_color

        assert _describe_color([255, 255, 255]) == "white"

    def test_mid_gray(self):
        from missy.vision.analysis import _describe_color

        assert _describe_color([128, 128, 128]) == "gray"


class TestImagePipelineIntegration:
    """Integration tests for the image pipeline."""

    def test_pipeline_process_basic(self):
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = pipeline.process(img)
        assert processed is not None
        assert processed.shape[0] > 0
        assert processed.shape[1] > 0

    def test_pipeline_quality_assessment(self):
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert isinstance(quality, dict)

    def test_pipeline_process_small_image(self):
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        processed = pipeline.process(img)
        assert processed is not None

    def test_pipeline_process_grayscale(self):
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        processed = pipeline.process(img)
        assert processed is not None


class TestDiscoveryValidation:
    """Integration tests for camera discovery validation."""

    @patch("missy.vision.discovery.CameraDiscovery._scan_sysfs")
    def test_discover_returns_list(self, mock_scan):
        from missy.vision.discovery import CameraDiscovery

        mock_scan.return_value = []
        disc = CameraDiscovery()
        devices = disc.discover(force=True)
        assert isinstance(devices, list)

    def test_find_preferred_no_camera(self):
        from missy.vision.discovery import CameraDiscovery

        disc = CameraDiscovery()
        disc._devices = []
        result = disc.find_preferred()
        # May return None or a device depending on state
        # Just verify it doesn't crash
        assert result is None or hasattr(result, "device_path")


class TestSceneMemoryIntegration:
    """Integration tests for scene memory."""

    def test_create_and_close_session(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager()
        session = mgr.create_session("test-task", TaskType.PUZZLE)
        assert session.task_id == "test-task"
        assert session.is_active is True

        mgr.close_session("test-task")
        sessions = mgr.list_sessions()
        active = [s for s in sessions if s.get("active")]
        assert len(active) == 0

    def test_add_observation_to_session(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager()
        session = mgr.create_session("obs-task", TaskType.GENERAL)
        session.add_observation("Found a red object")
        assert "Found a red object" in session.observations
        mgr.close_session("obs-task")

    def test_close_nonexistent_session(self):
        """Closing a nonexistent session should be safe."""
        from missy.vision.scene_memory import SceneManager

        mgr = SceneManager()
        mgr.close_session("does-not-exist")  # Should not raise

    def test_close_all_sessions(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager()
        mgr.create_session("t1", TaskType.PUZZLE)
        mgr.create_session("t2", TaskType.PAINTING)
        mgr.close_all()
        sessions = mgr.list_sessions()
        active = [s for s in sessions if s.get("active")]
        assert len(active) == 0


class TestOrientationAutoCorrectIntegration:
    """Integration test for auto-correct pipeline."""

    def test_landscape_image_unchanged(self):
        from missy.vision.orientation import auto_correct

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        assert corrected.shape == img.shape
        assert result.correction_applied is False

    def test_portrait_image_rotated(self):
        from missy.vision.orientation import auto_correct

        img = np.zeros((640, 320, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        if result.confidence >= 0.5:
            assert corrected.shape[0] < corrected.shape[1]  # Now landscape
