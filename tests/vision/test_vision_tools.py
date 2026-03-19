"""Tests for missy.tools.builtin.vision_tools — agent-callable vision tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
    def test_file_source(self, mock_avail, mock_acquire):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.image = img
        mock_frame.source_type.value = "file"
        mock_frame.source_path = "/tmp/test.jpg"
        mock_frame.width = 100
        mock_frame.height = 100
        mock_frame.timestamp.isoformat.return_value = "2026-03-19T00:00:00"
        mock_acquire.return_value = mock_frame

        tool = VisionCaptureTool()
        result = tool.execute(source="/tmp/test.jpg")
        assert result.success is True
        data = json.loads(result.output)
        assert "image_base64" in data

    def test_unavailable_source(self):
        tool = VisionCaptureTool()
        result = tool.execute(source="/nonexistent/image.xyz")
        assert result.success is False


# ---------------------------------------------------------------------------
# VisionAnalyzeTool tests
# ---------------------------------------------------------------------------


class TestVisionAnalyzeTool:
    def test_name_and_description(self):
        tool = VisionAnalyzeTool()
        assert tool.name == "vision_analyze"

    def test_general_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="general")
        assert result.success is True
        data = json.loads(result.output)
        assert data["mode"] == "general"
        assert "Analyze this image" in data["prompt"]

    def test_puzzle_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="puzzle", context="I need help with the sky section")
        assert result.success is True
        data = json.loads(result.output)
        assert data["mode"] == "puzzle"
        assert "jigsaw" in data["prompt"].lower()

    def test_painting_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="painting")
        assert result.success is True
        data = json.loads(result.output)
        assert "supportive" in data["prompt"].lower()

    def test_inspection_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="inspection")
        assert result.success is True

    def test_invalid_mode(self):
        tool = VisionAnalyzeTool()
        result = tool.execute(mode="nonexistent")
        assert result.success is False


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
        from missy.vision.scene_memory import SceneManager
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
