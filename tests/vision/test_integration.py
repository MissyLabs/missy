"""Integration tests for the vision subsystem pipeline.

Tests the full flow from image acquisition through preprocessing,
analysis prompt building, provider formatting, and scene memory management.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest
from missy.vision.intent import ActivationDecision, VisionIntent, VisionIntentClassifier
from missy.vision.pipeline import ImagePipeline, PipelineConfig
from missy.vision.provider_format import build_vision_message, format_image_for_provider
from missy.vision.scene_memory import SceneManager, TaskType
from missy.vision.sources import FileSource, ImageFrame, SourceType


class TestFullPuzzlePipeline:
    """Test the complete puzzle assistance pipeline."""

    def test_puzzle_first_look(self):
        """Simulate: capture → preprocess → analyze → format → scene memory."""
        # 1. Simulate captured image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.WEBCAM)

        # 2. Preprocess
        pipeline = ImagePipeline(PipelineConfig(
            normalize_exposure=False, denoise=False, sharpen=False,
            target_dimension=640,
        ))
        processed = pipeline.process(frame.image)
        assert processed.shape[0] <= 640
        assert processed.shape[1] <= 640

        # 3. Quality assessment
        quality = pipeline.assess_quality(processed)
        assert "quality" in quality

        # 4. Build analysis prompt
        request = AnalysisRequest(
            image=processed,
            mode=AnalysisMode.PUZZLE,
            context="I'm working on the sky section",
        )
        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(request)
        assert "Board State" in prompt
        assert "sky section" in prompt

        # 5. Format for provider
        # Simulate base64 encoding
        b64 = "fake_base64_data"
        msg = build_vision_message("anthropic", b64, prompt)
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"

        # 6. Scene memory
        mgr = SceneManager()
        session = mgr.create_session("puzzle-sky", TaskType.PUZZLE)
        session.add_frame(processed, source="webcam:/dev/video0")
        session.add_observation("Working on sky section")
        session.update_state(section="sky", progress="starting")

        assert session.frame_count == 1
        assert session.state["section"] == "sky"

    def test_puzzle_followup_with_scene_memory(self):
        """Simulate follow-up analysis using scene memory context."""
        mgr = SceneManager()
        session = mgr.create_session("puzzle-followup", TaskType.PUZZLE)

        # First observation
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        session.add_frame(img1)
        session.add_observation("Sky section is 30% complete")
        session.update_state(completed_sections=["top-left corner"])

        # Second capture with changes
        img2 = np.full((100, 100, 3), 100, dtype=np.uint8)
        session.add_frame(img2)

        # Detect change
        change = session.detect_latest_change()
        assert change is not None
        assert change.change_score > 0

        # Build followup prompt
        request = AnalysisRequest(
            image=img2,
            mode=AnalysisMode.PUZZLE,
            is_followup=True,
            previous_observations=session.observations,
            previous_state=session.state,
        )
        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(request)
        assert "30% complete" in prompt
        assert "progress" in prompt.lower()


class TestFullPaintingPipeline:
    """Test the complete painting feedback pipeline."""

    def test_painting_feedback_flow(self):
        """Simulate: capture → preprocess → warm feedback prompt."""
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        pipeline = ImagePipeline()
        processed = pipeline.process(img)

        request = AnalysisRequest(
            image=processed,
            mode=AnalysisMode.PAINTING,
            context="This is my first watercolor painting",
        )
        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(request)

        # Verify warm, encouraging tone
        assert "supportive" in prompt.lower()
        assert "warm" in prompt.lower()
        assert "encouraging" in prompt.lower()
        assert "first watercolor" in prompt

        # Format for OpenAI
        msg = build_vision_message("openai", "fake_b64", prompt)
        assert msg["content"][0]["type"] == "image_url"


class TestAudioTriggeredVisionFlow:
    """Test audio intent detection triggering vision capture."""

    def test_explicit_look_triggers_capture(self):
        classifier = VisionIntentClassifier()

        # Simulate transcribed utterance
        result = classifier.classify("Missy, look at this puzzle piece")

        assert result.decision == ActivationDecision.ACTIVATE
        assert result.intent in (VisionIntent.PUZZLE, VisionIntent.LOOK)
        assert result.confidence >= 0.80

        # Intent should suggest puzzle mode
        if result.intent == VisionIntent.PUZZLE:
            assert result.suggested_mode == "puzzle"

    def test_painting_intent_triggers_correct_mode(self):
        classifier = VisionIntentClassifier()

        result = classifier.classify("What do you think of this painting?")
        assert result.intent == VisionIntent.PAINTING
        assert result.suggested_mode == "painting"

    def test_general_question_no_activation(self):
        classifier = VisionIntentClassifier()

        result = classifier.classify("What is the weather today?")
        assert result.decision == ActivationDecision.SKIP

    def test_ambiguous_gets_ask(self):
        classifier = VisionIntentClassifier(auto_threshold=0.95, ask_threshold=0.50)

        result = classifier.classify("How does this look?")
        # "how does this look" has moderate confidence
        if 0.50 <= result.confidence < 0.95:
            assert result.decision == ActivationDecision.ASK


class TestMultiSourceIntegration:
    """Test using different image sources interchangeably."""

    @patch("missy.vision.sources._get_cv2")
    def test_file_source_to_analysis(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        # Create a test image file
        img = np.full((200, 300, 3), 128, dtype=np.uint8)
        mock_cv2.imread.return_value = img
        f = tmp_path / "test_painting.jpg"
        f.write_bytes(b"fake")

        # Acquire from file
        source = FileSource(str(f))
        frame = source.acquire()
        assert frame.source_type == SourceType.FILE
        assert frame.width == 300
        assert frame.height == 200

        # Process through pipeline
        pipeline = ImagePipeline(PipelineConfig(
            normalize_exposure=False, target_dimension=200
        ))
        processed = pipeline.process(frame.image)

        # Build analysis
        request = AnalysisRequest(
            image=processed,
            mode=AnalysisMode.PAINTING,
        )
        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(request)
        assert "supportive" in prompt.lower()


class TestSceneMemoryLifecycle:
    """Test scene memory across the full session lifecycle."""

    def test_full_lifecycle(self):
        mgr = SceneManager(max_sessions=3)

        # Create puzzle session
        s1 = mgr.create_session("puzzle-1", TaskType.PUZZLE, max_frames=5)

        # Add frames and observations over time
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            s1.add_frame(img, source="webcam:/dev/video0")
            s1.add_observation(f"Step {i+1} observation")

        s1.update_state(pieces_placed=15, total_pieces=500)

        # Verify state
        assert s1.frame_count == 3
        assert len(s1.observations) == 3
        assert s1.state["pieces_placed"] == 15

        # Summary for audit
        summary = s1.summarize()
        assert summary["task_type"] == "puzzle"
        assert summary["frame_count"] == 3
        assert summary["active"] is True

        # Close session
        s1.close()
        assert not s1.is_active

        # List should show closed session
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["active"] is False


class TestProviderFormatIntegration:
    """Test that different providers get correctly formatted messages."""

    def test_all_providers_produce_valid_messages(self):
        b64 = "test_image_data"
        prompt = "Describe what you see"

        for provider in ["anthropic", "openai", "ollama"]:
            msg = build_vision_message(provider, b64, prompt)
            assert msg["role"] == "user"
            assert len(msg["content"]) == 2

            # Image block should have correct type
            img_block = msg["content"][0]
            if provider == "anthropic":
                assert img_block["type"] == "image"
                assert "source" in img_block
            else:
                assert img_block["type"] == "image_url"
                assert "image_url" in img_block

            # Text block
            assert msg["content"][1]["type"] == "text"
            assert msg["content"][1]["text"] == prompt

    def test_format_consistency(self):
        """Same image data should produce consistent formatting."""
        b64 = "consistent_test_data"

        result1 = format_image_for_provider("anthropic", b64)
        result2 = format_image_for_provider("anthropic", b64)
        assert result1 == result2


class TestVisionToolIntegration:
    """Test vision tools with the scene memory system."""

    def test_vision_scene_tool_creates_and_uses_session(self):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool, VisionSceneMemoryTool

        scene_tool = VisionSceneMemoryTool()
        analyze_tool = VisionAnalyzeTool()

        # Create a session
        result = scene_tool.execute(
            action="create",
            task_id="integration-test",
            task_type="puzzle",
        )
        assert result.success
        data = json.loads(result.output)
        assert data["task_id"] == "integration-test"

        # Add observations
        result = scene_tool.execute(
            action="add_observation",
            task_id="integration-test",
            observation="Found blue edge pieces",
        )
        assert result.success

        # Build analysis prompt
        result = analyze_tool.execute(mode="puzzle", context="Help with edges")
        assert result.success
        data = json.loads(result.output)
        assert "jigsaw" in data["prompt"].lower()

        # Close session
        result = scene_tool.execute(action="close", task_id="integration-test")
        assert result.success
