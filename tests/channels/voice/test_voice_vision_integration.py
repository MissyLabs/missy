"""Tests for voice server → vision intent → capture integration.

Verifies the audio-triggered vision activation path: transcript classification,
camera auto-capture, metadata population, and error fallbacks.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from missy.channels.voice.stt.base import TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer

# ---------------------------------------------------------------------------
# Fixtures (matching test_voice_server.py patterns)
# ---------------------------------------------------------------------------


@dataclass
class MockEdgeNode:
    node_id: str = "node-1"
    friendly_name: str = "Test Node"
    room: str = "office"
    paired: bool = True
    policy_mode: str = "full"
    ip_address: str = "192.168.1.100"
    hardware_profile: dict = None
    audio_logging: bool = False
    audio_log_dir: str = ""

    def __post_init__(self):
        if self.hardware_profile is None:
            self.hardware_profile = {}


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.verify_token = MagicMock(return_value=True)
    reg.get_node = MagicMock(return_value=MockEdgeNode())
    reg.mark_online = MagicMock()
    reg.mark_offline = MagicMock()
    return reg


@pytest.fixture
def mock_stt():
    stt = MagicMock()
    stt.name = "test-stt"
    stt.load = MagicMock()
    stt.unload = MagicMock()
    return stt


@pytest.fixture
def mock_tts():
    tts = MagicMock()
    tts.name = "test-tts"
    tts.load = MagicMock()
    tts.unload = MagicMock()
    tts.synthesize = AsyncMock(return_value=AudioBuffer(
        data=b"\x00" * 44100, sample_rate=22050, channels=1, format="wav"
    ))
    return tts


@pytest.fixture
def mock_agent_callback():
    return AsyncMock(return_value="Agent response")


@pytest.fixture
def server(mock_registry, mock_stt, mock_tts, mock_agent_callback):
    from missy.channels.voice.server import VoiceServer
    mock_pairing = MagicMock()
    mock_presence = MagicMock()
    return VoiceServer(
        registry=mock_registry,
        pairing_manager=mock_pairing,
        presence_store=mock_presence,
        stt_engine=mock_stt,
        tts_engine=mock_tts,
        agent_callback=mock_agent_callback,
        host="127.0.0.1",
        port=0,
    )


# ---------------------------------------------------------------------------
# Vision intent detection from voice
# ---------------------------------------------------------------------------


class TestVoiceVisionIntentDetection:
    """Test that transcripts with vision-related phrases trigger intent classification."""

    @pytest.mark.asyncio
    async def test_puzzle_intent_detected(self, server, mock_stt, mock_agent_callback):
        """Puzzle-related transcript should populate vision metadata."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Where does this puzzle piece go?",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.PUZZLE,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="puzzle piece",
                suggested_mode="puzzle",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        # Agent should have been called with vision metadata
        mock_agent_callback.assert_awaited_once()
        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_intent") == "puzzle"
        assert metadata.get("vision_confidence") == 0.95
        assert metadata.get("vision_suggested_mode") == "puzzle"

    @pytest.mark.asyncio
    async def test_painting_intent_detected(self, server, mock_stt, mock_agent_callback):
        """Painting transcript triggers painting mode metadata."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="What do you think of my painting?",
            confidence=0.90, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.PAINTING,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.85,
                trigger_phrase="painting",
                suggested_mode="painting",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_intent") == "painting"
        assert metadata.get("vision_suggested_mode") == "painting"

    @pytest.mark.asyncio
    async def test_no_vision_intent(self, server, mock_stt, mock_agent_callback):
        """Normal transcript should not populate vision metadata."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="What is the weather today?",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        mock_agent_callback.assert_awaited_once()
        _, _, metadata = mock_agent_callback.call_args[0]
        assert "vision_intent" not in metadata
        assert "vision_image_base64" not in metadata

    @pytest.mark.asyncio
    async def test_look_at_this_intent(self, server, mock_stt, mock_agent_callback):
        """Explicit look request should trigger vision activation."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Missy, look at this",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.LOOK,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="look at this",
                suggested_mode="general",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_intent") == "look"


# ---------------------------------------------------------------------------
# Camera capture from voice trigger
# ---------------------------------------------------------------------------


class TestVoiceVisionCapture:
    """Test the camera capture flow triggered by voice intent."""

    @pytest.mark.asyncio
    async def test_successful_capture_populates_base64(self, server, mock_stt, mock_agent_callback):
        """When camera is available and capture succeeds, base64 image is in metadata."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Look at this puzzle piece",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        mock_camera = MagicMock()
        mock_camera.device_path = "/dev/video0"

        mock_capture_result = MagicMock()
        mock_capture_result.success = True
        mock_capture_result.image = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_capture_result.width = 640
        mock_capture_result.height = 480

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.PUZZLE,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="puzzle piece",
                suggested_mode="puzzle",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=mock_camera), \
                 patch("missy.vision.capture.CameraHandle") as MockHandle, \
                 patch("missy.vision.pipeline.ImagePipeline") as MockPipeline, \
                 patch("cv2.imencode") as mock_encode, \
                 patch("cv2.IMWRITE_JPEG_QUALITY", 1):

                mock_handle_inst = MagicMock()
                mock_handle_inst.capture.return_value = mock_capture_result
                MockHandle.return_value = mock_handle_inst

                mock_pipeline_inst = MagicMock()
                mock_pipeline_inst.process.return_value = np.full((480, 640, 3), 128, dtype=np.uint8)
                MockPipeline.return_value = mock_pipeline_inst

                mock_encode.return_value = (True, MagicMock(tobytes=MagicMock(return_value=b"jpegdata")))

                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_capture_success") is True
        assert "vision_image_base64" in metadata

    @pytest.mark.asyncio
    async def test_no_camera_sets_error(self, server, mock_stt, mock_agent_callback):
        """When no camera is found, metadata should indicate failure."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Look at this puzzle piece",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.LOOK,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="look at",
                suggested_mode="general",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_capture_success") is False
        assert "No camera" in metadata.get("vision_capture_error", "")

    @pytest.mark.asyncio
    async def test_capture_failure_sets_error(self, server, mock_stt, mock_agent_callback):
        """When capture fails, metadata should indicate the error."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Look at this",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        mock_camera = MagicMock()
        mock_camera.device_path = "/dev/video0"

        mock_capture_result = MagicMock()
        mock_capture_result.success = False
        mock_capture_result.error = "Device busy"

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.LOOK,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="look at",
                suggested_mode="general",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=mock_camera), \
                 patch("missy.vision.capture.CameraHandle") as MockHandle:

                mock_handle_inst = MagicMock()
                mock_handle_inst.capture.return_value = mock_capture_result
                MockHandle.return_value = mock_handle_inst

                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_capture_success") is False

    @pytest.mark.asyncio
    async def test_capture_exception_graceful(self, server, mock_stt, mock_agent_callback):
        """Exceptions during capture should not crash the audio handler."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Look at this",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        mock_camera = MagicMock()
        mock_camera.device_path = "/dev/video0"

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.LOOK,
                decision=ActivationDecision.ACTIVATE,
                confidence=0.95,
                trigger_phrase="look at",
                suggested_mode="general",
            )

            with patch("missy.vision.discovery.find_preferred_camera", return_value=mock_camera), \
                 patch("missy.vision.capture.CameraHandle", side_effect=RuntimeError("device broken")):

                await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        # Agent should still be called despite vision failure
        mock_agent_callback.assert_awaited_once()
        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata.get("vision_capture_success") is False

    @pytest.mark.asyncio
    async def test_vision_import_error_graceful(self, server, mock_stt, mock_agent_callback):
        """If vision module not installed, audio processing continues normally."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Look at this puzzle piece",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        # Should not crash — the import failure is caught
        with patch("builtins.__import__", side_effect=ImportError("no vision")), contextlib.suppress(ImportError):
            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)


# ---------------------------------------------------------------------------
# Metadata structure
# ---------------------------------------------------------------------------


class TestVoiceVisionMetadata:
    """Verify the metadata structure populated by vision integration."""

    @pytest.mark.asyncio
    async def test_base_metadata_always_present(self, server, mock_stt, mock_agent_callback):
        """Room, node_id, etc. should always be in metadata."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="Hello world",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode(room="kitchen", node_id="node-42")

        await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        assert metadata["room"] == "kitchen"
        assert metadata["node_id"] == "node-42"
        assert "confidence" in metadata
        assert "language" in metadata

    @pytest.mark.asyncio
    async def test_ask_decision_skips_capture(self, server, mock_stt, mock_agent_callback):
        """ASK decision should not auto-capture."""
        mock_stt.transcribe = AsyncMock(return_value=TranscriptionResult(
            text="How does this look?",
            confidence=0.95, processing_ms=100, language="en",
        ))
        ws = AsyncMock()
        node = MockEdgeNode()

        with patch("missy.vision.intent.classify_vision_intent") as mock_classify:
            from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent
            mock_classify.return_value = IntentResult(
                intent=VisionIntent.PAINTING,
                decision=ActivationDecision.ASK,
                confidence=0.65,
                trigger_phrase="how does this look",
                suggested_mode="painting",
            )

            await server._handle_audio(ws, node=node, audio_buffer=b"\x00" * 16000, sample_rate=16000)

        _, _, metadata = mock_agent_callback.call_args[0]
        # ASK decision means no auto-capture
        assert "vision_image_base64" not in metadata
        assert "vision_capture_success" not in metadata
