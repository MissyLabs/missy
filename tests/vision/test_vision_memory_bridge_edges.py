"""Tests for VisionMemoryBridge edge cases.

Covers:
- Metadata field protection
- Store without memory/vector stores
- Recall with no results
- Session context building
- Lazy initialization
"""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.vision.vision_memory import VisionMemoryBridge


class TestVisionMemoryBridge:
    """Test VisionMemoryBridge functionality."""

    def test_store_returns_uuid(self):
        """store_observation should return a UUID string."""
        bridge = VisionMemoryBridge(memory_store=MagicMock())
        obs_id = bridge.store_observation(
            session_id="s1",
            task_type="puzzle",
            observation="Found edge piece",
        )
        assert len(obs_id) == 36  # UUID format
        assert "-" in obs_id

    def test_metadata_reserved_keys_filtered(self):
        """Reserved keys in metadata should be filtered out."""
        mock_memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        bridge.store_observation(
            session_id="s1",
            task_type="puzzle",
            observation="test",
            metadata={
                "observation_id": "OVERRIDE_ATTEMPT",
                "session_id": "OVERRIDE_ATTEMPT",
                "custom_key": "allowed",
            },
        )

        # The add_turn call should have been made
        call_args = mock_memory.add_turn.call_args
        stored_metadata = call_args.kwargs.get("metadata", {})
        assert stored_metadata["observation_id"] != "OVERRIDE_ATTEMPT"
        assert stored_metadata["session_id"] == "s1"  # uses the real value
        assert stored_metadata.get("custom_key") == "allowed"

    def test_store_without_memory_store(self):
        """Storing without a memory store should not raise."""
        bridge = VisionMemoryBridge(memory_store=None)
        bridge._initialized = True  # skip lazy init
        obs_id = bridge.store_observation(
            session_id="s1",
            task_type="general",
            observation="test observation",
        )
        assert obs_id  # still returns an ID

    def test_store_handles_memory_exception(self):
        """Store should handle memory store exceptions gracefully."""
        mock_memory = MagicMock()
        mock_memory.add_turn.side_effect = RuntimeError("DB error")
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        obs_id = bridge.store_observation(
            session_id="s1",
            task_type="general",
            observation="test",
        )
        assert obs_id  # should still return ID

    def test_recall_empty_results(self):
        """Recall with no matching observations should return empty list."""
        mock_memory = MagicMock()
        mock_memory.get_recent.return_value = []
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        results = bridge.recall_observations(limit=5)
        assert results == []

    def test_session_context_empty(self):
        """Session context with no observations should return empty string."""
        mock_memory = MagicMock()
        mock_memory.get_session_turns.return_value = []
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        ctx = bridge.get_session_context("s1")
        assert ctx == ""

    def test_session_context_with_observations(self):
        """Session context should format observations into markdown."""
        mock_turn = MagicMock()
        mock_turn.role = "vision"
        mock_turn.content = "Found blue sky pieces"
        mock_turn.session_id = "s1"
        mock_turn.metadata = {
            "task_type": "puzzle",
            "observation": "Found blue sky pieces",
            "confidence": 0.85,
            "timestamp": "2026-03-19T10:00:00",
        }

        mock_memory = MagicMock()
        mock_memory.get_session_turns.return_value = [mock_turn]
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        ctx = bridge.get_session_context("s1")
        assert "Visual Observations" in ctx
        assert "puzzle" in ctx

    def test_clear_session_returns_count(self):
        """clear_session should return the number of removed observations."""
        mock_turn = MagicMock()
        mock_turn.role = "vision"
        mock_turn.id = "turn-1"

        mock_memory = MagicMock()
        mock_memory.get_session_turns.return_value = [mock_turn]
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        count = bridge.clear_session("s1")
        assert count == 1

    def test_clear_session_skips_non_vision_turns(self):
        """clear_session should only remove vision-role turns."""
        vision_turn = MagicMock()
        vision_turn.role = "vision"
        vision_turn.id = "v1"

        user_turn = MagicMock()
        user_turn.role = "user"
        user_turn.id = "u1"

        mock_memory = MagicMock()
        mock_memory.get_session_turns.return_value = [vision_turn, user_turn]
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        count = bridge.clear_session("s1")
        assert count == 1  # only vision turn removed

    def test_confidence_stored_correctly(self):
        """Confidence value should be stored in metadata."""
        mock_memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_memory)

        bridge.store_observation(
            session_id="s1",
            task_type="painting",
            observation="Beautiful color harmony",
            confidence=0.92,
        )

        call_args = mock_memory.add_turn.call_args
        stored_metadata = call_args.kwargs.get("metadata", {})
        assert stored_metadata["confidence"] == 0.92

    def test_vector_store_receives_tagged_text(self):
        """Vector store should receive task-type-prefixed text."""
        mock_vector = MagicMock()
        mock_memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_memory, vector_store=mock_vector)

        bridge.store_observation(
            session_id="s1",
            task_type="puzzle",
            observation="Edge piece found",
        )

        add_call = mock_vector.add.call_args
        assert "[puzzle]" in add_call.args[0]
        assert "Edge piece found" in add_call.args[0]
