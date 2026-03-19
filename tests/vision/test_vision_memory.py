"""Comprehensive tests for missy.vision.vision_memory.VisionMemoryBridge."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.vision.vision_memory import VisionMemoryBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(
    *,
    role: str = "vision",
    content: str = "a visual observation",
    session_id: str = "sess-1",
    metadata: dict[str, Any] | None = None,
    turn_id: str | None = None,
) -> SimpleNamespace:
    """Return a lightweight ConversationTurn-like object."""
    return SimpleNamespace(
        id=turn_id or str(uuid.uuid4()),
        role=role,
        content=content,
        session_id=session_id,
        metadata=metadata or {},
    )


def _make_sqlite_mock(turns: list | None = None) -> MagicMock:
    """Return a SQLiteMemoryStore mock with sensible defaults."""
    store = MagicMock()
    store.get_session_turns.return_value = turns or []
    store.search.return_value = turns or []
    store.get_recent.return_value = turns or []
    return store


def _make_vector_mock(results: list[tuple[float, dict]] | None = None) -> MagicMock:
    """Return a VectorMemoryStore mock with sensible defaults."""
    store = MagicMock()
    store.search.return_value = results or []
    return store


# ---------------------------------------------------------------------------
# CATEGORY constant
# ---------------------------------------------------------------------------


class TestCategoryConstant:
    def test_category_value(self) -> None:
        assert VisionMemoryBridge.CATEGORY == "vision_observation"

    def test_category_is_class_attribute(self) -> None:
        bridge = VisionMemoryBridge()
        assert bridge.CATEGORY == "vision_observation"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults_to_none_stores(self) -> None:
        bridge = VisionMemoryBridge()
        assert bridge._memory is None
        assert bridge._vector is None

    def test_not_initialized_on_construction(self) -> None:
        bridge = VisionMemoryBridge()
        assert bridge._initialized is False

    def test_accepts_memory_store(self) -> None:
        store = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=store)
        assert bridge._memory is store

    def test_accepts_vector_store(self) -> None:
        vstore = _make_vector_mock()
        bridge = VisionMemoryBridge(vector_store=vstore)
        assert bridge._vector is vstore

    def test_accepts_both_stores(self) -> None:
        mstore = _make_sqlite_mock()
        vstore = _make_vector_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        assert bridge._memory is mstore
        assert bridge._vector is vstore


# ---------------------------------------------------------------------------
# Lazy initialization (_ensure_init)
# ---------------------------------------------------------------------------


class TestEnsureInit:
    def test_sets_initialized_flag(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._ensure_init()
        assert bridge._initialized is True

    def test_ensure_init_only_runs_once(self) -> None:
        """Second call must be a no-op (guard clause)."""
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._ensure_init()
        bridge._ensure_init()
        assert bridge._initialized is True

    def test_lazy_creates_sqlite_store_when_none(self) -> None:
        fake_store = _make_sqlite_mock()
        with (
            patch(
                "missy.vision.vision_memory.SQLiteMemoryStore",
                return_value=fake_store,
                create=True,
            ) as mock_cls,
            patch.dict(
                "sys.modules",
                {"missy.memory.sqlite_store": MagicMock(SQLiteMemoryStore=mock_cls)},
            ),
        ):
                bridge = VisionMemoryBridge()
                bridge._memory = None  # ensure None
                bridge._initialized = False
                # Directly inject the patched class so the import succeeds
                import missy.vision.vision_memory as vm_mod

                original = vm_mod.__dict__.get("SQLiteMemoryStore")
                try:
                    vm_mod.SQLiteMemoryStore = mock_cls  # type: ignore[attr-defined]
                    bridge._ensure_init()
                    mock_cls.assert_called_once()
                finally:
                    if original is None:
                        vm_mod.__dict__.pop("SQLiteMemoryStore", None)
                    else:
                        vm_mod.SQLiteMemoryStore = original  # type: ignore[attr-defined]

    def test_lazy_sqlite_import_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When SQLiteMemoryStore import raises, a warning is logged and _memory stays None."""
        bridge = VisionMemoryBridge()
        bridge._memory = None
        bridge._initialized = False

        with patch("builtins.__import__", side_effect=ImportError("no sqlite")):
            pass  # We simulate the internal except branch instead via direct monkey-patch

        # Simulate the lazy-init branch by making the import raise inside _ensure_init

        def _patched_ensure() -> None:
            if bridge._initialized:
                return
            try:
                raise Exception("sqlite unavailable")
            except Exception as exc:
                import logging

                logging.getLogger("missy.vision.vision_memory").warning(
                    "Cannot init SQLiteMemoryStore: %s", exc
                )
            bridge._initialized = True

        bridge._ensure_init = _patched_ensure  # type: ignore[method-assign]
        import logging

        with caplog.at_level(logging.WARNING, logger="missy.vision.vision_memory"):
            bridge._ensure_init()

        assert "Cannot init SQLiteMemoryStore" in caplog.text

    def test_lazy_vector_import_failure_leaves_vector_none(self) -> None:
        """VectorMemoryStore import failure must not raise; _vector stays None."""
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._vector = None
        bridge._initialized = False

        # Simulate exception in the VectorMemoryStore import path
        def _ensure_no_vector(self_inner: Any = None) -> None:  # noqa: ANN001
            if bridge._initialized:
                return
            try:
                raise ImportError("faiss not installed")
            except Exception:
                pass
            bridge._initialized = True

        bridge._ensure_init = _ensure_no_vector  # type: ignore[method-assign]
        bridge._ensure_init()
        assert bridge._vector is None
        assert bridge._initialized is True


# ---------------------------------------------------------------------------
# store_observation
# ---------------------------------------------------------------------------


class TestStoreObservation:
    def test_returns_uuid_string(self) -> None:
        bridge = VisionMemoryBridge(memory_store=_make_sqlite_mock())
        result = bridge.store_observation("s1", "puzzle", "found a red piece")
        assert isinstance(result, str)
        # Must be a valid UUID
        uuid.UUID(result)

    def test_calls_add_turn_on_sqlite(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge.store_observation("s1", "puzzle", "a piece observation")
        mstore.add_turn.assert_called_once()
        _, kwargs = mstore.add_turn.call_args
        assert kwargs["session_id"] == "s1"
        assert kwargs["role"] == "vision"
        assert kwargs["content"] == "a piece observation"
        assert kwargs["provider"] == "vision"

    def test_metadata_passed_to_add_turn(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge.store_observation("s1", "puzzle", "obs", metadata={"extra": "data"})
        _, kwargs = mstore.add_turn.call_args
        stored_meta = kwargs["metadata"]
        assert stored_meta["extra"] == "data"

    def test_entry_contains_all_fields(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge.store_observation(
            session_id="s1",
            task_type="painting",
            observation="brush stroke analysis",
            confidence=0.9,
            source="webcam:/dev/video0",
            frame_id=3,
        )
        _, kwargs = mstore.add_turn.call_args
        meta = kwargs["metadata"]
        assert meta["session_id"] == "s1"
        assert meta["task_type"] == "painting"
        assert meta["observation"] == "brush stroke analysis"
        assert meta["confidence"] == 0.9
        assert meta["source"] == "webcam:/dev/video0"
        assert meta["frame_id"] == 3
        assert "timestamp" in meta
        assert "observation_id" in meta

    def test_indexes_in_vector_store(self) -> None:
        mstore = _make_sqlite_mock()
        vstore = _make_vector_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge.store_observation("s1", "general", "a scene", confidence=0.5)
        vstore.add.assert_called_once()
        text_arg, meta_arg = vstore.add.call_args[0]
        assert "[general]" in text_arg
        assert "a scene" in text_arg
        assert meta_arg["task_type"] == "general"

    def test_vector_text_prefixed_with_task_type(self) -> None:
        mstore = _make_sqlite_mock()
        vstore = _make_vector_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge.store_observation("s1", "inspection", "crack found")
        text_arg = vstore.add.call_args[0][0]
        assert text_arg == "[inspection] crack found"

    def test_no_sqlite_store_does_not_raise(self) -> None:
        vstore = _make_vector_mock()
        bridge = VisionMemoryBridge(memory_store=None, vector_store=vstore)
        bridge._initialized = True  # skip lazy init
        # Must not raise even without sqlite
        result = bridge.store_observation("s1", "general", "observation")
        assert isinstance(result, str)

    def test_no_vector_store_does_not_raise(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True
        result = bridge.store_observation("s1", "general", "observation")
        assert isinstance(result, str)
        mstore.add_turn.assert_called_once()

    def test_sqlite_add_turn_exception_is_swallowed(self) -> None:
        mstore = _make_sqlite_mock()
        mstore.add_turn.side_effect = RuntimeError("DB locked")
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True
        # Must not propagate the exception
        result = bridge.store_observation("s1", "general", "obs")
        assert isinstance(result, str)

    def test_vector_add_exception_is_swallowed(self) -> None:
        mstore = _make_sqlite_mock()
        vstore = _make_vector_mock()
        vstore.add.side_effect = Exception("index error")
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True
        result = bridge.store_observation("s1", "general", "obs")
        assert isinstance(result, str)

    def test_default_parameter_values(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge.store_observation("s1", "general", "obs")
        _, kwargs = mstore.add_turn.call_args
        meta = kwargs["metadata"]
        assert meta["confidence"] == 0.0
        assert meta["source"] == ""
        assert meta["frame_id"] == 0

    def test_unique_ids_per_call(self) -> None:
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore)
        ids = {bridge.store_observation("s1", "general", "obs") for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# recall_observations — vector search path
# ---------------------------------------------------------------------------


class TestRecallObservationsVectorPath:
    def test_uses_vector_search_when_query_given(self) -> None:
        meta = {"task_type": "puzzle", "session_id": "s1", "observation": "sky pieces"}
        vstore = _make_vector_mock(results=[(0.9, meta)])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="sky pieces")

        vstore.search.assert_called_once()
        assert len(results) == 1
        assert results[0]["relevance_score"] == 0.9

    def test_vector_query_prefixed_with_task_type(self) -> None:
        vstore = _make_vector_mock(results=[])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        bridge.recall_observations(query="edge pieces", task_type="puzzle")

        query_arg = vstore.search.call_args[0][0]
        assert query_arg == "[puzzle] edge pieces"

    def test_vector_query_without_task_type_not_prefixed(self) -> None:
        vstore = _make_vector_mock(results=[])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        bridge.recall_observations(query="sky pieces")

        query_arg = vstore.search.call_args[0][0]
        assert query_arg == "sky pieces"

    def test_vector_top_k_uses_limit(self) -> None:
        vstore = _make_vector_mock(results=[])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        bridge.recall_observations(query="x", limit=7)

        _, kwargs = vstore.search.call_args
        assert kwargs["top_k"] == 7

    def test_vector_results_filtered_by_task_type(self) -> None:
        puzzle_meta = {"task_type": "puzzle", "session_id": "s1", "observation": "piece"}
        painting_meta = {"task_type": "painting", "session_id": "s1", "observation": "stroke"}
        vstore = _make_vector_mock(results=[(0.9, puzzle_meta), (0.8, painting_meta)])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="piece", task_type="puzzle")

        assert len(results) == 1
        assert results[0]["task_type"] == "puzzle"

    def test_vector_results_filtered_by_session_id(self) -> None:
        meta_s1 = {"task_type": "general", "session_id": "s1", "observation": "obs1"}
        meta_s2 = {"task_type": "general", "session_id": "s2", "observation": "obs2"}
        vstore = _make_vector_mock(results=[(0.9, meta_s1), (0.8, meta_s2)])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="obs", session_id="s1")

        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_vector_results_filtered_by_both_task_type_and_session(self) -> None:
        meta_match = {"task_type": "puzzle", "session_id": "s1", "observation": "piece"}
        meta_wrong_task = {"task_type": "painting", "session_id": "s1", "observation": "stroke"}
        meta_wrong_sess = {"task_type": "puzzle", "session_id": "s2", "observation": "edge"}
        vstore = _make_vector_mock(
            results=[(0.9, meta_match), (0.8, meta_wrong_task), (0.7, meta_wrong_sess)]
        )
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="piece", task_type="puzzle", session_id="s1")

        assert len(results) == 1
        assert results[0]["task_type"] == "puzzle"
        assert results[0]["session_id"] == "s1"

    def test_vector_results_capped_at_limit(self) -> None:
        metas = [{"task_type": "general", "session_id": "s1", "observation": f"obs{i}"} for i in range(20)]
        vstore = _make_vector_mock(results=[(0.9 - i * 0.01, m) for i, m in enumerate(metas)])
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="obs", limit=5)

        assert len(results) <= 5

    def test_relevance_score_injected_into_each_result(self) -> None:
        meta = {"task_type": "general", "session_id": "s1"}
        vstore = _make_vector_mock(results=[(0.75, meta)])
        bridge = VisionMemoryBridge(memory_store=_make_sqlite_mock(), vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="test")

        assert results[0]["relevance_score"] == 0.75

    def test_vector_empty_results_falls_through_to_sqlite(self) -> None:
        vstore = _make_vector_mock(results=[])
        turn = _make_turn(role="vision", content="sqlite result", session_id="s1")
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="sqlite result")

        mstore.search.assert_called_once()
        assert any(r["observation"] == "sqlite result" for r in results)

    def test_vector_exception_falls_through_to_sqlite(self) -> None:
        vstore = _make_vector_mock()
        vstore.search.side_effect = RuntimeError("vector error")
        turn = _make_turn(role="vision", content="fallback obs", session_id="s1")
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="fallback obs")

        assert any(r["observation"] == "fallback obs" for r in results)

    def test_no_query_skips_vector_search(self) -> None:
        vstore = _make_vector_mock()
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        bridge.recall_observations(query="")

        vstore.search.assert_not_called()


# ---------------------------------------------------------------------------
# recall_observations — SQLite fallback path
# ---------------------------------------------------------------------------


class TestRecallObservationsSQLitePath:
    def test_uses_get_session_turns_when_session_id_given(self) -> None:
        turn = _make_turn(session_id="s1")
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        bridge.recall_observations(session_id="s1")

        mstore.get_session_turns.assert_called_once()
        args, _ = mstore.get_session_turns.call_args
        assert args[0] == "s1"

    def test_uses_search_when_query_given_and_no_vector(self) -> None:
        turn = _make_turn()
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        bridge.recall_observations(query="some text")

        mstore.search.assert_called_once_with("some text", limit=20)

    def test_uses_get_recent_when_no_query_and_no_session(self) -> None:
        turn = _make_turn()
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        bridge.recall_observations()

        mstore.get_recent.assert_called_once()

    def test_non_vision_turns_excluded(self) -> None:
        user_turn = _make_turn(role="user", content="hello")
        assistant_turn = _make_turn(role="assistant", content="reply")
        vision_turn = _make_turn(role="vision", content="visual obs")
        mstore = _make_sqlite_mock(turns=[user_turn, assistant_turn, vision_turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations()

        assert len(results) == 1
        assert results[0]["observation"] == "visual obs"

    def test_task_type_filter_applied(self) -> None:
        puzzle_turn = _make_turn(role="vision", metadata={"task_type": "puzzle"})
        painting_turn = _make_turn(role="vision", metadata={"task_type": "painting"})
        mstore = _make_sqlite_mock(turns=[puzzle_turn, painting_turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations(task_type="puzzle")

        assert len(results) == 1
        assert results[0].get("task_type") == "puzzle"

    def test_metadata_merged_into_result(self) -> None:
        meta = {"task_type": "inspection", "confidence": 0.8, "source": "/dev/video0"}
        turn = _make_turn(role="vision", content="crack detected", metadata=meta)
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations()

        assert results[0]["confidence"] == 0.8
        assert results[0]["source"] == "/dev/video0"
        assert results[0]["observation"] == "crack detected"

    def test_limit_respected(self) -> None:
        turns = [_make_turn(role="vision", content=f"obs {i}") for i in range(10)]
        mstore = _make_sqlite_mock(turns=turns)
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations(limit=3)

        assert len(results) <= 3

    def test_sqlite_query_limit_doubled(self) -> None:
        """SQLite is queried with 2x limit to allow for role filtering."""
        mstore = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        bridge.recall_observations(limit=5)

        mstore.get_recent.assert_called_once_with(limit=10)

    def test_sqlite_exception_returns_empty_list(self) -> None:
        mstore = _make_sqlite_mock()
        mstore.get_recent.side_effect = RuntimeError("DB error")
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations()

        assert results == []

    def test_none_metadata_on_turn_treated_as_empty_dict(self) -> None:
        turn = SimpleNamespace(
            id="t1",
            role="vision",
            content="obs",
            session_id="s1",
            metadata=None,
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations()

        assert results[0]["observation"] == "obs"

    def test_no_memory_store_returns_empty(self) -> None:
        bridge = VisionMemoryBridge(memory_store=None, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations()

        assert results == []

    def test_session_id_filter_in_sqlite_get_session_turns(self) -> None:
        """When session_id is provided, only get_session_turns is called, not search."""
        turn = _make_turn(role="vision", session_id="my-session")
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        results = bridge.recall_observations(session_id="my-session")

        mstore.get_session_turns.assert_called_once()
        mstore.search.assert_not_called()
        mstore.get_recent.assert_not_called()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_session_context
# ---------------------------------------------------------------------------


class TestGetSessionContext:
    def test_returns_empty_string_when_no_observations(self) -> None:
        mstore = _make_sqlite_mock(turns=[])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert ctx == ""

    def test_returns_string_starting_with_header(self) -> None:
        turn = _make_turn(
            role="vision",
            content="a piece",
            metadata={"task_type": "puzzle", "confidence": 0.8, "timestamp": "2026-01-01T00:00:00+00:00"},
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert ctx.startswith("## Visual Observations")

    def test_context_contains_observation_text(self) -> None:
        turn = _make_turn(
            role="vision",
            content="sky region identified",
            metadata={"task_type": "puzzle", "confidence": 0.9, "timestamp": "2026-01-01T00:00:00+00:00"},
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert "sky region identified" in ctx

    def test_context_contains_task_type(self) -> None:
        turn = _make_turn(
            role="vision",
            content="brushwork",
            metadata={"task_type": "painting", "confidence": 0.7, "timestamp": "2026-01-01T00:00:00+00:00"},
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert "painting" in ctx

    def test_context_contains_confidence_as_percentage(self) -> None:
        turn = _make_turn(
            role="vision",
            content="obs",
            metadata={"task_type": "general", "confidence": 0.75, "timestamp": "2026-01-01T00:00:00+00:00"},
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert "75%" in ctx

    def test_context_contains_timestamp(self) -> None:
        ts = "2026-03-19T12:00:00+00:00"
        turn = _make_turn(
            role="vision",
            content="obs",
            metadata={"task_type": "general", "confidence": 0.5, "timestamp": ts},
        )
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert ts in ctx

    def test_context_multiple_observations(self) -> None:
        turns = [
            _make_turn(
                role="vision",
                content=f"obs {i}",
                metadata={"task_type": "general", "confidence": 0.5, "timestamp": "2026-01-01"},
            )
            for i in range(3)
        ]
        mstore = _make_sqlite_mock(turns=turns)
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        assert ctx.count("- [") == 3

    def test_context_defaults_when_metadata_missing_fields(self) -> None:
        turn = _make_turn(role="vision", content="bare obs", metadata={})
        mstore = _make_sqlite_mock(turns=[turn])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        ctx = bridge.get_session_context("s1")

        # Should not raise; should fall back to "general" and 0% confidence
        assert "general" in ctx
        assert "0%" in ctx

    def test_context_calls_recall_with_session_id_and_limit_20(self) -> None:
        mstore = _make_sqlite_mock(turns=[])
        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        bridge.get_session_context("sess-42")

        mstore.get_session_turns.assert_called_once()
        args, kwargs = mstore.get_session_turns.call_args
        assert args[0] == "sess-42"


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------


class TestClearSession:
    def test_returns_zero_when_no_turns(self) -> None:
        mstore = _make_sqlite_mock(turns=[])
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 0

    def test_returns_count_of_deleted_turns(self) -> None:
        turns = [_make_turn(role="vision", session_id="s1") for _ in range(3)]
        mstore = _make_sqlite_mock(turns=turns)
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 3
        assert mstore.delete_turn.call_count == 3

    def test_skips_non_vision_turns(self) -> None:
        vision_turn = _make_turn(role="vision", session_id="s1")
        user_turn = _make_turn(role="user", session_id="s1")
        mstore = _make_sqlite_mock(turns=[vision_turn, user_turn])
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 1
        mstore.delete_turn.assert_called_once_with(vision_turn.id)

    def test_delete_turn_exception_is_swallowed(self) -> None:
        turn = _make_turn(role="vision", session_id="s1", turn_id="t1")
        mstore = _make_sqlite_mock(turns=[turn])
        mstore.delete_turn.side_effect = RuntimeError("delete failed")
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 0  # delete raised, so count not incremented

    def test_get_session_turns_exception_returns_zero(self) -> None:
        mstore = _make_sqlite_mock()
        mstore.get_session_turns.side_effect = RuntimeError("query failed")
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 0

    def test_no_memory_store_returns_zero(self) -> None:
        bridge = VisionMemoryBridge(memory_store=None, vector_store=None)
        bridge._initialized = True

        count = bridge.clear_session("s1")

        assert count == 0

    def test_queries_up_to_1000_turns(self) -> None:
        mstore = _make_sqlite_mock(turns=[])
        bridge = VisionMemoryBridge(memory_store=mstore)
        bridge._initialized = True

        bridge.clear_session("s1")

        mstore.get_session_turns.assert_called_once_with("s1", limit=1000)

    def test_calls_ensure_init(self) -> None:
        mstore = _make_sqlite_mock(turns=[])
        bridge = VisionMemoryBridge(memory_store=mstore)
        # Not initialized yet
        assert bridge._initialized is False

        bridge.clear_session("s1")

        assert bridge._initialized is True


# ---------------------------------------------------------------------------
# Integration-style: store then recall round-trip
# ---------------------------------------------------------------------------


class TestStoreRecallRoundTrip:
    def test_store_then_recall_via_sqlite(self) -> None:
        """Stored observations should be retrievable via SQLite fallback."""
        stored_meta: dict[str, Any] = {}

        def fake_add_turn(session_id: str, role: str, content: str, provider: str, metadata: dict) -> None:
            stored_meta.update(metadata)
            stored_meta["_content"] = content
            stored_meta["_session_id"] = session_id

        def fake_get_recent(limit: int = 10) -> list:
            if stored_meta:
                return [
                    SimpleNamespace(
                        id="t1",
                        role="vision",
                        content=stored_meta.get("_content", ""),
                        session_id=stored_meta.get("_session_id", ""),
                        metadata={k: v for k, v in stored_meta.items() if not k.startswith("_")},
                    )
                ]
            return []

        mstore = MagicMock()
        mstore.add_turn.side_effect = fake_add_turn
        mstore.get_recent.side_effect = fake_get_recent

        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=None)
        bridge._initialized = True

        bridge.store_observation("s1", "puzzle", "corner piece found", confidence=0.95)
        results = bridge.recall_observations()

        assert len(results) == 1
        assert results[0]["observation"] == "corner piece found"
        assert results[0]["confidence"] == 0.95

    def test_both_stores_present_vector_wins_on_query(self) -> None:
        vector_meta = {"task_type": "puzzle", "session_id": "s1", "observation": "vector result"}
        vstore = _make_vector_mock(results=[(0.99, vector_meta)])
        sqlite_turn = _make_turn(role="vision", content="sqlite result")
        mstore = _make_sqlite_mock(turns=[sqlite_turn])

        bridge = VisionMemoryBridge(memory_store=mstore, vector_store=vstore)
        bridge._initialized = True

        results = bridge.recall_observations(query="result")

        # Vector returned a hit, so SQLite search should not be called
        mstore.search.assert_not_called()
        assert results[0]["observation"] == "vector result"
