"""Tests for the graph_query agent tool (F04)."""

from __future__ import annotations

import os
import tempfile

import pytest

from missy.memory.graph_store import Entity, GraphMemoryStore, Relationship
from missy.tools.base import ToolResult
from missy.tools.builtin.graph_tools import GraphQueryTool


@pytest.fixture
def store() -> GraphMemoryStore:
    d = tempfile.mkdtemp()
    return GraphMemoryStore(db_path=os.path.join(d, "graph.db"))


@pytest.fixture
def seeded(store: GraphMemoryStore) -> GraphMemoryStore:
    vg = Entity.new("video_generate", "tool")
    comfy = Entity.new("comfyui", "tool")
    ffmpeg = Entity.new("ffmpeg", "tool")
    for e in (vg, comfy, ffmpeg):
        store.add_entity(e)
    store.add_relationship(
        Relationship.new(
            source_id=vg.id, target_id=comfy.id, relation_type="uses", context="vg uses comfyui"
        )
    )
    return store


class TestGraphQueryTool:
    def test_metadata(self, store: GraphMemoryStore) -> None:
        tool = GraphQueryTool(store=store)
        assert tool.name == "graph_query"
        assert tool.permissions.network is False
        assert tool.permissions.shell is False
        assert tool.permissions.filesystem_write is False

    def test_schema_has_required_query(self, store: GraphMemoryStore) -> None:
        schema = GraphQueryTool(store=store).get_schema()
        assert schema["name"] == "graph_query"
        assert "query" in schema["parameters"]["required"]
        assert "query" in schema["parameters"]["properties"]

    def test_returns_related_entities_and_relationships(self, seeded: GraphMemoryStore) -> None:
        result = GraphQueryTool(store=seeded).execute(query="video_generate")
        assert isinstance(result, ToolResult)
        assert result.success is True
        out = result.output
        names = {e["name"] for e in out["entities"]}
        assert "video_generate" in names
        assert "comfyui" in names
        assert out["relationship_count"] >= 1
        assert out["relationships"][0]["type"] == "uses"
        assert "video_generate" in out["subgraph"]

    def test_empty_query_is_rejected(self, seeded: GraphMemoryStore) -> None:
        result = GraphQueryTool(store=seeded).execute(query="   ")
        assert result.success is False
        assert "non-empty" in (result.error or "")

    def test_non_string_query_is_rejected(self, seeded: GraphMemoryStore) -> None:
        result = GraphQueryTool(store=seeded).execute(query=None)  # type: ignore[arg-type]
        assert result.success is False

    def test_unknown_query_succeeds_with_empty_result(self, seeded: GraphMemoryStore) -> None:
        result = GraphQueryTool(store=seeded).execute(query="nonexistent-topic-xyz")
        assert result.success is True
        assert result.output["entity_count"] == 0

    def test_max_entities_is_clamped(self, seeded: GraphMemoryStore) -> None:
        # Absurd value must not raise; it is clamped to the 1..50 cap.
        result = GraphQueryTool(store=seeded).execute(query="video_generate", max_entities=99999)
        assert result.success is True

    def test_max_entities_invalid_type_defaults(self, seeded: GraphMemoryStore) -> None:
        result = GraphQueryTool(store=seeded).execute(
            query="video_generate",
            max_entities="lots",  # type: ignore[arg-type]
        )
        assert result.success is True

    def test_registered_in_builtins(self) -> None:
        from missy.tools.builtin import _ALL_TOOL_CLASSES

        assert GraphQueryTool in _ALL_TOOL_CLASSES

    def test_default_store_is_constructed_when_not_injected(self) -> None:
        # Constructing without an explicit store must not raise (points at the
        # real ~/.missy graph store, which is created on demand).
        tool = GraphQueryTool()
        assert tool._store is not None
