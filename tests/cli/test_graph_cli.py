"""Tests for the `missy graph` CLI group (F04)."""

from __future__ import annotations

import inspect
import os
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli
from missy.memory.graph_store import Entity, GraphMemoryStore, Relationship


def _runner() -> CliRunner:
    sig = inspect.signature(CliRunner.__init__)
    kwargs = {"mix_stderr": False} if "mix_stderr" in sig.parameters else {}
    return CliRunner(**kwargs)


def _combined(result) -> str:
    out = result.output
    try:
        err = result.stderr
    except (ValueError, AttributeError):
        err = ""
    if err and err not in out:
        out += err
    return out


@pytest.fixture
def store() -> GraphMemoryStore:
    d = tempfile.mkdtemp()
    return GraphMemoryStore(db_path=os.path.join(d, "graph_cli.db"))


@pytest.fixture
def seeded(store: GraphMemoryStore) -> GraphMemoryStore:
    vg = Entity.new("video_generate", "tool")
    comfy = Entity.new("comfyui", "tool")
    store.add_entity(vg)
    store.add_entity(comfy)
    store.add_relationship(
        Relationship.new(source_id=vg.id, target_id=comfy.id, relation_type="uses", context="c")
    )
    return store


def _invoke(args, store_obj):
    with (
        patch("missy.cli.main._load_subsystems", return_value=object()),
        patch("missy.memory.graph_store.GraphMemoryStore", return_value=store_obj),
    ):
        return _runner().invoke(cli, args)


class TestGraphStats:
    def test_stats_reports_counts(self, seeded: GraphMemoryStore) -> None:
        result = _invoke(["graph", "stats"], seeded)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "2 entit" in out
        assert "1 relationship" in out
        assert "tool" in out  # entity-type breakdown

    def test_stats_empty_graph(self, store: GraphMemoryStore) -> None:
        result = _invoke(["graph", "stats"], store)
        assert result.exit_code == 0, _combined(result)
        assert "0" in _combined(result)


class TestGraphQuery:
    def test_query_shows_related(self, seeded: GraphMemoryStore) -> None:
        result = _invoke(["graph", "query", "video_generate"], seeded)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "video_generate" in out
        assert "comfyui" in out

    def test_query_unknown_reports_none(self, seeded: GraphMemoryStore) -> None:
        result = _invoke(["graph", "query", "zzz-nothing"], seeded)
        assert "No entities related" in _combined(result)


class TestGraphEntity:
    def test_entity_summary(self, seeded: GraphMemoryStore) -> None:
        result = _invoke(["graph", "entity", "video_generate"], seeded)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "video_generate" in out

    def test_entity_unknown(self, seeded: GraphMemoryStore) -> None:
        result = _invoke(["graph", "entity", "no-such-entity"], seeded)
        assert "No entity named" in _combined(result)


class TestGraphAddEntity:
    def test_add_entity_persists(self, store: GraphMemoryStore) -> None:
        result = _invoke(["graph", "add-entity", "AtlasProject", "--type", "project"], store)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "Added entity" in out
        # Name is normalised to lowercase by Entity.new.
        found = store.find_entities(name="atlasproject")
        assert any(e.name == "atlasproject" for e in found)

    def test_add_entity_default_type_is_concept(self, store: GraphMemoryStore) -> None:
        _invoke(["graph", "add-entity", "SomeIdea"], store)
        found = store.find_entities(name="someidea")
        assert found and found[0].entity_type == "concept"
