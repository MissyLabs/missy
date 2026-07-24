"""Tests for the rag_query agent tool (F03)."""

from __future__ import annotations

from missy.retrieval.embedder import HashingEmbedder
from missy.retrieval.engine import RetrievalEngine
from missy.tools.base import BaseTool
from missy.tools.builtin.rag_query import RagQueryTool


def _tool_with_engine():
    eng = RetrievalEngine(embedder=HashingEmbedder(256), max_chars=120, overlap=20)
    return RagQueryTool(engine=eng), eng


class TestQueryAction:
    def test_query_returns_citations(self) -> None:
        tool, eng = _tool_with_engine()
        eng.index_document("k.md", "rotate api keys via the vault set command then restart")
        res = tool.execute(action="query", query="rotate keys", top_k=2)
        assert res.success
        assert res.output["results"]
        top = res.output["results"][0]
        assert top["doc_id"] == "k.md"
        assert "citation" in top and top["citation"].startswith("k.md[")
        assert isinstance(top["source_span"], list) and len(top["source_span"]) == 2

    def test_query_defaults_to_query_action(self) -> None:
        tool, eng = _tool_with_engine()
        eng.index_document("d", "content about retrieval and citations")
        res = tool.execute(query="retrieval")  # no explicit action
        assert res.success

    def test_empty_query_errors(self) -> None:
        tool, _ = _tool_with_engine()
        res = tool.execute(action="query", query="   ")
        assert not res.success
        assert "non-empty" in res.error


class TestIndexActions:
    def test_index_text_then_query(self) -> None:
        tool, _ = _tool_with_engine()
        r1 = tool.execute(action="index_text", doc_id="doc1", text="hello retrieval world")
        assert r1.success and r1.output["chunks_indexed"] >= 1
        r2 = tool.execute(action="query", query="retrieval", top_k=1)
        assert r2.output["results"][0]["doc_id"] == "doc1"

    def test_index_text_requires_doc_id_and_text(self) -> None:
        tool, _ = _tool_with_engine()
        assert not tool.execute(action="index_text", doc_id="d").success
        assert not tool.execute(action="index_text", text="body").success

    def test_index_file(self, tmp_path) -> None:
        tool, engine = _tool_with_engine()
        p = tmp_path / "doc.txt"
        p.write_text("indexed file content about vaults and secrets")
        res = tool.execute(action="index_file", path=str(p))
        assert res.success and res.output["chunks_indexed"] >= 1
        assert res.output["path"] == str(p.resolve())
        assert res.output["doc_id"] == str(p.resolve())
        assert engine.query("indexed file content", top_k=1)[0].doc_id == str(p.resolve())

    def test_index_file_preserves_explicit_custom_doc_id(self, tmp_path) -> None:
        tool, engine = _tool_with_engine()
        p = tmp_path / "doc.txt"
        p.write_text("custom document identity")
        res = tool.execute(action="index_file", path=str(p), doc_id="requested-id")
        assert res.success
        assert res.output["doc_id"] == "requested-id"
        assert engine.query("custom identity", top_k=1)[0].doc_id == "requested-id"

    def test_index_file_missing_path_errors(self, tmp_path) -> None:
        tool, _ = _tool_with_engine()
        res = tool.execute(action="index_file", path=str(tmp_path / "nope.txt"))
        assert not res.success
        assert "not found" in res.error

    def test_index_file_requires_path(self) -> None:
        tool, _ = _tool_with_engine()
        assert not tool.execute(action="index_file").success


class TestStatsAndErrors:
    def test_stats_action(self) -> None:
        tool, eng = _tool_with_engine()
        eng.index_document("d", "some content")
        res = tool.execute(action="stats")
        assert res.success
        assert res.output["documents"] == 1

    def test_unknown_action_errors(self) -> None:
        tool, _ = _tool_with_engine()
        res = tool.execute(action="frobnicate")
        assert not res.success
        assert "unknown action" in res.error


class TestToolContract:
    def test_is_a_base_tool_with_readonly_defaults(self) -> None:
        tool = RagQueryTool()
        assert isinstance(tool, BaseTool)
        assert tool.name == "rag_query"
        assert tool.permissions.network is False
        assert tool.permissions.shell is False

    def test_resolve_filesystem_targets_declares_index_path(self) -> None:
        tool = RagQueryTool()
        reads, writes = tool.resolve_filesystem_targets({"path": "/tmp/x.txt"})
        assert reads == ["/tmp/x.txt"]
        assert writes == []

    def test_schema_is_wellformed(self) -> None:
        schema = RagQueryTool().get_schema()
        assert schema["name"] == "rag_query"
        assert "action" in schema["parameters"]["properties"]
        assert (
            "resolved absolute file path"
            in schema["parameters"]["properties"]["doc_id"]["description"]
        )

    def test_lazy_engine_uses_index_dir(self, tmp_path) -> None:
        # A tool built without an engine lazily constructs one at the given dir.
        tool = RagQueryTool(index_dir=str(tmp_path))
        r = tool.execute(action="index_text", doc_id="d", text="persisted content")
        assert r.success
        assert (tmp_path / "chunks.json").exists()

    def test_registered_in_builtins(self) -> None:
        from missy.tools.builtin import _ALL_TOOL_CLASSES

        assert RagQueryTool in _ALL_TOOL_CLASSES
