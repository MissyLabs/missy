"""``rag_query`` agent tool — citation-grounded local retrieval (F03).

Exposes the on-device :class:`~missy.retrieval.RetrievalEngine` to the agent.
Every result carries a ``source_span`` citation back into the indexed
document, so answers are grounded and verifiable rather than opaque snippets.

Actions (``action`` parameter):

* ``"query"`` (default) — return the top matching chunks for ``query``, each
  with its ``doc_id`` and ``source_span`` citation.
* ``"index_text"`` — add ``text`` to the index under ``doc_id``.
* ``"index_file"`` — read and index a local text file at ``path``.
* ``"stats"`` — report index size / embedder / document ids.

The engine persists to ``~/.missy/retrieval`` by default so the index is
durable across turns and processes. All embedding is on-device (no network),
matching Missy's self-hosted posture; the tool declares ``filesystem_read``
so ``index_file`` targets are enforced by the filesystem policy engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

if TYPE_CHECKING:
    from missy.retrieval import RetrievalEngine

# The retrieval engine pulls in NumPy (and optionally FAISS). Those are not
# base dependencies, so ``missy.retrieval`` is imported lazily inside methods —
# importing this module (and thus registering the built-in tool set) must not
# require the retrieval/vector extras to be installed. Duplicated here rather
# than imported so the module-level default needs no numpy.
DEFAULT_INDEX_DIR = "~/.missy/retrieval"


class RagQueryTool(BaseTool):
    """Local, citation-grounded retrieval over indexed documents."""

    name = "rag_query"
    description = (
        "Query a local, on-device retrieval index for passages relevant to a "
        "question and get citation-grounded results (doc id + character span). "
        "Also indexes text or local files. Fully offline; no cloud embeddings."
    )
    permissions = ToolPermissions(
        network=False,
        filesystem_read=True,
        filesystem_write=True,
        shell=False,
    )

    def __init__(
        self, engine: RetrievalEngine | None = None, *, index_dir: str | None = None
    ) -> None:
        # A caller (or test) may inject an engine; otherwise build one lazily
        # against the durable default index dir so state persists across turns.
        self._engine = engine
        self._index_dir = index_dir or DEFAULT_INDEX_DIR

    def _get_engine(self) -> RetrievalEngine:
        if self._engine is None:
            from missy.retrieval import RetrievalEngine

            self._engine = RetrievalEngine(index_dir=self._index_dir)
        return self._engine

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        # index_file reads an arbitrary path the model supplies; declare it so
        # the filesystem policy engine sees the real target.
        path = kwargs.get("path")
        reads = [str(path)] if path else []
        return (reads, [])

    def execute(self, **kwargs: Any) -> ToolResult:
        action = (kwargs.get("action") or "query").strip().lower()
        try:
            if action == "query":
                return self._do_query(kwargs)
            if action == "index_text":
                return self._do_index_text(kwargs)
            if action == "index_file":
                return self._do_index_file(kwargs)
            if action == "stats":
                return ToolResult(success=True, output=self._get_engine().stats())
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"unknown action {action!r}; expected one of "
                    "query, index_text, index_file, stats"
                ),
            )
        except FileNotFoundError as exc:
            return ToolResult(success=False, output=None, error=f"file not found: {exc}")
        except Exception as exc:  # keep tool failures inside ToolResult
            return ToolResult(success=False, output=None, error=str(exc))

    def _do_query(self, kwargs: dict[str, Any]) -> ToolResult:
        query = kwargs.get("query")
        if not query or not str(query).strip():
            return ToolResult(success=False, output=None, error="query must be a non-empty string")
        top_k = int(kwargs.get("top_k", 5) or 5)
        results = self._get_engine().query(str(query), top_k=top_k)
        payload = [
            {
                "doc_id": r.doc_id,
                "text": r.text,
                "source_span": list(r.source_span),
                "citation": r.citation(),
                "score": round(r.score, 6),
                "chunk_index": r.chunk_index,
                "metadata": r.metadata,
            }
            for r in results
        ]
        return ToolResult(success=True, output={"query": query, "results": payload})

    def _do_index_text(self, kwargs: dict[str, Any]) -> ToolResult:
        doc_id = kwargs.get("doc_id")
        text = kwargs.get("text")
        if not doc_id or text is None:
            return ToolResult(
                success=False,
                output=None,
                error="index_text requires 'doc_id' and 'text'",
            )
        n = self._get_engine().index_document(
            str(doc_id), str(text), metadata=kwargs.get("metadata")
        )
        return ToolResult(success=True, output={"doc_id": doc_id, "chunks_indexed": n})

    def _do_index_file(self, kwargs: dict[str, Any]) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, output=None, error="index_file requires 'path'")
        p = Path(str(path)).expanduser()
        if not p.exists():
            return ToolResult(success=False, output=None, error=f"file not found: {p}")
        resolved_path = p.resolve()
        supplied_doc_id = kwargs.get("doc_id")
        effective_doc_id = (
            str(supplied_doc_id).strip()
            if isinstance(supplied_doc_id, str) and supplied_doc_id.strip()
            else str(resolved_path)
        )
        n = self._get_engine().index_file(resolved_path, doc_id=effective_doc_id)
        return ToolResult(
            success=True,
            output={
                "path": str(resolved_path),
                "doc_id": effective_doc_id,
                "chunks_indexed": n,
            },
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "index_text", "index_file", "stats"],
                        "description": "What to do (default: query).",
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query (for action=query).",
                        "example": "how do I rotate API keys?",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return for a query (default 5).",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": (
                            "Document id for indexing actions. Required for index_text. "
                            "For index_file, omit this unless the user explicitly supplied "
                            "a custom id; omission uses the resolved absolute file path so "
                            "same-named files in different directories cannot collide."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "Text body to index (for action=index_text).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Local text file to index (for action=index_file).",
                    },
                },
                "required": [],
            },
        }
