"""Graph memory query tool (F04).

Exposes :class:`~missy.memory.graph_store.GraphMemoryStore` — a fully
implemented entity-relationship graph that previously had **zero** production
callers — to the agent as a read-only ``graph_query`` tool. Structured
knowledge retrieval: given a free-text query, it returns a human-readable
context subgraph plus the matched entities and relationships, so the agent can
recall *structured* facts ("what depends on X", "who owns Y") that flat FTS5
recall does not surface.

Read-only by construction: no ``add``/``ingest`` path is exposed to the model
(entity ingestion happens out-of-band; see the ``missy graph`` CLI for
operator seeding), and the tool declares no network/filesystem/shell
permissions, so it dispatches through the same reference monitor as any
built-in tool without widening capability.
"""

from __future__ import annotations

from typing import Any

from missy.memory.graph_store import GraphMemoryStore
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_ENTITIES_CAP = 50


class GraphQueryTool(BaseTool):
    """Read-only structured-knowledge retrieval over the entity graph."""

    name = "graph_query"
    description = (
        "Query the entity-relationship knowledge graph for structured facts: "
        "given a topic or entity name, returns related entities and their "
        "relationships (e.g. what depends on / uses / owns what). Read-only."
    )
    permissions = ToolPermissions(
        network=False,
        filesystem_read=False,
        filesystem_write=False,
        shell=False,
    )

    def __init__(self, store: GraphMemoryStore | None = None) -> None:
        # Allow injection for tests; default to the real ~/.missy graph store.
        self._store = store if store is not None else GraphMemoryStore()

    def execute(
        self,
        *,
        query: str = "",
        max_entities: int = 15,
        **_kwargs: Any,
    ) -> ToolResult:
        """Return structured knowledge related to *query*.

        Args:
            query: Free-text topic or entity name to look up.
            max_entities: Cap on the number of entities in the returned
                subgraph (clamped to 1..50).

        Returns:
            A :class:`ToolResult` whose ``output`` is a dict with a rendered
            ``subgraph`` string plus structured ``entities`` and
            ``relationships`` lists, or an error when the query is empty.
        """
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="query must be a non-empty string",
            )

        try:
            capped = max(1, min(int(max_entities), _MAX_ENTITIES_CAP))
        except (TypeError, ValueError):
            capped = 15

        q = query.strip()
        try:
            result = self._store.find_related(q, limit=capped)
            subgraph = self._store.get_context_subgraph(q, max_entities=capped)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(
                success=False,
                output=None,
                error=f"graph query failed: {exc}",
            )

        entities = [
            {
                "name": e.name,
                "type": e.entity_type,
                "mentions": e.mention_count,
            }
            for e in result.entities
        ]
        relationships = [
            {
                "source": r.source_id,
                "target": r.target_id,
                "type": r.relation_type,
            }
            for r in result.relationships
        ]

        return ToolResult(
            success=True,
            output={
                "query": q,
                "subgraph": subgraph,
                "entities": entities,
                "relationships": relationships,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the graph_query parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Topic or entity name to look up in the knowledge "
                            "graph, e.g. 'authentication' or 'video_generate'."
                        ),
                        "example": "video_generate",
                    },
                    "max_entities": {
                        "type": "integer",
                        "description": (
                            "Maximum entities to include in the returned "
                            "subgraph (1-50, default 15)."
                        ),
                        "default": 15,
                    },
                },
                "required": ["query"],
            },
        }
