"""Agent-callable tools for searching and expanding conversation history.

Provides three tools:
- ``memory_search``: FTS5 search across turns and summaries
- ``memory_describe``: Detailed metadata for a summary or large-content ref
- ``memory_expand``: Walk the DAG to retrieve source content behind a summary
"""

from __future__ import annotations

import logging
from typing import Any

from missy.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


def _ok(text: str) -> ToolResult:
    return ToolResult(success=True, output=text)


def _err(text: str) -> ToolResult:
    return ToolResult(success=False, output="", error=text)


class MemorySearchTool(BaseTool):
    """Search conversation history (turns and summaries) via full-text search."""

    name = "memory_search"
    description = (
        "Search your conversation history and summaries for a keyword or phrase. "
        "Returns matching turns and summary excerpts with timestamps."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (supports FTS5 syntax: phrases, AND/OR).",
            },
            "scope": {
                "type": "string",
                "enum": ["messages", "summaries", "both"],
                "description": "What to search: messages, summaries, or both.",
                "default": "both",
            },
            "session_id": {
                "type": "string",
                "description": "Restrict to a specific session. Empty = current session.",
                "default": "",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    requires_filesystem_read = []
    requires_filesystem_write = []
    requires_network = []
    requires_shell = False

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        scope = kwargs.get("scope", "both")
        session_id = kwargs.get("session_id", "") or kwargs.get("_session_id", "")
        limit = min(kwargs.get("limit", 10), 50)

        if not query:
            return _err("query is required.")

        store = kwargs.get("_memory_store")
        if store is None:
            return _err("Memory store is not available.")

        parts: list[str] = []

        if scope in ("messages", "both"):
            try:
                turns = store.search(query, limit=limit, session_id=session_id or None)
                if turns:
                    parts.append(f"### Messages ({len(turns)} matches)")
                    for t in turns:
                        ts = t.timestamp[:19] if t.timestamp else "?"
                        snippet = t.content[:200] + ("..." if len(t.content) > 200 else "")
                        parts.append(f"- [{ts}] {t.role}: {snippet}")
            except Exception as exc:
                parts.append(f"Message search error: {exc}")

        if scope in ("summaries", "both"):
            try:
                summaries = store.search_summaries(
                    query, session_id=session_id or None, limit=limit
                )
                if summaries:
                    parts.append(f"### Summaries ({len(summaries)} matches)")
                    for s in summaries:
                        time_info = ""
                        if s.time_range_start and s.time_range_end:
                            time_info = f" ({s.time_range_start[:19]} to {s.time_range_end[:19]})"
                        snippet = s.content[:300] + ("..." if len(s.content) > 300 else "")
                        parts.append(
                            f"- [{s.id}] depth={s.depth}{time_info}: {snippet}"
                        )
            except Exception as exc:
                parts.append(f"Summary search error: {exc}")

        if not parts:
            return _ok(f"No results found for '{query}'.")

        return _ok("\n".join(parts))


class MemoryDescribeTool(BaseTool):
    """Retrieve detailed metadata for a summary or large-content reference."""

    name = "memory_describe"
    description = (
        "Get full content and metadata for a summary ID (sum_*) or "
        "large-content reference (ref_*). Shows depth, time range, "
        "parent/child relationships, and source turn count."
    )
    parameters = {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "A summary ID (sum_...) or large-content ID (ref_...).",
            },
        },
        "required": ["item_id"],
    }

    requires_filesystem_read = []
    requires_filesystem_write = []
    requires_network = []
    requires_shell = False

    def execute(self, **kwargs: Any) -> ToolResult:
        item_id = kwargs.get("item_id", "")
        if not item_id:
            return _err("item_id is required.")

        store = kwargs.get("_memory_store")
        if store is None:
            return _err("Memory store is not available.")

        if item_id.startswith("sum_"):
            return self._describe_summary(store, item_id)
        if item_id.startswith("ref_"):
            return self._describe_large_content(store, item_id)

        return _err(f"Unknown ID format: {item_id}. Expected sum_* or ref_*.")

    @staticmethod
    def _describe_summary(store: Any, summary_id: str) -> ToolResult:
        summary = store.get_summary_by_id(summary_id)
        if summary is None:
            return _err(f"Summary '{summary_id}' not found.")

        lines = [
            f"## Summary: {summary.id}",
            f"- **Depth:** {summary.depth}",
            f"- **Tokens:** ~{summary.token_estimate}",
            f"- **Descendants:** {summary.descendant_count}",
            f"- **Time range:** {summary.time_range_start or '?'} to {summary.time_range_end or '?'}",
            f"- **Source turns:** {len(summary.source_turn_ids)}",
            f"- **Source summaries:** {len(summary.source_summary_ids)}",
            f"- **Parent:** {summary.parent_id or 'None (top-level)'}",
            f"- **Created:** {summary.created_at}",
            "",
            "### Content",
            summary.content,
        ]

        children = store.get_child_summaries(summary_id)
        if children:
            lines.append("")
            lines.append(f"### Children ({len(children)})")
            for c in children:
                lines.append(f"- {c.id} (depth={c.depth}, ~{c.token_estimate} tokens)")

        return _ok("\n".join(lines))

    @staticmethod
    def _describe_large_content(store: Any, content_id: str) -> ToolResult:
        record = store.get_large_content(content_id)
        if record is None:
            return _err(f"Large content '{content_id}' not found.")

        lines = [
            f"## Large Content: {record.id}",
            f"- **Tool:** {record.tool_name}",
            f"- **Size:** {record.original_chars} chars (~{record.original_chars // 4} tokens)",
            f"- **Session:** {record.session_id}",
            f"- **Created:** {record.created_at}",
            f"- **Summary:** {record.summary}",
            "",
            "### Preview (first 500 chars)",
            record.content[:500],
        ]
        return _ok("\n".join(lines))


class MemoryExpandTool(BaseTool):
    """Walk the summary DAG to retrieve source content behind a summary."""

    name = "memory_expand"
    description = (
        "Expand a summary or large-content reference to see its source content. "
        "For summaries: retrieves child summaries or original turns. "
        "For large-content refs: retrieves the full stored content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "A summary ID (sum_*) or large-content ID (ref_*).",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum tokens to return.",
                "default": 4000,
            },
        },
        "required": ["item_id"],
    }

    requires_filesystem_read = []
    requires_filesystem_write = []
    requires_network = []
    requires_shell = False

    def execute(self, **kwargs: Any) -> ToolResult:
        item_id = kwargs.get("item_id", "")
        max_tokens = min(kwargs.get("max_tokens", 4000), 20_000)

        if not item_id:
            return _err("item_id is required.")

        store = kwargs.get("_memory_store")
        if store is None:
            return _err("Memory store is not available.")

        if item_id.startswith("ref_"):
            return self._expand_large_content(store, item_id, max_tokens)
        if item_id.startswith("sum_"):
            return self._expand_summary(store, item_id, max_tokens)

        return _err(f"Unknown ID format: {item_id}. Expected sum_* or ref_*.")

    @staticmethod
    def _expand_large_content(store: Any, content_id: str, max_tokens: int) -> ToolResult:
        record = store.get_large_content(content_id)
        if record is None:
            return _err(f"Large content '{content_id}' not found.")

        max_chars = max_tokens * 4
        content = record.content
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        result = f"## Expanded: {content_id}\n\n{content}"
        if truncated:
            result += f"\n\n[TRUNCATED at {max_tokens} tokens — original was {record.original_chars} chars]"
        return _ok(result)

    @staticmethod
    def _expand_summary(store: Any, summary_id: str, max_tokens: int) -> ToolResult:
        summary = store.get_summary_by_id(summary_id)
        if summary is None:
            return _err(f"Summary '{summary_id}' not found.")

        parts: list[str] = [f"## Expanded: {summary_id} (depth={summary.depth})"]
        used_chars = 0
        max_chars = max_tokens * 4

        if summary.source_summary_ids:
            children = store.get_child_summaries(summary_id)
            if not children:
                for sid in summary.source_summary_ids:
                    child = store.get_summary_by_id(sid)
                    if child:
                        children.append(child)

            for child in children:
                block = (
                    f"\n### Child: {child.id} (depth={child.depth})\n{child.content}"
                )
                if used_chars + len(block) > max_chars:
                    parts.append(f"\n[TRUNCATED — reached {max_tokens} token limit]")
                    break
                parts.append(block)
                used_chars += len(block)

        if summary.source_turn_ids:
            source_turns = store.get_source_turns(summary_id)
            for turn in source_turns:
                ts = turn.timestamp[:19] if turn.timestamp else "?"
                block = f"\n[{ts}] {turn.role}: {turn.content}"
                if used_chars + len(block) > max_chars:
                    parts.append(f"\n[TRUNCATED — reached {max_tokens} token limit]")
                    break
                parts.append(block)
                used_chars += len(block)

        if len(parts) == 1:
            parts.append("\nNo source content found for this summary.")

        return _ok("\n".join(parts))
