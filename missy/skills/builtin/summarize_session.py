"""Built-in skill: summarize the current conversation session.

Loads recent turns from the memory store and returns a concise, formatted
transcript summary.  If no session_id is supplied an empty-session message is
returned.  If the memory store is unavailable a descriptive error is returned
rather than raising an exception.
"""

from __future__ import annotations

from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

_TURN_LIMIT = 20
_CONTENT_PREVIEW_LEN = 200  # chars shown per turn in the summary


def _format_turns(turns: list) -> str:  # type: ignore[type-arg]
    """Render a list of :class:`~missy.memory.store.ConversationTurn` objects.

    Each turn is shown as a labelled block with its timestamp and a
    truncated preview of the message content.

    Args:
        turns: Conversation turns in chronological order.

    Returns:
        A multi-line formatted string, or a note when the list is empty.
    """
    if not turns:
        return "(no turns recorded for this session)"

    lines: list[str] = []
    for turn in turns:
        ts = turn.timestamp.isoformat(timespec="seconds") if turn.timestamp else "unknown time"
        role_label = turn.role.capitalize() if turn.role else "Unknown"
        content = turn.content or ""
        if len(content) > _CONTENT_PREVIEW_LEN:
            content = content[:_CONTENT_PREVIEW_LEN].rstrip() + "…"
        lines.append(f"[{ts}] {role_label}: {content}")

    return "\n".join(lines)


class SummarizeSessionSkill(BaseSkill):
    """Summarizes the recent turns of a named conversation session."""

    name = "summarize_session"
    description = "Summarize the current conversation session."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)

    def execute(self, session_id: str = "", **kwargs: Any) -> SkillResult:
        """Load and format recent turns for *session_id*.

        Args:
            session_id: Identifier of the session to summarize.  When
                omitted or empty a helpful prompt is returned instead.
            **kwargs: Extra keyword arguments are accepted but ignored.

        Returns:
            :class:`~missy.skills.base.SkillResult` with a formatted
            transcript in ``output``, or an error message if the memory
            store cannot be reached.
        """
        if not session_id:
            return SkillResult(
                success=False,
                output="",
                error="session_id is required. Pass the active session identifier to summarize it.",
            )

        try:
            from missy.memory.store import MemoryStore  # local import to isolate failures

            store = MemoryStore()
            turns = store.get_session_turns(session_id, limit=_TURN_LIMIT)
        except Exception as exc:  # pragma: no cover
            return SkillResult(
                success=False,
                output="",
                error=f"Memory store unavailable: {exc}",
            )

        turn_count = len(turns)
        header_lines = [
            f"Session: {session_id}",
            f"Turns shown: {turn_count} (max {_TURN_LIMIT})",
            "-" * 60,
        ]
        body = _format_turns(turns)
        output = "\n".join(header_lines) + "\n" + body
        return SkillResult(success=True, output=output)
