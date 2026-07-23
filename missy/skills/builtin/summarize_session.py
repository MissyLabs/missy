"""Built-in skill: summarize the current conversation session.

Loads recent turns from the memory store and returns a concise, formatted
transcript summary.  If no session_id is supplied an empty-session message is
returned.  If the memory store is unavailable a descriptive error is returned
rather than raising an exception.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult, reject_unknown_arguments

_TURN_LIMIT = 20
_CONTENT_PREVIEW_LEN = 200  # chars shown per turn in the summary
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")


def _safe_content(value: object) -> str:
    try:
        from missy.security.censor import censor_response

        safe = censor_response(str(value or ""))
        if not isinstance(safe, str):
            raise TypeError("censor returned a non-string")
        return safe
    except Exception:
        return "[content unavailable: redaction failed]"


def _format_turns(turns: list) -> str:  # type: ignore[type-arg]
    """Render a list of :class:`~missy.memory.sqlite_store.ConversationTurn` objects.

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
        # SQLiteMemoryStore.ConversationTurn.timestamp is an ISO-8601
        # string (unlike the legacy JSON store's datetime object) --
        # truncate to seconds precision rather than calling .isoformat().
        ts = turn.timestamp[:19] if turn.timestamp else "unknown time"
        role = turn.role if turn.role in {"user", "assistant", "system", "tool"} else "unknown"
        role_label = role.capitalize()
        content = _safe_content(turn.content)
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

    def __init__(
        self,
        authorized_session_id: str | None = None,
        *,
        session_authorizer: Callable[[str], bool] | None = None,
        db_path: str = "~/.missy/memory.db",
    ) -> None:
        self._authorized_session_id = authorized_session_id
        self._session_authorizer = session_authorizer
        self._db_path = Path(db_path).expanduser().absolute()

    def execute(self, session_id: str = "", **kwargs: Any) -> SkillResult:
        """Load and format recent turns for *session_id*.

        Args:
            session_id: Identifier of the session to summarize.  When
                omitted or empty a helpful prompt is returned instead.
            **kwargs: Rejected; the built-in accepts only its documented
                arguments.

        Returns:
            :class:`~missy.skills.base.SkillResult` with a formatted
            transcript in ``output``, or an error message if the memory
            store cannot be reached.
        """
        if error := reject_unknown_arguments(kwargs):
            return error
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            return SkillResult(
                success=False,
                output=None,
                error="session_id is required. Pass the active session identifier to summarize it.",
            )
        if not self._session_allowed(session_id):
            return SkillResult(
                success=False,
                output=None,
                error="Session summary access denied for this session_id.",
            )
        if self._db_path.is_symlink():
            return SkillResult(
                success=False,
                output=None,
                error="Memory database symlink paths are not allowed.",
            )

        try:
            # SR-3.1/3.5: use the production SQLite backend, not the legacy
            # JSON store -- since FX-B, real conversation turns are written
            # to SQLiteMemoryStore, so reading from the JSON store here
            # always returned an empty/stale history regardless of what
            # actually happened in the session.
            from missy.memory.sqlite_store import (
                SQLiteMemoryStore,  # local import to isolate failures
            )

            store = SQLiteMemoryStore(str(self._db_path))
            turns = store.get_session_turns(session_id, limit=_TURN_LIMIT)
        except Exception:  # pragma: no cover
            return SkillResult(
                success=False,
                output=None,
                error="Memory store unavailable or unreadable.",
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

    def _session_allowed(self, session_id: str) -> bool:
        if self._authorized_session_id is not None:
            return session_id == self._authorized_session_id
        authorizer = self._session_authorizer
        if authorizer is None:
            return False
        try:
            return authorizer(session_id) is True
        except Exception:
            return False
