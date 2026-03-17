"""AI Playbook — auto-capture and replay of successful tool patterns.

When a task completes successfully with tool use, :meth:`Playbook.record`
extracts the pattern (task type + tool sequence) and stores it.  On future
runs, :meth:`Playbook.get_relevant` retrieves proven patterns so the agent
can inject them as context.  Patterns that succeed 3+ times are eligible
for promotion to skill proposals via :meth:`Playbook.get_promotable`.

Example::

    from missy.agent.playbook import Playbook

    pb = Playbook("/tmp/playbook.json")
    pb.record("shell", "deploy app", ["shell_exec", "file_write"], "use rsync")
    entries = pb.get_relevant("shell", top_k=3)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class PlaybookEntry:
    """A recorded successful tool-use pattern.

    Attributes:
        pattern_id: Deterministic hash of task_type + sorted tool_sequence.
        task_type: Coarse category (e.g. ``"shell"``, ``"file"``).
        description: Human-readable description of what the pattern does.
        tool_sequence: Ordered list of tool names used.
        prompt_template: Hint or prompt snippet for reproducing the pattern.
        success_count: Number of times this pattern has succeeded.
        created_at: ISO-8601 UTC timestamp of first recording.
        promoted: Whether this pattern has been promoted to a skill proposal.
    """

    pattern_id: str
    task_type: str
    description: str
    tool_sequence: list[str]
    prompt_template: str
    success_count: int = 1
    created_at: str = ""
    promoted: bool = False

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


def _compute_pattern_id(task_type: str, tool_sequence: list[str]) -> str:
    """Compute a deterministic pattern ID from task type and tool sequence.

    Args:
        task_type: The coarse task category.
        tool_sequence: List of tool names (will be sorted for hashing).

    Returns:
        A hex digest string identifying the pattern.
    """
    key = f"{task_type}:{','.join(sorted(tool_sequence))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Playbook:
    """Thread-safe playbook that persists successful patterns to JSON.

    Args:
        store_path: Path to the JSON persistence file.  Defaults to
            ``~/.missy/playbook.json``.
    """

    def __init__(self, store_path: str = "~/.missy/playbook.json") -> None:
        self._path = os.path.expanduser(store_path)
        self._lock = threading.Lock()
        self._entries: dict[str, PlaybookEntry] = {}
        self.load()

    def record(
        self,
        task_type: str,
        description: str,
        tool_sequence: list[str],
        prompt_hint: str,
    ) -> PlaybookEntry:
        """Record a successful pattern, creating or incrementing it.

        Matching is based on ``task_type`` + sorted ``tool_sequence`` hash.

        Args:
            task_type: Coarse task category.
            description: Human-readable description.
            tool_sequence: Ordered list of tool names used.
            prompt_hint: A hint or template for reproducing the approach.

        Returns:
            The created or updated :class:`PlaybookEntry`.
        """
        pattern_id = _compute_pattern_id(task_type, tool_sequence)
        with self._lock:
            if pattern_id in self._entries:
                entry = self._entries[pattern_id]
                entry.success_count += 1
                # Update description and hint to latest successful run
                entry.description = description
                entry.prompt_template = prompt_hint
            else:
                entry = PlaybookEntry(
                    pattern_id=pattern_id,
                    task_type=task_type,
                    description=description,
                    tool_sequence=list(tool_sequence),
                    prompt_template=prompt_hint,
                )
                self._entries[pattern_id] = entry
            self.save()
        return entry

    def get_relevant(self, task_type: str, top_k: int = 3) -> list[PlaybookEntry]:
        """Find patterns matching the given task type.

        Args:
            task_type: The task type to match.
            top_k: Maximum number of entries to return.

        Returns:
            A list of matching entries ordered by ``success_count`` descending.
        """
        with self._lock:
            matches = [e for e in self._entries.values() if e.task_type == task_type]
        matches.sort(key=lambda e: e.success_count, reverse=True)
        return matches[:top_k]

    def get_promotable(self, threshold: int = 3) -> list[PlaybookEntry]:
        """Return patterns eligible for promotion to skill proposals.

        Args:
            threshold: Minimum ``success_count`` required.

        Returns:
            A list of entries with ``success_count >= threshold`` that have
            not yet been promoted.
        """
        with self._lock:
            return [
                e for e in self._entries.values() if e.success_count >= threshold and not e.promoted
            ]

    def mark_promoted(self, pattern_id: str) -> None:
        """Mark a pattern as promoted.

        Args:
            pattern_id: The pattern ID to mark.

        Raises:
            KeyError: If the pattern ID is not found.
        """
        with self._lock:
            if pattern_id not in self._entries:
                raise KeyError(f"Pattern {pattern_id!r} not found")
            self._entries[pattern_id].promoted = True
            self.save()

    def save(self) -> None:
        """Persist entries to disk using atomic write."""
        data = [asdict(e) for e in self._entries.values()]
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path or ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                # Clean up temp file on failure
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except Exception:
            logger.debug("Failed to save playbook to %s", self._path, exc_info=True)

    def load(self) -> None:
        """Load entries from the JSON file.  No-op if file doesn't exist."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._entries = {}
            for item in data:
                entry = PlaybookEntry(**item)
                self._entries[entry.pattern_id] = entry
        except Exception:
            logger.debug("Failed to load playbook from %s", self._path, exc_info=True)


# Need contextlib for suppress in save()
import contextlib  # noqa: E402
