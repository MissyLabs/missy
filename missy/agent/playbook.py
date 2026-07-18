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

import contextlib
import fcntl
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


def classify_task_type(text: str) -> str:
    """Best-effort guess at the coarse task-type category a request will
    likely fall into, for prospective playbook retrieval before any tools
    have actually run this turn.

    :meth:`Playbook.record` itself is keyed on the more accurate
    ``extract_task_type()`` from :mod:`missy.agent.learnings`, computed from
    the tools *actually* used during a completed run. This function mirrors
    the same category vocabulary (``shell+web``, ``shell+file``, ``shell``,
    ``web``, ``file``, ``chat``) via lightweight keyword matching, so a
    caller retrieving relevant patterns *before* running (when actual tool
    usage isn't known yet) has a real chance of matching an already-recorded
    pattern instead of querying with an arbitrary raw string that could
    never match a recorded coarse category.
    """
    low = text.lower()
    has_shell = any(kw in low for kw in ("run", "execute", "shell", "command", "bash", "script"))
    has_web = any(kw in low for kw in ("fetch", "download", "url", "http", "website", "webpage"))
    has_file = any(kw in low for kw in ("file", "read", "write", "save", "edit"))
    if has_shell and has_web:
        return "shell+web"
    if has_shell and has_file:
        return "shell+file"
    if has_shell:
        return "shell"
    if has_web:
        return "web"
    if has_file:
        return "file"
    return "chat"


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
        self._lock_path = self._path + ".lock"
        self._lock = threading.Lock()
        self._entries: dict[str, PlaybookEntry] = {}
        self.load()

    @contextlib.contextmanager
    def _cross_process_locked(self):
        """Serialize read-modify-write cycles across separate Playbook
        instances (and processes), not just calls on this one object.

        Every production caller (``AgentRuntime``) constructs a fresh
        ``Playbook()`` per call rather than sharing one long-lived
        instance, so ``self._lock`` alone provided no real protection: two
        concurrently-completing tasks each load their own private snapshot
        of the store, and whichever finishes ``save()`` last silently
        overwrites the other's just-recorded pattern (its in-memory
        snapshot never saw the first writer's change). Live-reproduced:
        two ``Playbook()`` instances against the same path each recording
        a distinct pattern left only 1 of 2 entries surviving. A
        ``flock()`` on a dedicated lock file blocks other Playbook
        instances -- in this process or another -- holding a separate fd
        on the same lock file, the same fix already applied to
        :class:`~missy.security.vault.Vault` for an identical race.
        """
        dir_path = os.path.dirname(self._lock_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True, mode=0o700)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

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
        with self._cross_process_locked(), self._lock:
            self.load()  # refresh from disk under the lock before merging
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
        with self._cross_process_locked(), self._lock:
            self.load()  # refresh from disk under the lock before mutating
            if pattern_id not in self._entries:
                raise KeyError(f"Pattern {pattern_id!r} not found")
            self._entries[pattern_id].promoted = True
            self.save()

    def save(self) -> None:
        """Persist entries to disk using atomic write."""
        data = [asdict(e) for e in self._entries.values()]
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True, mode=0o700)
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

    # ------------------------------------------------------------------
    # F20: materialize promotable patterns into SKILL.md proposals
    # ------------------------------------------------------------------

    def write_skill_proposal(self, entry: PlaybookEntry, proposals_dir: str | None = None) -> str:
        """Write a discoverable SKILL.md draft for a promotable *entry*.

        Turns a proven playbook pattern into a real, parseable SKILL.md file
        (valid YAML frontmatter with the required ``name`` field) under a
        ``proposals`` directory. The file is a *draft for operator review* —
        SKILL.md discovery is read-only documentation, so an operator promotes
        it by moving it into the active skills directory. This closes the
        Playbook->Skill loop that previously stopped at "flagged promotable".

        Args:
            entry: The promotable pattern to materialize.
            proposals_dir: Target directory. Defaults to
                ``~/.missy/skills/proposals``.

        Returns:
            The path of the written SKILL.md file.
        """
        base = proposals_dir or os.path.expanduser("~/.missy/skills/proposals")
        # One subdirectory per proposal (SKILL.md discovery is rglob-based).
        safe_id = "".join(c for c in entry.pattern_id if c.isalnum() or c in "-_")[:32]
        skill_dir = os.path.join(base, f"playbook-{safe_id}")
        os.makedirs(skill_dir, exist_ok=True, mode=0o700)

        name = f"playbook-{entry.task_type}-{safe_id[:8]}"
        description = (
            entry.description
            or f"Proven {entry.task_type} pattern using {', '.join(entry.tool_sequence)}"
        )
        # Escape any double-quotes in free text so the frontmatter stays valid.
        desc_q = description.replace('"', "'").replace("\n", " ").strip()
        tools_list = ", ".join(entry.tool_sequence)
        content = (
            "---\n"
            f"name: {name}\n"
            f'description: "{desc_q}"\n'
            "version: 0.1.0\n"
            f"tools: [{tools_list}]\n"
            "status: proposed\n"
            "source: playbook-auto-promotion\n"
            "---\n\n"
            f"# {name}\n\n"
            f"Auto-generated skill proposal from a playbook pattern that "
            f"succeeded {entry.success_count} time(s).\n\n"
            f"- **Task type:** {entry.task_type}\n"
            f"- **Tool sequence:** {tools_list}\n"
            f"- **Pattern id:** {entry.pattern_id}\n\n"
            "## Review\n\n"
            "This is a *proposal draft*. To adopt it, review the steps below "
            "and move this file into your active skills directory "
            "(`~/.missy/skills`). The `tools` list is documentation, not a "
            "capability grant.\n\n"
            "## Steps\n\n"
            f"When performing a **{entry.task_type}** task, the proven approach "
            f"is to use these tools in order: {tools_list}.\n"
        )
        skill_path = os.path.join(skill_dir, "SKILL.md")
        fd, tmp = tempfile.mkstemp(dir=skill_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, skill_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return skill_path

    def promote_to_skills(
        self,
        threshold: int = 3,
        proposals_dir: str | None = None,
        dry_run: bool = False,
    ) -> list[dict]:
        """Materialize every promotable pattern into a SKILL.md proposal.

        For each pattern with ``success_count >= threshold`` that has not yet
        been promoted, writes a SKILL.md draft (unless *dry_run*) and marks it
        promoted so it is not re-materialized on the next call.

        Args:
            threshold: Minimum success count for promotion.
            proposals_dir: Where to write proposals (default
                ``~/.missy/skills/proposals``).
            dry_run: When True, report what *would* be promoted without
                writing files or marking anything promoted.

        Returns:
            A list of ``{"pattern_id", "task_type", "success_count", "path"}``
            dicts (``path`` is ``None`` on a dry run).
        """
        results: list[dict] = []
        for entry in self.get_promotable(threshold=threshold):
            record = {
                "pattern_id": entry.pattern_id,
                "task_type": entry.task_type,
                "success_count": entry.success_count,
                "path": None,
            }
            if not dry_run:
                try:
                    record["path"] = self.write_skill_proposal(entry, proposals_dir)
                    self.mark_promoted(entry.pattern_id)
                except Exception:
                    logger.debug(
                        "Playbook: failed to promote pattern %s", entry.pattern_id, exc_info=True
                    )
                    continue
            results.append(record)
        return results

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
