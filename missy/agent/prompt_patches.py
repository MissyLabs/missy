"""Prompt self-tuning patch system.

Manages a collection of :class:`PromptPatch` records that are appended to the
system prompt to guide model behaviour.  Patches can be proposed automatically
by the runtime, reviewed, approved or rejected by operators, and expire when
their success rate falls below threshold.

Example::

    from missy.agent.prompt_patches import PromptPatchManager, PatchType

    mgr = PromptPatchManager()
    # Note: TOOL_USAGE_HINT/DOMAIN_KNOWLEDGE/STYLE_PREFERENCE patches with
    # confidence >= 0.8 auto-approve immediately (see propose()) -- use
    # WORKFLOW_PATTERN or ERROR_AVOIDANCE (as below) to see a patch land
    # in PROPOSED status for manual `missy patches approve/reject` review.
    mgr.propose(PatchType.WORKFLOW_PATTERN, "Batch independent subtasks into one delegate_task call.", confidence=0.9)
    print(mgr.build_patch_prompt())
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Default patch store location (read at construction time, so tests can
#: redirect it).
DEFAULT_STORE_PATH = "~/.missy/patches.json"

#: Longest patch text accepted (SEC-04). Patches are appended to the system
#: prompt of every run, so an unbounded one is both a cost and an injection
#: amplification risk.
MAX_PATCH_CONTENT_CHARS = 2000


def scan_patch_content(content: str) -> list[str]:
    """Return prompt-injection findings for *content* (empty = clean).

    Fails closed: if the scanner itself is unavailable, report that as a
    finding so an unscanned patch is never silently treated as clean.
    """
    try:
        from missy.security.sanitizer import InputSanitizer

        return list(InputSanitizer().check_for_injection(content))
    except Exception as exc:  # pragma: no cover - defensive
        return [f"scanner unavailable: {exc}"]


class PatchType(StrEnum):
    """Category of a :class:`PromptPatch`."""

    TOOL_USAGE_HINT = "tool_usage_hint"
    ERROR_AVOIDANCE = "error_avoidance"
    WORKFLOW_PATTERN = "workflow_pattern"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    STYLE_PREFERENCE = "style_preference"


class PatchStatus(StrEnum):
    """Lifecycle status of a :class:`PromptPatch`."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class PromptPatch:
    """A single prompt guidance entry.

    Attributes:
        id: Short unique identifier (8-character UUID prefix).
        patch_type: Category of guidance.
        content: The guidance text to inject into the system prompt.
        confidence: Initial confidence score (0.0–1.0).
        status: Current lifecycle status.
        applications: Number of times this patch was active during a run.
        successes: Number of runs where this patch was active and the task
            succeeded.
        created_at: ISO-8601 UTC timestamp of creation.
    """

    id: str
    patch_type: PatchType
    content: str
    confidence: float
    status: PatchStatus = PatchStatus.PROPOSED
    applications: int = 0
    successes: int = 0
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    @property
    def success_rate(self) -> float:
        """Return success ratio (0.0 when never applied).

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        if self.applications == 0:
            return 0.0
        return self.successes / self.applications

    @property
    def is_expired(self) -> bool:
        """Return ``True`` when this patch has a poor enough track record to retire.

        A patch is expired when it has been applied at least 5 times and its
        success rate has fallen below 40 %.

        Returns:
            ``True`` if the patch should be retired.
        """
        if self.applications < 5:
            return False
        return self.success_rate < 0.4


class PatchContentRejected(ValueError):
    """Raised when approving a patch whose content the injection scanner flags."""

    def __init__(self, patch_id: str, findings: list[str]) -> None:
        self.patch_id = patch_id
        self.findings = findings
        super().__init__(
            f"Patch {patch_id} content matches prompt-injection patterns: {', '.join(findings[:5])}"
        )


class PatchStoreRefusedError(RuntimeError):
    """The patch store exists but was refused (untrusted permissions/owner or
    unparseable), so writing to it would silently erase its contents."""


class PromptPatchManager:
    """Manages proposed/approved system prompt patches with file persistence.

    Args:
        store_path: Path to the JSON file for persisting patches.  Tilde
            expansion is performed automatically.
    """

    MAX_PATCHES = 20

    def __init__(self, store_path: str | None = None) -> None:
        self._path = Path(store_path or DEFAULT_STORE_PATH).expanduser()
        self._lock = threading.Lock()
        self._signature: tuple[int, ...] | None = None
        #: Set by _load() when an existing store was not trusted/readable.
        #: While set, _save() refuses to write (it would replace the file's
        #: real contents with our empty view of it).
        self._refused_reason: str | None = None
        self._patches: list[PromptPatch] = self._load()
        self._signature = self._file_signature()

    # ------------------------------------------------------------------
    # Persistence (SEC-04)
    # ------------------------------------------------------------------

    def _file_signature(self) -> tuple[int, ...] | None:
        # Mode/owner are included so fixing a refused store with `chmod 600`
        # is picked up without a restart (chmod changes neither mtime nor size).
        try:
            st = self._path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_mode, st.st_uid)

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Exclusive cross-process lock so the runtime and `missy patches`
        never overwrite each other's changes."""
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        with contextlib.suppress(OSError):
            os.makedirs(self._path.parent, mode=0o700, exist_ok=True)
        try:
            fd = os.open(str(self._path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load(self) -> list[PromptPatch]:
        """Load patches from the JSON store file.

        Approved patches are injected into the system prompt, so the file is
        trusted only when owned by the current user and not group/world
        writable -- the same posture as ``jobs.json``/``mcp.json``.

        Returns:
            A list of :class:`PromptPatch` instances, or an empty list when
            the file is absent, untrusted, or malformed.
        """
        self._refused_reason = None
        if not self._path.exists():
            return []
        try:
            st = self._path.stat()
        except OSError as exc:
            logger.error("Cannot stat prompt patch store %s: %s", self._path, exc)
            self._refused_reason = f"cannot stat: {exc}"
            return []
        if st.st_uid != os.getuid():
            logger.error(
                "Prompt patch store %s is not owned by the current user — refusing to load",
                self._path,
            )
            self._refused_reason = "not owned by the current user"
            return []
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            logger.error(
                "Prompt patch store %s is group/world-writable (mode=%o) — refusing to "
                "load; run `chmod 600 %s`",
                self._path,
                st.st_mode & 0o777,
                self._path,
            )
            self._refused_reason = f"group/world-writable (mode {st.st_mode & 0o777:o})"
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            patches = []
            for d in data:
                d["patch_type"] = PatchType(d["patch_type"])
                d["status"] = PatchStatus(d["status"])
                patch = PromptPatch(**d)
                if len(patch.content) > MAX_PATCH_CONTENT_CHARS:
                    logger.warning("Skipping oversized prompt patch %s", patch.id)
                    continue
                patches.append(patch)
            return patches
        except Exception:
            logger.warning("Failed to load prompt patches from %s", self._path, exc_info=True)
            self._refused_reason = "unreadable or malformed"
            return []

    def _save(self) -> None:
        """Atomically persist the patch list with 0600 permissions.

        Raises:
            PatchStoreRefusedError: When the existing store was refused at
                load time -- overwriting it would erase patches we never read.
        """
        if self._refused_reason is not None:
            raise PatchStoreRefusedError(
                f"Not writing prompt patch store {self._path}: it was refused on load "
                f"({self._refused_reason}). Fix it (e.g. `chmod 600 {self._path}`) first."
            )
        os.makedirs(self._path.parent, mode=0o700, exist_ok=True)
        data = json.dumps([asdict(p) for p in self._patches], indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".patches-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._signature = self._file_signature()

    def _mutate(self, fn: Callable[[], _T]) -> _T:
        """Run *fn* against freshly reloaded state and persist the result.

        Serialized across threads (``self._lock``) and processes (flock), so
        e.g. `missy patches approve` and a running gateway's
        :meth:`record_outcome` never drop each other's updates.
        """
        with self._lock, self._file_lock():
            self._reload_if_changed_locked()
            if self._refused_reason is not None:
                raise PatchStoreRefusedError(
                    f"Prompt patch store {self._path} was refused on load "
                    f"({self._refused_reason}); fix it (e.g. `chmod 600 {self._path}`) "
                    "before changing patches."
                )
            return fn()

    def _reload_if_changed_locked(self) -> None:
        """Merge the on-disk store into memory if another process wrote it.

        Existing :class:`PromptPatch` objects are updated in place (identity
        preserved) so callers holding a reference see the new state.
        """
        signature = self._file_signature()
        if signature == self._signature:
            return
        existing = {p.id: p for p in self._patches}
        merged: list[PromptPatch] = []
        for disk_patch in self._load():
            live = existing.get(disk_patch.id)
            if live is None:
                merged.append(disk_patch)
                continue
            for field_name in PromptPatch.__dataclass_fields__:
                setattr(live, field_name, getattr(disk_patch, field_name))
            merged.append(live)
        self._patches = merged
        self._signature = signature

    def refresh(self) -> None:
        """Reload from disk if another process changed the store."""
        if self._file_signature() == self._signature:
            return
        with self._lock, self._file_lock():
            self._reload_if_changed_locked()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def propose(
        self,
        patch_type: PatchType,
        content: str,
        confidence: float = 0.7,
    ) -> PromptPatch | None:
        """Create and store a new proposed patch.

        Low-risk patch types (tool usage hints, domain knowledge, style
        preferences) with confidence >= 0.8 are auto-approved.

        Args:
            patch_type: Category of the patch.
            content: Guidance text.
            confidence: Initial confidence score (0.0–1.0).

        Returns:
            The created :class:`PromptPatch`, or ``None`` when the store is
            at capacity.
        """
        content = (content or "").strip()
        if not content or len(content) > MAX_PATCH_CONTENT_CHARS:
            logger.warning(
                "Refusing prompt patch proposal: content empty or over %d chars",
                MAX_PATCH_CONTENT_CHARS,
            )
            return None
        # Only clean content may skip human review (SEC-04).
        auto_approve = (
            patch_type
            in (
                PatchType.TOOL_USAGE_HINT,
                PatchType.DOMAIN_KNOWLEDGE,
                PatchType.STYLE_PREFERENCE,
            )
            and confidence >= 0.8
            and not scan_patch_content(content)
        )

        def _propose() -> PromptPatch | None:
            if len(self._patches) >= self.MAX_PATCHES:
                return None
            patch = PromptPatch(
                id=str(uuid.uuid4())[:8],
                patch_type=patch_type,
                content=content,
                confidence=confidence,
            )
            if auto_approve:
                patch.status = PatchStatus.APPROVED
            self._patches.append(patch)
            self._save()
            return patch

        return self._mutate(_propose)

    def approve(self, patch_id: str, *, force: bool = False) -> bool:
        """Approve the patch with the given ID.

        Only a patch currently in :attr:`PatchStatus.PROPOSED` can be
        approved -- without this guard, re-issuing an approve call against
        an already-``REJECTED`` or already-``EXPIRED`` patch (e.g. a stale
        CLI/API invocation replayed after the operator explicitly rejected
        it, or after it was auto-retired for a poor success rate) would
        silently reinstate it into :meth:`get_active_patches`'s active set
        with no further human review, contradicting both this method's own
        "approve a *proposed* patch" contract and the identical CLI help
        text (``missy patches approve``).

        Approved text is injected into every run's system prompt, so the
        content is scanned for prompt-injection patterns first (SEC-04); a
        flagged patch is refused unless *force* is set.

        Args:
            patch_id: Short patch identifier.
            force: Approve even when the injection scanner flags the content.

        Returns:
            ``True`` if the patch was found in ``PROPOSED`` status and was
            approved; ``False`` if not found or not currently proposed.

        Raises:
            PatchContentRejected: When the content is flagged and *force* is
                ``False``.
        """

        def _approve() -> bool:
            for p in self._patches:
                if p.id == patch_id:
                    if p.status != PatchStatus.PROPOSED:
                        return False
                    findings = scan_patch_content(p.content)
                    if findings and not force:
                        raise PatchContentRejected(patch_id, findings)
                    p.status = PatchStatus.APPROVED
                    self._save()
                    return True
            return False

        return self._mutate(_approve)

    def reject(self, patch_id: str) -> bool:
        """Reject the patch with the given ID.

        Only a patch currently in :attr:`PatchStatus.PROPOSED` can be
        rejected -- see :meth:`approve`'s docstring for why an unguarded
        status transition is a real bug, not just a hypothetical one.

        Args:
            patch_id: Short patch identifier.

        Returns:
            ``True`` if the patch was found in ``PROPOSED`` status and was
            rejected; ``False`` if not found or not currently proposed.
        """

        def _reject() -> bool:
            for p in self._patches:
                if p.id == patch_id:
                    if p.status != PatchStatus.PROPOSED:
                        return False
                    p.status = PatchStatus.REJECTED
                    self._save()
                    return True
            return False

        return self._mutate(_reject)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_patches(self) -> list[PromptPatch]:
        """Return all approved, non-expired patches.

        Side effect: expired patches are transitioned to
        :attr:`PatchStatus.EXPIRED` status.

        Returns:
            A list of currently active :class:`PromptPatch` instances.
        """
        self.refresh()
        with self._lock:
            active = [p for p in self._patches if p.status == PatchStatus.APPROVED]
            expired = [p.id for p in active if p.is_expired]
        if not expired:
            return active

        def _expire() -> list[PromptPatch]:
            live = []
            changed = False
            for p in self._patches:
                if p.status == PatchStatus.APPROVED:
                    if p.is_expired:
                        p.status = PatchStatus.EXPIRED
                        changed = True
                    else:
                        live.append(p)
            if changed:
                self._save()
            return live

        return self._mutate(_expire)

    def list_proposed(self) -> list[PromptPatch]:
        """Return all patches in PROPOSED status.

        Returns:
            A list of :class:`PromptPatch` instances awaiting review.
        """
        self.refresh()
        with self._lock:
            return [p for p in self._patches if p.status == PatchStatus.PROPOSED]

    def list_all(self) -> list[PromptPatch]:
        """Return a copy of all patches regardless of status.

        Returns:
            A list of all :class:`PromptPatch` instances.
        """
        self.refresh()
        with self._lock:
            return list(self._patches)

    def record_outcome(self, success: bool) -> None:
        """Record the outcome of a run for all currently active patches.

        Increments :attr:`~PromptPatch.applications` for every approved
        patch, and :attr:`~PromptPatch.successes` when *success* is ``True``.

        Args:
            success: ``True`` if the agent run was successful.
        """

        def _record() -> None:
            touched = False
            for p in self._patches:
                if p.status == PatchStatus.APPROVED:
                    p.applications += 1
                    if success:
                        p.successes += 1
                    touched = True
            if touched:
                self._save()

        self._mutate(_record)

    def build_patch_prompt(self) -> str:
        """Build a system prompt appendix from all active patches.

        Returns:
            A formatted multi-line string, or an empty string when there are
            no active patches.
        """
        active = self.get_active_patches()
        if not active:
            return ""
        lines = ["\n## Active Prompt Guidance"]
        for p in active:
            lines.append(f"- [{p.patch_type.value}] {p.content}")
        return "\n".join(lines)
