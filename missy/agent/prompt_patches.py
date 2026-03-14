"""Prompt self-tuning patch system.

Manages a collection of :class:`PromptPatch` records that are appended to the
system prompt to guide model behaviour.  Patches can be proposed automatically
by the runtime, reviewed, approved or rejected by operators, and expire when
their success rate falls below threshold.

Example::

    from missy.agent.prompt_patches import PromptPatchManager, PatchType

    mgr = PromptPatchManager()
    mgr.propose(PatchType.TOOL_USAGE_HINT, "Always verify file paths before writing.", confidence=0.9)
    print(mgr.build_patch_prompt())
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path


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


class PromptPatchManager:
    """Manages proposed/approved system prompt patches with file persistence.

    Args:
        store_path: Path to the JSON file for persisting patches.  Tilde
            expansion is performed automatically.
    """

    MAX_PATCHES = 20

    def __init__(self, store_path: str = "~/.missy/patches.json") -> None:
        self._path = Path(store_path).expanduser()
        self._lock = threading.Lock()
        self._patches: list[PromptPatch] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list[PromptPatch]:
        """Load patches from the JSON store file.

        Returns:
            A list of :class:`PromptPatch` instances, or an empty list when
            the file is absent or malformed.
        """
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            patches = []
            for d in data:
                d["patch_type"] = PatchType(d["patch_type"])
                d["status"] = PatchStatus(d["status"])
                patches.append(PromptPatch(**d))
            return patches
        except Exception:
            return []

    def _save(self) -> None:
        """Persist the current patch list to the JSON store file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps([asdict(p) for p in self._patches], indent=2))

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
        with self._lock:
            if len(self._patches) >= self.MAX_PATCHES:
                return None
            patch = PromptPatch(
                id=str(uuid.uuid4())[:8],
                patch_type=patch_type,
                content=content,
                confidence=confidence,
            )
            # Auto-approve low-risk patches with high confidence
            if (
                patch_type
                in (
                    PatchType.TOOL_USAGE_HINT,
                    PatchType.DOMAIN_KNOWLEDGE,
                    PatchType.STYLE_PREFERENCE,
                )
                and confidence >= 0.8
            ):
                patch.status = PatchStatus.APPROVED
            self._patches.append(patch)
            self._save()
            return patch

    def approve(self, patch_id: str) -> bool:
        """Approve the patch with the given ID.

        Args:
            patch_id: Short patch identifier.

        Returns:
            ``True`` if the patch was found and approved.
        """
        with self._lock:
            for p in self._patches:
                if p.id == patch_id:
                    p.status = PatchStatus.APPROVED
                    self._save()
                    return True
            return False

    def reject(self, patch_id: str) -> bool:
        """Reject the patch with the given ID.

        Args:
            patch_id: Short patch identifier.

        Returns:
            ``True`` if the patch was found and rejected.
        """
        with self._lock:
            for p in self._patches:
                if p.id == patch_id:
                    p.status = PatchStatus.REJECTED
                    self._save()
                    return True
            return False

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
        with self._lock:
            active = []
            changed = False
            for p in self._patches:
                if p.status == PatchStatus.APPROVED:
                    if p.is_expired:
                        p.status = PatchStatus.EXPIRED
                        changed = True
                    else:
                        active.append(p)
            if changed:
                self._save()
            return active

    def list_proposed(self) -> list[PromptPatch]:
        """Return all patches in PROPOSED status.

        Returns:
            A list of :class:`PromptPatch` instances awaiting review.
        """
        with self._lock:
            return [p for p in self._patches if p.status == PatchStatus.PROPOSED]

    def list_all(self) -> list[PromptPatch]:
        """Return a copy of all patches regardless of status.

        Returns:
            A list of all :class:`PromptPatch` instances.
        """
        with self._lock:
            return list(self._patches)

    def record_outcome(self, success: bool) -> None:
        """Record the outcome of a run for all currently active patches.

        Increments :attr:`~PromptPatch.applications` for every approved
        patch, and :attr:`~PromptPatch.successes` when *success* is ``True``.

        Args:
            success: ``True`` if the agent run was successful.
        """
        with self._lock:
            for p in self._patches:
                if p.status == PatchStatus.APPROVED:
                    p.applications += 1
                    if success:
                        p.successes += 1
            self._save()

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
