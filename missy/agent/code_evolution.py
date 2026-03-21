"""Self-evolving code modification engine.

Manages proposals for Missy to modify its own source code based on errors,
user requests, or learned patterns.  Every modification goes through a
multi-stage lifecycle: **proposed → approved → applied** (or **rejected**).

Applied changes are wrapped in a git commit so that rollback is always a
single ``git revert``.  Before any change is applied the full test suite is
run; the proposal is rejected automatically if tests fail.

Security model:
- All proposals require explicit human approval before application.
- Only files inside the Missy package directory are modifiable.
- A git stash safety net preserves uncommitted work.
- Full audit trail via :class:`~missy.core.events.AuditEvent`.

Example::

    from missy.agent.code_evolution import CodeEvolutionManager

    mgr = CodeEvolutionManager()
    prop = mgr.propose(
        title="Fix timeout handling in circuit breaker",
        description="Increase base timeout and add jitter",
        file_path="missy/agent/circuit_breaker.py",
        original_code="base_timeout=60",
        proposed_code="base_timeout=90",
        trigger="repeated_error",
    )
    mgr.approve(prop.id)
    result = mgr.apply(prop.id)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def restart_process() -> None:
    """Replace the current process with a fresh invocation of the same command.

    Uses :func:`os.execv` so the PID is reused and the caller's terminal
    session is preserved.  This function does **not** return on success.

    Falls back to :func:`sys.exit` with a restart hint if ``execv`` fails.
    """
    import os
    import sys

    logger.info("Restarting process: %s %s", sys.executable, sys.argv)
    try:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    except OSError as exc:
        logger.error("Failed to restart: %s", exc)
        print(f"[missy-evolve] Restart failed ({exc}). Please restart manually.")
        sys.exit(75)  # EX_TEMPFAIL


# Root of the Missy package — the only directory we allow edits in
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class EvolutionStatus(StrEnum):
    """Lifecycle status of a code evolution proposal."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class EvolutionTrigger(StrEnum):
    """What caused this evolution to be proposed."""

    REPEATED_ERROR = "repeated_error"
    USER_REQUEST = "user_request"
    LEARNING = "learning"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class FileDiff:
    """A single file-level change within an evolution proposal.

    Attributes:
        file_path: Path relative to the repository root.
        original_code: The exact text to be replaced.
        proposed_code: The replacement text.
        description: Why this specific change is needed.
    """

    file_path: str
    original_code: str
    proposed_code: str
    description: str = ""


@dataclass
class EvolutionProposal:
    """A proposed modification to Missy's own source code.

    Attributes:
        id: Short unique identifier (8-char UUID prefix).
        title: One-line summary of the change.
        description: Detailed rationale.
        diffs: One or more file-level changes.
        trigger: What caused this proposal.
        trigger_detail: Extra context (error message, learning ID, etc.).
        status: Current lifecycle status.
        confidence: Agent's confidence that this change is correct (0.0–1.0).
        error_pattern: If triggered by an error, the recurring pattern.
        git_commit_sha: SHA of the commit created when applied.
        test_output: Captured pytest output from validation.
        created_at: ISO-8601 UTC timestamp.
        resolved_at: ISO-8601 UTC timestamp when approved/rejected/applied.
    """

    id: str
    title: str
    description: str
    diffs: list[FileDiff]
    trigger: EvolutionTrigger
    trigger_detail: str = ""
    status: EvolutionStatus = EvolutionStatus.PROPOSED
    confidence: float = 0.5
    error_pattern: str = ""
    git_commit_sha: str = ""
    test_output: str = ""
    created_at: str = ""
    resolved_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


def _serialize_proposal(prop: EvolutionProposal) -> dict:
    """Convert a proposal to a JSON-safe dict."""
    d = asdict(prop)
    d["diffs"] = [asdict(diff) for diff in prop.diffs]
    return d


def _deserialize_proposal(d: dict) -> EvolutionProposal:
    """Reconstruct a proposal from a dict."""
    d["diffs"] = [FileDiff(**fd) for fd in d.get("diffs", [])]
    d["trigger"] = EvolutionTrigger(d["trigger"])
    d["status"] = EvolutionStatus(d["status"])
    return EvolutionProposal(**d)


class CodeEvolutionManager:
    """Manages the lifecycle of code self-modification proposals.

    All proposals are persisted to a JSON file.  Application of changes
    creates a git commit with a ``[missy-evolve]`` prefix so they can be
    identified and reverted.

    Args:
        store_path: Path to the JSON persistence file.
        repo_root: Root of the git repository.  Defaults to the Missy
            package's parent directory (the repo root).
        test_command: Shell command to validate changes before committing.
    """

    MAX_PROPOSALS = 50

    def __init__(
        self,
        store_path: str = "~/.missy/evolutions.json",
        repo_root: str | None = None,
        test_command: str = "python3 -m pytest tests/ -x -q --tb=short",
    ) -> None:
        self._path = Path(store_path).expanduser()
        self._repo_root = Path(repo_root) if repo_root else _PACKAGE_ROOT.parent
        self._test_command = test_command
        self._lock = threading.Lock()
        self._proposals: list[EvolutionProposal] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list[EvolutionProposal]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [_deserialize_proposal(d) for d in data]
        except Exception:
            logger.warning("Failed to load evolutions from %s", self._path, exc_info=True)
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path.write_text(
            json.dumps([_serialize_proposal(p) for p in self._proposals], indent=2)
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_path(self, file_path: str) -> Path:
        """Ensure *file_path* is inside the Missy package directory.

        When running against the real repository, this checks against the
        installed ``missy/`` package root.  When a custom *repo_root* was
        passed (e.g. in tests), it checks against ``<repo_root>/missy/``.

        Returns the resolved absolute path.

        Raises:
            ValueError: If the path escapes the allowed directory.
        """
        resolved = (self._repo_root / file_path).resolve()
        # Allow paths under the repo-local missy/ dir or the installed package
        allowed_root = (self._repo_root / "missy").resolve()
        if not (resolved.is_relative_to(allowed_root) or resolved.is_relative_to(_PACKAGE_ROOT)):
            raise ValueError(
                f"Path {file_path!r} resolves to {resolved}, "
                f"which is outside the Missy package ({allowed_root})"
            )
        return resolved

    def _validate_diffs(self, diffs: list[FileDiff]) -> None:
        """Validate that all diffs target existing code.

        Raises:
            ValueError: If any file doesn't exist or the original_code
                is not found in the file.
        """
        for diff in diffs:
            abs_path = self._validate_path(diff.file_path)
            if not abs_path.exists():
                raise ValueError(f"File does not exist: {diff.file_path}")
            content = abs_path.read_text()
            if diff.original_code not in content:
                raise ValueError(
                    f"Original code not found in {diff.file_path}. "
                    f"Expected:\n{diff.original_code[:200]}"
                )

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo root."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self._repo_root),
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        )

    def _has_uncommitted_changes(self) -> bool:
        result = self._git("status", "--porcelain", check=False)
        return bool(result.stdout.strip())

    def _stash_if_dirty(self) -> bool:
        """Stash uncommitted work. Returns True if a stash was created."""
        if self._has_uncommitted_changes():
            self._git("stash", "push", "-m", "missy-evolve: safety stash")
            return True
        return False

    def _stash_pop(self) -> None:
        """Restore stashed work (best-effort)."""
        self._git("stash", "pop", check=False)

    # ------------------------------------------------------------------
    # Proposal lifecycle
    # ------------------------------------------------------------------

    def propose(
        self,
        title: str,
        description: str,
        file_path: str,
        original_code: str,
        proposed_code: str,
        trigger: str = "user_request",
        trigger_detail: str = "",
        confidence: float = 0.5,
        error_pattern: str = "",
    ) -> EvolutionProposal:
        """Create a new evolution proposal with a single file diff.

        Args:
            title: One-line summary.
            description: Detailed rationale.
            file_path: Relative path from repo root.
            original_code: Exact text to replace.
            proposed_code: Replacement text.
            trigger: What caused this proposal.
            trigger_detail: Extra context.
            confidence: 0.0–1.0 agent confidence.
            error_pattern: Recurring error pattern if any.

        Returns:
            The created :class:`EvolutionProposal`.

        Raises:
            ValueError: If the path is outside the package or the
                original code is not found.
        """
        return self.propose_multi(
            title=title,
            description=description,
            diffs=[FileDiff(file_path, original_code, proposed_code)],
            trigger=trigger,
            trigger_detail=trigger_detail,
            confidence=confidence,
            error_pattern=error_pattern,
        )

    def propose_multi(
        self,
        title: str,
        description: str,
        diffs: list[FileDiff],
        trigger: str = "user_request",
        trigger_detail: str = "",
        confidence: float = 0.5,
        error_pattern: str = "",
    ) -> EvolutionProposal:
        """Create a proposal with multiple file diffs.

        Args:
            title: One-line summary.
            description: Detailed rationale.
            diffs: List of :class:`FileDiff` instances.
            trigger: What caused this proposal.
            trigger_detail: Extra context.
            confidence: 0.0–1.0 agent confidence.
            error_pattern: Recurring error pattern if any.

        Returns:
            The created :class:`EvolutionProposal`.

        Raises:
            ValueError: If validation fails or store is full.
        """
        with self._lock:
            if len(self._proposals) >= self.MAX_PROPOSALS:
                raise ValueError(
                    f"Evolution store at capacity ({self.MAX_PROPOSALS}). "
                    "Reject or clean up old proposals first."
                )

            self._validate_diffs(diffs)

            proposal = EvolutionProposal(
                id=str(uuid.uuid4())[:8],
                title=title,
                description=description,
                diffs=diffs,
                trigger=EvolutionTrigger(trigger),
                trigger_detail=trigger_detail,
                confidence=confidence,
                error_pattern=error_pattern,
            )
            self._proposals.append(proposal)
            self._save()

            self._emit_event(
                "code_evolution.proposed",
                "allow",
                detail={
                    "proposal_id": proposal.id,
                    "title": title,
                    "trigger": trigger,
                    "files": [d.file_path for d in diffs],
                    "confidence": confidence,
                },
            )
            return proposal

    def approve(self, proposal_id: str) -> bool:
        """Mark a proposal as approved (ready to apply).

        Returns:
            ``True`` if the proposal was found and approved.
        """
        with self._lock:
            prop = self._find(proposal_id)
            if not prop or prop.status != EvolutionStatus.PROPOSED:
                return False
            prop.status = EvolutionStatus.APPROVED
            prop.resolved_at = datetime.now(UTC).isoformat()
            self._save()
            self._emit_event(
                "code_evolution.approved",
                "allow",
                detail={"proposal_id": proposal_id, "title": prop.title},
            )
            return True

    def reject(self, proposal_id: str) -> bool:
        """Reject a proposal.

        Returns:
            ``True`` if the proposal was found and rejected.
        """
        with self._lock:
            prop = self._find(proposal_id)
            if not prop or prop.status not in (
                EvolutionStatus.PROPOSED,
                EvolutionStatus.APPROVED,
            ):
                return False
            prop.status = EvolutionStatus.REJECTED
            prop.resolved_at = datetime.now(UTC).isoformat()
            self._save()
            self._emit_event(
                "code_evolution.rejected",
                "allow",
                detail={"proposal_id": proposal_id, "title": prop.title},
            )
            return True

    def apply(self, proposal_id: str) -> dict[str, Any]:
        """Apply an approved proposal: patch files, run tests, commit.

        This is the critical path.  Steps:

        1. Verify the proposal is in APPROVED status.
        2. Stash any uncommitted changes.
        3. Apply each diff to the source files.
        4. Run the test suite.
        5. If tests pass: commit with ``[missy-evolve]`` prefix.
        6. If tests fail: revert all changes, mark FAILED.
        7. Restore stash.

        Args:
            proposal_id: ID of the proposal to apply.

        Returns:
            A dict with keys ``success``, ``message``, ``commit_sha``,
            ``test_output``.

        Raises:
            ValueError: If proposal is not found or not in APPROVED status.
        """
        with self._lock:
            prop = self._find(proposal_id)
            if not prop:
                raise ValueError(f"Proposal {proposal_id!r} not found.")
            if prop.status != EvolutionStatus.APPROVED:
                raise ValueError(f"Proposal {proposal_id!r} is {prop.status.value}, not approved.")

        # Validate diffs still match current source
        try:
            self._validate_diffs(prop.diffs)
        except ValueError as exc:
            with self._lock:
                prop.status = EvolutionStatus.FAILED
                prop.test_output = str(exc)
                self._save()
            return {
                "success": False,
                "message": f"Diff validation failed: {exc}",
                "commit_sha": "",
                "test_output": "",
            }

        stashed = self._stash_if_dirty()

        try:
            # Apply diffs
            for diff in prop.diffs:
                abs_path = (self._repo_root / diff.file_path).resolve()
                # Prevent path traversal: ensure resolved path is under repo root
                if not abs_path.is_relative_to(self._repo_root.resolve()):
                    self._revert_diffs(prop.diffs)
                    with self._lock:
                        prop.status = EvolutionStatus.FAILED
                        self._save()
                    self._emit_event(
                        "code_evolution.apply",
                        "deny",
                        {"proposal_id": proposal_id, "error": f"Path traversal: {diff.file_path}"},
                    )
                    return {
                        "success": False,
                        "message": f"Path traversal blocked: {diff.file_path} resolves outside repo",
                        "commit_sha": "",
                        "test_output": "",
                    }
                content = abs_path.read_text()
                content = content.replace(diff.original_code, diff.proposed_code, 1)
                abs_path.write_text(content)

            # Run tests — sanitize environment to prevent API key leakage
            _SAFE_ENV_VARS = frozenset(
                {
                    "PATH",
                    "HOME",
                    "USER",
                    "LOGNAME",
                    "SHELL",
                    "LANG",
                    "LC_ALL",
                    "LC_CTYPE",
                    "LANGUAGE",
                    "TERM",
                    "COLORTERM",
                    "COLUMNS",
                    "LINES",
                    "XDG_RUNTIME_DIR",
                    "TMPDIR",
                    "PWD",
                    "DISPLAY",
                }
            )
            safe_env = {k: os.environ[k] for k in _SAFE_ENV_VARS if k in os.environ}
            # Use shlex.split to avoid shell=True injection risks.
            import shlex

            test_argv = shlex.split(self._test_command)
            test_result = subprocess.run(
                test_argv,
                shell=False,
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                timeout=300,
                env=safe_env,
            )
            test_output = test_result.stdout + test_result.stderr

            if test_result.returncode != 0:
                # Tests failed — revert
                self._revert_diffs(prop.diffs)
                with self._lock:
                    prop.status = EvolutionStatus.FAILED
                    prop.test_output = test_output[-2000:]  # cap output
                    prop.resolved_at = datetime.now(UTC).isoformat()
                    self._save()
                self._emit_event(
                    "code_evolution.test_failed",
                    "error",
                    detail={
                        "proposal_id": proposal_id,
                        "returncode": test_result.returncode,
                    },
                )
                return {
                    "success": False,
                    "message": "Tests failed. Changes reverted.",
                    "commit_sha": "",
                    "test_output": test_output[-2000:],
                }

            # Tests passed — commit
            changed_files = [d.file_path for d in prop.diffs]
            for f in changed_files:
                self._git("add", f)

            commit_msg = (
                f"[missy-evolve] {prop.title}\n\n"
                f"Proposal: {prop.id}\n"
                f"Trigger: {prop.trigger.value}\n"
                f"Confidence: {prop.confidence:.0%}\n"
                f"Files: {', '.join(changed_files)}\n\n"
                f"{prop.description}"
            )
            self._git("commit", "-m", commit_msg)

            # Get the commit SHA
            sha_result = self._git("rev-parse", "HEAD")
            commit_sha = sha_result.stdout.strip()

            with self._lock:
                prop.status = EvolutionStatus.APPLIED
                prop.git_commit_sha = commit_sha
                prop.test_output = test_output[-2000:]
                prop.resolved_at = datetime.now(UTC).isoformat()
                self._save()

            self._emit_event(
                "code_evolution.applied",
                "allow",
                detail={
                    "proposal_id": proposal_id,
                    "commit_sha": commit_sha,
                    "files": changed_files,
                },
            )
            return {
                "success": True,
                "message": f"Evolution applied and committed: {commit_sha[:8]}",
                "commit_sha": commit_sha,
                "test_output": test_output[-2000:],
            }

        except Exception as exc:
            # Something went wrong — revert everything
            logger.exception("Failed to apply evolution %s", proposal_id)
            self._revert_diffs(prop.diffs)
            with self._lock:
                prop.status = EvolutionStatus.FAILED
                prop.test_output = str(exc)
                self._save()
            return {
                "success": False,
                "message": f"Application failed: {exc}",
                "commit_sha": "",
                "test_output": str(exc),
            }
        finally:
            if stashed:
                self._stash_pop()

    def rollback(self, proposal_id: str) -> dict[str, Any]:
        """Revert a previously applied evolution via ``git revert``.

        Args:
            proposal_id: ID of the applied proposal to roll back.

        Returns:
            A dict with ``success`` and ``message``.
        """
        with self._lock:
            prop = self._find(proposal_id)
            if not prop:
                return {"success": False, "message": f"Proposal {proposal_id!r} not found."}
            if prop.status != EvolutionStatus.APPLIED:
                return {
                    "success": False,
                    "message": f"Proposal is {prop.status.value}, not applied.",
                }
            if not prop.git_commit_sha:
                return {"success": False, "message": "No commit SHA recorded."}

        try:
            self._git("revert", "--no-edit", prop.git_commit_sha)
            with self._lock:
                prop.status = EvolutionStatus.ROLLED_BACK
                prop.resolved_at = datetime.now(UTC).isoformat()
                self._save()
            self._emit_event(
                "code_evolution.rolled_back",
                "allow",
                detail={
                    "proposal_id": proposal_id,
                    "reverted_sha": prop.git_commit_sha,
                },
            )
            return {
                "success": True,
                "message": f"Reverted commit {prop.git_commit_sha[:8]}.",
            }
        except subprocess.CalledProcessError as exc:
            return {
                "success": False,
                "message": f"git revert failed: {exc.stderr or exc.stdout}",
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, proposal_id: str) -> EvolutionProposal | None:
        """Look up a proposal by ID."""
        with self._lock:
            return self._find(proposal_id)

    def list_all(self) -> list[EvolutionProposal]:
        """Return all proposals."""
        with self._lock:
            return list(self._proposals)

    def list_pending(self) -> list[EvolutionProposal]:
        """Return proposals awaiting review."""
        with self._lock:
            return [
                p
                for p in self._proposals
                if p.status in (EvolutionStatus.PROPOSED, EvolutionStatus.APPROVED)
            ]

    def list_applied(self) -> list[EvolutionProposal]:
        """Return successfully applied proposals."""
        with self._lock:
            return [p for p in self._proposals if p.status == EvolutionStatus.APPLIED]

    # ------------------------------------------------------------------
    # Error analysis for auto-proposal
    # ------------------------------------------------------------------

    def analyze_error_for_evolution(
        self,
        error_message: str,
        traceback_text: str,
        tool_name: str = "",
        failure_count: int = 1,
    ) -> EvolutionProposal | None:
        """Analyze a recurring error and propose a fix if the error
        originates from Missy's own source code.

        This method inspects the traceback to determine which Missy source
        file is involved.  It does *not* generate the fix itself — it
        creates a proposal skeleton that the agent can fill in via the
        ``code_evolve`` tool.

        Args:
            error_message: The error string.
            traceback_text: Full traceback text.
            tool_name: Name of the tool that failed.
            failure_count: How many times this error has recurred.

        Returns:
            An :class:`EvolutionProposal` skeleton if the error is in
            Missy source, otherwise ``None``.
        """
        # Only propose after repeated failures
        if failure_count < 3:
            return None

        # Find Missy source files in the traceback
        missy_files: list[str] = []
        for line in traceback_text.splitlines():
            line = line.strip()
            if "missy/" in line and 'File "' in line:
                # Extract path from: File "/path/to/missy/foo.py", line 42
                try:
                    path_part = line.split('File "')[1].split('"')[0]
                    # Convert to relative path
                    try:
                        rel = Path(path_part).resolve().relative_to(self._repo_root)
                        missy_files.append(str(rel))
                    except ValueError:
                        pass
                except (IndexError, ValueError):
                    pass

        if not missy_files:
            return None

        # Create a skeleton proposal — the agent fills in the actual fix
        target_file = missy_files[-1]  # Most specific frame
        title = f"Fix {error_message[:60]} in {Path(target_file).name}"

        return EvolutionProposal(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=(
                f"Recurring error ({failure_count}x) in tool '{tool_name}':\n"
                f"{error_message}\n\n"
                f"Traceback points to: {', '.join(missy_files)}\n\n"
                "This is a skeleton proposal. The agent should read the "
                "source file, diagnose the root cause, and fill in the "
                "original_code and proposed_code via the code_evolve tool."
            ),
            diffs=[],  # Agent fills these in
            trigger=EvolutionTrigger.REPEATED_ERROR,
            trigger_detail=traceback_text[-1000:],
            error_pattern=error_message,
            confidence=0.0,  # Skeleton — no fix yet
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find(self, proposal_id: str) -> EvolutionProposal | None:
        for p in self._proposals:
            if p.id == proposal_id:
                return p
        return None

    def _revert_diffs(self, diffs: list[FileDiff]) -> None:
        """Best-effort revert of applied diffs via ``git checkout``."""
        for diff in diffs:
            try:
                self._git("checkout", "--", diff.file_path, check=False)
            except Exception:
                logger.warning("Failed to revert %s", diff.file_path, exc_info=True)

    def _emit_event(
        self,
        event_type: str,
        result: str,
        detail: dict | None = None,
    ) -> None:
        """Publish an audit event (best-effort)."""
        try:
            from missy.core.events import AuditEvent, event_bus

            event_bus.publish(
                AuditEvent.now(
                    session_id="system",
                    task_id="code_evolution",
                    event_type=event_type,
                    category="plugin",
                    result=result,
                    detail=detail or {},
                )
            )
        except Exception:
            logger.debug("Failed to emit code evolution audit event", exc_info=True)
