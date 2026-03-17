"""Prompt drift detection for the Missy framework.

:class:`PromptDriftDetector` registers system prompts with SHA-256 hashes
and verifies they have not been tampered with during a conversation (e.g.
via prompt injection in tool outputs).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _PromptRecord:
    """Internal record of a registered prompt."""

    prompt_id: str
    expected_hash: str


class PromptDriftDetector:
    """Detects tampering of registered prompts by comparing SHA-256 hashes.

    Usage::

        detector = PromptDriftDetector()
        detector.register("system", system_prompt)
        # ... later, before each provider call ...
        if not detector.verify("system", system_prompt):
            # drift detected!
    """

    def __init__(self) -> None:
        self._records: dict[str, _PromptRecord] = {}

    @staticmethod
    def _hash(content: str) -> str:
        """Return the SHA-256 hex digest of *content*."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def register(self, prompt_id: str, content: str) -> None:
        """Register a prompt with its SHA-256 hash.

        Args:
            prompt_id: Unique identifier for this prompt.
            content: The prompt text to fingerprint.
        """
        self._records[prompt_id] = _PromptRecord(
            prompt_id=prompt_id,
            expected_hash=self._hash(content),
        )

    def verify(self, prompt_id: str, content: str) -> bool:
        """Verify that *content* matches the registered hash for *prompt_id*.

        Args:
            prompt_id: The prompt identifier previously passed to :meth:`register`.
            content: The current prompt text to check.

        Returns:
            ``True`` if the content is unchanged, ``False`` if drift is detected.
            Returns ``True`` if *prompt_id* was never registered (nothing to check).
        """
        record = self._records.get(prompt_id)
        if record is None:
            return True
        return self._hash(content) == record.expected_hash

    def get_drift_report(self) -> list[dict]:
        """Return a report of all registered prompts and their drift status.

        Returns:
            List of dicts with keys ``prompt_id``, ``expected_hash``,
            ``actual_hash``, and ``drifted``.  Since this method has no
            access to the current content, ``actual_hash`` and ``drifted``
            reflect the last :meth:`verify` result — but this method
            reports the *stored* state only.  Use :meth:`verify_all` for
            a live check.
        """
        return [
            {
                "prompt_id": record.prompt_id,
                "expected_hash": record.expected_hash,
            }
            for record in self._records.values()
        ]

    def verify_all(self, contents: dict[str, str]) -> list[dict]:
        """Verify multiple prompts and return a drift report.

        Args:
            contents: Mapping of prompt_id to current content.

        Returns:
            List of dicts with ``prompt_id``, ``expected_hash``,
            ``actual_hash``, and ``drifted``.
        """
        report = []
        for prompt_id, record in self._records.items():
            content = contents.get(prompt_id)
            if content is not None:
                actual_hash = self._hash(content)
                report.append(
                    {
                        "prompt_id": record.prompt_id,
                        "expected_hash": record.expected_hash,
                        "actual_hash": actual_hash,
                        "drifted": actual_hash != record.expected_hash,
                    }
                )
            else:
                report.append(
                    {
                        "prompt_id": record.prompt_id,
                        "expected_hash": record.expected_hash,
                        "actual_hash": None,
                        "drifted": False,
                    }
                )
        return report
