"""Trust scoring system for providers, MCP servers, and tools.

Tracks reliability on a 0-1000 scale.  New entities start at 500.
Successes increase the score; failures and policy violations decrease it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Default score assigned to entities seen for the first time.
DEFAULT_SCORE = 500

#: Maximum trust score.
MAX_SCORE = 1000

#: Minimum trust score.
MIN_SCORE = 0


class TrustScorer:
    """In-memory trust scorer for entity reliability tracking.

    Scores range from 0 (untrusted) to 1000 (fully trusted).
    """

    def __init__(self) -> None:
        self._scores: dict[str, int] = {}

    def score(self, entity_id: str) -> int:
        """Return the current trust score for *entity_id* (default 500)."""
        return self._scores.get(entity_id, DEFAULT_SCORE)

    def record_success(self, entity_id: str, weight: int = 10) -> None:
        """Increase the score for *entity_id* by *weight* (capped at 1000)."""
        current = self.score(entity_id)
        self._scores[entity_id] = min(current + weight, MAX_SCORE)

    def record_failure(self, entity_id: str, weight: int = 50) -> None:
        """Decrease the score for *entity_id* by *weight* (floored at 0)."""
        current = self.score(entity_id)
        self._scores[entity_id] = max(current - weight, MIN_SCORE)

    def record_violation(self, entity_id: str, weight: int = 200) -> None:
        """Major decrease for a policy violation (floored at 0)."""
        current = self.score(entity_id)
        self._scores[entity_id] = max(current - weight, MIN_SCORE)

    def is_trusted(self, entity_id: str, threshold: int = 200) -> bool:
        """Return ``True`` if the entity's score is above *threshold*."""
        return self.score(entity_id) > threshold

    def get_scores(self) -> dict[str, int]:
        """Return a copy of all current scores."""
        return dict(self._scores)

    def reset(self, entity_id: str) -> None:
        """Reset *entity_id* back to the default score (500)."""
        self._scores[entity_id] = DEFAULT_SCORE
