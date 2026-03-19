"""Audio-triggered vision intent detection.

Classifies user utterances to determine whether visual input is needed.
Integrates with the existing ``IntentInterpreter`` from the behavior layer
but adds vision-specific detection.

Design rules
------------
- Wake word detection remains separate from task execution.
- Vision activation is SCOPED to the active task — never becomes always-on.
- All activation decisions, captures, and outcomes are logged.
- Auto-activation requires strong confidence; ambiguous cases ask the user.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class VisionIntent(str, Enum):
    """Classified vision intent from user utterance."""

    NONE = "none"  # no vision needed
    LOOK = "look"  # explicit "look at this"
    PUZZLE = "puzzle"  # puzzle-related visual task
    PAINTING = "painting"  # painting/art feedback
    INSPECT = "inspect"  # general visual inspection
    CHECK = "check"  # "check what's on the table"
    COMPARE = "compare"  # "does this match" / comparison
    READ = "read"  # "read what this says"
    SCREENSHOT = "screenshot"  # desktop screenshot request


class ActivationDecision(str, Enum):
    """Whether to auto-activate vision."""

    ACTIVATE = "activate"  # strong signal — auto-activate
    ASK = "ask"  # ambiguous — ask the user
    SKIP = "skip"  # no vision needed


@dataclass
class IntentResult:
    """Result of vision intent classification."""

    intent: VisionIntent
    decision: ActivationDecision
    confidence: float  # 0.0 to 1.0
    trigger_phrase: str = ""  # the phrase that triggered detection
    suggested_mode: str = ""  # "puzzle", "painting", "general"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "trigger_phrase": self.trigger_phrase,
            "suggested_mode": self.suggested_mode,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Explicit vision request patterns (high confidence)
_EXPLICIT_LOOK_PATTERNS: list[tuple[re.Pattern[str], VisionIntent, float]] = [
    (re.compile(r"\b(look\s+at|show\s+me|let\s+me\s+see|take\s+a\s+look)\b", re.I), VisionIntent.LOOK, 0.95),
    (re.compile(r"\b(can\s+you\s+see|do\s+you\s+see|what\s+do\s+you\s+see)\b", re.I), VisionIntent.LOOK, 0.90),
    (re.compile(r"\bcheck\s+(what('s|\s+is)\s+(on|at)|this|that)\b", re.I), VisionIntent.CHECK, 0.85),
    (re.compile(r"\b(take\s+a\s+(photo|picture|snapshot)|capture|snap)\b", re.I), VisionIntent.LOOK, 0.90),
    (re.compile(r"\bscreenshot\b", re.I), VisionIntent.SCREENSHOT, 0.95),
]

# Puzzle-related patterns
_PUZZLE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bpuzzle\s+piece\b", re.I), 0.95),
    (re.compile(r"\b(where\s+does\s+this|this\s+piece)\s+(go|fit|belong)\b", re.I), 0.90),
    (re.compile(r"\b(edge|corner|border)\s+piece\b", re.I), 0.90),
    (re.compile(r"\b(sky|water|grass|tree)\s+section\b", re.I), 0.80),
    (re.compile(r"\bjigsaw\b", re.I), 0.85),
    (re.compile(r"\bfit\s+in\s+the\s+\w+\s+section\b", re.I), 0.85),
    (re.compile(r"\b(sort|group|cluster)\s+(the\s+)?(pieces|parts)\b", re.I), 0.80),
    (re.compile(r"\bboard\s+state\b", re.I), 0.85),
    (re.compile(r"\b(puzzle|piece)\b.*\b(help|assist|guide)\b", re.I), 0.80),
]

# Painting/art patterns
_PAINTING_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(painting|canvas|artwork|art\s+piece)\b", re.I), 0.85),
    (re.compile(r"\bwhat\s+do\s+you\s+think\s+of\s+(this|my)\b", re.I), 0.75),
    (re.compile(r"\b(how\s+(can|do)\s+I\s+improve|feedback|critique)\b.*\b(paint|draw|art)\b", re.I), 0.90),
    (re.compile(r"\b(color|composition|brushwork|technique|palette)\b", re.I), 0.60),
    (re.compile(r"\b(sketch|drawing|watercolor|oil|acrylic)\b", re.I), 0.70),
    (re.compile(r"\bhow\s+(does\s+)?this\s+look\b", re.I), 0.65),
]

# Reading/inspection patterns
_INSPECT_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bread\s+(this|what|the)\b", re.I), 0.80),
    (re.compile(r"\bwhat\s+(does\s+)?(this|it|that)\s+say\b", re.I), 0.85),
    (re.compile(r"\b(identify|recognize|detect)\b", re.I), 0.70),
    (re.compile(r"\bwhat('s|\s+is)\s+(on|at)\s+(the\s+)?(table|desk|screen|board)\b", re.I), 0.80),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class VisionIntentClassifier:
    """Classifies user text for vision-related intent.

    Parameters
    ----------
    auto_threshold:
        Minimum confidence to auto-activate vision (default 0.80).
    ask_threshold:
        Minimum confidence to ask user about activation (default 0.50).
    """

    def __init__(
        self,
        auto_threshold: float = 0.80,
        ask_threshold: float = 0.50,
    ) -> None:
        if not (0.0 <= auto_threshold <= 1.0):
            raise ValueError(f"auto_threshold must be 0.0-1.0, got {auto_threshold}")
        if not (0.0 <= ask_threshold <= 1.0):
            raise ValueError(f"ask_threshold must be 0.0-1.0, got {ask_threshold}")
        self._auto_threshold = auto_threshold
        self._ask_threshold = ask_threshold
        self._activation_log: list[IntentResult] = []
        self._log_lock = __import__("threading").Lock()

    def classify(self, text: str) -> IntentResult:
        """Classify a user utterance for vision intent."""
        if not text or not text.strip():
            return IntentResult(
                intent=VisionIntent.NONE,
                decision=ActivationDecision.SKIP,
                confidence=0.0,
            )

        # Check each category and take the highest confidence match
        best_intent = VisionIntent.NONE
        best_confidence = 0.0
        best_trigger = ""
        best_mode = "general"

        # Explicit look/check patterns
        for pattern, intent, base_conf in _EXPLICIT_LOOK_PATTERNS:
            m = pattern.search(text)
            if m:
                if base_conf > best_confidence:
                    best_intent = intent
                    best_confidence = base_conf
                    best_trigger = m.group(0)

        # Puzzle patterns
        for pattern, base_conf in _PUZZLE_PATTERNS:
            m = pattern.search(text)
            if m:
                effective = base_conf
                if best_intent in (VisionIntent.LOOK, VisionIntent.CHECK):
                    effective = min(1.0, base_conf + 0.1)  # boost if combined
                if effective > best_confidence:
                    best_intent = VisionIntent.PUZZLE
                    best_confidence = effective
                    best_trigger = m.group(0)
                    best_mode = "puzzle"

        # Painting patterns
        for pattern, base_conf in _PAINTING_PATTERNS:
            m = pattern.search(text)
            if m:
                effective = base_conf
                if best_intent in (VisionIntent.LOOK, VisionIntent.CHECK):
                    effective = min(1.0, base_conf + 0.1)
                if effective > best_confidence:
                    best_intent = VisionIntent.PAINTING
                    best_confidence = effective
                    best_trigger = m.group(0)
                    best_mode = "painting"

        # Inspect/read patterns
        for pattern, base_conf in _INSPECT_PATTERNS:
            m = pattern.search(text)
            if m:
                if base_conf > best_confidence:
                    best_intent = VisionIntent.INSPECT
                    best_confidence = base_conf
                    best_trigger = m.group(0)
                    best_mode = "inspection"

        # Determine activation decision
        if best_confidence >= self._auto_threshold:
            decision = ActivationDecision.ACTIVATE
        elif best_confidence >= self._ask_threshold:
            decision = ActivationDecision.ASK
        else:
            decision = ActivationDecision.SKIP

        result = IntentResult(
            intent=best_intent,
            decision=decision,
            confidence=best_confidence,
            trigger_phrase=best_trigger,
            suggested_mode=best_mode,
        )

        with self._log_lock:
            self._activation_log.append(result)
        logger.debug(
            "Vision intent: %s (%.2f) — %s [trigger: %r]",
            result.intent.value,
            result.confidence,
            result.decision.value,
            result.trigger_phrase,
        )

        return result

    @property
    def activation_log(self) -> list[IntentResult]:
        """Return the activation decision log for auditing."""
        with self._log_lock:
            return list(self._activation_log)

    def clear_log(self) -> None:
        with self._log_lock:
            self._activation_log.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_classifier: VisionIntentClassifier | None = None


def get_intent_classifier() -> VisionIntentClassifier:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = VisionIntentClassifier()
    return _default_classifier


def classify_vision_intent(text: str) -> IntentResult:
    """Convenience: classify text for vision intent."""
    return get_intent_classifier().classify(text)
