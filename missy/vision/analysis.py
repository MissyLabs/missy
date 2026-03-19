"""Domain-specific visual analysis for puzzle assistance and painting feedback.

Provides structured prompts and local preprocessing for two primary visual
task modes:

1. **Puzzle assistance** — board-state tracking, piece identification,
   edge/corner detection, color clustering, placement guidance.
2. **Painting feedback** — warm, encouraging composition critique with
   supportive coaching tone.

The heavy analysis is delegated to the LLM via vision-capable API calls.
This module handles prompt construction, local preprocessing, and
response formatting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image analysis constants
# ---------------------------------------------------------------------------

# Canny edge detection thresholds
_CANNY_EDGE_LOW = 50
_CANNY_EDGE_HIGH = 150
_CANNY_CONTOUR_LOW = 30
_CANNY_CONTOUR_HIGH = 100

# K-means color clustering
_KMEANS_CLUSTERS = 8
_KMEANS_MAX_ITER = 20
_KMEANS_EPSILON = 1.0
_KMEANS_DOWNSAMPLE_SIZE = (200, 200)
_KMEANS_MIN_COLOR_PCT = 2

# Edge overlay blending weights
_EDGE_OVERLAY_ORIGINAL = 0.8
_EDGE_OVERLAY_EDGE = 0.2

# Color classification thresholds
_COLOR_BLACK_MAX = 50
_COLOR_WHITE_MIN = 200

# Change detection (scene_memory uses its own constants)
_CHANGE_COMPARE_SIZE = (64, 64)

# ---------------------------------------------------------------------------
# Analysis modes
# ---------------------------------------------------------------------------


class AnalysisMode(StrEnum):
    GENERAL = "general"
    PUZZLE = "puzzle"
    PAINTING = "painting"
    INSPECTION = "inspection"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

GENERAL_ANALYSIS_PROMPT = """\
Analyze this image and provide a detailed description of what you see.
Include observations about:
- Main subjects and objects
- Layout and composition
- Colors and lighting
- Any text or labels visible
- Notable details
"""

PUZZLE_ANALYSIS_PROMPT = """\
You are helping with a tabletop jigsaw puzzle. Analyze this image carefully.

Provide structured observations:

1. **Board State**: Describe the current state of the puzzle — what areas are \
completed, what's in progress, what's empty. Estimate completion percentage.

2. **Piece Identification**: Identify any loose pieces visible. For each:
   - Describe its colors, patterns, and textures
   - Note if it's an edge piece (one flat side), corner piece (two flat sides), \
or interior piece (no flat sides)
   - Note the piece's tab/blank pattern (e.g. "2 tabs, 2 blanks")
   - Estimate which region of the puzzle it likely belongs to

3. **Orientation Hints**: For pieces that seem close to fitting:
   - Suggest rotation (0°, 90°, 180°, 270°)
   - Note which direction the image/pattern flows
   - Identify alignment cues from neighboring pieces

4. **Grouping Suggestions**: Suggest how loose pieces could be grouped by:
   - Color similarity (group sky blues, greens, etc.)
   - Texture/pattern matching (similar brushstrokes, gradients)
   - Edge type (border pieces first, then interior)
   - Region affinity (pieces that likely go near each other)

5. **Placement Guidance**: If you can identify likely placement locations:
   - Describe where specific pieces might fit
   - Note any pattern continuity clues (lines, color gradients)
   - Identify "anchor" pieces that connect completed regions
   - Suggest which area to focus on next

6. **Strategy Tips**: Provide 2-3 actionable next steps, such as:
   - "Sort all remaining edge pieces and complete the border"
   - "The blue gradient in the top-left suggests these 3 pieces connect"
   - "Try rotating the piece with the tree pattern 90° clockwise"

Be specific and reference actual colors, patterns, and positions you observe.
"""

PUZZLE_FOLLOWUP_PROMPT = """\
Compare this new view of the puzzle with what we observed before.

Previous observations:
{previous_observations}

Previous state:
{previous_state}

Describe:
1. What progress has been made since last time?
2. Which pieces have been placed?
3. What should be tried next?
4. Any pieces that look like they fit together?

Be encouraging about progress made.
"""

PAINTING_ANALYSIS_PROMPT = """\
You are a supportive painting coach reviewing a painter's work. Your role is \
to provide warm, encouraging feedback that builds confidence while offering \
genuinely helpful guidance.

Approach this with the warmth and patience of a beloved art teacher. The \
painter has put their heart into this work, and your feedback should honor that.

Provide feedback on:

1. **First Impression** — Share your genuine first reaction. Lead with what \
catches your eye and what works well. Be specific about what draws you in.

2. **Color & Light** — Comment on the color choices and how light plays \
through the work. Note harmonies, contrasts, and mood created by the palette. \
If something could be enhanced, frame it as an exciting possibility, not a \
correction.

3. **Composition** — Discuss how the eye moves through the painting. \
Mention strong compositional choices. If balance could be improved, suggest \
it gently as something to explore.

4. **Technique** — Appreciate the brushwork, texture, and technique. Note \
areas of particular skill or expressiveness. Offer technique suggestions \
as "you might enjoy trying..." rather than corrections.

5. **Emotional Impact** — Describe what the painting makes you feel. Art is \
communication — tell the painter what their work communicates to you.

6. **Growth Opportunity** — Offer ONE specific, actionable suggestion framed \
positively: "One thing that could take this even further..."

Guidelines:
- Be warm, genuine, and encouraging — never harsh or dismissive
- Lead every section with something positive
- Frame suggestions as possibilities, not corrections
- Use language like "I love how...", "This really works because...", \
"You might enjoy exploring..."
- Never use words like "wrong", "bad", "weak", "poor", "fail"
- End with genuine encouragement about their artistic journey
"""

PAINTING_FOLLOWUP_PROMPT = """\
Let's revisit this painting with fresh eyes. Here's what we noticed before:

{previous_observations}

Now look at this new view and:
1. Notice any new details you missed before
2. See how the painting looks from this angle/lighting
3. Offer any additional encouragement or gentle suggestions
4. Celebrate any changes or progress if this is an updated version

Remember: warm, supportive, encouraging. Like a favorite art teacher.
"""

INSPECTION_PROMPT = """\
Examine this image carefully and provide a detailed inspection report.

Structure your report as:

1. **Overview**: What is the primary subject? Describe the scene at a high level.

2. **Condition Assessment**: Rate the overall condition (if applicable) and \
note any wear, damage, or irregularities.

3. **Details & Observations**:
   - List specific items, components, or features visible
   - Note any text, labels, serial numbers, or markings
   - Describe materials, textures, and finishes
   - Identify any tools, equipment, or objects present

4. **Measurements & Quantities**: Estimate sizes, counts, or quantities \
where visible. Note relative scale using reference objects.

5. **Concerns & Anomalies**: Flag anything unusual, potentially problematic, \
or requiring attention.

6. **Recommendations**: Suggest next steps or actions based on observations.

Be precise and factual. Distinguish between observations and inferences.
"""


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


@dataclass
class AnalysisRequest:
    """A request to analyze an image."""

    image: np.ndarray
    mode: AnalysisMode = AnalysisMode.GENERAL
    context: str = ""  # additional user context
    previous_observations: list[str] = field(default_factory=list)
    previous_state: dict[str, Any] = field(default_factory=dict)
    is_followup: bool = False


@dataclass
class AnalysisResult:
    """Result of visual analysis."""

    text: str  # the analysis text from the LLM
    mode: AnalysisMode
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_used: str = ""


class AnalysisPromptBuilder:
    """Builds analysis prompts based on mode and context.

    User-provided context is truncated and wrapped in a clearly-delimited
    block to mitigate prompt injection (the LLM may still follow injected
    instructions, but the delimiters signal that the text is user-supplied).
    """

    #: Maximum character length for user-supplied context strings.
    MAX_CONTEXT_LENGTH = 2000

    def build_prompt(self, request: AnalysisRequest) -> str:
        """Build the appropriate analysis prompt."""
        if request.mode == AnalysisMode.PUZZLE:
            return self._build_puzzle_prompt(request)
        elif request.mode == AnalysisMode.PAINTING:
            return self._build_painting_prompt(request)
        elif request.mode == AnalysisMode.INSPECTION:
            return self._build_inspection_prompt(request)
        else:
            return self._build_general_prompt(request)

    @classmethod
    def _sanitize_context(cls, context: str) -> str:
        """Truncate and delimit user-supplied context."""
        if not context:
            return ""
        truncated = context[: cls.MAX_CONTEXT_LENGTH]
        if len(context) > cls.MAX_CONTEXT_LENGTH:
            truncated += " [truncated]"
        return truncated

    def _build_general_prompt(self, request: AnalysisRequest) -> str:
        prompt = GENERAL_ANALYSIS_PROMPT
        ctx = self._sanitize_context(request.context)
        if ctx:
            prompt += f"\n\n[User-provided context]: {ctx}"
        return prompt

    def _build_puzzle_prompt(self, request: AnalysisRequest) -> str:
        if request.is_followup and request.previous_observations:
            return PUZZLE_FOLLOWUP_PROMPT.format(
                previous_observations="\n".join(
                    f"- {obs}" for obs in request.previous_observations
                ),
                previous_state=_format_state(request.previous_state),
            )
        prompt = PUZZLE_ANALYSIS_PROMPT
        ctx = self._sanitize_context(request.context)
        if ctx:
            prompt += f"\n\n[User note]: {ctx}"
        return prompt

    def _build_painting_prompt(self, request: AnalysisRequest) -> str:
        if request.is_followup and request.previous_observations:
            return PAINTING_FOLLOWUP_PROMPT.format(
                previous_observations="\n".join(
                    f"- {obs}" for obs in request.previous_observations
                ),
            )
        prompt = PAINTING_ANALYSIS_PROMPT
        ctx = self._sanitize_context(request.context)
        if ctx:
            prompt += f"\n\n[The painter says]: {ctx}"
        return prompt

    def _build_inspection_prompt(self, request: AnalysisRequest) -> str:
        prompt = INSPECTION_PROMPT
        ctx = self._sanitize_context(request.context)
        if ctx:
            prompt += f"\n\n[Inspection focus]: {ctx}"
        return prompt


def _format_state(state: dict[str, Any]) -> str:
    """Format state dict as readable text."""
    if not state:
        return "No previous state recorded."
    lines = []
    for key, value in state.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local preprocessing for puzzle images
# ---------------------------------------------------------------------------


class PuzzlePreprocessor:
    """Local image preprocessing for puzzle analysis.

    Enhances puzzle images before sending to the LLM by improving
    edge visibility and color separation.
    """

    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance piece edges for better visibility."""
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, _CANNY_EDGE_LOW, _CANNY_EDGE_HIGH)
            # Overlay edges on original
            edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(
                image, _EDGE_OVERLAY_ORIGINAL, edge_colored, _EDGE_OVERLAY_EDGE, 0
            )
        except Exception:
            return image

    def extract_color_regions(self, image: np.ndarray) -> dict[str, Any]:
        """Identify dominant color regions in the image."""
        try:
            import cv2

            # Downsample for speed
            small = cv2.resize(image, _KMEANS_DOWNSAMPLE_SIZE)
            # Convert to RGB for reporting
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Simple k-means clustering
            pixels = rgb.reshape(-1, 3).astype(np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                _KMEANS_MAX_ITER,
                _KMEANS_EPSILON,
            )
            _, labels, centers = cv2.kmeans(
                pixels, _KMEANS_CLUSTERS, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )

            # Count pixels per cluster
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)

            regions = []
            for idx in np.argsort(-counts):
                center = centers[idx].astype(int)
                pct = counts[idx] / total * 100
                if pct < _KMEANS_MIN_COLOR_PCT:
                    continue
                regions.append({
                    "color_rgb": center.tolist(),
                    "percentage": round(float(pct), 1),
                    "description": _describe_color(center.tolist()),
                })

            return {"dominant_colors": regions}
        except Exception as exc:
            logger.warning("Color extraction failed: %s", exc)
            return {"dominant_colors": [], "error": str(exc)}

    def detect_edges_and_corners(self, image: np.ndarray) -> dict[str, Any]:
        """Count potential edge and corner pieces based on contour analysis."""
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, _CANNY_CONTOUR_LOW, _CANNY_CONTOUR_HIGH)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            return {
                "contour_count": len(contours),
                "analysis": "Contour detection complete — detailed piece analysis "
                "is performed by the vision model.",
            }
        except Exception as exc:
            return {"error": str(exc)}


def _describe_color(rgb: list[int]) -> str:
    """Provide a human-readable color description."""
    r, g, b = rgb

    # Simple color naming
    if max(r, g, b) < _COLOR_BLACK_MAX:
        return "black"
    if min(r, g, b) > _COLOR_WHITE_MIN:
        return "white"
    if r > 180 and g < 80 and b < 80:
        return "red"
    if r < 80 and g > 150 and b < 80:
        return "green"
    if r < 80 and g < 80 and b > 150:
        return "blue"
    if r > 180 and g > 150 and b < 80:
        return "yellow"
    if r > 180 and g > 100 and b < 60:
        return "orange"
    if r > 140 and g < 80 and b > 140:
        return "purple"
    # Gray check before tan/brown — otherwise neutral grays match tan/brown
    if abs(r - g) < 30 and abs(g - b) < 30:
        if r > 160:
            return "light gray"
        return "gray"
    if r > 140 and g > 100 and b > 80:
        return "tan/brown"

    return f"rgb({r},{g},{b})"
