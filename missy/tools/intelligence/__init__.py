"""Tool intelligence — request tracking, pattern detection, and candidate synthesis.

Public API
----------
- :class:`RequestTracker` / :func:`get_request_tracker` — record turns, detect patterns.
- :class:`RequestEvent` — a single recorded user turn.
- :class:`RequestPattern` — a detected high-frequency workflow cluster.
- :class:`CandidateStore` / :func:`get_candidate_store` — tool candidate lifecycle.
- :class:`ToolCandidate` — a proposed or active structured tool.
- :class:`ToolLifecycleState` — lifecycle state enum (proposed → enabled → disabled).
- :class:`BenchmarkSummary` — per-provider benchmark outcome stored on a candidate.
- :class:`CandidateGenerator` — generate candidates from patterns.
- :class:`GenerationResult` — outcome of a generation attempt.
"""

from .candidate_generator import CandidateGenerator, GenerationResult
from .candidate_store import (
    BenchmarkSummary,
    CandidateStore,
    ToolCandidate,
    ToolLifecycleState,
    get_candidate_store,
)
from .provider_gate import (
    GateDecision,
    ProviderGateStore,
    ToolProviderGate,
    get_provider_gate_store,
)
from .request_tracker import RequestEvent, RequestPattern, RequestTracker, get_request_tracker

__all__ = [
    "BenchmarkSummary",
    "CandidateGenerator",
    "CandidateStore",
    "GateDecision",
    "GenerationResult",
    "ProviderGateStore",
    "RequestEvent",
    "RequestPattern",
    "RequestTracker",
    "ToolCandidate",
    "ToolLifecycleState",
    "ToolProviderGate",
    "get_candidate_store",
    "get_provider_gate_store",
    "get_request_tracker",
]
