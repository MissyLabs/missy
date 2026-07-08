"""Tool benchmark infrastructure — suites, scoring, and result storage.

Public API
----------
- :class:`BenchmarkTask` — a single (tool, input, expected) test case.
- :class:`BenchmarkSuite` — named collection of tasks for one tool.
- :class:`BenchmarkRunner` — execute suites and record results.
- :class:`SuiteRunReport` — aggregated outcome of a suite run.
- :class:`BenchmarkResult` — raw measurement from one task run.
- :class:`ScoredResult` — measurement enriched with dimension scores.
- :class:`BenchmarkScorer` — convert raw results into ``[0,1]`` scores.
- :class:`ScoreWeights` — dimension weights for the composite.
- :class:`BenchmarkStore` / :func:`get_benchmark_store` — result persistence.
- :class:`ProviderSummary` — aggregated per-provider statistics.
"""

from .benchmark_store import BenchmarkStore, ProviderSummary, get_benchmark_store
from .runner import BenchmarkRunner, BenchmarkSuite, BenchmarkTask, SuiteRunReport
from .scoring import BenchmarkResult, BenchmarkScorer, ScoredResult, ScoreWeights

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkScorer",
    "BenchmarkStore",
    "BenchmarkSuite",
    "BenchmarkTask",
    "ProviderSummary",
    "ScoreWeights",
    "ScoredResult",
    "SuiteRunReport",
    "get_benchmark_store",
]
