"""Tool benchmark infrastructure — suites, scoring, and result storage.

Public API
----------
- :class:`BenchmarkTask` — a single (tool, input, expected) test case.
- :class:`BenchmarkSuite` — named collection of tasks for one tool.
- :class:`BenchmarkRunner` — execute suites directly against the registry.
- :class:`SuiteRunReport` — aggregated outcome of a suite run.
- :class:`LLMBenchmarkTask` — a natural-language test case for provider
  tool-calling behavior.
- :class:`LLMBenchmarkRunner` — drive a real/mock provider through
  ``complete_with_tools`` and score what it actually did.
- :class:`LLMSuiteRunReport` — aggregated outcome of an LLM benchmark run.
- :class:`MockToolProvider` — deterministic offline provider for local/mock
  benchmarking without real credentials.
- :class:`BenchmarkResult` — raw measurement from one task run.
- :class:`ScoredResult` — measurement enriched with dimension scores.
- :class:`BenchmarkScorer` — convert raw results into ``[0,1]`` scores.
- :class:`ScoreWeights` — dimension weights for the composite.
- :class:`BenchmarkStore` / :func:`get_benchmark_store` — result persistence.
- :class:`ProviderSummary` — aggregated per-provider statistics.
"""

from .benchmark_store import BenchmarkStore, ProviderSummary, get_benchmark_store
from .llm_runner import LLMBenchmarkRunner, LLMBenchmarkTask, LLMSuiteRunReport
from .mock_provider import MockToolProvider
from .runner import BenchmarkRunner, BenchmarkSuite, BenchmarkTask, SuiteRunReport
from .scoring import BenchmarkResult, BenchmarkScorer, ScoredResult, ScoreWeights

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkScorer",
    "BenchmarkStore",
    "BenchmarkSuite",
    "BenchmarkTask",
    "LLMBenchmarkRunner",
    "LLMBenchmarkTask",
    "LLMSuiteRunReport",
    "MockToolProvider",
    "ProviderSummary",
    "ScoreWeights",
    "ScoredResult",
    "SuiteRunReport",
    "get_benchmark_store",
]
