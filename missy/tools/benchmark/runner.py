"""Benchmark runner — execute tool tasks against providers and collect results.

A :class:`BenchmarkSuite` is a named collection of :class:`BenchmarkTask`
objects describing inputs and expected outputs for a single tool.
:class:`BenchmarkRunner` executes the suite against one or more providers,
uses :class:`~.scoring.BenchmarkScorer` to grade each run, and persists
the results via :class:`~.benchmark_store.BenchmarkStore`.

The runner is deliberately provider-agnostic: it calls the tool directly
through the :class:`~missy.tools.registry.ToolRegistry` rather than routing
through a provider LLM.  Provider-level runs (where the LLM must choose and
call the tool) are a separate ``llm_benchmark`` path that can be layered on
top.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus

from .benchmark_store import BenchmarkStore, get_benchmark_store
from .scoring import BenchmarkResult, BenchmarkScorer, ScoredResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """A single test case for a tool.

    Attributes:
        id: Unique task identifier.
        tool_name: Name of the tool to invoke.
        input_args: Keyword arguments passed to the tool's ``execute`` method.
        expected_output: Expected ``output`` value from :class:`~missy.tools.base.ToolResult`.
            ``None`` means no ground truth — success alone scores correctly.
        schema_required_params: Required parameter names used for schema scoring.
            Defaults to the keys of *input_args*.
        timeout_s: Maximum execution time; ``0`` means no timeout.
        tags: Arbitrary string labels for filtering.
    """

    id: str
    tool_name: str
    input_args: dict[str, Any]
    expected_output: Any = None
    schema_required_params: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        tool_name: str,
        input_args: dict[str, Any],
        expected_output: Any = None,
        schema_required_params: list[str] | None = None,
        timeout_s: float = 30.0,
        tags: list[str] | None = None,
    ) -> BenchmarkTask:
        return cls(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            input_args=dict(input_args),
            expected_output=expected_output,
            schema_required_params=list(schema_required_params or input_args.keys()),
            timeout_s=timeout_s,
            tags=list(tags or []),
        )


@dataclass
class BenchmarkSuite:
    """A named collection of :class:`BenchmarkTask` objects for one tool.

    Attributes:
        name: Human-readable suite name (e.g. ``"calculator_basic"``).
        tool_name: The tool all tasks in this suite target.
        tasks: Ordered list of :class:`BenchmarkTask` instances.
        description: Optional free-text description.
        tags: Arbitrary string labels.
    """

    name: str
    tool_name: str
    tasks: list[BenchmarkTask] = field(default_factory=list)
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def add_task(self, task: BenchmarkTask) -> None:
        """Append *task* to this suite."""
        self.tasks.append(task)

    def task_count(self) -> int:
        """Return the number of tasks."""
        return len(self.tasks)


@dataclass
class SuiteRunReport:
    """Aggregated outcome of running a :class:`BenchmarkSuite`.

    Attributes:
        suite_name: Name of the executed suite.
        tool_name: Tool that was benchmarked.
        provider: Provider identifier (``"direct"`` for registry-only runs).
        scored_results: Individual :class:`~.scoring.ScoredResult` per task.
        aggregate: Mean scores across all tasks.
        started_at: ISO-8601 start timestamp.
        finished_at: ISO-8601 finish timestamp.
        error_count: Number of tasks that failed with an exception.
    """

    suite_name: str
    tool_name: str
    provider: str
    scored_results: list[ScoredResult]
    aggregate: dict[str, float]
    started_at: str
    finished_at: str
    error_count: int = 0


class BenchmarkRunner:
    """Execute :class:`BenchmarkSuite` instances and store the results.

    Args:
        scorer: :class:`~.scoring.BenchmarkScorer` for grading runs.
            Defaults to a scorer with default weights.
        store: :class:`~.benchmark_store.BenchmarkStore` for persistence.
            Defaults to the module-level singleton.
        provider: Provider label attached to results (default ``"direct"``).
    """

    def __init__(
        self,
        scorer: BenchmarkScorer | None = None,
        store: BenchmarkStore | None = None,
        provider: str = "direct",
    ) -> None:
        self._scorer = scorer or BenchmarkScorer()
        self._store = store or get_benchmark_store()
        self._provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_suite(
        self,
        suite: BenchmarkSuite,
        registry: Any = None,
        persist: bool = True,
    ) -> SuiteRunReport:
        """Execute every task in *suite* and return a :class:`SuiteRunReport`.

        Args:
            suite: The suite to run.
            registry: Optional :class:`~missy.tools.registry.ToolRegistry`
                instance.  When ``None`` the module-level registry is used.
            persist: Whether to save results to the store.

        Returns:
            :class:`SuiteRunReport` with aggregate scores.
        """
        from datetime import UTC, datetime

        started_at = datetime.now(UTC).isoformat()
        scored_results: list[ScoredResult] = []
        error_count = 0

        if registry is None:
            try:
                from missy.tools.registry import get_tool_registry

                registry = get_tool_registry()
            except Exception:
                registry = None

        for task in suite.tasks:
            sr = self._run_task(task, registry)
            scored_results.append(sr)
            if not sr.result.success:
                error_count += 1

        finished_at = datetime.now(UTC).isoformat()
        aggregate = self._scorer.aggregate(scored_results)

        if persist and scored_results:
            self._store.save_batch(scored_results)

        _emit_audit(
            "tool.benchmark.suite_completed",
            {
                "suite": suite.name,
                "tool": suite.tool_name,
                "provider": self._provider,
                "tasks": len(suite.tasks),
                "errors": error_count,
                "composite": aggregate.get("composite", 0.0),
            },
        )
        logger.info(
            "BenchmarkRunner: suite=%s tool=%s provider=%s tasks=%d composite=%.3f",
            suite.name,
            suite.tool_name,
            self._provider,
            len(suite.tasks),
            aggregate.get("composite", 0.0),
        )
        return SuiteRunReport(
            suite_name=suite.name,
            tool_name=suite.tool_name,
            provider=self._provider,
            scored_results=scored_results,
            aggregate=aggregate,
            started_at=started_at,
            finished_at=finished_at,
            error_count=error_count,
        )

    def run_task(
        self,
        task: BenchmarkTask,
        registry: Any = None,
        persist: bool = True,
    ) -> ScoredResult:
        """Execute a single *task* and return the scored result.

        Args:
            task: Task to execute.
            registry: Optional tool registry.
            persist: Whether to save the result to the store.

        Returns:
            :class:`~.scoring.ScoredResult`.
        """
        if registry is None:
            try:
                from missy.tools.registry import get_tool_registry

                registry = get_tool_registry()
            except Exception:
                registry = None

        sr = self._run_task(task, registry)
        if persist:
            self._store.save(sr)
        return sr

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_task(self, task: BenchmarkTask, registry: Any) -> ScoredResult:
        t0 = time.monotonic()
        success = False
        actual_output: Any = None
        error = ""

        try:
            if registry is None:
                raise RuntimeError("No tool registry available")
            tool = registry.get(task.tool_name)
            if tool is None:
                raise KeyError(f"Tool {task.tool_name!r} not registered")
            result = tool.execute(**task.input_args)
            success = result.success
            actual_output = result.output
            if not result.success:
                error = result.error or "tool returned success=False"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            success = False

        latency_ms = (time.monotonic() - t0) * 1000.0

        raw = BenchmarkResult(
            task_id=task.id,
            tool_name=task.tool_name,
            provider=self._provider,
            success=success,
            latency_ms=latency_ms,
            cost_usd=0.0,
            actual_output=actual_output,
            expected_output=task.expected_output,
            tool_call_made=True,
            tool_call_args=task.input_args,
            schema_required_params=task.schema_required_params,
            safety_violation=False,
            error=error,
        )
        return self._scorer.score(raw)


def _emit_audit(event_type: str, detail: dict[str, Any]) -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result="allow",
                detail=detail,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("BenchmarkRunner: audit emit failed: %s", exc)
