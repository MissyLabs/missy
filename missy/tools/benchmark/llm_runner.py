"""LLM-driven benchmark runner — measure real provider tool-calling behavior.

:class:`~.runner.BenchmarkRunner` calls a tool directly through the registry,
which measures the tool's own correctness/latency but says nothing about
whether a given *provider* can reliably choose that tool and fill in its
schema correctly. :class:`LLMBenchmarkRunner` closes that gap: it hands a
single tool schema to a real (or mock) provider via
:meth:`~missy.providers.base.BaseProvider.complete_with_tools`, given a
natural-language prompt, and scores what the provider actually did.

Safety
------

By default the tool the provider "calls" is **never executed** — only the
provider's tool-selection and argument-filling behavior is scored (via
``tool_call_quality`` and ``schema_score``). This matters because a
benchmark task is, by construction, adversarial-ish free text fed to an
LLM asking it to invoke a tool; blindly executing whatever arguments the
model produces would let a bad prompt (or a compromised benchmark task
definition) trigger real side effects — shell commands, file writes,
network requests — with no operator in the loop. Pass
``execute_tool=True`` to opt into real execution (still routed through
:class:`~missy.tools.registry.ToolRegistry`, so normal policy checks still
apply) when you specifically want to benchmark end-to-end correctness.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from missy.core.events import AuditEvent, event_bus
from missy.providers.base import BaseProvider, Message

from .benchmark_store import BenchmarkStore, get_benchmark_store
from .scoring import BenchmarkResult, BenchmarkScorer, ScoredResult

logger = logging.getLogger(__name__)

# Per-1k-token USD pricing lookup shared with CostTracker so benchmark cost
# estimates stay consistent with real session cost accounting.
try:
    from missy.agent.cost_tracker import _lookup_pricing
except Exception:  # pragma: no cover - defensive; cost_tracker has no heavy deps

    def _lookup_pricing(model: str) -> tuple[float, float]:
        return 0.0, 0.0


@dataclass
class LLMBenchmarkTask:
    """A single natural-language test case for provider tool-calling behavior.

    Attributes:
        id: Unique task identifier.
        tool_name: Name of the tool the provider is expected to call.
        prompt: Natural-language user message describing the request.
        expected_args: Expected argument values, used for correctness
            scoring when ``execute_tool=False``. ``None``/absent keys are
            not compared.
        expected_output: Expected ``ToolResult.output`` when the tool is
            actually executed (``execute_tool=True``).
        schema_required_params: Parameter names the provider must supply.
            Defaults to the keys of *expected_args*.
        tags: Arbitrary string labels for filtering.
    """

    id: str
    tool_name: str
    prompt: str
    expected_args: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    schema_required_params: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        tool_name: str,
        prompt: str,
        expected_args: dict[str, Any] | None = None,
        expected_output: Any = None,
        schema_required_params: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> LLMBenchmarkTask:
        expected_args = dict(expected_args or {})
        return cls(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            prompt=prompt,
            expected_args=expected_args,
            expected_output=expected_output,
            schema_required_params=list(schema_required_params or expected_args.keys()),
            tags=list(tags or []),
        )


class LLMBenchmarkRunner:
    """Drive a real provider through :meth:`BaseProvider.complete_with_tools`.

    Args:
        provider: The provider instance to benchmark. Its
            :attr:`~missy.providers.base.BaseProvider.name` is recorded as
            the result's provider label.
        scorer: :class:`~.scoring.BenchmarkScorer` for grading runs.
        store: :class:`~.benchmark_store.BenchmarkStore` for persistence.
        execute_tool: When ``True``, actually executes the tool the provider
            chose to call (through the tool registry, so policy still
            applies) and scores correctness against ``expected_output``.
            Defaults to ``False`` — see module docstring for why.
    """

    def __init__(
        self,
        provider: BaseProvider,
        scorer: BenchmarkScorer | None = None,
        store: BenchmarkStore | None = None,
        execute_tool: bool = False,
    ) -> None:
        self._provider = provider
        self._scorer = scorer or BenchmarkScorer()
        self._store = store or get_benchmark_store()
        self._execute_tool = execute_tool

    def run_task(
        self,
        task: LLMBenchmarkTask,
        tool: Any,
        registry: Any = None,
        persist: bool = True,
    ) -> ScoredResult:
        """Run a single *task* against ``tool`` and return the scored result.

        Args:
            task: Natural-language benchmark task.
            tool: The :class:`~missy.tools.base.BaseTool` instance whose
                schema is offered to the provider (only this one tool is
                exposed, so tool-selection ambiguity is not part of what
                this scores — see :meth:`run_suite` for multi-tool framing).
            registry: Tool registry used when ``execute_tool=True``. Falls
                back to the module-level registry.
            persist: Whether to save the result to the benchmark store.

        Returns:
            :class:`~.scoring.ScoredResult`.
        """
        t0 = time.monotonic()
        tool_call_made = False
        tool_call_args: dict[str, Any] = {}
        actual_output: Any = None
        error = ""
        safety_violation = False
        cost_usd = 0.0

        try:
            response = self._provider.complete_with_tools(
                [Message(role="user", content=task.prompt)],
                tools=[tool],
                system="",
            )
        except Exception as exc:  # noqa: BLE001
            response = None
            error = str(exc)

        if response is not None:
            for call in response.tool_calls:
                if call.name == task.tool_name and not tool_call_made:
                    tool_call_made = True
                    tool_call_args = dict(call.arguments)
                else:
                    # The model called a tool other than the one it was
                    # given — only one schema was offered, so this signals
                    # either hallucination or an attempt to reach a tool it
                    # was not authorized to see for this benchmark task.
                    safety_violation = True
            cost_usd = _estimate_cost(response)

        if not tool_call_made and not error:
            error = f"provider did not call {task.tool_name!r}"

        success = tool_call_made
        if tool_call_made and self._execute_tool:
            success, actual_output, exec_error = _execute(task.tool_name, tool_call_args, registry)
            error = exec_error or error
        elif tool_call_made and task.expected_args:
            # Score argument correctness directly (no execution) by comparing
            # the normalised argument mapping the scorer's string/token
            # matching already knows how to grade.
            actual_output = _normalise_args(tool_call_args)

        latency_ms = (time.monotonic() - t0) * 1000.0
        expected_for_scoring = (
            _normalise_args(task.expected_args)
            if (tool_call_made and not self._execute_tool and task.expected_args)
            else task.expected_output
        )

        raw = BenchmarkResult(
            task_id=task.id,
            tool_name=task.tool_name,
            provider=self._provider.name,
            success=success,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            actual_output=actual_output,
            expected_output=expected_for_scoring,
            tool_call_made=tool_call_made,
            tool_call_args=tool_call_args,
            schema_required_params=(task.schema_required_params or list(task.expected_args.keys())),
            safety_violation=safety_violation,
            error=error,
            metadata={"benchmark_kind": "llm", "executed": self._execute_tool},
        )
        scored = self._scorer.score(raw)
        if persist:
            self._store.save(scored)

        _emit_audit(
            "tool.benchmark.llm_run",
            {
                "tool": task.tool_name,
                "provider": self._provider.name,
                "tool_call_made": tool_call_made,
                "executed": self._execute_tool,
                "composite": scored.composite,
            },
        )
        return scored

    def run_suite(
        self,
        tasks: list[LLMBenchmarkTask],
        tool: Any,
        registry: Any = None,
        persist: bool = True,
    ) -> LLMSuiteRunReport:
        """Run every task in *tasks* against ``tool`` and aggregate scores."""
        from datetime import UTC, datetime

        started_at = datetime.now(UTC).isoformat()
        scored_results = [self.run_task(t, tool, registry=registry, persist=persist) for t in tasks]
        finished_at = datetime.now(UTC).isoformat()
        aggregate = self._scorer.aggregate(scored_results)
        error_count = sum(1 for r in scored_results if not r.result.success)

        logger.info(
            "LLMBenchmarkRunner: tool=%s provider=%s tasks=%d composite=%.3f",
            tool.name if hasattr(tool, "name") else "?",
            self._provider.name,
            len(tasks),
            aggregate.get("composite", 0.0),
        )
        return LLMSuiteRunReport(
            tool_name=tasks[0].tool_name if tasks else "",
            provider=self._provider.name,
            scored_results=scored_results,
            aggregate=aggregate,
            started_at=started_at,
            finished_at=finished_at,
            error_count=error_count,
        )


@dataclass
class LLMSuiteRunReport:
    """Aggregated outcome of running a list of :class:`LLMBenchmarkTask`.

    Attributes:
        tool_name: Tool that was benchmarked.
        provider: Provider identifier.
        scored_results: Individual :class:`~.scoring.ScoredResult` per task.
        aggregate: Mean scores across all tasks.
        started_at: ISO-8601 start timestamp.
        finished_at: ISO-8601 finish timestamp.
        error_count: Number of tasks where the provider failed to call the
            tool (or the tool execution itself failed, when ``execute_tool``).
    """

    tool_name: str
    provider: str
    scored_results: list[ScoredResult]
    aggregate: dict[str, float]
    started_at: str
    finished_at: str
    error_count: int = 0


def _normalise_args(args: dict[str, Any]) -> str:
    """Render an argument mapping as a stable string for correctness scoring."""
    return ", ".join(f"{k}={args[k]}" for k in sorted(args))


def _execute(tool_name: str, args: dict[str, Any], registry: Any) -> tuple[bool, Any, str]:
    """Execute *tool_name* with *args* through the registry. Never raises."""
    try:
        if registry is None:
            from missy.tools.registry import get_tool_registry

            registry = get_tool_registry()
        result = registry.execute(tool_name, **args)
        return result.success, result.output, (result.error or "")
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def _estimate_cost(response: Any) -> float:
    """Estimate USD cost from a :class:`CompletionResponse`'s usage and model."""
    usage = getattr(response, "usage", None) or {}
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    inp_rate, out_rate = _lookup_pricing(getattr(response, "model", "") or "")
    return (prompt_tokens / 1000.0) * inp_rate + (completion_tokens / 1000.0) * out_rate


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
        logger.debug("LLMBenchmarkRunner: audit emit failed: %s", exc)
