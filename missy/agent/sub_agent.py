"""Sub-agent delegation for parallel/sequential task decomposition.

Parses compound prompts into :class:`SubTask` instances and executes them
respecting dependency ordering and concurrency limits.

Example::

    from missy.agent.sub_agent import parse_subtasks, SubAgentRunner

    tasks = parse_subtasks("1. Search the web  2. Summarise results")
    runner = SubAgentRunner(runtime_factory=lambda: my_runtime)
    results = runner.run_all(tasks)
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_SUB_AGENTS = 10
MAX_CONCURRENT = 3


@dataclass
class SubTask:
    """A single step within a decomposed compound task.

    Attributes:
        id: Zero-based index within the parent task list.
        description: Human-readable description of the step.
        tool_hints: Names of tools expected to be useful for this step.
        depends_on: IDs of steps that must complete before this one begins.
        result: Output string set after the step completes successfully.
        error: Error message set when the step fails.
    """

    id: int
    description: str
    tool_hints: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None


def parse_subtasks(prompt: str) -> list[SubTask]:
    """Extract subtasks from a compound prompt.

    Attempts to parse numbered lists first, then sequential connectives
    (``"then"``, ``"and then"``, ``"after that"``, ``"finally"``).  Falls
    back to a single subtask wrapping the whole prompt.

    Args:
        prompt: The raw user prompt string.

    Returns:
        A list of :class:`SubTask` instances.  Sequentially dependent steps
        (from the connective pattern) have ``depends_on`` set appropriately.
    """
    tasks: list[SubTask] = []

    # Try numbered list pattern first
    numbered = re.findall(r'^\s*(\d+)[\.\)]\s+(.+)', prompt, re.M)
    if numbered:
        for i, (_, desc) in enumerate(numbered):
            tasks.append(SubTask(id=i, description=desc.strip()))
        return tasks

    # Try sequential connective pattern
    parts = re.split(r'\b(then|and then|after that|finally)\b', prompt, flags=re.I)
    if len(parts) > 2:
        descs = [
            p.strip()
            for p in parts
            if p.strip() and not re.match(
                r'^(then|and then|after that|finally)$', p.strip(), re.I
            )
        ]
        for i, desc in enumerate(descs):
            tasks.append(
                SubTask(
                    id=i,
                    description=desc,
                    depends_on=[i - 1] if i > 0 else [],
                )
            )
        return tasks

    return [SubTask(id=0, description=prompt)]


class SubAgentRunner:
    """Runs :class:`SubTask` instances using a shared :class:`AgentRuntime`.

    Each subtask is run inside a semaphore that caps concurrent in-flight
    calls to :data:`MAX_CONCURRENT`.  Sequential dependencies are respected
    in :meth:`run_all` by processing tasks in order.

    Args:
        runtime_factory: A callable that returns a fresh
            :class:`~missy.agent.runtime.AgentRuntime` instance per call.
            Using a factory (rather than sharing one runtime) avoids session
            state collisions between concurrent subtasks.
    """

    def __init__(self, runtime_factory: Callable) -> None:
        self._factory = runtime_factory
        self._semaphore = threading.Semaphore(MAX_CONCURRENT)

    def run_subtask(self, subtask: SubTask, context: str = "") -> str:
        """Execute a single subtask, optionally prepending *context*.

        Args:
            subtask: The step to execute.
            context: Optional accumulated context from prior steps.

        Returns:
            The string result from the runtime, or an error message string
            when the runtime raises.
        """
        prompt = subtask.description
        if context:
            prompt = f"Context: {context}\n\nTask: {prompt}"

        with self._semaphore:
            try:
                runtime = self._factory()
                result = runtime.run(prompt)
                subtask.result = result
                return result
            except Exception as exc:
                subtask.error = str(exc)
                return f"[Error in subtask {subtask.id}: {exc}]"

    def run_all(
        self,
        subtasks: list[SubTask],
        max_total: int = MAX_SUB_AGENTS,
    ) -> list[str]:
        """Run all subtasks respecting dependencies and concurrency limits.

        Tasks are processed in list order.  Before each task executes, the
        results of any declared dependencies are prepended as context.  The
        total number of tasks is capped at *max_total*.

        Args:
            subtasks: Ordered list of steps to execute.
            max_total: Hard cap on total subtask executions.

        Returns:
            A list of result strings in the same order as *subtasks*.
        """
        if len(subtasks) > max_total:
            subtasks = subtasks[:max_total]

        results: list[str] = []
        context_accumulator = ""

        for task in subtasks:
            # Accumulate context from completed dependencies
            for dep_id in task.depends_on:
                dep = next((t for t in subtasks if t.id == dep_id), None)
                if dep and dep.result:
                    context_accumulator += f"\nResult of step {dep_id}: {dep.result[:200]}"

            result = self.run_subtask(task, context=context_accumulator)
            results.append(result)

        return results
