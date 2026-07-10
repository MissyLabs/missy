"""Sub-agent delegation for parallel/sequential task decomposition.

Parses compound prompts into :class:`SubTask` instances and executes them
respecting dependency ordering and concurrency limits.

Wiring status (SR-4.2): :class:`SubAgentRunner` is wired into production
via the ``delegate_task`` tool (``missy/tools/builtin/delegate_task.py``),
dispatched through :meth:`~missy.agent.runtime.AgentRuntime._execute_tool`.
Each sub-agent call reuses the *same* :class:`~missy.agent.runtime.AgentRuntime`
instance and ``session_id`` as its parent -- not a fresh, independent
runtime -- so it goes through identical policy/capability_mode
enforcement and its spend is tracked by the exact same per-session
``CostTracker`` the parent's budget cap already checks (no separate
cross-child budget-aggregation logic is needed; it falls out of reusing
the parent's own session-scoped accounting). Recursion is bounded by
:data:`MAX_SUB_AGENT_DEPTH`, threaded down from
``AgentRuntime.run(_delegation_depth=...)``. Independent tasks (no
shared dependency) genuinely execute concurrently via
:class:`concurrent.futures.ThreadPoolExecutor`, capped at
:data:`MAX_CONCURRENT` in-flight calls at a time.

Example::

    from missy.agent.sub_agent import parse_subtasks, SubAgentRunner

    tasks = parse_subtasks("1. Search the web  2. Summarise results")
    runner = SubAgentRunner(runtime=my_runtime, session_id="sess-1", depth=1)
    results = runner.run_all(tasks)
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

MAX_SUB_AGENTS = 10
MAX_CONCURRENT = 3

#: Hard cap on delegate_task nesting. depth=0 is a genuine top-level
#: call; a delegate_task tool call made *from within* a sub-agent's own
#: run increments depth by 1. Once depth reaches this value, further
#: delegation is refused -- unbounded recursive delegation is a real
#: resource-exhaustion vector (each level can fan out to MAX_SUB_AGENTS
#: more calls), so this must be enforced regardless of how deep a
#: determined/compromised prompt tries to nest.
MAX_SUB_AGENT_DEPTH = 2


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
    result: str | None = None
    error: str | None = None


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
    numbered = re.findall(r"^\s*(\d+)[\.\)]\s+(.+)", prompt, re.M)
    if numbered:
        for i, (_, desc) in enumerate(numbered):
            tasks.append(SubTask(id=i, description=desc.strip()))
        return tasks

    # Try sequential connective pattern
    parts = re.split(r"\b(then|and then|after that|finally)\b", prompt, flags=re.I)
    if len(parts) > 2:
        descs = [
            p.strip()
            for p in parts
            if p.strip() and not re.match(r"^(then|and then|after that|finally)$", p.strip(), re.I)
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
    """Runs :class:`SubTask` instances against a shared :class:`AgentRuntime`.

    Args:
        runtime: The live :class:`~missy.agent.runtime.AgentRuntime` to run
            sub-tasks through. Deliberately the *same* instance the parent
            call is using (not a fresh one) so every sub-task goes through
            identical policy/capability_mode enforcement and shares the
            parent's per-session cost tracking -- a sub-agent cannot spend
            outside the budget the parent call is already bound by.
        session_id: The session ID to run sub-tasks under. Passing the
            parent's own session_id is what makes budget aggregation work
            (see :meth:`~missy.agent.runtime.AgentRuntime._get_cost_tracker`).
        depth: Current delegation depth (0 = top-level). Forwarded to each
            sub-task's ``runtime.run(_delegation_depth=depth)`` call so a
            sub-task that itself calls ``delegate_task`` is one level
            deeper, eventually hitting :data:`MAX_SUB_AGENT_DEPTH`.
    """

    def __init__(self, runtime, session_id: str, depth: int) -> None:
        self._runtime = runtime
        self._session_id = session_id
        self._depth = depth
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

        # Defense in depth: run_all() already caps concurrency via its own
        # ThreadPoolExecutor(max_workers=MAX_CONCURRENT), but a caller that
        # invokes run_subtask() directly from several threads (bypassing
        # run_all() entirely) must still be bounded -- this semaphore is
        # what enforces MAX_CONCURRENT for that path too.
        with self._semaphore:
            try:
                result = self._runtime.run(
                    prompt, session_id=self._session_id, _delegation_depth=self._depth
                )
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

        Tasks whose dependencies are all already complete are genuinely
        executed in parallel (up to :data:`MAX_CONCURRENT` at a time) via a
        :class:`~concurrent.futures.ThreadPoolExecutor`; a task with an
        unmet dependency waits for the next wave. The total number of
        tasks is capped at *max_total*.

        Args:
            subtasks: Ordered list of steps to execute.
            max_total: Hard cap on total subtask executions.

        Returns:
            A list of result strings in the same order as *subtasks*.
        """
        if len(subtasks) > max_total:
            subtasks = subtasks[:max_total]

        by_id = {t.id: t for t in subtasks}
        result_strings: dict[int, str] = {}
        done: set[int] = set()
        remaining = list(subtasks)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
            while remaining:
                ready = [t for t in remaining if all(d in done for d in t.depends_on)]
                if not ready:
                    # Circular/unsatisfiable dependency -- run everything
                    # left sequentially rather than deadlocking forever.
                    ready = remaining

                def _run_one(task: SubTask) -> SubTask:
                    context = "\n".join(
                        f"Result of step {dep_id}: {(by_id[dep_id].result or '')[:200]}"
                        for dep_id in task.depends_on
                        if dep_id in by_id and by_id[dep_id].result
                    )
                    # Capture run_subtask()'s actual return value (including
                    # the "[Error in subtask N: ...]" wrapper on failure) --
                    # reconstructing this from task.result/.error afterwards
                    # would silently drop the wrapper text for failed tasks.
                    result_strings[task.id] = self.run_subtask(task, context=context)
                    return task

                for finished in pool.map(_run_one, ready):
                    done.add(finished.id)
                remaining = [t for t in remaining if t.id not in done]

        return [result_strings.get(t.id, "") for t in subtasks]
