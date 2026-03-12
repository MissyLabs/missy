"""Per-tool failure tracking with strategy-rotation injection.

After FAILURE_THRESHOLD consecutive failures of the same tool,
the FailureTracker injects a strategy-rotation prompt asking the
agent to analyse the failure, list three alternatives, and execute
the best one.

Example::

    from missy.agent.failure_tracker import FailureTracker

    tracker = FailureTracker(threshold=3)
    for _ in range(3):
        should_inject = tracker.record_failure("shell_exec", "permission denied")

    if should_inject:
        prompt = tracker.get_strategy_prompt("shell_exec", "permission denied")
        # Inject prompt into the agent's loop_messages as a user message.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _ToolState:
    """Mutable counters for a single tool."""

    consecutive: int = 0
    total: int = 0


class FailureTracker:
    """Track consecutive failures per tool and emit strategy-rotation prompts.

    A fresh :class:`FailureTracker` should be created at the start of each
    top-level task so that failure counters do not bleed between unrelated
    user requests.

    Args:
        threshold: Number of consecutive failures before
            :meth:`record_failure` returns ``True`` and
            :meth:`should_inject_strategy` returns ``True``.
            Defaults to ``3``.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold!r}")
        self.threshold = threshold
        self._state: dict[str, _ToolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_failure(self, tool_name: str, error: str) -> bool:
        """Record a single consecutive failure for *tool_name*.

        Args:
            tool_name: Name of the tool that failed.
            error: The error message or content returned by the tool.

        Returns:
            ``True`` when the consecutive failure count has just reached
            (or is already at or above) the configured *threshold*, signalling
            that the caller should inject a strategy-rotation prompt.
            ``False`` otherwise.
        """
        state = self._get_or_create(tool_name)
        state.consecutive += 1
        state.total += 1
        return state.consecutive >= self.threshold

    def record_success(self, tool_name: str) -> None:
        """Reset the consecutive failure counter for *tool_name*.

        The total-failures counter is preserved so that statistics remain
        accurate across the lifetime of the tracker.

        Args:
            tool_name: Name of the tool that succeeded.
        """
        state = self._get_or_create(tool_name)
        state.consecutive = 0

    def should_inject_strategy(self, tool_name: str) -> bool:
        """Return ``True`` if the consecutive failure count meets the threshold.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            ``True`` when ``consecutive_failures >= threshold``.
        """
        state = self._state.get(tool_name)
        if state is None:
            return False
        return state.consecutive >= self.threshold

    def get_strategy_prompt(self, tool_name: str, last_error: str) -> str:
        """Build the strategy-rotation user prompt for *tool_name*.

        Args:
            tool_name: The tool whose repeated failure triggered rotation.
            last_error: The most recent error message from that tool.

        Returns:
            A plain-text prompt string suitable for injection as a
            ``{"role": "user", "content": ...}`` message.
        """
        count = self._state[tool_name].consecutive if tool_name in self._state else self.threshold
        return (
            f"The tool '{tool_name}' has failed {count} times consecutively."
            f" Last error: {last_error}\n\n"
            f"Before continuing, please:\n"
            f"1. Analyse why this tool keeps failing\n"
            f"2. List 3 alternative approaches that do not use '{tool_name}'\n"
            f"3. Execute the best alternative\n\n"
            f"Do not attempt '{tool_name}' again until you have tried an alternative."
        )

    def reset(self, tool_name: str) -> None:
        """Reset both counters for a specific tool.

        Args:
            tool_name: Name of the tool whose state should be cleared.
        """
        self._state.pop(tool_name, None)

    def reset_all(self) -> None:
        """Reset all counters.

        Call at the start of each top-level task to ensure a clean slate.
        """
        self._state.clear()

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return per-tool failure statistics.

        Returns:
            A dict of the form::

                {
                    "tool_name": {
                        "failures": <consecutive_failures: int>,
                        "total_failures": <total_failures: int>,
                    },
                    ...
                }

            Only tools that have been seen (failure or success recorded) are
            included.
        """
        return {
            name: {"failures": s.consecutive, "total_failures": s.total}
            for name, s in self._state.items()
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, tool_name: str) -> _ToolState:
        if tool_name not in self._state:
            self._state[tool_name] = _ToolState()
        return self._state[tool_name]
