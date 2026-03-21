"""MCP Tool Annotation support.

Provides :class:`ToolAnnotation` (a structured wrapper around the MCP spec's
``tool.annotations`` field) and :class:`AnnotationRegistry` (a thread-safe
store that maps tool names to their annotations).

MCP spec fields (from tool.annotations):
- ``readOnlyHint``   – tool does not modify external state
- ``destructiveHint`` – tool modifies or destroys external state
- ``idempotentHint`` – calling multiple times has the same effect as once
- ``openWorldHint``  – tool interacts with external entities (network, etc.)

Missy-specific extensions (optional, not in MCP spec):
- ``costHint``            – "none" | "low" | "medium" | "high"
- ``estimatedLatencyMs``  – approximate execution time in milliseconds
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, fields
from typing import Any

# ---------------------------------------------------------------------------
# Cost hint ordering used for ``max_cost`` filtering
# ---------------------------------------------------------------------------

_COST_ORDER: dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass
class ToolAnnotation:
    """Structured metadata describing an MCP (or built-in) tool's behaviour.

    Fields map directly to the MCP spec ``tool.annotations`` object where
    named equivalents exist; additional fields capture hints used by Missy's
    policy and approval layers.

    Attributes:
        read_only: Tool does not modify external state (maps to
            ``readOnlyHint``).  Defaults to ``True`` (safe assumption).
        mutating: Tool modifies or destroys external state (maps to
            ``destructiveHint``).
        idempotent: Calling the tool multiple times has the same net effect
            as calling it once (maps to ``idempotentHint``).
        estimated_latency_ms: Approximate wall-clock execution time in
            milliseconds, or ``None`` if unknown.
        cost_hint: Qualitative execution-cost band — one of
            ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
        requires_approval: Tool invocation should trigger a human-approval
            prompt before execution.
        network_access: Tool makes outbound network requests (maps to
            ``openWorldHint``).
        filesystem_access: Tool reads from or writes to the local filesystem.
        category: Broad behavioural category — ``"general"``, ``"search"``,
            ``"write"``, ``"admin"``, or ``"dangerous"``.
    """

    # Core behaviour hints
    read_only: bool = True
    mutating: bool = False
    idempotent: bool = False

    # Cost / performance hints
    estimated_latency_ms: int | None = None
    cost_hint: str = "none"

    # Security hints
    requires_approval: bool = False
    network_access: bool = False
    filesystem_access: bool = False

    # Categorisation
    category: str = "general"

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_mcp_dict(cls, data: dict[str, Any]) -> ToolAnnotation:
        """Parse a :class:`ToolAnnotation` from an MCP ``tool.annotations`` dict.

        Unknown keys are silently ignored so that future MCP spec additions do
        not break existing deployments.

        Args:
            data: The ``annotations`` mapping from an MCP tool definition.

        Returns:
            A populated :class:`ToolAnnotation` instance.

        Example::

            ann = ToolAnnotation.from_mcp_dict({
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": True,
            })
        """
        destructive = bool(data.get("destructiveHint", False))
        read_only = bool(data.get("readOnlyHint", True))
        idempotent = bool(data.get("idempotentHint", False))
        open_world = bool(data.get("openWorldHint", False))

        cost_hint = str(data.get("costHint", "none"))
        if cost_hint not in _COST_ORDER:
            cost_hint = "none"

        latency_raw = data.get("estimatedLatencyMs")
        estimated_latency_ms: int | None = None
        if isinstance(latency_raw, int) and latency_raw >= 0:
            estimated_latency_ms = latency_raw

        category = cls._infer_category(data)

        return cls(
            read_only=read_only,
            mutating=destructive,
            idempotent=idempotent,
            network_access=open_world,
            requires_approval=destructive,
            cost_hint=cost_hint,
            estimated_latency_ms=estimated_latency_ms,
            category=category,
        )

    @staticmethod
    def _infer_category(data: dict[str, Any]) -> str:
        """Derive a category string from raw MCP annotation data.

        Priority: ``destructiveHint`` > absence of ``readOnlyHint`` > default.

        Args:
            data: Raw ``annotations`` dict from the MCP tool manifest.

        Returns:
            One of ``"dangerous"``, ``"write"``, ``"search"``, or
            ``"general"``.
        """
        if data.get("destructiveHint"):
            return "dangerous"
        if not data.get("readOnlyHint", True):
            return "write"
        if data.get("openWorldHint"):
            return "search"
        return "general"

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the annotation to a plain dictionary.

        Suitable for JSON storage or display.

        Returns:
            A ``dict`` with one key per dataclass field.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_policy_hints(self) -> dict[str, bool]:
        """Convert annotation to policy-engine consumable hints.

        Returns:
            A dict with the following boolean keys:

            - ``requires_approval`` – ``True`` when approval is needed.
            - ``network_access`` – ``True`` when the tool uses the network.
            - ``filesystem_access`` – ``True`` when the tool touches the FS.
            - ``is_safe`` – ``True`` when read-only and non-mutating.
        """
        return {
            "requires_approval": self.requires_approval or self.mutating,
            "network_access": self.network_access,
            "filesystem_access": self.filesystem_access,
            "is_safe": self.read_only and not self.mutating,
        }


# ---------------------------------------------------------------------------
# Default annotations for Missy's built-in tools
# ---------------------------------------------------------------------------

BUILTIN_ANNOTATIONS: dict[str, ToolAnnotation] = {
    "shell_exec": ToolAnnotation(
        read_only=False,
        mutating=True,
        requires_approval=True,
        category="dangerous",
    ),
    "file_read": ToolAnnotation(
        read_only=True,
        filesystem_access=True,
        category="general",
    ),
    "file_write": ToolAnnotation(
        read_only=False,
        mutating=True,
        filesystem_access=True,
        category="write",
    ),
    "web_fetch": ToolAnnotation(
        read_only=True,
        network_access=True,
        category="search",
    ),
}


# ---------------------------------------------------------------------------
# AnnotationRegistry
# ---------------------------------------------------------------------------


class AnnotationRegistry:
    """Thread-safe store mapping tool names to :class:`ToolAnnotation` objects.

    Tools without an explicit annotation can still be queried via
    :meth:`get_or_default`, which returns a conservative default
    (``ToolAnnotation()`` — read-only, no special access, no approval).

    Example::

        registry = AnnotationRegistry()
        registry.register("my_tool", ToolAnnotation(mutating=True))
        ann = registry.get("my_tool")
        hints = ann.to_policy_hints()
    """

    def __init__(self) -> None:
        self._annotations: dict[str, ToolAnnotation] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, tool_name: str, annotation: ToolAnnotation) -> None:
        """Register *annotation* under *tool_name*, replacing any prior entry.

        Args:
            tool_name: Fully-qualified tool name (e.g. ``"server__tool"``).
            annotation: The :class:`ToolAnnotation` to store.
        """
        with self._lock:
            self._annotations[tool_name] = annotation

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> ToolAnnotation | None:
        """Return the annotation for *tool_name*, or ``None`` if absent.

        Args:
            tool_name: Tool name to look up.

        Returns:
            The stored :class:`ToolAnnotation`, or ``None``.
        """
        with self._lock:
            return self._annotations.get(tool_name)

    def get_or_default(self, tool_name: str) -> ToolAnnotation:
        """Return the annotation for *tool_name*, falling back to a safe default.

        The default annotation is ``ToolAnnotation()`` — read-only, no network
        or filesystem access, no approval required.

        Args:
            tool_name: Tool name to look up.

        Returns:
            A :class:`ToolAnnotation` (never ``None``).
        """
        with self._lock:
            return self._annotations.get(tool_name, ToolAnnotation())

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def filter_tools(
        self,
        tools: list[dict[str, Any]],
        *,
        read_only: bool | None = None,
        category: str | None = None,
        max_cost: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a subset of *tools* matching the given annotation criteria.

        Args:
            tools: List of tool definition dicts (each must have a ``"name"``
                key).
            read_only: If ``True``, keep only tools marked read-only; if
                ``False``, keep only mutating tools; ``None`` applies no filter.
            category: If provided, keep only tools whose annotation category
                matches this string exactly.
            max_cost: If provided, keep only tools whose ``cost_hint`` is at
                or below this level (``"none"`` < ``"low"`` < ``"medium"`` <
                ``"high"``).

        Returns:
            A new list containing only the matching tool dicts.
        """
        max_cost_level = _COST_ORDER.get(max_cost or "", _COST_ORDER["high"])
        result = []
        for tool in tools:
            ann = self.get_or_default(tool.get("name", ""))
            if read_only is not None and ann.read_only is not read_only:
                continue
            if category is not None and ann.category != category:
                continue
            if max_cost is not None:
                tool_cost_level = _COST_ORDER.get(ann.cost_hint, 0)
                if tool_cost_level > max_cost_level:
                    continue
            result.append(tool)
        return result

    def get_safe_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only tools that are read-only and non-mutating.

        Args:
            tools: List of tool definition dicts.

        Returns:
            Filtered list of safe tool dicts.
        """
        return [
            t
            for t in tools
            if self.get_or_default(t.get("name", "")).to_policy_hints()["is_safe"]
        ]

    def get_approval_required(self, tools: list[dict[str, Any]]) -> list[str]:
        """Return the names of tools that require human approval.

        Args:
            tools: List of tool definition dicts.

        Returns:
            A list of tool name strings whose annotations have
            ``requires_approval`` set (or whose ``mutating`` flag is ``True``).
        """
        return [
            t["name"]
            for t in tools
            if t.get("name")
            and self.get_or_default(t["name"]).to_policy_hints()["requires_approval"]
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_all_annotations(self) -> dict[str, ToolAnnotation]:
        """Return a shallow-copy snapshot of all registered annotations.

        Returns:
            A dict mapping tool name to :class:`ToolAnnotation`.
        """
        with self._lock:
            return dict(self._annotations)

    def summarize(self) -> str:
        """Return a human-readable summary of all registered annotations.

        Returns:
            A multi-line string with one entry per registered tool.
        """
        with self._lock:
            items = sorted(self._annotations.items())
        if not items:
            return "No tool annotations registered."
        lines = ["Tool Annotations:"]
        for name, ann in items:
            hints = ann.to_policy_hints()
            safe_label = "safe" if hints["is_safe"] else "mutating"
            approval_label = " [approval required]" if hints["requires_approval"] else ""
            lines.append(
                f"  {name}: {safe_label}, category={ann.category!r},"
                f" cost={ann.cost_hint!r}{approval_label}"
            )
        return "\n".join(lines)
