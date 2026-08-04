"""Cross-provider LLM benchmark tool, agent-callable.

A Discord user asked Missy to "benchmark a tool on openai vs ollama" and the
agent, having no real tool for this, improvised a shell script that checked
``which openai``/``which ollama`` and the ``OPENAI_API_KEY`` environment
variable -- both entirely disconnected from how Missy actually authenticates
its own providers (``~/.missy/config.yaml``'s ``providers:`` section, backed
by vault references / OAuth token files / config ``api_key``, resolved
through :class:`~missy.providers.registry.ProviderRegistry`, not the shell
environment or a CLI binary). The benchmark predictably "failed" for a
provider that was never actually broken.

:class:`ProviderBenchmarkTool` gives the agent a real way to do this: it
looks up each requested provider directly in the process-level
:class:`~missy.providers.registry.ProviderRegistry` (the same registry real
chat dispatch uses) and either times a plain
:meth:`~missy.providers.base.BaseProvider.complete` call, or -- when
``tool_name`` is given -- reuses
:class:`~missy.tools.benchmark.llm_runner.LLMBenchmarkRunner` to score the
provider's real tool-selection/argument-filling behavior against one
registered tool's schema (the same machinery ``missy tools benchmark
run-llm`` already uses from the CLI, just made agent-callable and able to
compare several providers in one call instead of one at a time).

No shell commands, environment variable inspection, or CLI binaries are
involved anywhere in this tool.
"""

from __future__ import annotations

import time
from typing import Any

from missy.providers.base import Message
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

#: Truncate a benchmarked provider's raw reply before it re-enters the
#: conversation as tool output -- a benchmark prompt is, by construction,
#: free text handed to N different providers, and nothing here needs (or
#: should echo back) an unbounded reply.
_MAX_RESPONSE_CHARS = 2000

#: Hard cap on how many providers a single call fans out to. Unlike
#: vision_analyze's single-provider-with-fallback call, this tool
#: deliberately dispatches to every requested provider -- capping it bounds
#: the worst case (an operator/agent asking to benchmark an unbounded
#: `providers` list) to a small, predictable number of real provider calls.
_MAX_PROVIDERS_PER_CALL = 8


class ProviderBenchmarkTool(BaseTool):
    """Benchmark the same prompt (or tool-call task) across real providers."""

    name = "provider_benchmark"
    description = (
        "Benchmark the same prompt across one or more configured AI providers "
        "(e.g. anthropic, openai-codex, ollama) using Missy's own provider "
        "registry -- the real configured credentials (vault/OAuth/API key), "
        "never shell CLIs or environment variables. Pass tool_name to "
        "benchmark a specific tool's tool-calling behavior instead of a "
        "plain reply. ALWAYS use this tool when asked to benchmark, compare, "
        "or race providers against each other -- never shell out to "
        "`openai`/`ollama` CLI binaries or check OPENAI_API_KEY/etc. "
        "yourself; those are unrelated to how Missy actually authenticates "
        "its providers and will report a false failure."
    )
    permissions = ToolPermissions()

    def execute(self, **kwargs: Any) -> ToolResult:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(success=False, output=None, error="'prompt' is required.")

        tool_name = kwargs.get("tool_name")
        tool_name = str(tool_name).strip() if tool_name else None
        persist = bool(kwargs.get("persist", True))

        from missy.providers.registry import get_registry

        try:
            registry = get_registry()
        except RuntimeError as exc:
            return ToolResult(
                success=False, output=None, error=f"Provider registry unavailable: {exc}"
            )

        requested = kwargs.get("providers")
        if requested:
            names = [str(n).strip() for n in requested if str(n).strip()]
        else:
            names = list(registry.list_providers())
        if not names:
            return ToolResult(success=False, output=None, error="No providers configured.")

        truncated = len(names) > _MAX_PROVIDERS_PER_CALL
        names = names[:_MAX_PROVIDERS_PER_CALL]

        tool = None
        if tool_name:
            from missy.tools.registry import get_tool_registry

            try:
                tool_registry = get_tool_registry()
            except RuntimeError as exc:
                return ToolResult(
                    success=False, output=None, error=f"Tool registry unavailable: {exc}"
                )
            tool = tool_registry.get(tool_name)
            if tool is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        f"Tool {tool_name!r} is not registered; cannot benchmark its "
                        "tool-calling behavior."
                    ),
                )

        results: dict[str, dict[str, Any]] = {
            name: self._benchmark_one(registry, name, prompt, tool, tool_name, persist)
            for name in names
        }

        output: dict[str, Any] = {
            "prompt": prompt,
            "tool_name": tool_name,
            "providers": names,
            "results": results,
        }
        if truncated:
            output["note"] = f"Only the first {_MAX_PROVIDERS_PER_CALL} providers were benchmarked."
        return ToolResult(success=True, output=output)

    def _benchmark_one(
        self,
        registry: Any,
        name: str,
        prompt: str,
        tool: Any,
        tool_name: str | None,
        persist: bool,
    ) -> dict[str, Any]:
        provider = registry.get(name)
        if provider is None:
            return {"ok": False, "error": f"provider {name!r} is not configured"}

        try:
            available = bool(provider.is_available())
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"availability check failed: {exc}"}
        if not available:
            return {
                "ok": False,
                "error": (
                    "not available -- check its real configured credentials via "
                    "`missy providers list`/`missy doctor`, not the shell environment"
                ),
            }

        if tool is not None:
            return self._benchmark_tool_call(provider, tool, tool_name or "", prompt, persist)
        return self._benchmark_plain_completion(provider, prompt)

    def _benchmark_plain_completion(self, provider: Any, prompt: str) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            response = provider.complete([Message(role="user", content=prompt)])
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "ms": round((time.monotonic() - t0) * 1000, 2),
                "error": str(exc),
            }
        ms = round((time.monotonic() - t0) * 1000, 2)
        content = str(getattr(response, "content", "") or "")
        return {
            "ok": True,
            "ms": ms,
            "model": str(getattr(response, "model", "") or ""),
            "response": content[:_MAX_RESPONSE_CHARS],
        }

    def _benchmark_tool_call(
        self, provider: Any, tool: Any, tool_name: str, prompt: str, persist: bool
    ) -> dict[str, Any]:
        from missy.tools.benchmark import LLMBenchmarkRunner, LLMBenchmarkTask

        task = LLMBenchmarkTask.create(tool_name=tool_name, prompt=prompt)
        runner = LLMBenchmarkRunner(provider=provider)
        try:
            scored = runner.run_task(task, tool, persist=persist)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {
            "ok": bool(scored.result.success),
            "ms": round(scored.result.latency_ms, 2),
            "tool_call_made": bool(scored.result.tool_call_made),
            "tool_call_args": dict(scored.result.tool_call_args),
            "composite": round(scored.composite, 3),
            "error": scored.result.error or None,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/message to send to each provider.",
                        "example": "ping",
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Registry keys of the providers to benchmark, e.g. "
                            "['openai-codex', 'ollama']. Omit to benchmark every "
                            "currently configured provider."
                        ),
                    },
                    "tool_name": {
                        "type": "string",
                        "description": (
                            "Optional: benchmark this registered tool's tool-calling "
                            "behavior (does the provider choose it and fill its "
                            "schema correctly) instead of a plain text reply."
                        ),
                    },
                    "persist": {
                        "type": "boolean",
                        "description": (
                            "Save tool-call benchmark results for later review via "
                            "`missy tools benchmark results` (only applies when "
                            "tool_name is given). Defaults to true."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        }
