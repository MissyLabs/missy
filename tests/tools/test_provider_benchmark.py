"""Tests for the agent-callable ProviderBenchmarkTool.

Regression context: a Discord user asked Missy to benchmark a tool across
providers, and the agent -- lacking any real tool for this -- shelled out
to `which openai`/`which ollama` and checked `$OPENAI_API_KEY`, none of
which has anything to do with how Missy actually authenticates its own
providers. These tests assert the tool talks only to the real, in-process
ProviderRegistry/ToolRegistry -- never a shell or environment variable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.providers.base import BaseProvider, CompletionResponse, ToolCall
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin.provider_benchmark import ProviderBenchmarkTool


class _StubProvider(BaseProvider):
    """Provider double with per-instance availability/response/failure."""

    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        reply: str = "pong",
        model: str = "stub-model",
        raises: Exception | None = None,
        tool_calls: list[ToolCall] | None = None,
    ):
        self.name = name
        self._available = available
        self._reply = reply
        self._model = model
        self._raises = raises
        self._tool_calls = tool_calls or []

    def is_available(self) -> bool:
        return self._available

    def complete(self, messages, **kwargs) -> CompletionResponse:
        if self._raises:
            raise self._raises
        return CompletionResponse(
            content=self._reply,
            model=self._model,
            provider=self.name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
        )

    def complete_with_tools(self, messages, tools, system: str = "") -> CompletionResponse:
        if self._raises:
            raise self._raises
        return CompletionResponse(
            content="",
            model=self._model,
            provider=self.name,
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            raw={},
            tool_calls=self._tool_calls,
            finish_reason="tool_calls" if self._tool_calls else "stop",
        )


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo the given text."
    permissions = ToolPermissions()

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("text"))

    def get_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }


def _make_registry(*providers: BaseProvider) -> MagicMock:
    by_name = {p.name: p for p in providers}
    registry = MagicMock()
    registry.get.side_effect = lambda name: by_name.get(name)
    registry.list_providers.return_value = sorted(by_name)
    return registry


class TestPromptValidation:
    def test_missing_prompt_fails(self):
        tool = ProviderBenchmarkTool()
        result = tool.execute()
        assert result.success is False
        assert "prompt" in result.error.lower()

    def test_blank_prompt_fails(self):
        tool = ProviderBenchmarkTool()
        result = tool.execute(prompt="   ")
        assert result.success is False


class TestRegistryUnavailable:
    def test_registry_not_initialised_fails_cleanly(self):
        tool = ProviderBenchmarkTool()
        with patch(
            "missy.providers.registry.get_registry",
            side_effect=RuntimeError("not initialised"),
        ):
            result = tool.execute(prompt="ping")
        assert result.success is False
        assert "registry" in result.error.lower()


class TestPlainCompletionBenchmark:
    def test_default_providers_uses_every_registered_provider(self):
        registry = _make_registry(
            _StubProvider("ollama", reply="pong"),
            _StubProvider("openai-codex", reply="hi there"),
        )
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping")

        assert result.success is True
        assert set(result.output["results"]) == {"ollama", "openai-codex"}
        for name in ("ollama", "openai-codex"):
            entry = result.output["results"][name]
            assert entry["ok"] is True
            assert entry["ms"] >= 0
            assert entry["model"] == "stub-model"

    def test_explicit_providers_list_restricts_selection(self):
        registry = _make_registry(
            _StubProvider("ollama"),
            _StubProvider("openai-codex"),
        )
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping", providers=["ollama"])

        assert set(result.output["results"]) == {"ollama"}

    def test_unconfigured_provider_reports_ok_false_not_exception(self):
        registry = _make_registry(_StubProvider("ollama"))
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping", providers=["nonexistent"])

        assert result.success is True  # the overall tool call succeeded
        entry = result.output["results"]["nonexistent"]
        assert entry["ok"] is False
        assert "not configured" in entry["error"]

    def test_unavailable_provider_reports_ok_false(self):
        registry = _make_registry(_StubProvider("openai-codex", available=False))
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping", providers=["openai-codex"])

        entry = result.output["results"]["openai-codex"]
        assert entry["ok"] is False
        assert "not available" in entry["error"]

    def test_provider_exception_is_captured_not_raised(self):
        from missy.core.exceptions import ProviderError

        registry = _make_registry(
            _StubProvider("ollama", raises=ProviderError("upstream exploded"))
        )
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping", providers=["ollama"])

        assert result.success is True
        entry = result.output["results"]["ollama"]
        assert entry["ok"] is False
        assert "upstream exploded" in entry["error"]
        assert "ms" in entry

    def test_response_is_truncated(self):
        registry = _make_registry(_StubProvider("ollama", reply="x" * 5000))
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping", providers=["ollama"])

        assert len(result.output["results"]["ollama"]["response"]) <= 2000

    def test_no_providers_configured_fails(self):
        registry = _make_registry()
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping")

        assert result.success is False
        assert "no providers" in result.error.lower()

    def test_provider_count_is_capped(self):
        providers = [_StubProvider(f"p{i}") for i in range(12)]
        registry = _make_registry(*providers)
        tool = ProviderBenchmarkTool()
        with patch("missy.providers.registry.get_registry", return_value=registry):
            result = tool.execute(prompt="ping")

        assert len(result.output["results"]) == 8
        assert "note" in result.output


class TestToolCallBenchmark:
    def test_missing_tool_fails_whole_call(self):
        registry = _make_registry(_StubProvider("ollama"))
        tool_registry = MagicMock()
        tool_registry.get.return_value = None
        tool = ProviderBenchmarkTool()
        with (
            patch("missy.providers.registry.get_registry", return_value=registry),
            patch("missy.tools.registry.get_tool_registry", return_value=tool_registry),
        ):
            result = tool.execute(prompt="say hi", tool_name="echo")

        assert result.success is False
        assert "echo" in result.error

    def test_tool_call_benchmark_reports_composite_and_args(self):
        call = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        registry = _make_registry(_StubProvider("openai-codex", tool_calls=[call]))
        tool_registry = MagicMock()
        tool_registry.get.return_value = _EchoTool()
        tool = ProviderBenchmarkTool()
        with (
            patch("missy.providers.registry.get_registry", return_value=registry),
            patch("missy.tools.registry.get_tool_registry", return_value=tool_registry),
            patch("missy.tools.benchmark.llm_runner.get_benchmark_store") as mock_store,
        ):
            mock_store.return_value.save = MagicMock()
            result = tool.execute(
                prompt="say hi", tool_name="echo", providers=["openai-codex"], persist=False
            )

        assert result.success is True
        entry = result.output["results"]["openai-codex"]
        assert entry["ok"] is True
        assert entry["tool_call_made"] is True
        assert entry["tool_call_args"] == {"text": "hi"}
        assert 0.0 <= entry["composite"] <= 1.0

    def test_tool_call_benchmark_no_call_made_reports_error(self):
        registry = _make_registry(_StubProvider("ollama", tool_calls=[]))
        tool_registry = MagicMock()
        tool_registry.get.return_value = _EchoTool()
        tool = ProviderBenchmarkTool()
        with (
            patch("missy.providers.registry.get_registry", return_value=registry),
            patch("missy.tools.registry.get_tool_registry", return_value=tool_registry),
            patch("missy.tools.benchmark.llm_runner.get_benchmark_store") as mock_store,
        ):
            mock_store.return_value.save = MagicMock()
            result = tool.execute(
                prompt="say hi", tool_name="echo", providers=["ollama"], persist=False
            )

        entry = result.output["results"]["ollama"]
        assert entry["ok"] is False
        assert "did not call" in entry["error"]


class TestSchema:
    def test_schema_requires_prompt(self):
        schema = ProviderBenchmarkTool().get_schema()
        assert schema["parameters"]["required"] == ["prompt"]
        assert "providers" in schema["parameters"]["properties"]
        assert "tool_name" in schema["parameters"]["properties"]


class TestNoShellOrEnvUsage:
    def test_execute_never_touches_subprocess_or_environ(self):
        """Regression guard: the tool must resolve providers purely through
        the registry, never by shelling out or reading env vars."""
        registry = _make_registry(_StubProvider("ollama"))
        tool = ProviderBenchmarkTool()
        with (
            patch("missy.providers.registry.get_registry", return_value=registry),
            patch("subprocess.run", side_effect=AssertionError("must not shell out")),
            patch("subprocess.Popen", side_effect=AssertionError("must not shell out")),
            patch("os.system", side_effect=AssertionError("must not shell out")),
        ):
            result = tool.execute(prompt="ping")

        assert result.success is True
