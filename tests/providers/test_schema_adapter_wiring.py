"""Tests for schema_adapter wiring in provider get_tool_schema() methods."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_tool(name: str = "test_tool", description: str = "A test tool") -> MagicMock:
    """Create a minimal BaseTool-like mock."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.get_schema.return_value = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
            },
            "required": ["query"],
        },
    }
    return tool


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


def test_anthropic_get_tool_schema_uses_adapter() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="claude-sonnet-4-6"))
    tool = _make_tool("search", "Search the web")
    schemas = provider.get_tool_schema([tool])

    assert len(schemas) == 1
    s = schemas[0]
    # Anthropic format: name + description + input_schema
    assert s["name"] == "search"
    assert s["description"] == "Search the web"
    assert "input_schema" in s
    assert "parameters" not in s
    # input_schema must have properties
    assert "properties" in s["input_schema"]
    assert "query" in s["input_schema"]["properties"]


def test_anthropic_get_tool_schema_required_passthrough() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="claude-sonnet-4-6"))
    tool = _make_tool()
    schemas = provider.get_tool_schema([tool])
    assert schemas[0]["input_schema"].get("required") == ["query"]


def test_anthropic_get_tool_schema_multiple_tools() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="claude-sonnet-4-6"))
    tools = [_make_tool("t1"), _make_tool("t2"), _make_tool("t3")]
    schemas = provider.get_tool_schema(tools)
    assert len(schemas) == 3
    names = [s["name"] for s in schemas]
    assert names == ["t1", "t2", "t3"]


def test_anthropic_get_tool_schema_no_get_schema_method() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="claude-sonnet-4-6"))
    tool = MagicMock(spec=[])  # no get_schema attribute
    tool.name = "minimal"
    tool.description = "Minimal tool"
    schemas = provider.get_tool_schema([tool])
    assert schemas[0]["name"] == "minimal"
    assert "input_schema" in schemas[0]


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


def test_openai_get_tool_schema_uses_adapter() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig(name="openai", model="gpt-4o"))
    tool = _make_tool("calculator", "Evaluate math expressions")
    schemas = provider.get_tool_schema([tool])

    assert len(schemas) == 1
    s = schemas[0]
    # OpenAI format: {type: function, function: {...}}
    assert s["type"] == "function"
    assert "function" in s
    fn = s["function"]
    assert fn["name"] == "calculator"
    assert fn["description"] == "Evaluate math expressions"
    assert "parameters" in fn
    assert "query" in fn["parameters"]["properties"]


def test_openai_get_tool_schema_required_passthrough() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig(name="openai", model="gpt-4o"))
    tool = _make_tool()
    schemas = provider.get_tool_schema([tool])
    assert schemas[0]["function"]["parameters"].get("required") == ["query"]


def test_openai_get_tool_schema_multiple_tools() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig(name="openai", model="gpt-4o"))
    tools = [_make_tool("t1"), _make_tool("t2")]
    schemas = provider.get_tool_schema(tools)
    assert len(schemas) == 2
    assert all(s["type"] == "function" for s in schemas)


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


def test_ollama_get_tool_schema_uses_adapter() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(ProviderConfig(name="ollama", model="llama3"))
    tool = _make_tool("file_read", "Read a file")
    schemas = provider.get_tool_schema([tool])

    assert len(schemas) == 1
    s = schemas[0]
    # Ollama uses OpenAI-compatible format
    assert s["type"] == "function"
    assert "function" in s
    fn = s["function"]
    assert fn["name"] == "file_read"


def test_ollama_get_tool_schema_no_get_schema_fallback() -> None:
    from missy.config.settings import ProviderConfig
    from missy.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(ProviderConfig(name="ollama", model="llama3"))
    tool = MagicMock(spec=[])
    tool.name = "bare_tool"
    tool.description = "No schema"
    schemas = provider.get_tool_schema([tool])
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "bare_tool"


# ---------------------------------------------------------------------------
# Schema preservation through adapter round-trip
# ---------------------------------------------------------------------------


def test_schema_properties_preserved_through_adapter() -> None:
    """Ensure the adapter doesn't silently drop properties."""
    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig(name="anthropic", model="claude-sonnet-4-6"))
    tool = MagicMock()
    tool.name = "rich_tool"
    tool.description = "Multi-param tool"
    tool.get_schema.return_value = {
        "name": "rich_tool",
        "description": "Multi-param tool",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "max_depth": {"type": "integer"},
            },
            "required": ["path"],
        },
    }
    schemas = provider.get_tool_schema([tool])
    props = schemas[0]["input_schema"]["properties"]
    assert "path" in props
    assert "recursive" in props
    assert "max_depth" in props
