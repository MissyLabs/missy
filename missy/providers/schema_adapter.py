"""Per-provider tool schema normalization (OpenClaw A6).

Every LLM provider expects tool definitions in a slightly different format.
This module provides a single :func:`normalize_for_provider` function that
converts a canonical tool schema (the format returned by
:meth:`~missy.tools.base.BaseTool.get_schema`) into the wire format expected
by a specific provider.

Canonical format
----------------
::

    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "..."}
            },
            "required": ["expression"],
        },
    }

Provider formats
----------------

**Anthropic** — wraps ``parameters`` under ``input_schema``::

    {
        "name": "calculator",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...],
        },
    }

**OpenAI / Codex / Compatible** — wraps the whole thing in a
``type: function`` envelope::

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "...",
            "parameters": {...},
        },
    }

**Ollama** — identical to OpenAI function-calling format.

**Mistral** — like OpenAI but nested object properties are flattened
(Mistral struggles with deeply nested schemas).

**Gemini** — drops unsupported JSON Schema keywords (``default``,
``examples``, ``$schema``, ``$id``, ``additionalProperties``).

Usage
-----
::

    from missy.tools.base import BaseTool
    from missy.providers.schema_adapter import normalize_for_provider

    schema = some_tool.get_schema()
    anthropic_schema = normalize_for_provider(schema, "anthropic")
    openai_schema   = normalize_for_provider(schema, "openai")
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Providers whose canonical name maps to a normalization strategy.
_OPENAI_COMPATIBLE = frozenset({"openai", "ollama", "codex", "openai_compatible"})
_GEMINI_DROP_KEYS = frozenset(
    # Both the plural ``examples`` (JSON Schema) and the singular ``example``
    # (OpenAPI annotation, used by e.g. CalculatorTool's expression param) are
    # unsupported by Gemini's stricter Schema proto and must be scrubbed.
    {"default", "example", "examples", "$schema", "$id", "additionalProperties", "title"}
)


def normalize_for_provider(
    schema: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Return *schema* converted to the wire format expected by *provider*.

    Args:
        schema: Canonical tool schema as returned by
            :meth:`~missy.tools.base.BaseTool.get_schema`.
        provider: Provider identifier string (``"anthropic"``, ``"openai"``,
            ``"ollama"``, ``"mistral"``, ``"gemini"``).  Unknown providers
            receive the canonical form unchanged.

    Returns:
        A new ``dict`` in the provider-specific format (never mutates input).
    """
    s = copy.deepcopy(schema)
    key = provider.lower().strip()

    if key == "anthropic":
        return _to_anthropic(s)
    if key in _OPENAI_COMPATIBLE:
        return _to_openai(s)
    if key == "mistral":
        return _to_mistral(s)
    if key == "gemini":
        return _to_gemini(s)

    logger.debug("schema_adapter: unknown provider %r — returning canonical schema", provider)
    return s


def normalize_batch_for_provider(
    schemas: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, Any]]:
    """Apply :func:`normalize_for_provider` to each schema in *schemas*.

    Args:
        schemas: List of canonical tool schemas.
        provider: Provider identifier.

    Returns:
        New list of converted schemas in the same order.
    """
    return [normalize_for_provider(s, provider) for s in schemas]


def canonical_from_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic-format schema back to canonical form.

    Args:
        schema: Anthropic tool dict with ``input_schema`` key.

    Returns:
        Canonical tool schema dict.
    """
    s = copy.deepcopy(schema)
    input_schema = s.pop("input_schema", {})
    s["parameters"] = input_schema
    return s


def canonical_from_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-format schema back to canonical form.

    Handles both the ``{"type": "function", "function": {...}}`` envelope
    and the bare function dict.

    Args:
        schema: OpenAI tool definition.

    Returns:
        Canonical tool schema dict.
    """
    s = copy.deepcopy(schema)
    if s.get("type") == "function" and "function" in s:
        return s["function"]
    return s


# ---------------------------------------------------------------------------
# Provider-specific converters
# ---------------------------------------------------------------------------


def _to_anthropic(s: dict[str, Any]) -> dict[str, Any]:
    """Anthropic Messages API format."""
    params = s.pop("parameters", {})
    s["input_schema"] = params or {"type": "object", "properties": {}, "required": []}
    return s


def _to_openai(s: dict[str, Any]) -> dict[str, Any]:
    """OpenAI / Ollama function-calling format."""
    if "parameters" not in s:
        s["parameters"] = {"type": "object", "properties": {}, "required": []}
    return {"type": "function", "function": s}


def _to_mistral(s: dict[str, Any]) -> dict[str, Any]:
    """Mistral format — like OpenAI but with flattened nested objects.

    Mistral has limited support for nested object parameters.  We flatten
    one level deep: a nested object property ``address.city`` becomes
    ``address__city``.
    """
    if "parameters" not in s:
        s["parameters"] = {"type": "object", "properties": {}, "required": []}
    s["parameters"] = _flatten_nested_objects(s["parameters"])
    return {"type": "function", "function": s}


def _to_gemini(s: dict[str, Any]) -> dict[str, Any]:
    """Gemini format — remove unsupported JSON Schema keys recursively."""
    if "parameters" in s:
        s["parameters"] = _scrub_gemini_keys(s["parameters"])
    return s


# ---------------------------------------------------------------------------
# Schema manipulation helpers
# ---------------------------------------------------------------------------


def _flatten_nested_objects(params: dict[str, Any]) -> dict[str, Any]:
    """Flatten one level of nested ``type: object`` properties.

    For example::

        {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                }
            }
        }

    becomes::

        {
            "address__city": {"type": "string"},
            "address__zip": {"type": "string"},
        }

    Non-object properties are left unchanged.
    """
    props = params.get("properties", {})
    required = list(params.get("required", []))
    new_props: dict[str, Any] = {}
    new_required: list[str] = []

    for key, val in props.items():
        if val.get("type") == "object" and "properties" in val:
            nested_req = set(val.get("required", []))
            for sub_key, sub_val in val["properties"].items():
                flat_key = f"{key}__{sub_key}"
                new_props[flat_key] = sub_val
                if sub_key in nested_req:
                    new_required.append(flat_key)
        else:
            new_props[key] = val
            if key in required:
                new_required.append(key)

    result = dict(params)
    result["properties"] = new_props
    result["required"] = new_required
    return result


def _scrub_gemini_keys(obj: Any) -> Any:
    """Recursively remove Gemini-unsupported JSON Schema keys from *obj*."""
    if isinstance(obj, dict):
        return {k: _scrub_gemini_keys(v) for k, v in obj.items() if k not in _GEMINI_DROP_KEYS}
    if isinstance(obj, list):
        return [_scrub_gemini_keys(item) for item in obj]
    return obj
