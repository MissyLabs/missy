"""Tests for missy.providers.schema_adapter (OpenClaw A6)."""

from __future__ import annotations

import pytest

from missy.providers.schema_adapter import (
    _flatten_nested_objects,
    _scrub_gemini_keys,
    canonical_from_anthropic,
    canonical_from_openai,
    normalize_batch_for_provider,
    normalize_for_provider,
)

CANONICAL = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The expression."},
        },
        "required": ["expression"],
    },
}


class TestNormalizeForProvider:
    def test_unknown_provider_returns_canonical(self):
        result = normalize_for_provider(CANONICAL, "unknown_provider")
        assert result["name"] == "calculator"
        assert "parameters" in result

    def test_does_not_mutate_input(self):
        original = dict(CANONICAL)
        normalize_for_provider(CANONICAL, "anthropic")
        assert original == CANONICAL

    def test_returns_new_dict(self):
        result = normalize_for_provider(CANONICAL, "anthropic")
        assert result is not CANONICAL


class TestAnthropicFormat:
    def test_has_input_schema(self):
        result = normalize_for_provider(CANONICAL, "anthropic")
        assert "input_schema" in result
        assert "parameters" not in result

    def test_input_schema_has_properties(self):
        result = normalize_for_provider(CANONICAL, "anthropic")
        assert "properties" in result["input_schema"]
        assert "expression" in result["input_schema"]["properties"]

    def test_empty_parameters_becomes_empty_schema(self):
        schema = {"name": "t", "description": "d"}
        result = normalize_for_provider(schema, "anthropic")
        assert result["input_schema"] == {"type": "object", "properties": {}, "required": []}


class TestOpenAIFormat:
    def test_wrapped_in_type_function(self):
        result = normalize_for_provider(CANONICAL, "openai")
        assert result["type"] == "function"
        assert "function" in result

    def test_function_has_name(self):
        result = normalize_for_provider(CANONICAL, "openai")
        assert result["function"]["name"] == "calculator"

    def test_function_has_parameters(self):
        result = normalize_for_provider(CANONICAL, "openai")
        assert "parameters" in result["function"]

    def test_ollama_same_as_openai(self):
        oa = normalize_for_provider(CANONICAL, "openai")
        ol = normalize_for_provider(CANONICAL, "ollama")
        assert oa["type"] == ol["type"]
        assert oa["function"]["name"] == ol["function"]["name"]

    def test_codex_same_as_openai(self):
        oa = normalize_for_provider(CANONICAL, "openai")
        cx = normalize_for_provider(CANONICAL, "codex")
        assert oa["function"]["name"] == cx["function"]["name"]


class TestMistralFormat:
    def test_wrapped_in_function(self):
        result = normalize_for_provider(CANONICAL, "mistral")
        assert result["type"] == "function"

    def test_nested_object_flattened(self):
        schema = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["city"],
                    }
                },
                "required": [],
            },
        }
        result = normalize_for_provider(schema, "mistral")
        props = result["function"]["parameters"]["properties"]
        assert "address__city" in props
        assert "address__zip" in props
        assert "address" not in props


class TestGeminiFormat:
    def test_removes_default_key(self):
        schema = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "default": "hello", "description": "..."},
                },
                "required": [],
            },
        }
        result = normalize_for_provider(schema, "gemini")
        props = result["parameters"]["properties"]
        assert "default" not in props["x"]

    def test_removes_examples_key(self):
        schema = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "examples": ["a", "b"]},
                },
                "required": [],
            },
        }
        result = normalize_for_provider(schema, "gemini")
        props = result["parameters"]["properties"]
        assert "examples" not in props["x"]

    def test_keeps_type_and_description(self):
        schema = {
            "name": "t",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "kept"},
                },
            },
        }
        result = normalize_for_provider(schema, "gemini")
        props = result["parameters"]["properties"]
        assert props["x"]["type"] == "string"
        assert props["x"]["description"] == "kept"


class TestBatchNormalize:
    def test_normalizes_all(self):
        schemas = [CANONICAL, dict(CANONICAL)]
        results = normalize_batch_for_provider(schemas, "anthropic")
        assert len(results) == 2
        for r in results:
            assert "input_schema" in r

    def test_empty_list(self):
        assert normalize_batch_for_provider([], "openai") == []


class TestCanonicalRoundTrip:
    def test_anthropic_round_trip(self):
        anthropic = normalize_for_provider(CANONICAL, "anthropic")
        canonical = canonical_from_anthropic(anthropic)
        assert "parameters" in canonical
        assert "input_schema" not in canonical

    def test_openai_round_trip(self):
        openai = normalize_for_provider(CANONICAL, "openai")
        canonical = canonical_from_openai(openai)
        assert canonical["name"] == "calculator"


class TestFlattenNestedObjects:
    def test_flat_params_unchanged(self):
        params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        result = _flatten_nested_objects(params)
        assert "x" in result["properties"]

    def test_nested_required_not_promoted_when_parent_optional(self):
        # `addr` itself is optional (not in the outer `required` list), so a
        # field required only *within* `addr` must not become an
        # unconditionally required flat key — the model can still omit the
        # whole address. See PADAPT-003.
        params = {
            "type": "object",
            "properties": {
                "addr": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "required": [],
        }
        result = _flatten_nested_objects(params)
        assert "addr__city" not in result["required"]
        assert "addr__zip" not in result["required"]

    def test_nested_required_preserved_when_parent_required(self):
        params = {
            "type": "object",
            "properties": {
                "addr": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "required": ["addr"],
        }
        result = _flatten_nested_objects(params)
        assert "addr__city" in result["required"]
        assert "addr__zip" not in result["required"]

    def test_collision_between_literal_and_nested_key_raises(self):
        params = {
            "type": "object",
            "properties": {
                "address__city": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            "required": [],
        }
        with pytest.raises(ValueError, match="collision"):
            _flatten_nested_objects(params)

    def test_collision_between_two_nested_objects_raises(self):
        params = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {"x__y": {"type": "string"}},
                },
                "a__x": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                },
            },
            "required": [],
        }
        with pytest.raises(ValueError, match="collision"):
            _flatten_nested_objects(params)


class TestScrubGeminiKeys:
    def test_removes_schema_key(self):
        obj = {"$schema": "http://...", "type": "object"}
        result = _scrub_gemini_keys(obj)
        assert "$schema" not in result
        assert "type" in result

    def test_recurses_into_nested(self):
        obj = {"properties": {"x": {"default": "val", "type": "string"}}}
        result = _scrub_gemini_keys(obj)
        assert "default" not in result["properties"]["x"]

    def test_removes_singular_example_key(self):
        # OpenAPI-style singular ``example`` (as used by CalculatorTool's
        # expression param) is unsupported by Gemini and must be scrubbed too,
        # not just the plural JSON Schema ``examples``.
        obj = {"properties": {"x": {"example": "(2 + 3) * 4", "type": "string"}}}
        result = _scrub_gemini_keys(obj)
        assert "example" not in result["properties"]["x"]
        assert result["properties"]["x"]["type"] == "string"

    def test_handles_lists(self):
        obj = [{"default": "x"}, {"type": "string"}]
        result = _scrub_gemini_keys(obj)
        assert "default" not in result[0]
        assert result[1]["type"] == "string"

    def test_primitives_unchanged(self):
        assert _scrub_gemini_keys(42) == 42
        assert _scrub_gemini_keys("hello") == "hello"
