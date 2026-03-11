"""Tests for missy.tools.builtin.calculator.CalculatorTool."""

from __future__ import annotations

import pytest

from missy.tools.builtin.calculator import CalculatorTool
from missy.tools.base import ToolResult


@pytest.fixture()
def calculator() -> CalculatorTool:
    return CalculatorTool()


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------


class TestBasicArithmetic:
    def test_addition(self, calculator):
        result = calculator.execute(expression="1 + 1")
        assert result.success is True
        assert result.output == 2

    def test_multiplication(self, calculator):
        result = calculator.execute(expression="2 * 3")
        assert result.success is True
        assert result.output == 6

    def test_division(self, calculator):
        result = calculator.execute(expression="10 / 2")
        assert result.success is True
        assert result.output == 5.0

    def test_subtraction(self, calculator):
        result = calculator.execute(expression="10 - 4")
        assert result.success is True
        assert result.output == 6

    def test_floor_division(self, calculator):
        result = calculator.execute(expression="7 // 2")
        assert result.success is True
        assert result.output == 3

    def test_modulo(self, calculator):
        result = calculator.execute(expression="10 % 3")
        assert result.success is True
        assert result.output == 1


# ---------------------------------------------------------------------------
# Exponentiation
# ---------------------------------------------------------------------------


class TestExponentiation:
    def test_exponent(self, calculator):
        result = calculator.execute(expression="2 ** 3")
        assert result.success is True
        assert result.output == 8

    def test_exponent_result(self, calculator):
        result = calculator.execute(expression="2 ** 10")
        assert result.success is True
        assert result.output == 1024

    def test_exponent_cap_exceeds_maximum(self, calculator):
        result = calculator.execute(expression="2 ** 1001")
        assert result.success is False
        assert result.error is not None
        assert "1001" in result.error or "Exponent" in result.error

    def test_exponent_at_maximum_allowed(self, calculator):
        result = calculator.execute(expression="2 ** 1000")
        assert result.success is True

    def test_negative_exponent_within_limit(self, calculator):
        result = calculator.execute(expression="2 ** -2")
        assert result.success is True
        assert result.output == 0.25


# ---------------------------------------------------------------------------
# Float results
# ---------------------------------------------------------------------------


class TestFloatResults:
    def test_float_division(self, calculator):
        result = calculator.execute(expression="1 / 3")
        assert result.success is True
        assert abs(result.output - 1 / 3) < 1e-10

    def test_float_literal(self, calculator):
        result = calculator.execute(expression="3.14 * 2")
        assert result.success is True
        assert abs(result.output - 6.28) < 1e-10

    def test_negative_float(self, calculator):
        result = calculator.execute(expression="-1.5 + 0.5")
        assert result.success is True
        assert result.output == -1.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_division_by_zero(self, calculator):
        result = calculator.execute(expression="1 / 0")
        assert result.success is False
        assert result.error is not None

    def test_invalid_syntax(self, calculator):
        result = calculator.execute(expression="1 +* 2")
        assert result.success is False
        assert result.error is not None

    def test_empty_expression(self, calculator):
        result = calculator.execute(expression="")
        assert result.success is False
        assert "empty" in (result.error or "").lower()

    def test_whitespace_only_expression(self, calculator):
        result = calculator.execute(expression="   ")
        assert result.success is False

    def test_non_string_expression(self, calculator):
        result = calculator.execute(expression=42)  # type: ignore[arg-type]
        assert result.success is False
        assert result.error is not None

    def test_string_literal_rejected(self, calculator):
        result = calculator.execute(expression='"hello"')
        assert result.success is False

    def test_comparison_operator_rejected(self, calculator):
        result = calculator.execute(expression="1 == 1")
        assert result.success is False

    def test_boolean_operator_rejected(self, calculator):
        result = calculator.execute(expression="True and False")
        assert result.success is False


# ---------------------------------------------------------------------------
# Shell injection / sandbox bypass attempts
# ---------------------------------------------------------------------------


class TestShellInjectionBlocked:
    def test_import_blocked(self, calculator):
        result = calculator.execute(expression="__import__('os')")
        assert result.success is False

    def test_function_call_blocked(self, calculator):
        result = calculator.execute(expression="print('hello')")
        assert result.success is False

    def test_attribute_access_blocked(self, calculator):
        result = calculator.execute(expression="(1).bit_length()")
        assert result.success is False

    def test_list_comprehension_blocked(self, calculator):
        result = calculator.execute(expression="[x for x in range(10)]")
        assert result.success is False

    def test_lambda_blocked(self, calculator):
        result = calculator.execute(expression="lambda x: x")
        assert result.success is False


# ---------------------------------------------------------------------------
# Bitwise operators
# ---------------------------------------------------------------------------


class TestBitwiseOperators:
    def test_bitwise_and(self, calculator):
        result = calculator.execute(expression="12 & 10")
        assert result.success is True
        assert result.output == (12 & 10)

    def test_bitwise_or(self, calculator):
        result = calculator.execute(expression="12 | 10")
        assert result.success is True
        assert result.output == (12 | 10)

    def test_bitwise_xor(self, calculator):
        result = calculator.execute(expression="12 ^ 10")
        assert result.success is True
        assert result.output == (12 ^ 10)

    def test_left_shift(self, calculator):
        result = calculator.execute(expression="1 << 4")
        assert result.success is True
        assert result.output == 16

    def test_right_shift(self, calculator):
        result = calculator.execute(expression="16 >> 2")
        assert result.success is True
        assert result.output == 4

    def test_bitwise_invert(self, calculator):
        result = calculator.execute(expression="~5")
        assert result.success is True
        assert result.output == ~5


# ---------------------------------------------------------------------------
# get_schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    def test_returns_dict(self, calculator):
        schema = calculator.get_schema()
        assert isinstance(schema, dict)

    def test_schema_has_name(self, calculator):
        schema = calculator.get_schema()
        assert schema["name"] == "calculator"

    def test_schema_has_description(self, calculator):
        schema = calculator.get_schema()
        assert "description" in schema
        assert isinstance(schema["description"], str)

    def test_schema_has_parameters(self, calculator):
        schema = calculator.get_schema()
        assert "parameters" in schema
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "expression" in params["properties"]
        assert "expression" in params["required"]


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_name_is_calculator(self, calculator):
        assert calculator.name == "calculator"

    def test_permissions_are_all_false(self, calculator):
        perms = calculator.permissions
        assert perms.network is False
        assert perms.filesystem_read is False
        assert perms.filesystem_write is False
        assert perms.shell is False

    def test_result_is_tool_result(self, calculator):
        result = calculator.execute(expression="1 + 1")
        assert isinstance(result, ToolResult)
