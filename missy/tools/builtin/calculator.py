"""Safe arithmetic calculator tool for the Missy framework.

Evaluates mathematical expressions using Python's :mod:`ast` module,
restricting execution to a safe subset of operations.  No shell commands,
file I/O, or network access are performed.

Supported constructs:

* Numeric literals (int, float, complex)
* Unary operators: ``+``, ``-``
* Binary operators: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
* Bitwise operators: ``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``
* Parenthesised sub-expressions

Anything outside this set (function calls, attribute access, comparisons,
boolean operators, etc.) raises a :class:`ValueError`.

Example::

    from missy.tools.builtin.calculator import CalculatorTool

    tool = CalculatorTool()
    result = tool.execute(expression="(2 ** 10) / 4")
    assert result.success
    assert result.output == 256.0
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

# ---------------------------------------------------------------------------
# Allowed AST node types and operator mappings
# ---------------------------------------------------------------------------

_ALLOWED_NODE_TYPES = frozenset(
    {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        # Operators - binary
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        # Operators - unary
        ast.UAdd,
        ast.USub,
        ast.Invert,
    }
)

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}

# Guard against exponentiation DoS (e.g. 9**9**9**9)
_MAX_EXPONENT = 1_000

# Guard against left-shift memory exhaustion (e.g. 1 << 10000000000)
_MAX_SHIFT = 10_000


def _safe_eval(node: ast.AST) -> int | float | complex:
    """Recursively evaluate an AST node from a numeric expression.

    Args:
        node: An AST node to evaluate.

    Returns:
        The numeric result.

    Raises:
        ValueError: When the node contains unsupported constructs.
        ZeroDivisionError: When a division by zero is attempted.
        OverflowError: When the result exceeds representable bounds.
    """
    if type(node) not in _ALLOWED_NODE_TYPES:
        raise ValueError(
            f"Unsupported expression construct: {type(node).__name__}. "
            "Only arithmetic operators and numeric literals are allowed."
        )

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, complex)):
            raise ValueError(f"Non-numeric literal {node.value!r} is not permitted.")
        return node.value

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)
        op_fn = _BINARY_OPS.get(op_type)
        if op_fn is None:  # pragma: no cover - covered by _ALLOWED_NODE_TYPES guard
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        # Prevent exponent DoS
        if (
            isinstance(node.op, ast.Pow)
            and isinstance(right, (int, float))
            and abs(right) > _MAX_EXPONENT
        ):
            raise ValueError(
                f"Exponent {right} exceeds the maximum allowed value of {_MAX_EXPONENT}."
            )
        # Prevent left-shift memory exhaustion
        if isinstance(node.op, ast.LShift) and isinstance(right, int) and right > _MAX_SHIFT:
            raise ValueError(
                f"Left shift by {right} exceeds the maximum allowed value of {_MAX_SHIFT}."
            )
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)
        op_fn = _UNARY_OPS.get(op_type)
        if op_fn is None:  # pragma: no cover
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return op_fn(operand)

    # Should be unreachable due to the _ALLOWED_NODE_TYPES guard at the top.
    raise ValueError(f"Unhandled node type: {type(node).__name__}")  # pragma: no cover


class CalculatorTool(BaseTool):
    """Safe arithmetic evaluator with no network, filesystem, or shell access.

    Expressions are parsed and evaluated entirely within Python using the
    :mod:`ast` module.  No ``eval`` or ``exec`` call is made; only
    explicitly whitelisted AST node types are accepted.

    Attributes:
        name: ``"calculator"``
        description: One-line description for function-calling schemas.
        permissions: All permission flags are ``False``.
    """

    name = "calculator"
    description = "Evaluate arithmetic expressions safely without using eval or shell"
    permissions = ToolPermissions(
        network=False,
        filesystem_read=False,
        filesystem_write=False,
        shell=False,
    )

    def execute(self, *, expression: str, **_kwargs: Any) -> ToolResult:
        """Evaluate *expression* and return the numeric result.

        Args:
            expression: A string containing a pure arithmetic expression,
                e.g. ``"(2 + 3) * 4"`` or ``"2 ** 8"``.

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` and ``output`` set to the numeric result on
              success.
            * ``success=False`` and ``error`` describing the problem on
              parse or evaluation failure.
        """
        if not isinstance(expression, str):
            return ToolResult(
                success=False,
                output=None,
                error=f"expression must be a string, got {type(expression).__name__}",
            )

        expr = expression.strip()
        if not expr:
            return ToolResult(
                success=False,
                output=None,
                error="expression must not be empty",
            )

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Syntax error in expression: {exc}",
            )

        try:
            result = _safe_eval(tree)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
            )

        return ToolResult(success=True, output=result)

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the calculator's parameters.

        Returns:
            A dict suitable for use in an OpenAI-style function-calling
            payload.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "An arithmetic expression to evaluate, e.g. '(2 + 3) * 4'. "
                            "Supports +, -, *, /, //, %, ** and bitwise operators. "
                            "No function calls or variable references are permitted."
                        ),
                    }
                },
                "required": ["expression"],
            },
        }
