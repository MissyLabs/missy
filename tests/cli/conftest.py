"""Shared fixtures for CLI tests."""

from __future__ import annotations

import inspect

import pytest
from click.testing import CliRunner


def _make_cli_runner(**kwargs):
    """Create a CliRunner, dropping unsupported kwargs for the installed Click version."""
    sig = inspect.signature(CliRunner.__init__)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return CliRunner(**supported)


def combined_output(result) -> str:
    """Return a Result's stdout and stderr as one string, on any Click version.

    On Click < 8.2 our runners pass ``mix_stderr=False``, so error text
    (``_print_error`` writes to stderr) never appears in ``result.output``;
    on Click >= 8.2 the kwarg no longer exists and ``result.output``
    already combines both streams. Assertions about error messages must
    go through this helper to pass on both.
    """
    output = result.output
    try:
        stderr = result.stderr
    except (ValueError, AttributeError):
        stderr = ""
    if stderr and stderr not in output:
        output += stderr
    return output


@pytest.fixture()
def runner():
    """CliRunner compatible with all Click versions."""
    return _make_cli_runner(mix_stderr=False)
