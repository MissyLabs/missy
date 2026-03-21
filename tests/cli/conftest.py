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


@pytest.fixture()
def runner():
    """CliRunner compatible with all Click versions."""
    return _make_cli_runner(mix_stderr=False)
