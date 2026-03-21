"""Coverage-gap tests for missy/tools/builtin/__init__.py.

Targets uncovered lines:
  194-196: register_builtin_tools(registry=None) → fetches process-level registry
            via get_tool_registry() when no explicit registry is provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin import _ALL_TOOL_CLASSES, register_builtin_tools


class TestRegisterBuiltinToolsWithRegistry:
    """register_builtin_tools(registry=explicit) path — already covered; sanity checks."""

    def test_registers_all_tool_classes_into_provided_registry(self):
        """Each class in _ALL_TOOL_CLASSES is instantiated and registered."""
        mock_registry = MagicMock()
        register_builtin_tools(registry=mock_registry)

        # register() should have been called once per tool class.
        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)

    def test_registered_objects_are_tool_instances(self):
        """Each argument to registry.register() is an instance of the corresponding class."""
        from missy.tools.base import BaseTool

        mock_registry = MagicMock()
        register_builtin_tools(registry=mock_registry)

        for call_args in mock_registry.register.call_args_list:
            obj = call_args.args[0]
            assert isinstance(obj, BaseTool), f"{obj!r} is not a BaseTool instance"


class TestRegisterBuiltinToolsWithoutRegistry:
    """Lines 194-196: register_builtin_tools() with registry=None uses get_tool_registry().

    The function has a lazy import:
        from missy.tools.registry import get_tool_registry
    so we must patch at the registry module level, not on the __init__ module.
    """

    def test_calls_get_tool_registry_when_no_registry_provided(self):
        """When registry=None, get_tool_registry() is called to obtain the singleton."""
        mock_registry = MagicMock()

        with patch(
            "missy.tools.registry.get_tool_registry",
            return_value=mock_registry,
        ) as mock_get:
            register_builtin_tools()

        mock_get.assert_called_once_with()

    def test_registers_all_tools_into_process_level_registry(self):
        """When registry=None, all tools are registered into the returned singleton."""
        mock_registry = MagicMock()

        with patch(
            "missy.tools.registry.get_tool_registry",
            return_value=mock_registry,
        ):
            register_builtin_tools()

        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)

    def test_get_tool_registry_raises_runtime_error_propagates(self):
        """RuntimeError from get_tool_registry (not yet initialised) propagates to caller."""
        with (
            patch(
                "missy.tools.registry.get_tool_registry",
                side_effect=RuntimeError("registry not initialised"),
            ),
            pytest.raises(RuntimeError, match="registry not initialised"),
        ):
            register_builtin_tools()

    def test_none_is_treated_same_as_omitting_registry(self):
        """Explicitly passing registry=None is identical to calling with no argument."""
        mock_registry = MagicMock()

        with patch(
            "missy.tools.registry.get_tool_registry",
            return_value=mock_registry,
        ) as mock_get:
            register_builtin_tools(registry=None)

        mock_get.assert_called_once_with()
        assert mock_registry.register.call_count == len(_ALL_TOOL_CLASSES)


class TestAllToolClassesList:
    """Sanity checks on _ALL_TOOL_CLASSES to guard against accidental omissions."""

    def test_all_tool_classes_non_empty(self):
        assert len(_ALL_TOOL_CLASSES) > 0

    def test_all_entries_are_classes(self):
        import inspect

        for cls in _ALL_TOOL_CLASSES:
            assert inspect.isclass(cls), f"{cls!r} is not a class"

    def test_tts_tools_present(self):
        from missy.tools.builtin.tts_speak import (
            AudioListDevicesTool,
            AudioSetVolumeTool,
            TTSSpeakTool,
        )

        assert TTSSpeakTool in _ALL_TOOL_CLASSES
        assert AudioListDevicesTool in _ALL_TOOL_CLASSES
        assert AudioSetVolumeTool in _ALL_TOOL_CLASSES
