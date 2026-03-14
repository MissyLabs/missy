"""Coverage gap tests for missy/tools/builtin/atspi_tools.py.

Targets uncovered lines:
  45-47  : _get_desktop() imports and calls pyatspi.Registry.getDesktop
  89     : _get_focused_application — STATE_ACTIVE match (returns active child)
  151-154: _walk_tree — state_set.contains raises, states falls back to []
  178-179: _walk_tree — outer exception returns []
  238    : _find_element — inner exception path in _search
  256-259: _find_element — child search exception continues
  349-350: AtSpiGetTreeTool.execute — outer exception from _walk_tree/_format_tree
  510-511: AtSpiGetTextTool.execute — desktop connection exception
  521-529: AtSpiGetTextTool.execute — app not found by name / no focused app
  541    : AtSpiGetTextTool.execute — element not found
  565-566: AtSpiGetTextTool.execute — outer exception
  634-636: AtSpiSetValueTool.execute — app not found by name
  655-656: AtSpiSetValueTool.execute — no focused app
  667-668: AtSpiSetValueTool.execute — element not found
  698-699: AtSpiSetValueTool.execute — outer exception
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pyatspi_mock():
    """Return a MagicMock that acts as the pyatspi module."""
    mock = MagicMock()
    mock.STATE_ACTIVE = "STATE_ACTIVE"
    mock.STATE_FOCUSABLE = "STATE_FOCUSABLE"
    mock.STATE_FOCUSED = "STATE_FOCUSED"
    mock.STATE_SENSITIVE = "STATE_SENSITIVE"
    mock.STATE_EDITABLE = "STATE_EDITABLE"
    mock.STATE_CHECKED = "STATE_CHECKED"
    mock.STATE_SELECTED = "STATE_SELECTED"
    mock.STATE_VISIBLE = "STATE_VISIBLE"
    mock.STATE_SHOWING = "STATE_SHOWING"
    return mock


def _make_accessible(name="TestApp", role_name="application", child_count=0,
                     states=None, text=None):
    """Return a mock AT-SPI accessible node."""
    node = MagicMock()
    node.name = name
    node.getRoleName.return_value = role_name
    node.childCount = child_count
    node.getChildAtIndex.return_value = None

    state_set = MagicMock()
    _active_states = set(states or [])
    state_set.contains.side_effect = lambda s: s in _active_states
    node.getState.return_value = state_set

    if text is not None:
        text_iface = MagicMock()
        text_iface.getText.return_value = text
        node.queryText.return_value = text_iface
    else:
        node.queryText.side_effect = Exception("no text iface")

    return node


# ---------------------------------------------------------------------------
# _get_desktop — lines 45-47
# ---------------------------------------------------------------------------


class TestGetDesktop:
    def test_get_desktop_calls_registry(self):
        """_get_desktop imports pyatspi and calls Registry.getDesktop(0)."""
        from missy.tools.builtin import atspi_tools

        mock_pyatspi = _make_pyatspi_mock()
        fake_desktop = MagicMock()
        mock_pyatspi.Registry.getDesktop.return_value = fake_desktop

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            # Force re-import inside the function
            result = atspi_tools._get_desktop()

        assert result is fake_desktop
        mock_pyatspi.Registry.getDesktop.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# _get_focused_application — line 89 (STATE_ACTIVE match)
# ---------------------------------------------------------------------------


class TestGetFocusedApplication:
    def test_returns_active_child(self):
        """Returns the first child whose state set contains STATE_ACTIVE."""
        from missy.tools.builtin import atspi_tools

        mock_pyatspi = _make_pyatspi_mock()

        inactive_child = _make_accessible("Inactive")
        active_child = _make_accessible("Active", states=["STATE_ACTIVE"])

        desktop = MagicMock()
        desktop.childCount = 2
        desktop.getChildAtIndex.side_effect = [inactive_child, active_child]

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = atspi_tools._get_focused_application(desktop)

        assert result is active_child

    def test_falls_back_to_first_nonnone_child(self):
        """When no STATE_ACTIVE child exists, returns first non-None child."""
        from missy.tools.builtin import atspi_tools

        mock_pyatspi = _make_pyatspi_mock()

        only_child = _make_accessible("Only")

        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.return_value = only_child

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = atspi_tools._get_focused_application(desktop)

        assert result is only_child

    def test_returns_none_when_no_children(self):
        """Returns None when desktop has no children."""
        from missy.tools.builtin import atspi_tools

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = atspi_tools._get_focused_application(desktop)

        assert result is None


# ---------------------------------------------------------------------------
# _walk_tree — lines 151-154 and 178-179
# ---------------------------------------------------------------------------


class TestWalkTree:
    def test_state_contains_raises_falls_back_to_empty_states(self):
        """Lines 151-154: when state_set.contains() raises, states defaults to []."""
        from missy.tools.builtin import atspi_tools

        mock_pyatspi = _make_pyatspi_mock()

        node = MagicMock()
        node.name = "Btn"
        node.getRoleName.return_value = "push button"
        node.childCount = 0

        # getState succeeds but contains() raises
        bad_state_set = MagicMock()
        bad_state_set.contains.side_effect = Exception("state error")
        node.getState.return_value = bad_state_set

        node.queryText.side_effect = Exception("no text")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = atspi_tools._walk_tree(node, max_depth=1)

        assert len(result) == 1
        assert result[0]["states"] == []

    def test_walk_tree_outer_exception_returns_empty(self):
        """Lines 178-179: when the node itself raises on getRoleName, returns []."""
        from missy.tools.builtin.atspi_tools import _walk_tree

        bad_node = MagicMock()
        bad_node.getRoleName.side_effect = RuntimeError("boom")
        bad_node.name = None  # accessing .name after getRoleName raises

        result = _walk_tree(bad_node, max_depth=2)
        assert result == []

    def test_walk_tree_none_node_returns_empty(self):
        """None node returns []."""
        from missy.tools.builtin.atspi_tools import _walk_tree

        assert _walk_tree(None, max_depth=2) == []

    def test_walk_tree_depth_exceeded_returns_empty(self):
        """current_depth > max_depth returns []."""
        from missy.tools.builtin.atspi_tools import _walk_tree

        node = _make_accessible()
        assert _walk_tree(node, max_depth=0, current_depth=1) == []


# ---------------------------------------------------------------------------
# _find_element — lines 238, 256-259
# ---------------------------------------------------------------------------


class TestFindElement:
    def test_inner_exception_continues_to_children(self):
        """Lines 256-259: getChildAtIndex raises → continues to next child."""
        from missy.tools.builtin.atspi_tools import _find_element

        # child that raises on access
        bad_child = MagicMock()
        bad_child.name = "target"
        bad_child.getRoleName.return_value = "push button"
        bad_child.childCount = 0

        app = MagicMock()
        app.name = "App"
        app.getRoleName.return_value = "application"
        app.childCount = 2
        # First child raises, second child is the target
        app.getChildAtIndex.side_effect = [Exception("access denied"), bad_child]

        result = _find_element(app, name="target", role=None)
        assert result is bad_child

    def test_find_element_returns_none_when_not_found(self):
        """Returns None when no element matches the search criteria."""
        from missy.tools.builtin.atspi_tools import _find_element

        app = _make_accessible("App", child_count=0)
        result = _find_element(app, name="nonexistent", role=None)
        assert result is None

    def test_node_exception_in_search_returns_none(self):
        """Lines 238: outer try block catches node-level exception."""
        from missy.tools.builtin.atspi_tools import _find_element

        bad_app = MagicMock()
        bad_app.name = MagicMock(side_effect=Exception("attr fail"))
        bad_app.getRoleName.side_effect = Exception("role fail")
        bad_app.childCount = 0

        result = _find_element(bad_app, name="x", role=None)
        assert result is None


# ---------------------------------------------------------------------------
# AtSpiGetTreeTool — lines 349-350 (outer exception in execute)
# ---------------------------------------------------------------------------


class TestAtSpiGetTreeToolCoverage:
    def test_pyatspi_missing_returns_error(self):
        """When pyatspi is not installed, returns error ToolResult."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        with patch.dict(sys.modules, {"pyatspi": None}):
            result = AtSpiGetTreeTool().execute()

        assert result.success is False
        assert "pyatspi" in result.error

    def test_desktop_exception_returns_error(self):
        """When _get_desktop raises, returns error with message."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        mock_pyatspi = _make_pyatspi_mock()
        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   side_effect=Exception("dbus not available")):
            result = AtSpiGetTreeTool().execute()

        assert result.success is False
        assert "AT-SPI desktop" in result.error

    def test_app_not_found_by_name(self):
        """When named app is not on desktop, returns error."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiGetTreeTool().execute(app_name="NoSuchApp")

        assert result.success is False
        assert "not found" in result.error

    def test_no_focused_app_returns_error(self):
        """When no app name given and no focused app, returns error."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiGetTreeTool().execute()

        assert result.success is False
        assert "No focused application" in result.error

    def test_outer_exception_in_execute(self):
        """Lines 349-350: outer try/except catches exceptions from _walk_tree etc."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._walk_tree",
                   side_effect=RuntimeError("tree walk exploded")):
            result = AtSpiGetTreeTool().execute()

        assert result.success is False
        assert "Failed to read accessibility tree" in result.error

    def test_success_with_focused_app(self):
        """Happy path: focused app found, tree built successfully."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("MyApp")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._walk_tree",
                   return_value=[{"depth": 0, "role": "application", "name": "MyApp",
                                  "states": []}]):
            result = AtSpiGetTreeTool().execute()

        assert result.success is True
        assert result.output["app_name"] == "MyApp"


# ---------------------------------------------------------------------------
# AtSpiClickTool — coverage gaps
# ---------------------------------------------------------------------------


class TestAtSpiClickToolCoverage:
    def test_pyatspi_missing(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        with patch.dict(sys.modules, {"pyatspi": None}):
            result = AtSpiClickTool().execute(name="OK")
        assert result.success is False
        assert "pyatspi" in result.error

    def test_no_name_or_role_returns_error(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = AtSpiClickTool().execute()
        assert result.success is False
        assert "name" in result.error or "role" in result.error

    def test_element_not_found_returns_error(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=None):
            result = AtSpiClickTool().execute(name="NonExistentButton")

        assert result.success is False
        assert "not found" in result.error

    def test_successful_click(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("OK", role_name="push button")
        action_iface = MagicMock()
        element.queryAction.return_value = action_iface

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiClickTool().execute(name="OK")

        assert result.success is True
        action_iface.doAction.assert_called_once_with(0)

    def test_click_with_app_name(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("Firefox")
        element = _make_accessible("Close", role_name="push button")
        action_iface = MagicMock()
        element.queryAction.return_value = action_iface

        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.return_value = app

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiClickTool().execute(name="Close", app_name="Firefox")

        assert result.success is True

    def test_click_app_not_found(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiClickTool().execute(name="OK", app_name="NoSuchApp")

        assert result.success is False
        assert "not found" in result.error

    def test_click_exception_returns_error(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("Btn", role_name="push button")
        element.queryAction.side_effect = Exception("action error")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiClickTool().execute(name="Btn")

        assert result.success is False
        assert "AT-SPI click failed" in result.error


# ---------------------------------------------------------------------------
# AtSpiGetTextTool — lines 510-511, 521-529, 541, 565-566
# ---------------------------------------------------------------------------


class TestAtSpiGetTextToolCoverage:
    def test_pyatspi_missing(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        with patch.dict(sys.modules, {"pyatspi": None}):
            result = AtSpiGetTextTool().execute(name="label")
        assert result.success is False
        assert "pyatspi" in result.error

    def test_no_name_or_role(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}):
            result = AtSpiGetTextTool().execute()
        assert result.success is False

    def test_desktop_exception(self):
        """Lines 510-511: desktop connection fails."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   side_effect=Exception("no dbus")):
            result = AtSpiGetTextTool().execute(name="label")

        assert result.success is False
        assert "AT-SPI desktop" in result.error

    def test_app_not_found_by_name(self):
        """Lines 521-525: named app not found."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiGetTextTool().execute(name="label", app_name="Ghost")

        assert result.success is False
        assert "not found" in result.error

    def test_no_focused_app(self):
        """Lines 526-529: no focused app when no app_name given."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiGetTextTool().execute(name="label")

        assert result.success is False
        assert "No focused application" in result.error

    def test_element_not_found(self):
        """Line 541: element not found in app tree."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=None):
            result = AtSpiGetTextTool().execute(name="ghost_label")

        assert result.success is False
        assert "not found" in result.error

    def test_outer_exception(self):
        """Lines 565-566: outer exception caught and returned as error."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element",
                   side_effect=RuntimeError("deep failure")):
            result = AtSpiGetTextTool().execute(name="lbl")

        assert result.success is False
        assert "AT-SPI get text failed" in result.error

    def test_success_via_text_interface(self):
        """Happy path: element found and text interface returns content."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("StatusLabel", text="Hello world")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiGetTextTool().execute(name="Status")

        assert result.success is True
        assert result.output["text"] == "Hello world"

    def test_success_falls_back_to_element_name(self):
        """When text interface raises, falls back to element.name."""
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("MyLabel")
        element.queryText.side_effect = Exception("no text iface")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiGetTextTool().execute(name="MyLabel")

        assert result.success is True
        assert result.output["text"] == "MyLabel"


# ---------------------------------------------------------------------------
# AtSpiSetValueTool — lines 634-636, 655-656, 667-668, 698-699
# ---------------------------------------------------------------------------


class TestAtSpiSetValueToolCoverage:
    def test_pyatspi_missing(self):
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        with patch.dict(sys.modules, {"pyatspi": None}):
            result = AtSpiSetValueTool().execute(name="field", value="hello")
        assert result.success is False
        assert "pyatspi" in result.error

    def test_desktop_exception(self):
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   side_effect=Exception("dbus gone")):
            result = AtSpiSetValueTool().execute(name="field", value="hi")

        assert result.success is False
        assert "AT-SPI desktop" in result.error

    def test_app_not_found_by_name(self):
        """Lines 634-636: named app not found."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiSetValueTool().execute(
                name="input", value="text", app_name="Ghost"
            )

        assert result.success is False
        assert "not found" in result.error

    def test_no_focused_app(self):
        """Lines 655-656: no focused app."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        desktop = MagicMock()
        desktop.childCount = 0

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=desktop):
            result = AtSpiSetValueTool().execute(name="field", value="v")

        assert result.success is False
        assert "No focused application" in result.error

    def test_element_not_found(self):
        """Lines 667-668: element not found."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=None):
            result = AtSpiSetValueTool().execute(name="no_field", value="v")

        assert result.success is False
        assert "not found" in result.error

    def test_outer_exception(self):
        """Lines 698-699: outer try/except catches unexpected errors."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element",
                   side_effect=RuntimeError("unexpected")):
            result = AtSpiSetValueTool().execute(name="field", value="v")

        assert result.success is False
        assert "AT-SPI set value failed" in result.error

    def test_success_via_editable_text_interface(self):
        """Happy path: set value via editable text interface."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("Username")

        # text interface for character count — return a real int so > 0 works
        text_iface = MagicMock()
        text_iface.getCharacterCount.return_value = 4
        # Override the side_effect from _make_accessible so queryText succeeds here
        element.queryText = MagicMock(return_value=text_iface)

        # editable text interface
        editable = MagicMock()
        element.queryEditableText.return_value = editable

        # component interface for focus
        component = MagicMock()
        element.queryComponent.return_value = component

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiSetValueTool().execute(name="Username", value="admin")

        assert result.success is True
        assert result.output["value_set"] == "admin"
        assert result.output["method"] == "editable_text_interface"
        editable.deleteText.assert_called_once_with(0, 4)
        editable.insertText.assert_called_once_with(0, "admin", 5)

    def test_success_via_value_interface_fallback(self):
        """Falls back to value interface when editable text interface unavailable."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("Spinner")

        element.queryText.side_effect = Exception("no text")
        element.queryEditableText.side_effect = Exception("no editable")

        value_iface = MagicMock()
        element.queryValue.return_value = value_iface
        element.queryComponent.side_effect = Exception("no component")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiSetValueTool().execute(name="Spinner", value="42")

        assert result.success is True
        assert result.output["method"] == "value_interface"
        assert value_iface.currentValue == 42.0

    def test_no_interface_returns_error(self):
        """When neither editable text nor value interface available, returns error."""
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        mock_pyatspi = _make_pyatspi_mock()
        app = _make_accessible("App")
        element = _make_accessible("ReadOnly")

        element.queryText.side_effect = Exception("no text")
        element.queryEditableText.side_effect = Exception("no editable")
        element.queryValue.side_effect = Exception("no value")
        element.queryComponent.side_effect = Exception("no component")

        with patch.dict(sys.modules, {"pyatspi": mock_pyatspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop",
                   return_value=MagicMock(childCount=0)), \
             patch("missy.tools.builtin.atspi_tools._get_focused_application",
                   return_value=app), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=element):
            result = AtSpiSetValueTool().execute(name="ReadOnly", value="x")

        assert result.success is False
        assert "editable text" in result.error
