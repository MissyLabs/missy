"""AT-SPI accessibility tree tools for GTK/X11 application interaction.

These tools use the AT-SPI2 (Assistive Technology Service Provider Interface)
protocol to inspect and interact with GTK applications running under X11 or
Wayland.  They are more reliable than coordinate-based clicking because they
address UI elements by their semantic accessible name and role rather than by
pixel position.

Requires ``pyatspi``::

    pip install pyatspi

All tools fail gracefully with an informative message when pyatspi is not
installed so they never break the runtime on headless or minimal systems.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_PYATSPI_MISSING = "pyatspi is not installed. Install it with: pip install pyatspi"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_desktop() -> Any:
    """Return the AT-SPI desktop object.

    Returns:
        The ``pyatspi`` desktop accessible object.

    Raises:
        ImportError: When pyatspi is not available.
    """
    import pyatspi  # type: ignore[import]

    return pyatspi.Registry.getDesktop(0)


def _find_application(desktop: Any, app_name: str) -> Any:
    """Locate an application on the AT-SPI desktop by name.

    Args:
        desktop: The AT-SPI desktop object.
        app_name: Case-insensitive application name to search for.

    Returns:
        The first matching application accessible, or ``None``.
    """
    lower_name = app_name.lower()
    for i in range(desktop.childCount):
        try:
            child = desktop.getChildAtIndex(i)
            if child and child.name and lower_name in child.name.lower():
                return child
        except Exception:  # noqa: BLE001
            continue
    return None


def _get_focused_application(desktop: Any) -> Any:
    """Return the currently focused application on the desktop.

    Iterates over desktop children and returns the first one whose state
    set includes ``STATE_ACTIVE``. Some GTK/GNOME combinations never expose
    that state on the application root, so a second pass finds an application
    with a descendant carrying ``STATE_FOCUSED``.

    Args:
        desktop: The AT-SPI desktop object.

    Returns:
        The focused application accessible, or ``None``.
    """
    import pyatspi  # type: ignore[import]

    desktop_children: list[Any] = []
    for child_index in range(desktop.childCount):
        try:
            desktop_children.append(desktop.getChildAtIndex(child_index))
        except Exception:  # noqa: BLE001
            continue

    def _deepest_focused_node(node: Any, depth: int = 0) -> int:
        # Keep traversal bounded for malformed or cyclic accessibility trees.
        if node is None or depth > 32:
            return -1
        deepest = -1
        try:
            if node.getState().contains(pyatspi.STATE_FOCUSED):
                deepest = depth
        except Exception:  # noqa: BLE001
            logger.debug("Unable to inspect AT-SPI focus state", exc_info=True)
        try:
            child_count = int(node.childCount)
        except Exception:  # noqa: BLE001
            return deepest
        for child_index in range(min(max(child_count, 0), 4096)):
            try:
                deepest = max(
                    deepest,
                    _deepest_focused_node(node.getChildAtIndex(child_index), depth + 1),
                )
            except Exception:  # noqa: BLE001
                continue
        return deepest

    for child in desktop_children:
        try:
            if child is None:
                continue
            state_set = child.getState()
            if state_set.contains(pyatspi.STATE_ACTIVE):
                return child
        except Exception:  # noqa: BLE001
            continue

    # AT-SPI focus states can remain set on several applications at once. On
    # X11, map the actual active window title back to the application tree
    # before falling back to heuristic focus depth. This is a fixed read-only
    # query; no model-controlled shell text is involved.
    active_window_title = ""
    try:
        active_window_env = os.environ.copy()
        if not active_window_env.get("DISPLAY"):
            for display_number in range(10):
                if os.path.exists(f"/tmp/.X11-unix/X{display_number}"):
                    active_window_env["DISPLAY"] = f":{display_number}"
                    break
        completed = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env=active_window_env,
        )
        if completed.returncode == 0:
            active_window_title = completed.stdout.strip().casefold()
    except (OSError, subprocess.SubprocessError):
        pass

    def _contains_named_window(node: Any, depth: int = 0) -> bool:
        if node is None or depth > 32:
            return False
        try:
            node_name = node.name
            if (
                active_window_title
                and isinstance(node_name, str)
                and node_name.strip().casefold() == active_window_title
            ):
                return True
        except Exception:  # noqa: BLE001
            logger.debug("Unable to inspect AT-SPI node name", exc_info=True)
        try:
            child_count = int(node.childCount)
        except Exception:  # noqa: BLE001
            return False
        for child_index in range(min(max(child_count, 0), 4096)):
            try:
                if _contains_named_window(node.getChildAtIndex(child_index), depth + 1):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    if active_window_title:
        title_matches: list[tuple[int, int, Any]] = []
        for child in desktop_children:
            try:
                if _contains_named_window(child):
                    try:
                        child_name = str(child.name or "").casefold()
                    except Exception:  # noqa: BLE001
                        child_name = ""
                    is_decorator = int("mutter" in child_name or child_name.endswith("-frames"))
                    title_matches.append((_deepest_focused_node(child), -is_decorator, child))
            except Exception:  # noqa: BLE001
                continue
        if title_matches:
            return max(title_matches, key=lambda candidate: candidate[:2])[2]

    focused_app = None
    focused_depth = -1
    for child in desktop_children:
        try:
            candidate_depth = _deepest_focused_node(child)
            if candidate_depth > focused_depth:
                focused_app = child
                focused_depth = candidate_depth
        except Exception:  # noqa: BLE001
            continue
    if focused_app is not None:
        return focused_app

    # Fall back to first non-None child
    for child in desktop_children:
        if child is not None:
            return child
    return None


def _walk_tree(node, max_depth: int, current_depth: int = 0) -> list[dict[str, Any]]:
    """Recursively walk the AT-SPI accessibility tree.

    Args:
        node: Current accessible node.
        max_depth: Maximum tree depth to descend.
        current_depth: Current recursion depth (starts at 0).

    Returns:
        A list of dicts describing each node in subtree order.
    """
    if node is None or current_depth > max_depth:
        return []

    try:
        role_name = node.getRoleName() if hasattr(node, "getRoleName") else ""
        node_name = node.name or ""

        # Collect text content if available
        text_content = ""
        try:
            text_iface = node.queryText()
            text_content = text_iface.getText(0, -1)
        except Exception:  # noqa: BLE001
            logger.debug("AT-SPI text query failed", exc_info=True)

        # Collect state flags of interest
        try:
            state_set = node.getState()
            import pyatspi  # type: ignore[import]

            states: list[str] = []
            for state_name, state_val in [
                ("focusable", pyatspi.STATE_FOCUSABLE),
                ("focused", pyatspi.STATE_FOCUSED),
                ("sensitive", pyatspi.STATE_SENSITIVE),
                ("editable", pyatspi.STATE_EDITABLE),
                ("checked", pyatspi.STATE_CHECKED),
                ("selected", pyatspi.STATE_SELECTED),
                ("visible", pyatspi.STATE_VISIBLE),
                ("showing", pyatspi.STATE_SHOWING),
            ]:
                try:
                    if state_set.contains(state_val):
                        states.append(state_name)
                except Exception:  # noqa: BLE001
                    logger.debug("AT-SPI state query failed for %s", state_name, exc_info=True)
        except Exception:  # noqa: BLE001
            states = []

        entry: dict[str, Any] = {
            "depth": current_depth,
            "role": role_name,
            "name": node_name,
            "states": states,
        }
        if text_content:
            entry["text"] = text_content

        result = [entry]

        if current_depth < max_depth:
            child_count = node.childCount if hasattr(node, "childCount") else 0
            for i in range(child_count):
                try:
                    child = node.getChildAtIndex(i)
                    result.extend(_walk_tree(child, max_depth, current_depth + 1))
                except Exception:  # noqa: BLE001
                    continue

        return result

    except Exception:  # noqa: BLE001
        return []


def _format_tree(nodes: list[dict[str, Any]]) -> str:
    """Format a flat node list into an indented tree string.

    Args:
        nodes: Output of :func:`_walk_tree`.

    Returns:
        A human-readable, indented string representation.
    """
    lines: list[str] = []
    for node in nodes:
        indent = "  " * node["depth"]
        role = node.get("role", "")
        name = node.get("name", "")
        states = node.get("states", [])
        text = node.get("text", "")

        parts = [f"{indent}[{role}]"]
        if name:
            parts.append(f" {name!r}")
        if text and text != name:
            preview = text[:60].replace("\n", " ")
            parts.append(f" text={preview!r}")
        if states:
            parts.append(f" ({', '.join(states)})")

        lines.append("".join(parts))

    return "\n".join(lines)


def _find_element(
    app,
    name: str | None,
    role: str | None,
    max_depth: int = 20,
):
    """Search an application's accessibility tree for a matching element.

    Matching is case-insensitive.  If both ``name`` and ``role`` are given,
    both must match.  If only one is given, only that attribute is checked.

    Args:
        app: AT-SPI application accessible.
        name: Accessible name to search for, or ``None``.
        role: Role name to search for, or ``None``.
        max_depth: Maximum tree depth to search. Real GTK4 applications
            nest interactive elements much deeper than the container
            structure a quick manual inspection suggests -- live
            verification against a real, running gnome-calculator found
            its push buttons at depth 11, one level beyond the previous
            default of 10, which made every button invisible to
            ``atspi_click``/``atspi_set_value`` ("Element not found")
            despite being present, named, and exposed correctly in the
            real accessibility tree. 20 gives comfortable real-world
            headroom without meaningfully changing search cost (bounded
            by actual child counts, not exponential in depth).

    Returns:
        The first matching accessible, or ``None``.
    """
    lower_name = name.lower() if name else None
    lower_role = role.lower() if role else None

    def _search(node: Any, depth: int) -> Any:
        if node is None or depth > max_depth:
            return None
        try:
            node_role = (node.getRoleName() or "").lower() if hasattr(node, "getRoleName") else ""
            node_name = (node.name or "").lower()

            name_match = lower_name is None or lower_name in node_name
            role_match = lower_role is None or lower_role in node_role

            if name_match and role_match and (lower_name is not None or lower_role is not None):
                return node

            child_count = node.childCount if hasattr(node, "childCount") else 0
            for i in range(child_count):
                try:
                    child = node.getChildAtIndex(i)
                    found = _search(child, depth + 1)
                    if found is not None:
                        return found
                except Exception:  # noqa: BLE001
                    logger.debug("AT-SPI child access failed at index %d", i, exc_info=True)
                    continue
        except Exception:  # noqa: BLE001
            logger.debug("AT-SPI element search failed", exc_info=True)
        return None

    return _search(app, 0)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class AtSpiGetTreeTool(BaseTool):
    """Inspect the accessibility tree of a GTK/X11 application.

    Returns a formatted tree of all visible roles, names, text, and states
    so the agent can understand what UI elements are present before acting.
    """

    name = "atspi_get_tree"
    description = (
        "Get the accessibility tree of the focused application or a named application. "
        "Shows all clickable buttons, inputs, labels, and menus. "
        "Use this to understand an app's UI before interacting."
    )
    permissions = ToolPermissions(shell=False)

    parameters: dict[str, Any] = {
        "app_name": {
            "type": "string",
            "description": "Name of the application to inspect (optional; uses focused app if omitted).",
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum tree depth to traverse (default 10, max 20).",
            "default": 10,
        },
    }

    def execute(
        self,
        *,
        app_name: str = "",
        max_depth: int = 10,
        **_: Any,
    ) -> ToolResult:
        try:
            import pyatspi  # type: ignore[import] # noqa: F401
        except ImportError:
            return ToolResult(success=False, output=None, error=_PYATSPI_MISSING)

        max_depth = min(max(int(max_depth), 1), 20)

        try:
            desktop = _get_desktop()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"Could not connect to AT-SPI desktop: {exc}",
            )

        try:
            if app_name:
                app = _find_application(desktop, app_name)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Application {app_name!r} not found on the desktop.",
                    )
            else:
                app = _get_focused_application(desktop)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No focused application found. Pass app_name to specify one.",
                    )

            nodes = _walk_tree(app, max_depth=max_depth)
            tree_str = _format_tree(nodes)

            return ToolResult(
                success=True,
                output={
                    "app_name": app.name or app_name,
                    "node_count": len(nodes),
                    "tree": tree_str,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to read accessibility tree: {exc}",
            )


class AtSpiClickTool(BaseTool):
    """Click a UI element in a GTK application by accessible name and/or role.

    Uses AT-SPI action interface (``doAction(0)``), which maps to the
    element's primary action (activate, click, etc.).  More reliable than
    coordinate clicking for GTK apps.
    """

    name = "atspi_click"
    description = (
        "Click a UI element in an application by its accessible name and/or role. "
        "More reliable than coordinate clicking for GTK apps."
    )
    permissions = ToolPermissions(shell=False)

    parameters: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "Accessible name of the element to click (case-insensitive partial match).",
        },
        "role": {
            "type": "string",
            "description": "ARIA/AT-SPI role, e.g. 'push button', 'menu item', 'check box'.",
        },
        "app_name": {
            "type": "string",
            "description": "Application name (optional; uses focused app if omitted).",
        },
    }

    def execute(
        self,
        *,
        name: str = "",
        role: str = "",
        app_name: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            import pyatspi  # type: ignore[import] # noqa: F401
        except ImportError:
            return ToolResult(success=False, output=None, error=_PYATSPI_MISSING)

        if not name and not role:
            return ToolResult(
                success=False,
                output=None,
                error="At least one of 'name' or 'role' must be provided.",
            )

        try:
            desktop = _get_desktop()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"Could not connect to AT-SPI desktop: {exc}",
            )

        try:
            if app_name:
                app = _find_application(desktop, app_name)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Application {app_name!r} not found.",
                    )
            else:
                app = _get_focused_application(desktop)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No focused application. Pass app_name to specify one.",
                    )

            element = _find_element(app, name or None, role or None)
            if element is None:
                desc = []
                if name:
                    desc.append(f"name={name!r}")
                if role:
                    desc.append(f"role={role!r}")
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Element not found: {', '.join(desc)} in {app.name!r}.",
                )

            action_iface = element.queryAction()
            action_iface.doAction(0)

            return ToolResult(
                success=True,
                output={
                    "clicked": element.name,
                    "role": element.getRoleName() if hasattr(element, "getRoleName") else "",
                    "app": app.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"AT-SPI click failed: {exc}",
            )


class AtSpiGetTextTool(BaseTool):
    """Get text content from a GTK application's UI element."""

    name = "atspi_get_text"
    description = "Get text content from an application's UI element by name or role."
    permissions = ToolPermissions(shell=False)

    parameters: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "Accessible name of the element (case-insensitive partial match).",
        },
        "role": {
            "type": "string",
            "description": "AT-SPI role of the element, e.g. 'text', 'label', 'entry'.",
        },
        "app_name": {
            "type": "string",
            "description": "Application name (optional; uses focused app if omitted).",
        },
    }

    def execute(
        self,
        *,
        name: str = "",
        role: str = "",
        app_name: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            import pyatspi  # type: ignore[import] # noqa: F401
        except ImportError:
            return ToolResult(success=False, output=None, error=_PYATSPI_MISSING)

        if not name and not role:
            return ToolResult(
                success=False,
                output=None,
                error="At least one of 'name' or 'role' must be provided.",
            )

        try:
            desktop = _get_desktop()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"Could not connect to AT-SPI desktop: {exc}",
            )

        try:
            if app_name:
                app = _find_application(desktop, app_name)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Application {app_name!r} not found.",
                    )
            else:
                app = _get_focused_application(desktop)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No focused application. Pass app_name to specify one.",
                    )

            element = _find_element(app, name or None, role or None)
            if element is None:
                desc = []
                if name:
                    desc.append(f"name={name!r}")
                if role:
                    desc.append(f"role={role!r}")
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Element not found: {', '.join(desc)} in {app.name!r}.",
                )

            # Try AT-SPI text interface first; fall back to element.name
            text_content = ""
            try:
                text_iface = element.queryText()
                text_content = text_iface.getText(0, -1)
            except Exception:  # noqa: BLE001
                text_content = element.name or ""

            return ToolResult(
                success=True,
                output={
                    "text": text_content,
                    "element_name": element.name,
                    "role": element.getRoleName() if hasattr(element, "getRoleName") else "",
                    "app": app.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"AT-SPI get text failed: {exc}",
            )


class AtSpiSetValueTool(BaseTool):
    """Set the value of a text input or editable field in a GTK application.

    Locates the element by accessible name, focuses it via AT-SPI, then
    sets its value using the AT-SPI editable text interface.  Falls back to
    xdotool type when the editable text interface is unavailable.
    """

    name = "atspi_set_value"
    description = "Set the value of a text input or editable field in a GTK application."
    permissions = ToolPermissions(shell=False)

    parameters: dict[str, Any] = {
        "name": {
            "type": "string",
            "description": "Accessible name of the input field (case-insensitive partial match).",
            "required": True,
        },
        "value": {
            "type": "string",
            "description": "The value to set.",
            "required": True,
        },
        "app_name": {
            "type": "string",
            "description": "Application name (optional; uses focused app if omitted).",
        },
    }

    def execute(
        self,
        *,
        name: str,
        value: str,
        app_name: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            import pyatspi  # type: ignore[import]  # noqa: F401
        except ImportError:
            return ToolResult(success=False, output=None, error=_PYATSPI_MISSING)

        try:
            desktop = _get_desktop()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"Could not connect to AT-SPI desktop: {exc}",
            )

        try:
            if app_name:
                app = _find_application(desktop, app_name)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Application {app_name!r} not found.",
                    )
            else:
                app = _get_focused_application(desktop)
                if app is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No focused application. Pass app_name to specify one.",
                    )

            # Labels and their inputs commonly share the same accessible name.
            # Prefer text/entry roles so a name-only search does not stop on
            # the read-only label before reaching the editable control.
            element = (
                _find_element(app, name, role="text")
                or _find_element(app, name, role="entry")
                or _find_element(app, name, role=None)
            )
            if element is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Element {name!r} not found in {app.name!r}.",
                )

            # Focus the element via AT-SPI component interface
            try:
                component = element.queryComponent()
                component.grabFocus()
            except Exception:  # noqa: BLE001
                logger.debug("AT-SPI grabFocus failed for %s", name, exc_info=True)

            # Try AT-SPI editable text interface
            set_via: str
            try:
                editable = element.queryEditableText()
                # Clear existing text first
                current_len = 0
                try:
                    text_iface = element.queryText()
                    character_count = getattr(text_iface, "characterCount", None)
                    if isinstance(character_count, int):
                        current_len = character_count
                    else:
                        current_len = text_iface.getCharacterCount()
                except Exception:  # noqa: BLE001
                    logger.debug("AT-SPI character count query failed", exc_info=True)
                if current_len > 0:
                    editable.deleteText(0, current_len)
                editable.insertText(0, value, len(value))
                set_via = "editable_text_interface"
            except Exception:  # noqa: BLE001
                # Fall back to AT-SPI value interface (e.g. spin buttons)
                try:
                    value_iface = element.queryValue()
                    value_iface.currentValue = float(value)
                    set_via = "value_interface"
                except Exception:  # noqa: BLE001
                    return ToolResult(
                        success=False,
                        output=None,
                        error=(
                            f"Element {name!r} does not support editable text or value interface. "
                            "Try atspi_click to focus it first, then use x11_type to type into it."
                        ),
                    )

            verified_value: bool | None = None
            readback: str | None = None
            try:
                observed = element.queryText().getText(0, -1)
                if isinstance(observed, str):
                    readback = observed
                    verified_value = observed == value
            except Exception:  # noqa: BLE001
                logger.debug("AT-SPI value readback failed for %s", name, exc_info=True)

            if verified_value is False:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        f"AT-SPI reported setting {name!r}, but readback was {readback!r} "
                        f"instead of {value!r}."
                    ),
                )

            return ToolResult(
                success=True,
                output={
                    "element_name": element.name,
                    "value_set": value,
                    "method": set_via,
                    "app": app.name,
                    "verified": verified_value,
                    "readback": readback,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                error=f"AT-SPI set value failed: {exc}",
            )
