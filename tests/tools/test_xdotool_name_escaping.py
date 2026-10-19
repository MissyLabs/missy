"""xdotool --name arguments are POSIX ERE; window titles must be escaped.

Live validation: ``desktop_focus_window(window_name="Report (Draft) - Writer")``
never matched because xdotool compiled ``(Draft)`` as a regex group.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
from unittest.mock import patch

import pytest

from missy.tools.builtin._desktop_shared import xdotool_literal_pattern

TITLE = "Report (Draft) - Writer"


def _posix_ere_icase_match(pattern: str, text: str) -> bool:
    """Match like libxdo: glibc regcomp(REG_EXTENDED | REG_ICASE)."""
    name = ctypes.util.find_library("c")
    if not name:
        pytest.skip("libc not found")
    libc = ctypes.CDLL(name)
    buf = ctypes.create_string_buffer(1024)  # >= sizeof(regex_t)
    assert libc.regcomp(buf, pattern.encode(), 1 | 2) == 0
    try:
        return libc.regexec(buf, text.encode(), 0, None, 0) == 0
    finally:
        libc.regfree(buf)


@pytest.mark.parametrize(
    "title",
    [
        TITLE,
        "a.b [x] {1} $5 ^ | + ? * \\ end",
        "Price: $10 (approx.)",
        "]} plain",
        "Untitled - Mousepad",
    ],
)
def test_escaped_title_matches_itself_literally(title):
    pat = xdotool_literal_pattern(title)
    assert _posix_ere_icase_match(pat, title)
    assert _posix_ere_icase_match(pat, "prefix " + title.upper() + " suffix")


def test_dot_is_not_a_wildcard():
    assert not _posix_ere_icase_match(xdotool_literal_pattern("a.b"), "axb")


def test_raw_title_with_parentheses_does_not_match():
    # Documents the bug: unescaped, "(Draft)" is a group and never matches.
    assert not _posix_ere_icase_match(TITLE, TITLE)


def _ok(cmd, *a, **k):
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_desktop_focus_window_escapes_but_allowlist_sees_raw():
    from missy.tools.builtin import desktop_tools

    seen: list[str] = []
    with (
        patch.object(desktop_tools, "_check_rate_limit", return_value=None),
        patch.object(
            desktop_tools,
            "check_window_allowed",
            side_effect=lambda n: seen.append(n),
        ),
        patch.object(desktop_tools.subprocess, "run", side_effect=_ok) as run,
    ):
        result = desktop_tools.DesktopFocusWindowTool().execute(window_name=TITLE)
    assert result.success
    assert seen == [TITLE]
    argv = run.call_args.args[0]
    assert argv[argv.index("--name") + 1] == r"Report \(Draft\) - Writer"


@pytest.mark.parametrize("tool_name", ["X11ClickTool", "X11TypeTool", "X11KeyTool"])
def test_x11_tools_escape_window_name(tool_name):
    from missy.tools.builtin import x11_tools

    cmds: list[str] = []

    def fake_run(cmd, **_):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    kwargs = {"window_name": TITLE}
    if tool_name == "X11ClickTool":
        kwargs.update(x=1, y=1)
    elif tool_name == "X11TypeTool":
        kwargs.update(text="hi")
    else:
        kwargs.update(key="Return")
    with (
        patch.object(x11_tools, "_check_rate_limit", return_value=None),
        patch.object(x11_tools, "_check_window_allowed", return_value=None),
        patch.object(x11_tools, "_run", side_effect=fake_run),
    ):
        getattr(x11_tools, tool_name)().execute(**kwargs)
    focus = [c for c in cmds if "search --name" in c]
    assert focus, cmds
    assert r"'Report \(Draft\) - Writer'" in focus[0]
