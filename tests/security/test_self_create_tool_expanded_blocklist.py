"""Tests for session 25 security fixes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Self-create tool: expanded blocklist
# ---------------------------------------------------------------------------


class TestSelfCreateToolBlocklist:
    """Self-create tool rejects newly added dangerous patterns."""

    def _create(self, script: str, name: str = "test_tool"):
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        tool = SelfCreateTool()
        return tool.execute(
            action="create",
            tool_name=name,
            tool_description="test",
            language="python",
            script=script,
        )

    def test_blocks_builtins_access(self):
        result = self._create("x = __builtins__.__dict__['exec']")
        assert not result.success
        assert "__builtins__" in result.error

    def test_blocks_open(self):
        result = self._create("f = open('/etc/passwd')")
        assert not result.success
        assert "open(" in result.error

    def test_blocks_os_exec(self):
        result = self._create("os.execv('/bin/sh', ['sh'])")
        assert not result.success
        assert "os.exec" in result.error

    def test_blocks_os_fork(self):
        result = self._create("pid = os.fork()")
        assert not result.success
        assert "os.fork" in result.error

    def test_blocks_os_spawn(self):
        result = self._create("os.spawnl(os.P_WAIT, '/bin/ls')")
        assert not result.success
        assert "os.spawn" in result.error

    def test_blocks_os_popen(self):
        result = self._create("p = os.popen('ls')")
        assert not result.success
        # May match "open(" or "os.popen(" — both are blocked
        assert "dangerous pattern" in result.error.lower() or "open(" in result.error

    def test_blocks_shutil_rmtree(self):
        result = self._create("import shutil\nshutil.rmtree('/tmp/data')")
        assert not result.success
        assert "shutil.rmtree" in result.error

    def test_blocks_shutil_move(self):
        result = self._create("shutil.move('/etc/passwd', '/tmp/stolen')")
        assert not result.success
        assert "shutil.move" in result.error

    def test_blocks_os_remove(self):
        result = self._create("os.remove('/etc/passwd')")
        assert not result.success
        assert "os.remove(" in result.error

    def test_blocks_os_unlink(self):
        result = self._create("os.unlink('/var/log/auth.log')")
        assert not result.success
        assert "os.unlink(" in result.error

    def test_blocks_os_rmdir(self):
        result = self._create("os.rmdir('/tmp/mydir')")
        assert not result.success
        assert "os.rmdir(" in result.error

    def test_allows_safe_script(self):
        """Safe scripts should still be accepted."""
        result = self._create("print('hello world')\nresult = 2 + 2")
        # May fail for other reasons (no dir), but not for pattern match
        if not result.success:
            assert "dangerous pattern" not in (result.error or "").lower()


# ---------------------------------------------------------------------------
# X11 tools: integer coercion
# ---------------------------------------------------------------------------


class TestX11IntegerCoercion:
    """X11 tools enforce integer types for coordinate/delay params."""

    def test_click_rejects_string_x(self):
        from missy.tools.builtin.x11_tools import X11ClickTool

        tool = X11ClickTool()
        with pytest.raises((ValueError, TypeError)):
            tool.execute(x="0; rm -rf /", y=100)

    def test_click_rejects_string_y(self):
        from missy.tools.builtin.x11_tools import X11ClickTool

        tool = X11ClickTool()
        with pytest.raises((ValueError, TypeError)):
            tool.execute(x=100, y="0; rm -rf /")

    def test_click_accepts_integer_params(self):
        from missy.tools.builtin.x11_tools import X11ClickTool

        tool = X11ClickTool()
        with patch("missy.tools.builtin.x11_tools._run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            tool.execute(x=100, y=200)
            # Should have called _run with integer values in the command
            cmd = mock_run.call_args[0][0]
            assert "100" in cmd
            assert "200" in cmd

    def test_type_rejects_string_delay(self):
        from missy.tools.builtin.x11_tools import X11TypeTool

        tool = X11TypeTool()
        with pytest.raises((ValueError, TypeError)):
            tool.execute(text="hello", delay_ms="12; rm -rf /")

    def test_type_accepts_integer_delay(self):
        from missy.tools.builtin.x11_tools import X11TypeTool

        tool = X11TypeTool()
        with patch("missy.tools.builtin.x11_tools._run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            tool.execute(text="hello", delay_ms=50)
            cmd = mock_run.call_args[0][0]
            assert "--delay 50" in cmd


# ---------------------------------------------------------------------------
# FallbackSandbox: environment sanitization
# ---------------------------------------------------------------------------


class TestFallbackSandboxEnv:
    """FallbackSandbox sanitizes subprocess environment."""

    def test_env_excludes_api_keys(self):
        from missy.security.sandbox import FallbackSandbox, SandboxConfig

        config = SandboxConfig()
        sandbox = FallbackSandbox(config)

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-secret", "PATH": "/usr/bin"}),
            patch("missy.security.sandbox.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("echo hello")

            # Check that env was passed and doesn't contain API key
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs.get("env", {})
            assert "ANTHROPIC_API_KEY" not in env
            assert "PATH" in env

    def test_env_includes_safe_vars(self):
        from missy.security.sandbox import FallbackSandbox, SandboxConfig

        config = SandboxConfig()
        sandbox = FallbackSandbox(config)

        with (
            patch.dict(
                os.environ,
                {"HOME": "/home/test", "LANG": "en_US.UTF-8", "OPENAI_API_KEY": "sk-xxx"},
            ),
            patch("missy.security.sandbox.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("echo test")

            call_kwargs = mock_run.call_args[1]
            env = call_kwargs.get("env", {})
            assert "HOME" in env
            assert "LANG" in env
            assert "OPENAI_API_KEY" not in env


# ---------------------------------------------------------------------------
# Code evolution: path traversal prevention
# ---------------------------------------------------------------------------


class TestCodeEvolutionPathTraversal:
    """Code evolution blocks path traversal in diff file paths."""

    def test_traversal_blocked(self, tmp_path):
        from missy.agent.code_evolution import (
            CodeEvolutionManager,
            EvolutionProposal,
            EvolutionStatus,
            EvolutionTrigger,
            FileDiff,
        )

        repo = tmp_path / "repo"
        repo.mkdir()

        # Create a target file outside the repo
        target = tmp_path / "etc" / "cron.d"
        target.mkdir(parents=True)
        (target / "backdoor").write_text("original")

        engine = CodeEvolutionManager.__new__(CodeEvolutionManager)
        engine._repo_root = repo
        engine._test_command = "true"
        engine._proposals = []
        engine._lock = __import__("threading").Lock()
        engine._save = MagicMock()
        engine._emit_event = MagicMock()
        engine._stash_if_dirty = MagicMock(return_value=False)
        engine._unstash = MagicMock()
        engine._revert_diffs = MagicMock()
        engine._validate_diffs = MagicMock()  # Skip validation

        # Create a proposal with traversal path
        diff = FileDiff(
            file_path="../../etc/cron.d/backdoor",
            original_code="original",
            proposed_code="malicious",
        )
        prop = EvolutionProposal(
            id="test-1",
            title="test title",
            description="test",
            diffs=[diff],
            status=EvolutionStatus.APPROVED,
            trigger=EvolutionTrigger.USER_REQUEST,
            confidence=1.0,
        )
        engine._proposals = [prop]
        engine._find = MagicMock(return_value=prop)

        result = engine.apply("test-1")
        assert result["success"] is False
        assert "Path traversal" in result["message"]

    def test_safe_path_allowed(self, tmp_path):
        from missy.agent.code_evolution import (
            CodeEvolutionManager,
            EvolutionProposal,
            EvolutionStatus,
            EvolutionTrigger,
            FileDiff,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("old_code")

        engine = CodeEvolutionManager.__new__(CodeEvolutionManager)
        engine._repo_root = repo
        engine._test_command = "true"
        engine._proposals = []
        engine._lock = __import__("threading").Lock()
        engine._save = MagicMock()
        engine._emit_event = MagicMock()
        engine._stash_if_dirty = MagicMock(return_value=False)
        engine._unstash = MagicMock()
        engine._validate_diffs = MagicMock()
        engine._git = MagicMock(return_value="abc123")

        diff = FileDiff(
            file_path="test.py",
            original_code="old_code",
            proposed_code="new_code",
        )
        prop = EvolutionProposal(
            id="test-2",
            title="test title",
            description="test",
            diffs=[diff],
            status=EvolutionStatus.APPROVED,
            trigger=EvolutionTrigger.USER_REQUEST,
            confidence=1.0,
        )
        engine._proposals = [prop]
        engine._find = MagicMock(return_value=prop)

        with patch("missy.agent.code_evolution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = engine.apply("test-2")

        # Should succeed (not blocked by path traversal)
        assert result.get("success") is not False or "Path traversal" not in result.get(
            "message", ""
        )


# ---------------------------------------------------------------------------
# Code evolution: environment sanitization
# ---------------------------------------------------------------------------


class TestCodeEvolutionEnvSanitization:
    """Code evolution test subprocess uses sanitized environment."""

    def test_test_command_env_excludes_api_keys(self, tmp_path):
        from missy.agent.code_evolution import (
            CodeEvolutionManager,
            EvolutionProposal,
            EvolutionStatus,
            EvolutionTrigger,
            FileDiff,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test.py").write_text("old_code")

        engine = CodeEvolutionManager.__new__(CodeEvolutionManager)
        engine._repo_root = repo
        engine._test_command = "python -m pytest"
        engine._proposals = []
        engine._lock = __import__("threading").Lock()
        engine._save = MagicMock()
        engine._emit_event = MagicMock()
        engine._stash_if_dirty = MagicMock(return_value=False)
        engine._unstash = MagicMock()
        engine._validate_diffs = MagicMock()  # Skip validation
        engine._git = MagicMock(return_value="abc123")

        diff = FileDiff(
            file_path="test.py",
            original_code="old_code",
            proposed_code="new_code",
        )
        prop = EvolutionProposal(
            id="test-env",
            title="test title",
            description="test",
            diffs=[diff],
            status=EvolutionStatus.APPROVED,
            trigger=EvolutionTrigger.USER_REQUEST,
            confidence=1.0,
        )
        engine._proposals = [prop]
        engine._find = MagicMock(return_value=prop)

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret", "PATH": "/usr/bin"}),
            patch("missy.agent.code_evolution.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            engine.apply("test-env")

            call_kwargs = mock_run.call_args[1]
            env = call_kwargs.get("env", {})
            assert "ANTHROPIC_API_KEY" not in env
            assert "PATH" in env
