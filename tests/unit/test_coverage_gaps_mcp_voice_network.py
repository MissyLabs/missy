"""Session 16 coverage gap tests.

Targets uncovered paths in:
- missy/mcp/manager.py              (lines 189-195 — _save_config error: fd close + tmp unlink)
- missy/channels/voice/registry.py  (lines 165-171, 172-178 — non-owner and world-writable)
- missy/tools/registry.py           (lines 242-243 — censor_response import exception swallowed)
- missy/policy/network.py           (lines 157-158 — invalid IP string skipped via ValueError)
- missy/channels/voice/server.py    (line 360 — ConnectionClosed before auth logs remote_addr)
- missy/channels/voice/edge_client.py (lines 48-50 — missing websockets exits; main() no creds)
"""
from __future__ import annotations

import importlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.core.events import event_bus


@pytest.fixture(autouse=True)
def _clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# ===========================================================================
# 1. McpManager._save_config — os.write fails: fd must be closed, tmp unlinked
# ===========================================================================
# os and tempfile are imported *inside* _save_config, so patches must target
# the standard library modules directly (not missy.mcp.manager.os).


class TestMcpManagerSaveConfigWriteError:
    """_save_config lines 189-195: when os.write raises the except block closes
    the fd and unlinks the temp file before re-raising."""

    def _make_manager(self, tmp_path: Path):
        from missy.mcp.manager import McpManager

        cfg = tmp_path / "mcp.json"
        return McpManager(config_path=str(cfg))

    def test_fd_closed_and_tmp_unlinked_when_write_fails(self, tmp_path: Path) -> None:
        """When os.write raises OSError the except block closes the fd and unlinks
        the temp file before re-raising the exception."""
        mgr = self._make_manager(tmp_path)

        closed_fds: list[int] = []
        unlinked_paths: list[str] = []

        # Capture a real tmp path, then return a fake fd so we can track close().
        real_mkstemp = tempfile.mkstemp
        fake_fd = 999

        def fake_mkstemp(dir, suffix):
            _, path = real_mkstemp(dir=dir, suffix=suffix)
            return fake_fd, path

        def fake_write(fd, data):
            raise OSError("disk full")

        def fake_close(fd):
            closed_fds.append(fd)

        def fake_unlink(path):
            unlinked_paths.append(path)

        with (
            patch("tempfile.mkstemp", side_effect=fake_mkstemp),
            patch("os.write", side_effect=fake_write),
            patch("os.close", side_effect=fake_close),
            patch("os.unlink", side_effect=fake_unlink),
            patch("os.fchmod"),
            patch("os.replace"),
            pytest.raises(OSError, match="disk full"),
        ):
            mgr._save_config()

        # fd must have been closed exactly once (closed=False path)
        assert fake_fd in closed_fds
        # tmp file must have been unlinked
        assert len(unlinked_paths) == 1

    def test_tmp_unlinked_but_fd_not_reclosed_after_successful_close(
        self, tmp_path: Path
    ) -> None:
        """When os.close succeeds but os.replace then fails, closed=True prevents
        a double-close, and the tmp file is still unlinked."""
        mgr = self._make_manager(tmp_path)

        closed_fds: list[int] = []
        unlinked_paths: list[str] = []
        fake_fd = 777

        real_mkstemp = tempfile.mkstemp

        def fake_mkstemp(dir, suffix):
            _, path = real_mkstemp(dir=dir, suffix=suffix)
            return fake_fd, path

        def fake_close(fd):
            closed_fds.append(fd)

        def fake_unlink(path):
            unlinked_paths.append(path)

        with (
            patch("tempfile.mkstemp", side_effect=fake_mkstemp),
            patch("os.write"),
            patch("os.fchmod"),
            patch("os.close", side_effect=fake_close),
            patch("os.replace", side_effect=OSError("replace failed")),
            patch("os.unlink", side_effect=fake_unlink),
            pytest.raises(OSError, match="replace failed"),
        ):
            mgr._save_config()

        # closed=True when os.replace failed, so fd must NOT be re-closed in except.
        assert closed_fds.count(fake_fd) == 1  # only the successful explicit close
        # tmp must still be unlinked
        assert len(unlinked_paths) == 1

    def test_exception_is_reraised_after_cleanup(self, tmp_path: Path) -> None:
        """The original exception propagates out of _save_config after cleanup."""
        mgr = self._make_manager(tmp_path)

        fake_fd = 555
        real_mkstemp = tempfile.mkstemp

        def fake_mkstemp(dir, suffix):
            _, path = real_mkstemp(dir=dir, suffix=suffix)
            return fake_fd, path

        with (
            patch("tempfile.mkstemp", side_effect=fake_mkstemp),
            patch("os.write", side_effect=PermissionError("no write perm")),
            patch("os.close"),
            patch("os.fchmod"),
            patch("os.replace"),
            patch("os.unlink"),
            pytest.raises(PermissionError, match="no write perm"),
        ):
            mgr._save_config()

    def test_unlink_oserror_is_suppressed(self, tmp_path: Path) -> None:
        """When os.unlink raises OSError the exception does not propagate; only
        the original write error is raised (contextlib.suppress path)."""
        mgr = self._make_manager(tmp_path)

        fake_fd = 444
        real_mkstemp = tempfile.mkstemp

        def fake_mkstemp(dir, suffix):
            _, path = real_mkstemp(dir=dir, suffix=suffix)
            return fake_fd, path

        with (
            patch("tempfile.mkstemp", side_effect=fake_mkstemp),
            patch("os.write", side_effect=OSError("write failed")),
            patch("os.close"),
            patch("os.fchmod"),
            patch("os.replace"),
            patch("os.unlink", side_effect=OSError("unlink also failed")),
            # Only the original write error propagates; unlink OSError is suppressed.
            pytest.raises(OSError, match="write failed"),
        ):
            mgr._save_config()


# ===========================================================================
# 2. DeviceRegistry.load — non-owner and world-writable permission checks
# ===========================================================================


class TestDeviceRegistryPermissionChecks:
    """DeviceRegistry.load permission guards: non-owner and world-writable paths."""

    def _write_registry_file(self, path: Path) -> None:
        path.write_text(json.dumps([]), encoding="utf-8")

    def test_non_owner_file_is_refused(self, tmp_path: Path, caplog) -> None:
        """When st_uid != os.getuid() the registry refuses to load and stays empty."""
        import logging

        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        fake_stat = MagicMock()
        fake_stat.st_uid = os.getuid() + 1  # different owner
        fake_stat.st_mode = stat.S_IFREG | 0o600

        with (
            patch("missy.channels.voice.registry.os.getuid", return_value=os.getuid()),
            caplog.at_level(logging.ERROR, logger="missy.channels.voice.registry"),
        ):
            reg = DeviceRegistry(registry_path=str(reg_file))
            # Patch Path.stat on the specific instance path
            with patch.object(type(reg._path), "stat", return_value=fake_stat):
                reg.load()

        assert reg._nodes == {}
        assert "not owned by current user" in caplog.text

    def test_world_writable_file_is_refused(self, tmp_path: Path, caplog) -> None:
        """When the file has the world-write bit set the registry refuses to load."""
        import logging

        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        fake_stat = MagicMock()
        fake_stat.st_uid = os.getuid()
        fake_stat.st_mode = stat.S_IFREG | 0o646  # world-write bit set

        with (
            patch("missy.channels.voice.registry.os.getuid", return_value=os.getuid()),
            caplog.at_level(logging.ERROR, logger="missy.channels.voice.registry"),
        ):
            reg = DeviceRegistry(registry_path=str(reg_file))
            with patch.object(type(reg._path), "stat", return_value=fake_stat):
                reg.load()

        assert reg._nodes == {}
        assert "group/world-writable" in caplog.text

    def test_group_writable_file_is_refused(self, tmp_path: Path, caplog) -> None:
        """When the file has the group-write bit set the registry refuses to load."""
        import logging

        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        fake_stat = MagicMock()
        fake_stat.st_uid = os.getuid()
        fake_stat.st_mode = stat.S_IFREG | 0o620  # group-write bit set

        with (
            patch("missy.channels.voice.registry.os.getuid", return_value=os.getuid()),
            caplog.at_level(logging.ERROR, logger="missy.channels.voice.registry"),
        ):
            reg = DeviceRegistry(registry_path=str(reg_file))
            with patch.object(type(reg._path), "stat", return_value=fake_stat):
                reg.load()

        assert reg._nodes == {}
        assert "group/world-writable" in caplog.text

    def test_correct_owner_and_perms_loads_normally(self, tmp_path: Path) -> None:
        """A file owned by the current user with safe permissions loads correctly."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        fake_stat = MagicMock()
        fake_stat.st_uid = os.getuid()
        fake_stat.st_mode = stat.S_IFREG | 0o600  # owner rw only — no group/world bits

        with (
            patch("missy.channels.voice.registry.os.getuid", return_value=os.getuid()),
        ):
            reg = DeviceRegistry(registry_path=str(reg_file))
            with patch.object(type(reg._path), "stat", return_value=fake_stat):
                reg.load()

        assert reg._nodes == {}


# ===========================================================================
# 3. ToolRegistry._emit_event — censor_response import exception swallowed
# ===========================================================================


class TestToolRegistryEmitEventCensorFallback:
    """_emit_event lines 242-243: when the inner censor import raises the raw
    detail_msg is used and no exception propagates to the caller."""

    def test_censor_import_error_falls_back_to_raw_message(self) -> None:
        """An ImportError from censor_response is silently swallowed and the
        raw detail_msg appears in the published audit event."""
        from missy.tools.registry import ToolRegistry

        registry = ToolRegistry()
        captured_events: list = []
        event_bus.subscribe("tool_execute", lambda e: captured_events.append(e))

        # Patch the import inside the function body by making the module unavailable.
        with patch.dict(sys.modules, {"missy.security.censor": None}):
            registry._emit_event(
                tool_name="test_tool",
                session_id="s1",
                task_id="t1",
                result="allow",
                detail_msg="raw_message",
            )

        assert len(captured_events) == 1
        detail = captured_events[0].detail
        assert detail["tool"] == "test_tool"
        # Raw message used when censor fails
        assert detail["message"] == "raw_message"

    def test_censor_exception_does_not_propagate(self) -> None:
        """Even when censor is unavailable the emit method does not raise."""
        from missy.tools.registry import ToolRegistry

        registry = ToolRegistry()

        with patch.dict(sys.modules, {"missy.security.censor": None}):
            # Must not raise
            registry._emit_event(
                tool_name="my_tool",
                session_id="s2",
                task_id="t2",
                result="deny",
                detail_msg="some detail",
            )

    def test_outer_emit_exception_logged_not_raised(self, caplog) -> None:
        """When event_bus.publish raises, the outer except logs and does not raise."""
        import logging

        from missy.tools.registry import ToolRegistry

        registry = ToolRegistry()

        with (
            patch("missy.tools.registry.event_bus") as mock_bus,
            caplog.at_level(logging.ERROR, logger="missy.tools.registry"),
        ):
            mock_bus.publish.side_effect = RuntimeError("bus down")
            registry._emit_event(
                tool_name="broken_tool",
                session_id="s3",
                task_id="t3",
                result="error",
                detail_msg="detail",
            )

        assert "Failed to emit audit event" in caplog.text


# ===========================================================================
# 4. NetworkPolicyEngine — invalid IP string skipped via ValueError (lines 157-158)
# ===========================================================================


class TestNetworkPolicyInvalidIPSkipped:
    """Lines 157-158: when ipaddress.ip_address raises ValueError for an address
    string returned by getaddrinfo the entry is skipped via ``continue``."""

    def test_invalid_ip_string_from_getaddrinfo_is_skipped(self) -> None:
        """An unparseable address in getaddrinfo results is skipped; the request
        proceeds and is allowed by the domain allowlist."""
        from missy.policy.network import NetworkPolicy, NetworkPolicyEngine

        policy = NetworkPolicy(
            default_deny=True,
            allowed_domains=["example.com"],
        )
        engine = NetworkPolicyEngine(policy)

        # Return one garbage IP (triggers ValueError → continue) and one valid
        # public IP (passes the private-address check).
        fake_infos = [
            (2, 1, 6, "", ("not_a_valid_ip", 0)),
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]

        with patch("missy.policy.network.socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host("example.com", session_id="s", task_id="t")

        assert result is True

    def test_all_invalid_ips_from_getaddrinfo_allows_via_domain(self) -> None:
        """When every getaddrinfo entry has an unparseable IP the resolved list is
        empty; the private-check loop is skipped and the domain-level allow stands."""
        from missy.policy.network import NetworkPolicy, NetworkPolicyEngine

        policy = NetworkPolicy(
            default_deny=True,
            allowed_domains=["trusted.example.com"],
        )
        engine = NetworkPolicyEngine(policy)

        fake_infos = [
            (2, 1, 6, "", ("garbage_ip_1", 0)),
            (2, 1, 6, "", ("garbage_ip_2", 0)),
        ]

        with patch("missy.policy.network.socket.getaddrinfo", return_value=fake_infos):
            result = engine.check_host(
                "trusted.example.com", session_id="s", task_id="t"
            )

        assert result is True


# ===========================================================================
# 5. VoiceServer — ConnectionClosed before auth logs remote_addr (line 360)
# ===========================================================================


def _make_voice_server():
    """Return a VoiceServer with all dependencies mocked out."""
    from missy.channels.voice.server import VoiceServer

    return VoiceServer(
        registry=MagicMock(),
        pairing_manager=MagicMock(),
        presence_store=MagicMock(),
        stt_engine=MagicMock(),
        tts_engine=MagicMock(),
        agent_callback=AsyncMock(),
    )


class TestVoiceServerConnectionClosedBeforeAuth:
    """Line 360: when ConnectionClosed fires before auth completes and node is None
    the debug message includes the remote address."""

    @pytest.mark.asyncio
    async def test_connection_closed_before_auth_logs_remote_addr(
        self, caplog
    ) -> None:
        """ConnectionClosed raised from websocket.recv() before auth: node is None
        so the line-360 branch logs the remote_addr."""
        import logging

        import websockets.exceptions

        server = _make_voice_server()

        mock_ws = AsyncMock()
        mock_ws.remote_address = ("192.0.2.1", 54321)
        # recv() raises ConnectionClosed immediately — before any auth frame.
        mock_ws.recv.side_effect = websockets.exceptions.ConnectionClosed(
            rcvd=None, sent=None
        )

        with caplog.at_level(logging.DEBUG, logger="missy.channels.voice.server"):
            await server._handle_connection(mock_ws)

        # The log message at line 360 references remote_addr when node is None.
        assert any(
            "before auth" in record.message or "192.0.2.1" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unexpected_exception_triggers_error_response(self, caplog) -> None:
        """A generic exception propagating from the handler causes an error log
        and an attempt to send an error frame (lines 361-373)."""
        import logging

        server = _make_voice_server()

        mock_ws = AsyncMock()
        mock_ws.remote_address = ("192.0.2.2", 54322)
        mock_ws.recv.side_effect = ValueError("unexpected frame")
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        with caplog.at_level(logging.ERROR, logger="missy.channels.voice.server"):
            await server._handle_connection(mock_ws)

        assert any(
            "unexpected error" in record.message.lower()
            for record in caplog.records
        )


# ===========================================================================
# 6. edge_client — missing websockets exits with sys.exit(1) (lines 48-50)
# ===========================================================================


class TestEdgeClientMissingWebsockets:
    """Lines 48-50: when websockets is not installed the module prints an error
    and calls sys.exit(1)."""

    def test_missing_websockets_causes_sys_exit(self) -> None:
        """Re-importing edge_client with websockets absent triggers sys.exit(1)."""
        mod_key = "missy.channels.voice.edge_client"
        ws_key = "websockets"

        saved_mod = sys.modules.pop(mod_key, None)
        saved_ws = sys.modules.get(ws_key)

        # Block websockets import
        sys.modules[ws_key] = None  # type: ignore[assignment]

        try:
            with pytest.raises(SystemExit) as exc_info:
                importlib.import_module(mod_key)
            assert exc_info.value.code == 1
        finally:
            if saved_ws is not None:
                sys.modules[ws_key] = saved_ws
            elif ws_key in sys.modules:
                del sys.modules[ws_key]
            if saved_mod is not None:
                sys.modules[mod_key] = saved_mod
            elif mod_key in sys.modules:
                del sys.modules[mod_key]

    def test_missing_websockets_prints_install_hint(self, capsys) -> None:
        """The error message printed to stderr mentions pip install websockets."""
        mod_key = "missy.channels.voice.edge_client"
        ws_key = "websockets"

        saved_mod = sys.modules.pop(mod_key, None)
        saved_ws = sys.modules.get(ws_key)

        sys.modules[ws_key] = None  # type: ignore[assignment]

        try:
            with pytest.raises(SystemExit):
                importlib.import_module(mod_key)
            captured = capsys.readouterr()
            assert "websockets" in captured.err
        finally:
            if saved_ws is not None:
                sys.modules[ws_key] = saved_ws
            elif ws_key in sys.modules:
                del sys.modules[ws_key]
            if saved_mod is not None:
                sys.modules[mod_key] = saved_mod
            elif mod_key in sys.modules:
                del sys.modules[mod_key]


# ===========================================================================
# 7. edge_client main() — no node_id/token exits with sys.exit(1)
# ===========================================================================


def _call_main_with_args(edge_client_module, args):
    """Invoke edge_client.main() by patching argparse to return our args object."""
    with patch("argparse.ArgumentParser.parse_args", return_value=args):
        edge_client_module.main()


class TestEdgeClientMainMissingCredentials:
    """main() lines 461-467: when node_id or token is absent the process exits."""

    def test_main_exits_when_no_node_id_or_token(self) -> None:
        """main() with no node_id/token and no config exits with code 1."""
        from missy.channels.voice import edge_client

        args = MagicMock()
        args.verbose = False
        args.config = "/nonexistent/path/edge.json"
        args.server = "ws://127.0.0.1:8765"
        args.node_id = None
        args.token = None
        args.pair = False
        args.name = "test-node"
        args.room = "living-room"
        args.duration = 5
        args.sample_rate = 16000
        args.continuous = False

        with (
            patch.object(edge_client, "_load_config", return_value={}),
            patch.object(edge_client, "_save_config"),
            pytest.raises(SystemExit) as exc_info,
        ):
            _call_main_with_args(edge_client, args)

        assert exc_info.value.code == 1

    def test_main_exits_when_token_missing_but_node_id_present(self) -> None:
        """When node_id is set but token is absent sys.exit(1) is called."""
        from missy.channels.voice import edge_client

        args = MagicMock()
        args.verbose = False
        args.config = "/nonexistent/path/edge.json"
        args.server = "ws://127.0.0.1:8765"
        args.node_id = "my-node"
        args.token = None
        args.pair = False
        args.name = "test-node"
        args.room = "living-room"
        args.duration = 5
        args.sample_rate = 16000
        args.continuous = False

        with (
            patch.object(edge_client, "_load_config", return_value={}),
            patch.object(edge_client, "_save_config"),
            pytest.raises(SystemExit) as exc_info,
        ):
            _call_main_with_args(edge_client, args)

        assert exc_info.value.code == 1

    def test_main_exits_when_node_id_missing_but_token_present(self) -> None:
        """When token is set but node_id is absent sys.exit(1) is called."""
        from missy.channels.voice import edge_client

        args = MagicMock()
        args.verbose = False
        args.config = "/nonexistent/path/edge.json"
        args.server = "ws://127.0.0.1:8765"
        args.node_id = None
        args.token = "my-secret-token"
        args.pair = False
        args.name = "test-node"
        args.room = "living-room"
        args.duration = 5
        args.sample_rate = 16000
        args.continuous = False

        with (
            patch.object(edge_client, "_load_config", return_value={}),
            patch.object(edge_client, "_save_config"),
            pytest.raises(SystemExit) as exc_info,
        ):
            _call_main_with_args(edge_client, args)

        assert exc_info.value.code == 1
