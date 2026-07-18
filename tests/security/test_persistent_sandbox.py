"""Tests for F08 — PersistentContainerSandbox activation via get_sandbox."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.security.sandbox import (
    DockerSandbox,
    FallbackSandbox,
    PersistentContainerSandbox,
    RefusingSandbox,
    SandboxConfig,
    get_sandbox,
)


class TestConfigFlag:
    def test_persistent_defaults_false(self) -> None:
        assert SandboxConfig().persistent is False

    def test_parsed_from_yaml_section(self) -> None:
        from missy.security.sandbox import parse_sandbox_config

        assert parse_sandbox_config({"enabled": True, "persistent": True}).persistent is True
        assert parse_sandbox_config({"enabled": True}).persistent is False


class TestGetSandboxSelection:
    def test_persistent_selected_when_available(self) -> None:
        cfg = SandboxConfig(enabled=True, persistent=True)
        with patch.object(PersistentContainerSandbox, "is_available", return_value=True):
            sb = get_sandbox(cfg)
        assert isinstance(sb, PersistentContainerSandbox)

    def test_falls_back_to_docker_when_persistent_unavailable(self) -> None:
        cfg = SandboxConfig(enabled=True, persistent=True, require_isolation=False)
        with (
            patch.object(PersistentContainerSandbox, "is_available", return_value=False),
            patch.object(DockerSandbox, "is_available", return_value=True),
        ):
            sb = get_sandbox(cfg)
        assert isinstance(sb, DockerSandbox)

    def test_persistent_unavailable_require_isolation_refuses(self) -> None:
        cfg = SandboxConfig(enabled=True, persistent=True, require_isolation=True)
        with (
            patch.object(PersistentContainerSandbox, "is_available", return_value=False),
            patch.object(DockerSandbox, "is_available", return_value=False),
        ):
            sb = get_sandbox(cfg)
        assert isinstance(sb, RefusingSandbox)

    def test_non_persistent_uses_docker_path(self) -> None:
        cfg = SandboxConfig(enabled=True, persistent=False)
        with (
            patch.object(PersistentContainerSandbox, "is_available", return_value=True),
            patch.object(DockerSandbox, "is_available", return_value=True),
        ):
            sb = get_sandbox(cfg)
        assert isinstance(sb, DockerSandbox)  # persistent not requested

    def test_disabled_returns_fallback(self) -> None:
        assert isinstance(get_sandbox(SandboxConfig(enabled=False)), FallbackSandbox)

    def test_magicmock_config_does_not_select_persistent(self) -> None:
        # `persistent` truthy-Mock must not select the persistent path (the same
        # is-True guard lesson as the Landlock gate).
        cfg = MagicMock()
        cfg.enabled = True
        # cfg.persistent is a truthy MagicMock, cfg.require_isolation truthy too.
        with (
            patch.object(PersistentContainerSandbox, "is_available", return_value=True),
            patch.object(DockerSandbox, "is_available", return_value=False),
        ):
            sb = get_sandbox(cfg)
        assert not isinstance(sb, PersistentContainerSandbox)


class TestAdapterExecute:
    def _adapter_with_container(self, container):
        adapter = PersistentContainerSandbox(SandboxConfig(enabled=True, persistent=True))
        adapter._container = container
        adapter._started = True
        return adapter

    def test_success_maps_exit_zero(self) -> None:
        container = MagicMock()
        container.execute.return_value = ("hello\n", 0)
        adapter = self._adapter_with_container(container)
        r = adapter.execute("echo hello")
        assert r.success is True
        assert r.output == "hello\n"
        assert r.sandboxed is True

    def test_nonzero_exit_is_failure(self) -> None:
        container = MagicMock()
        container.execute.return_value = ("boom", 2)
        adapter = self._adapter_with_container(container)
        r = adapter.execute("false")
        assert r.success is False
        assert "exited with 2" in r.error

    def test_container_exception_reported(self) -> None:
        container = MagicMock()
        container.execute.side_effect = RuntimeError("docker gone")
        adapter = self._adapter_with_container(container)
        r = adapter.execute("x")
        assert r.success is False
        assert "docker gone" in r.error

    def test_unavailable_container_returns_error(self) -> None:
        adapter = PersistentContainerSandbox(SandboxConfig())
        # Force start to fail (no container).
        with patch.object(adapter, "_ensure_started", return_value=False):
            r = adapter.execute("x")
        assert r.success is False
        assert "unavailable" in r.error

    def test_lazy_start_then_execute(self) -> None:
        fake_container = MagicMock()
        fake_container.execute.return_value = ("ok", 0)
        with patch("missy.security.container.ContainerSandbox", return_value=fake_container):
            adapter = PersistentContainerSandbox(SandboxConfig(enabled=True, persistent=True))
            r = adapter.execute("echo ok")
        fake_container.start.assert_called_once()
        assert r.success is True

    def test_is_available_delegates(self) -> None:
        with patch("missy.security.container.ContainerSandbox") as CS:
            CS.is_available.return_value = True
            assert PersistentContainerSandbox.is_available() is True
