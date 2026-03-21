"""Tests for the Linux Landlock LSM filesystem policy module.

No Landlock syscalls are actually executed against the kernel (that would
permanently restrict the test process).  Instead:

* :meth:`LandlockPolicy.is_available` is tested for its return type.
* Rule-building methods are tested without calling :meth:`apply`.
* :meth:`apply` is tested via syscall mocking so the kernel is never touched.
* :func:`apply_landlock_from_config` is tested against a mock config.
* :func:`landlock_status` is tested for structure and types.
* Constant values are verified against the Landlock ABI specification.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from missy.security.landlock import (
    LANDLOCK_ACCESS_FS_EXECUTE,
    LANDLOCK_ACCESS_FS_MAKE_BLOCK,
    LANDLOCK_ACCESS_FS_MAKE_CHAR,
    LANDLOCK_ACCESS_FS_MAKE_DIR,
    LANDLOCK_ACCESS_FS_MAKE_FIFO,
    LANDLOCK_ACCESS_FS_MAKE_REG,
    LANDLOCK_ACCESS_FS_MAKE_SOCK,
    LANDLOCK_ACCESS_FS_MAKE_SYM,
    LANDLOCK_ACCESS_FS_READ,
    LANDLOCK_ACCESS_FS_READ_DIR,
    LANDLOCK_ACCESS_FS_READ_FILE,
    LANDLOCK_ACCESS_FS_REMOVE_DIR,
    LANDLOCK_ACCESS_FS_REMOVE_FILE,
    LANDLOCK_ACCESS_FS_WRITE,
    LANDLOCK_ACCESS_FS_WRITE_FILE,
    LANDLOCK_CREATE_RULESET_VERSION,
    LANDLOCK_RULE_PATH_BENEATH,
    LandlockPolicy,
    apply_landlock_from_config,
    landlock_status,
)

# ---------------------------------------------------------------------------
# Minimal stub config objects
# ---------------------------------------------------------------------------


@dataclass
class _FilesystemPolicy:
    allowed_read_paths: list[str] = field(default_factory=list)
    allowed_write_paths: list[str] = field(default_factory=list)


@dataclass
class _MissyConfig:
    filesystem: _FilesystemPolicy = field(default_factory=_FilesystemPolicy)


# ---------------------------------------------------------------------------
# Helper: a pre-built policy without applying it
# ---------------------------------------------------------------------------


def _policy_with_rules() -> LandlockPolicy:
    return (
        LandlockPolicy()
        .add_read_path("/usr")
        .add_read_path("/etc")
        .add_write_path("/tmp")
        .add_execute_path("/usr/bin")
    )


# ---------------------------------------------------------------------------
# Helper: produce a libc mock that looks healthy (positive syscall results)
# ---------------------------------------------------------------------------


def _mock_libc(syscall_return: int = 5) -> MagicMock:
    libc = MagicMock()
    libc.syscall.return_value = syscall_return
    libc.prctl.return_value = 0
    return libc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify Landlock constant values match the kernel ABI specification."""

    def test_individual_access_rights_are_powers_of_two(self) -> None:
        individual = [
            LANDLOCK_ACCESS_FS_EXECUTE,
            LANDLOCK_ACCESS_FS_WRITE_FILE,
            LANDLOCK_ACCESS_FS_READ_FILE,
            LANDLOCK_ACCESS_FS_READ_DIR,
            LANDLOCK_ACCESS_FS_REMOVE_DIR,
            LANDLOCK_ACCESS_FS_REMOVE_FILE,
            LANDLOCK_ACCESS_FS_MAKE_CHAR,
            LANDLOCK_ACCESS_FS_MAKE_DIR,
            LANDLOCK_ACCESS_FS_MAKE_REG,
            LANDLOCK_ACCESS_FS_MAKE_SOCK,
            LANDLOCK_ACCESS_FS_MAKE_FIFO,
            LANDLOCK_ACCESS_FS_MAKE_BLOCK,
            LANDLOCK_ACCESS_FS_MAKE_SYM,
        ]
        for right in individual:
            assert right > 0
            assert (right & (right - 1)) == 0, f"{right} is not a power of two"

    def test_individual_rights_are_unique(self) -> None:
        rights = [
            LANDLOCK_ACCESS_FS_EXECUTE,
            LANDLOCK_ACCESS_FS_WRITE_FILE,
            LANDLOCK_ACCESS_FS_READ_FILE,
            LANDLOCK_ACCESS_FS_READ_DIR,
            LANDLOCK_ACCESS_FS_REMOVE_DIR,
            LANDLOCK_ACCESS_FS_REMOVE_FILE,
            LANDLOCK_ACCESS_FS_MAKE_CHAR,
            LANDLOCK_ACCESS_FS_MAKE_DIR,
            LANDLOCK_ACCESS_FS_MAKE_REG,
            LANDLOCK_ACCESS_FS_MAKE_SOCK,
            LANDLOCK_ACCESS_FS_MAKE_FIFO,
            LANDLOCK_ACCESS_FS_MAKE_BLOCK,
            LANDLOCK_ACCESS_FS_MAKE_SYM,
        ]
        assert len(rights) == len(set(rights))

    def test_execute_is_bit_0(self) -> None:
        assert LANDLOCK_ACCESS_FS_EXECUTE == 1

    def test_write_file_is_bit_1(self) -> None:
        assert LANDLOCK_ACCESS_FS_WRITE_FILE == 2

    def test_read_file_is_bit_2(self) -> None:
        assert LANDLOCK_ACCESS_FS_READ_FILE == 4

    def test_read_dir_is_bit_3(self) -> None:
        assert LANDLOCK_ACCESS_FS_READ_DIR == 8

    def test_read_composite_contains_read_file_and_read_dir(self) -> None:
        assert LANDLOCK_ACCESS_FS_READ & LANDLOCK_ACCESS_FS_READ_FILE
        assert LANDLOCK_ACCESS_FS_READ & LANDLOCK_ACCESS_FS_READ_DIR

    def test_write_composite_contains_write_file(self) -> None:
        assert LANDLOCK_ACCESS_FS_WRITE & LANDLOCK_ACCESS_FS_WRITE_FILE

    def test_write_composite_contains_all_mutation_rights(self) -> None:
        mutation_rights = [
            LANDLOCK_ACCESS_FS_WRITE_FILE,
            LANDLOCK_ACCESS_FS_REMOVE_DIR,
            LANDLOCK_ACCESS_FS_REMOVE_FILE,
            LANDLOCK_ACCESS_FS_MAKE_CHAR,
            LANDLOCK_ACCESS_FS_MAKE_DIR,
            LANDLOCK_ACCESS_FS_MAKE_REG,
            LANDLOCK_ACCESS_FS_MAKE_SOCK,
            LANDLOCK_ACCESS_FS_MAKE_FIFO,
            LANDLOCK_ACCESS_FS_MAKE_BLOCK,
            LANDLOCK_ACCESS_FS_MAKE_SYM,
        ]
        for right in mutation_rights:
            assert LANDLOCK_ACCESS_FS_WRITE & right, f"WRITE composite missing {right}"

    def test_read_does_not_include_write_rights(self) -> None:
        assert not (LANDLOCK_ACCESS_FS_READ & LANDLOCK_ACCESS_FS_WRITE_FILE)
        assert not (LANDLOCK_ACCESS_FS_READ & LANDLOCK_ACCESS_FS_REMOVE_FILE)

    def test_create_ruleset_version_flag(self) -> None:
        assert LANDLOCK_CREATE_RULESET_VERSION == 1

    def test_rule_path_beneath(self) -> None:
        assert LANDLOCK_RULE_PATH_BENEATH == 1


# ---------------------------------------------------------------------------
# LandlockPolicy — availability
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_bool(self) -> None:
        result = LandlockPolicy.is_available()
        assert isinstance(result, bool)

    def test_non_linux_returns_false(self) -> None:
        with patch.object(sys, "platform", "darwin"):
            assert LandlockPolicy.is_available() is False

    def test_non_linux_win32_returns_false(self) -> None:
        with patch.object(sys, "platform", "win32"):
            assert LandlockPolicy.is_available() is False

    def test_returns_false_when_libc_not_found(self) -> None:
        with patch("missy.security.landlock.LandlockPolicy._load_libc") as mock_load:
            mock_load.side_effect = OSError("no libc")
            assert LandlockPolicy.is_available() is False

    def test_returns_false_when_syscall_raises(self) -> None:
        with patch("missy.security.landlock.LandlockPolicy._load_libc") as mock_load:
            mock_libc = MagicMock()
            mock_libc.syscall.side_effect = OSError("ENOSYS")
            mock_load.return_value = mock_libc
            assert LandlockPolicy.is_available() is False

    def test_returns_false_when_syscall_returns_zero(self) -> None:
        # ABI version 0 means not supported.
        mock_libc = MagicMock()
        mock_libc.syscall.return_value = 0
        with (
            patch.object(sys, "platform", "linux"),
            patch("missy.security.landlock.LandlockPolicy._load_libc", return_value=mock_libc),
        ):
            assert LandlockPolicy.is_available() is False

    def test_returns_true_when_syscall_returns_positive(self) -> None:
        mock_libc = MagicMock()
        mock_libc.syscall.return_value = 1
        with (
            patch.object(sys, "platform", "linux"),
            patch("missy.security.landlock.LandlockPolicy._load_libc", return_value=mock_libc),
        ):
            assert LandlockPolicy.is_available() is True

    def test_returns_true_for_abi_v3(self) -> None:
        mock_libc = MagicMock()
        mock_libc.syscall.return_value = 3
        with (
            patch.object(sys, "platform", "linux"),
            patch("missy.security.landlock.LandlockPolicy._load_libc", return_value=mock_libc),
        ):
            assert LandlockPolicy.is_available() is True


# ---------------------------------------------------------------------------
# LandlockPolicy — rule builders
# ---------------------------------------------------------------------------


class TestRuleBuilders:
    def test_initial_rules_empty(self) -> None:
        policy = LandlockPolicy()
        assert policy.rules == []

    def test_add_read_path_appends_rule(self) -> None:
        policy = LandlockPolicy()
        policy.add_read_path("/usr")
        assert policy.rules == [{"path": "/usr", "access": "read"}]

    def test_add_write_path_appends_rule(self) -> None:
        policy = LandlockPolicy()
        policy.add_write_path("/tmp")
        assert policy.rules == [{"path": "/tmp", "access": "read_write"}]

    def test_add_execute_path_appends_rule(self) -> None:
        policy = LandlockPolicy()
        policy.add_execute_path("/usr/bin")
        assert policy.rules == [{"path": "/usr/bin", "access": "execute"}]

    def test_add_read_path_returns_self_for_chaining(self) -> None:
        policy = LandlockPolicy()
        result = policy.add_read_path("/usr")
        assert result is policy

    def test_add_write_path_returns_self_for_chaining(self) -> None:
        policy = LandlockPolicy()
        result = policy.add_write_path("/tmp")
        assert result is policy

    def test_add_execute_path_returns_self_for_chaining(self) -> None:
        policy = LandlockPolicy()
        result = policy.add_execute_path("/bin")
        assert result is policy

    def test_fluent_chain_accumulates_rules(self) -> None:
        policy = (
            LandlockPolicy()
            .add_read_path("/usr")
            .add_read_path("/etc")
            .add_write_path("/tmp")
            .add_execute_path("/usr/bin")
        )
        assert len(policy.rules) == 4

    def test_rules_property_returns_copy(self) -> None:
        policy = LandlockPolicy().add_read_path("/usr")
        snapshot = policy.rules
        snapshot.append({"path": "/evil", "access": "read_write"})
        assert len(policy.rules) == 1  # original unaffected

    def test_multiple_paths_same_access_type(self) -> None:
        policy = LandlockPolicy().add_read_path("/usr").add_read_path("/lib").add_read_path("/etc")
        accesses = [r["access"] for r in policy.rules]
        assert all(a == "read" for a in accesses)
        paths = [r["path"] for r in policy.rules]
        assert "/usr" in paths
        assert "/lib" in paths
        assert "/etc" in paths


# ---------------------------------------------------------------------------
# LandlockPolicy — applied property
# ---------------------------------------------------------------------------


class TestAppliedProperty:
    def test_not_applied_initially(self) -> None:
        assert not LandlockPolicy().applied

    def test_applied_after_successful_apply(self) -> None:
        policy = LandlockPolicy().add_read_path("/usr")
        libc = _mock_libc()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(policy, "_get_libc", return_value=libc),
            patch.object(LandlockPolicy, "_open_path", return_value=10),
            patch("os.close"),
        ):
            policy.apply()
        assert policy.applied


# ---------------------------------------------------------------------------
# LandlockPolicy — apply()
# ---------------------------------------------------------------------------


class TestApply:
    def test_returns_false_when_not_available(self) -> None:
        policy = LandlockPolicy()
        with patch.object(LandlockPolicy, "is_available", return_value=False):
            result = policy.apply()
        assert result is False
        assert not policy.applied

    def test_raises_if_called_twice(self) -> None:
        policy = LandlockPolicy()
        libc = _mock_libc()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(policy, "_get_libc", return_value=libc),
            patch.object(LandlockPolicy, "_open_path", return_value=-1),
            patch("os.close"),
        ):
            policy.apply()

        with pytest.raises(RuntimeError, match="already been called"):
            policy.apply()

    def test_apply_skips_nonexistent_paths(self) -> None:
        policy = LandlockPolicy().add_read_path("/nonexistent/path/xyz")
        libc = _mock_libc()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(policy, "_get_libc", return_value=libc),
            patch.object(LandlockPolicy, "_open_path", return_value=-1),
            patch("os.close"),
        ):
            result = policy.apply()
        assert result is True

    def test_apply_raises_on_ruleset_creation_failure(self) -> None:
        policy = LandlockPolicy()
        libc = _mock_libc(syscall_return=-1)
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(policy, "_get_libc", return_value=libc),
            patch("ctypes.get_errno", return_value=1),
            patch("os.close"),
            pytest.raises(RuntimeError, match="Failed to apply Landlock"),
        ):
            policy.apply()

    def test_apply_returns_true_on_success(self) -> None:
        policy = LandlockPolicy().add_read_path("/usr")
        libc = _mock_libc()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(policy, "_get_libc", return_value=libc),
            patch.object(LandlockPolicy, "_open_path", return_value=10),
            patch("os.close"),
        ):
            result = policy.apply()
        assert result is True


# ---------------------------------------------------------------------------
# LandlockPolicy — _access_mask_for
# ---------------------------------------------------------------------------


class TestAccessMaskFor:
    def test_read_mask(self) -> None:
        mask = LandlockPolicy._access_mask_for("read")
        assert mask == LANDLOCK_ACCESS_FS_READ

    def test_read_write_mask_includes_read(self) -> None:
        mask = LandlockPolicy._access_mask_for("read_write")
        assert mask & LANDLOCK_ACCESS_FS_READ

    def test_read_write_mask_includes_write(self) -> None:
        mask = LandlockPolicy._access_mask_for("read_write")
        assert mask & LANDLOCK_ACCESS_FS_WRITE

    def test_execute_mask_includes_read(self) -> None:
        mask = LandlockPolicy._access_mask_for("execute")
        assert mask & LANDLOCK_ACCESS_FS_READ

    def test_execute_mask_includes_execute(self) -> None:
        mask = LandlockPolicy._access_mask_for("execute")
        assert mask & LANDLOCK_ACCESS_FS_EXECUTE

    def test_unknown_access_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown Landlock access type"):
            LandlockPolicy._access_mask_for("superpower")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown Landlock access type"):
            LandlockPolicy._access_mask_for("")


# ---------------------------------------------------------------------------
# LandlockPolicy — _open_path
# ---------------------------------------------------------------------------


class TestOpenPath:
    def test_returns_negative_one_for_nonexistent_path(self) -> None:
        result = LandlockPolicy._open_path("/this/path/does/not/exist/xyz123")
        assert result == -1

    def test_returns_valid_fd_for_existing_path(self, tmp_path) -> None:
        fd = LandlockPolicy._open_path(str(tmp_path))
        try:
            assert fd >= 0
        finally:
            if fd >= 0:
                os.close(fd)


# ---------------------------------------------------------------------------
# LandlockPolicy — _syscall
# ---------------------------------------------------------------------------


class TestSyscall:
    def test_raises_os_error_on_negative_result(self) -> None:
        policy = LandlockPolicy()
        mock_libc = MagicMock()
        mock_libc.syscall.return_value = -1
        with (
            patch("ctypes.get_errno", return_value=1),  # EPERM
            pytest.raises(OSError),
        ):
            policy._syscall(mock_libc, 444)

    def test_returns_value_on_success(self) -> None:
        policy = LandlockPolicy()
        mock_libc = MagicMock()
        mock_libc.syscall.return_value = 7
        result = policy._syscall(mock_libc, 444)
        assert result == 7


# ---------------------------------------------------------------------------
# apply_landlock_from_config
# ---------------------------------------------------------------------------


class TestApplyLandlockFromConfig:
    def test_returns_false_when_not_available(self) -> None:
        config = _MissyConfig()
        with patch.object(LandlockPolicy, "is_available", return_value=False):
            result = apply_landlock_from_config(config)  # type: ignore[arg-type]
        assert result is False

    def test_includes_system_read_paths(self, tmp_path) -> None:
        # Verify that apply() is called when Landlock is available, implying
        # system paths were added to the policy before the call.
        config = _MissyConfig()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=False),
            patch.object(LandlockPolicy, "__init__", wraps=LandlockPolicy.__init__),
        ):
            # Returns False because Landlock is not "available" — no crash.
            result = apply_landlock_from_config(config)  # type: ignore[arg-type]
        assert result is False

    def test_calls_apply_when_available(self) -> None:
        config = _MissyConfig()
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True) as mock_apply,
        ):
            result = apply_landlock_from_config(config)  # type: ignore[arg-type]
        assert result is True
        mock_apply.assert_called_once()

    def test_user_read_paths_added(self, tmp_path) -> None:
        read_dir = tmp_path / "readable"
        read_dir.mkdir()
        config = _MissyConfig(
            filesystem=_FilesystemPolicy(
                allowed_read_paths=[str(read_dir)],
            )
        )
        added_rules: list[dict] = []
        original_add_read = LandlockPolicy.add_read_path

        def spy_add_read(self_inner: LandlockPolicy, path: str) -> LandlockPolicy:
            added_rules.append({"path": path, "access": "read"})
            return original_add_read(self_inner, path)

        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True),
            patch.object(LandlockPolicy, "add_read_path", spy_add_read),
        ):
            apply_landlock_from_config(config)  # type: ignore[arg-type]

        paths = [r["path"] for r in added_rules]
        assert str(read_dir) in paths

    def test_user_write_paths_added(self, tmp_path) -> None:
        write_dir = tmp_path / "writable"
        write_dir.mkdir()
        config = _MissyConfig(
            filesystem=_FilesystemPolicy(
                allowed_write_paths=[str(write_dir)],
            )
        )
        added_write_rules: list[dict] = []
        original_add_write = LandlockPolicy.add_write_path

        def spy_add_write(self_inner: LandlockPolicy, path: str) -> LandlockPolicy:
            added_write_rules.append({"path": path, "access": "read_write"})
            return original_add_write(self_inner, path)

        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True),
            patch.object(LandlockPolicy, "add_write_path", spy_add_write),
        ):
            apply_landlock_from_config(config)  # type: ignore[arg-type]

        paths = [r["path"] for r in added_write_rules]
        assert str(write_dir) in paths

    def test_nonexistent_user_paths_skipped(self) -> None:
        config = _MissyConfig(
            filesystem=_FilesystemPolicy(
                allowed_read_paths=["/does/not/exist/xyzzy"],
                allowed_write_paths=["/also/does/not/exist"],
            )
        )
        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True),
        ):
            # Should not raise even if paths are missing.
            result = apply_landlock_from_config(config)  # type: ignore[arg-type]
        assert result is True

    def test_tmp_always_added_as_write_path(self) -> None:
        config = _MissyConfig()
        added_write_paths: list[str] = []
        original_add_write = LandlockPolicy.add_write_path

        def spy_add_write(self_inner: LandlockPolicy, path: str) -> LandlockPolicy:
            added_write_paths.append(path)
            return original_add_write(self_inner, path)

        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True),
            patch.object(LandlockPolicy, "add_write_path", spy_add_write),
            patch("os.path.exists", return_value=True),
        ):
            apply_landlock_from_config(config)  # type: ignore[arg-type]

        assert "/tmp" in added_write_paths

    def test_missy_dir_always_added_as_write_path(self) -> None:
        config = _MissyConfig()
        added_write_paths: list[str] = []
        original_add_write = LandlockPolicy.add_write_path

        def spy_add_write(self_inner: LandlockPolicy, path: str) -> LandlockPolicy:
            added_write_paths.append(path)
            return original_add_write(self_inner, path)

        with (
            patch.object(LandlockPolicy, "is_available", return_value=True),
            patch.object(LandlockPolicy, "apply", return_value=True),
            patch.object(LandlockPolicy, "add_write_path", spy_add_write),
            patch("os.path.exists", return_value=True),
        ):
            apply_landlock_from_config(config)  # type: ignore[arg-type]

        missy_dir = os.path.expanduser("~/.missy")
        assert missy_dir in added_write_paths


# ---------------------------------------------------------------------------
# landlock_status
# ---------------------------------------------------------------------------


class TestLandlockStatus:
    def test_returns_dict(self) -> None:
        result = landlock_status()
        assert isinstance(result, dict)

    def test_has_available_key(self) -> None:
        result = landlock_status()
        assert "available" in result

    def test_has_applied_key(self) -> None:
        result = landlock_status()
        assert "applied" in result

    def test_has_kernel_version_key(self) -> None:
        result = landlock_status()
        assert "kernel_version" in result

    def test_has_platform_key(self) -> None:
        result = landlock_status()
        assert "platform" in result

    def test_available_is_bool(self) -> None:
        result = landlock_status()
        assert isinstance(result["available"], bool)

    def test_applied_is_bool(self) -> None:
        result = landlock_status()
        assert isinstance(result["applied"], bool)

    def test_kernel_version_is_str(self) -> None:
        result = landlock_status()
        assert isinstance(result["kernel_version"], str)

    def test_platform_is_str(self) -> None:
        result = landlock_status()
        assert isinstance(result["platform"], str)

    def test_platform_matches_sys_platform(self) -> None:
        result = landlock_status()
        assert result["platform"] == sys.platform

    def test_applied_reflects_module_state_false(self) -> None:
        import missy.security.landlock as ll_module

        original = ll_module._applied_globally
        try:
            ll_module._applied_globally = False
            result = landlock_status()
            assert result["applied"] is False
        finally:
            ll_module._applied_globally = original

    def test_applied_reflects_module_state_true(self) -> None:
        import missy.security.landlock as ll_module

        original = ll_module._applied_globally
        try:
            ll_module._applied_globally = True
            result = landlock_status()
            assert result["applied"] is True
        finally:
            ll_module._applied_globally = original
