from __future__ import annotations

import os
from pathlib import Path

import pytest

from missy.config.settings import NomadConfig
from missy.nomad.credentials import build_cli_environment, load_credentials
from missy.nomad.errors import NomadConfigurationError

from .conftest import make_bundle


def test_load_completed_bundle(nomad_config: NomadConfig) -> None:
    credentials = load_credentials(nomad_config)
    assert credentials.identity_cn == "missy"
    assert credentials.token == "test-token-value"


def test_disabled_integration_fails_closed(nomad_config: NomadConfig) -> None:
    nomad_config.enabled = False
    with pytest.raises(NomadConfigurationError, match="disabled"):
        load_credentials(nomad_config)


def test_missing_completed_artifact_is_rejected(nomad_config: NomadConfig) -> None:
    Path(nomad_config.bundle_dir, "missy.pem").unlink()
    with pytest.raises(NomadConfigurationError, match="missing"):
        load_credentials(nomad_config)


def test_group_readable_secret_is_rejected(nomad_config: NomadConfig) -> None:
    token = Path(nomad_config.bundle_dir, "missy.token")
    os.chmod(token, 0o640)
    with pytest.raises(NomadConfigurationError, match="group or others"):
        load_credentials(nomad_config)


def test_wrong_identity_is_rejected(tmp_path: Path, nomad_config: NomadConfig) -> None:
    bundle = make_bundle(tmp_path / "wrong", common_name="odin")
    nomad_config.bundle_dir = str(bundle)
    with pytest.raises(NomadConfigurationError, match="expected 'missy'"):
        load_credentials(nomad_config)


def test_mismatched_key_is_rejected(tmp_path: Path, nomad_config: NomadConfig) -> None:
    bundle = make_bundle(tmp_path / "mismatch", mismatched_key=True)
    nomad_config.bundle_dir = str(bundle)
    with pytest.raises(NomadConfigurationError, match="does not match"):
        load_credentials(nomad_config)


def test_expired_certificate_is_rejected(tmp_path: Path, nomad_config: NomadConfig) -> None:
    bundle = make_bundle(tmp_path / "expired", expired=True)
    nomad_config.bundle_dir = str(bundle)
    with pytest.raises(NomadConfigurationError, match="expired"):
        load_credentials(nomad_config)


def test_cli_environment_contains_no_skip_verify(nomad_config: NomadConfig) -> None:
    credentials = load_credentials(nomad_config)
    env = build_cli_environment(nomad_config, credentials)
    assert env["NOMAD_TOKEN"] == "test-token-value"
    assert "NOMAD_SKIP_VERIFY" not in env
    assert "HOME" not in env


def test_symbolic_linked_credential_is_rejected(tmp_path: Path, nomad_config: NomadConfig) -> None:
    certificate = Path(nomad_config.bundle_dir, "missy.pem")
    real_certificate = tmp_path / "real-missy.pem"
    certificate.rename(real_certificate)
    certificate.symlink_to(real_certificate)
    with pytest.raises(NomadConfigurationError, match="symbolic link"):
        load_credentials(nomad_config)
