"""Credential validation and protected Nomad CLI environments."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import NameOID

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadConfigurationError


@dataclass(frozen=True)
class NomadCredentials:
    """Validated paths and secret material for one Nomad CLI invocation."""

    ca_file: Path
    certificate_file: Path
    key_file: Path
    token_file: Path
    token: str
    certificate_expires_at: datetime
    identity_cn: str


def _protected_file(bundle: Path, name: str, *, secret: bool) -> Path:
    candidate = bundle / name
    if candidate.is_symlink():
        raise NomadConfigurationError(f"Nomad credential must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NomadConfigurationError(f"Nomad credential file is missing: {candidate}") from exc
    try:
        resolved.relative_to(bundle)
    except ValueError as exc:
        raise NomadConfigurationError(
            f"Nomad credential path escapes configured bundle directory: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise NomadConfigurationError(f"Nomad credential is not a regular file: {candidate}")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if secret and mode & 0o077:
        raise NomadConfigurationError(
            f"Nomad secret file must not be accessible by group or others: {candidate}"
        )
    return resolved


def load_credentials(config: NomadConfig, *, now: datetime | None = None) -> NomadCredentials:
    """Validate the completed credential bundle without exposing its token."""
    if not config.enabled:
        raise NomadConfigurationError("Nomad integration is disabled in config.yaml.")
    bundle = Path(config.bundle_dir).expanduser().resolve()
    if not bundle.is_dir():
        raise NomadConfigurationError(f"Nomad bundle directory does not exist: {bundle}")

    ca_file = _protected_file(bundle, "ca.pem", secret=False)
    certificate_file = _protected_file(bundle, "missy.pem", secret=False)
    key_file = _protected_file(bundle, "missy-key.pem", secret=True)
    token_file = _protected_file(bundle, "missy.token", secret=True)

    try:
        certificate = x509.load_pem_x509_certificate(certificate_file.read_bytes())
        ca_certificate = x509.load_pem_x509_certificate(ca_file.read_bytes())
        private_key = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
    except Exception as exc:  # cryptography intentionally exposes many concrete parse errors
        raise NomadConfigurationError("Nomad certificate, CA, or private key is invalid.") from exc

    cert_public = certificate.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_public = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if cert_public != key_public:
        raise NomadConfigurationError("Nomad client certificate does not match its private key.")
    try:
        certificate.verify_directly_issued_by(ca_certificate)
    except Exception as exc:
        raise NomadConfigurationError(
            "Nomad client certificate was not issued by the configured CA."
        ) from exc

    common_names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    identity_cn = common_names[0].value if common_names else ""
    if identity_cn != config.identity_cn:
        raise NomadConfigurationError(
            f"Nomad client certificate identity is {identity_cn!r}, expected {config.identity_cn!r}."
        )
    current = now or datetime.now(tz=UTC)
    expires = certificate.not_valid_after_utc
    if certificate.not_valid_before_utc > current:
        raise NomadConfigurationError("Nomad client certificate is not valid yet.")
    if expires <= current:
        raise NomadConfigurationError("Nomad client certificate has expired.")

    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise NomadConfigurationError("Nomad ACL token could not be read.") from exc
    if not token or len(token) > 4096 or any(char.isspace() for char in token):
        raise NomadConfigurationError("Nomad ACL token file is empty or malformed.")

    return NomadCredentials(
        ca_file=ca_file,
        certificate_file=certificate_file,
        key_file=key_file,
        token_file=token_file,
        token=token,
        certificate_expires_at=expires,
        identity_cn=identity_cn,
    )


def resolve_binary(config: NomadConfig) -> str:
    """Return an absolute Nomad CLI path, rejecting missing/non-executable files."""
    binary = os.path.expanduser(config.binary)
    resolved = shutil.which(binary)
    if not resolved:
        raise NomadConfigurationError(f"Nomad CLI binary was not found: {config.binary}")
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise NomadConfigurationError(f"Nomad CLI binary is not executable: {path}")
    return str(path)


def build_cli_environment(config: NomadConfig, credentials: NomadCredentials) -> dict[str, str]:
    """Build a minimal subprocess environment containing only Nomad's secret token."""
    env: dict[str, str] = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NOMAD_ADDR": config.address,
        "NOMAD_CACERT": str(credentials.ca_file),
        "NOMAD_CLIENT_CERT": str(credentials.certificate_file),
        "NOMAD_CLIENT_KEY": str(credentials.key_file),
        "NOMAD_TOKEN": credentials.token,
        "NOMAD_CLI_NO_COLOR": "1",
    }
    # HOME is not necessary for the fixed CLI operations and omitting it
    # prevents accidental loading of unrelated operator configuration.
    return env


def credential_metadata(credentials: NomadCredentials) -> dict[str, Any]:
    """Return safe identity metadata suitable for status output and audit logs."""
    return {
        "identity_cn": credentials.identity_cn,
        "certificate_expires_at": credentials.certificate_expires_at.isoformat(),
    }
