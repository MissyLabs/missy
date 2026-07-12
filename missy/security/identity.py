"""Cryptographic agent identity using Ed25519 keypairs.

Each :class:`AgentIdentity` wraps an Ed25519 private key and provides
methods for signing, verification, fingerprinting, and JWK export.
Keys are persisted as PEM files with restrictive permissions (0o600).
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat as stat_module
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

#: Default location for the agent identity key file.
DEFAULT_KEY_PATH = os.path.expanduser("~/.missy/identity.pem")


class IdentityError(Exception):
    """Raised when the persisted agent identity key fails a safety check."""


class AgentIdentity:
    """Ed25519-based cryptographic identity for an agent instance.

    Use :meth:`generate` to create a fresh keypair or :meth:`from_key_file`
    to load an existing one from disk.
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key
        self._public_key: Ed25519PublicKey = private_key.public_key()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> AgentIdentity:
        """Generate a new Ed25519 keypair and return an identity."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_key_file(cls, path: str) -> AgentIdentity:
        """Load a private key from a PEM file at *path*.

        This key signs every audit event (see :mod:`missy.observability`),
        so it is as sensitive as the private key material it wraps. Before
        reading it, refuse a symlink or multi-hard-linked file (either
        could point to attacker-controlled content substituted after the
        file was created — the same TOCTOU/symlink class of attack
        :class:`missy.security.vault.Vault` already guards against for its
        own key file), refuse a file not owned by the current user, and
        refuse a file that is group- or world-readable/writable. Silently
        loading a compromised or wrongly-permissioned key would let another
        local user (or anyone who can substitute the symlink target) sign
        audit events that pass verification as this agent's own —
        defeating the tamper-evidence guarantee with no operator-visible
        signal short of manually auditing file permissions.
        """
        p = Path(path)
        if p.is_symlink():
            raise IdentityError(f"Identity key file {p} is a symlink; refusing to read.")
        st = p.stat()
        if st.st_nlink > 1:
            raise IdentityError(f"Identity key file {p} has multiple hard links; refusing to read.")
        if st.st_uid != os.getuid():
            raise IdentityError(f"Identity key file {p} is not owned by current user.")
        if st.st_mode & (stat_module.S_IRWXG | stat_module.S_IRWXO):
            raise IdentityError(
                f"Identity key file {p} has permissive mode "
                f"0o{st.st_mode & 0o777:o}; expected 0o600."
            )
        data = p.read_bytes()
        private_key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"Expected Ed25519 private key, got {type(private_key).__name__}")
        return cls(private_key)

    @classmethod
    def load_or_generate(cls, path: str | None = None) -> AgentIdentity:
        """Load the identity at *path*, generating and persisting one if absent.

        Single source of truth for "the" process-level agent identity, so
        every caller (the agent runtime's own event signing, the audit
        log's signing/verification) resolves to the *same* keypair rather
        than each independently reimplementing this load-or-create
        sequence and risking drift.

        Args:
            path: PEM key file path. When ``None`` (default),
                :data:`DEFAULT_KEY_PATH` is looked up dynamically at call
                time rather than bound once at import time, so
                monkeypatching the module-level constant (as tests do)
                takes effect.

        Returns:
            The loaded or newly generated :class:`AgentIdentity`.
        """
        if path is None:
            path = DEFAULT_KEY_PATH
        if os.path.exists(path):
            return cls.from_key_file(path)
        identity = cls.generate()
        identity.save(path)
        return identity

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the private key to *path* as PEM with 0o600 permissions."""
        pem_bytes = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Write with restrictive permissions: create file 0o600.
        # os.open's mode argument only applies when O_CREAT creates a new file;
        # for pre-existing paths the inode permissions are unchanged.  Explicitly
        # chmod after writing so overwrite also enforces 0o600.
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, pem_bytes)
        finally:
            os.close(fd)
        os.chmod(str(p), 0o600)

    # ------------------------------------------------------------------
    # Cryptographic operations
    # ------------------------------------------------------------------

    def sign(self, message: bytes) -> bytes:
        """Sign *message* with the private key and return the signature."""
        return self._private_key.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Return ``True`` if *signature* is valid for *message*."""
        try:
            self._public_key.verify(signature, message)
            return True
        except Exception:  # noqa: BLE001  — cryptography.exceptions.InvalidSignature + general errors
            return False

    # ------------------------------------------------------------------
    # Key inspection
    # ------------------------------------------------------------------

    def public_key_fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the public key (hex-encoded)."""
        raw = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return hashlib.sha256(raw).hexdigest()

    def to_jwk(self) -> dict:
        """Export the public key as a JWK dictionary.

        Returns a dict with keys ``kty``, ``crv``, ``x`` suitable for
        use in JSON Web Key sets.
        """
        raw = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        # JWK uses base64url-no-padding encoding
        x_b64 = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        return {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": x_b64,
        }
