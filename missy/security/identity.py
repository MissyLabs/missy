"""Cryptographic agent identity using Ed25519 keypairs.

Each :class:`AgentIdentity` wraps an Ed25519 private key and provides
methods for signing, verification, fingerprinting, and JWK export.
Keys are persisted as PEM files with restrictive permissions (0o600).
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

#: Default location for the agent identity key file.
DEFAULT_KEY_PATH = os.path.expanduser("~/.missy/identity.pem")


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
        """Load a private key from a PEM file at *path*."""
        data = Path(path).read_bytes()
        private_key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"Expected Ed25519 private key, got {type(private_key).__name__}")
        return cls(private_key)

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
        except Exception:
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
