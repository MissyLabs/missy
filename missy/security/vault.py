"""Encrypted local secrets vault using ChaCha20-Poly1305.

Secrets are stored encrypted at ~/.missy/secrets/vault.enc.
The encryption key is derived from a key file at ~/.missy/secrets/vault.key.

Usage::

    vault = Vault()
    vault.set("OPENAI_API_KEY", "sk-...")
    key = vault.get("OPENAI_API_KEY")
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


class VaultError(Exception):
    pass


class Vault:
    """ChaCha20-Poly1305 encrypted key-value store for secrets.

    Args:
        vault_dir: Directory containing vault.key and vault.enc.
    """

    KEY_FILE = "vault.key"
    VAULT_FILE = "vault.enc"

    def __init__(self, vault_dir: str = "~/.missy/secrets"):
        if not _CRYPTO_AVAILABLE:
            raise VaultError(
                "cryptography package is required for vault support. "
                "Install it with: pip install cryptography"
            )
        self._dir = Path(vault_dir).expanduser()
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._key_path = self._dir / self.KEY_FILE
        self._vault_path = self._dir / self.VAULT_FILE
        self._key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        # Try atomic exclusive create first to avoid TOCTOU race.
        # O_CREAT | O_EXCL fails if the file already exists, guaranteeing
        # we never overwrite an existing key and the file is born with 0o600.
        try:
            fd = os.open(
                str(self._key_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                key = secrets.token_bytes(32)
                os.write(fd, key)
            finally:
                os.close(fd)
            return key
        except FileExistsError:
            pass

        # File already exists -- verify it is a regular file, not a symlink
        # or hard link that could point to an attacker-controlled file.
        if self._key_path.is_symlink():
            raise VaultError("Vault key file is a symlink; refusing to read.")
        st = self._key_path.stat()
        if st.st_nlink > 1:
            raise VaultError("Vault key file has multiple hard links; refusing to read.")
        key = self._key_path.read_bytes()
        if len(key) != 32:
            raise VaultError("Invalid vault key length; expected 32 bytes.")
        return key

    def _encrypt(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        ct = ChaCha20Poly1305(self._key).encrypt(nonce, data, None)
        return nonce + ct

    def _decrypt(self, data: bytes) -> bytes:
        nonce, ct = data[:12], data[12:]
        return ChaCha20Poly1305(self._key).decrypt(nonce, ct, None)

    def _load_store(self) -> dict:
        if not self._vault_path.exists():
            return {}
        try:
            raw = self._vault_path.read_bytes()
            return json.loads(self._decrypt(raw))
        except Exception as exc:
            raise VaultError(f"Cannot decrypt vault: {exc}") from exc

    def _save_store(self, store: dict) -> None:
        raw = json.dumps(store).encode()
        encrypted = self._encrypt(raw)
        # Atomic write: create temp file with correct permissions, then rename.
        # This prevents data loss if the process is interrupted mid-write.
        import tempfile

        self._vault_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._vault_path.parent), suffix=".tmp"
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, encrypted)
            os.fsync(fd)
            os.close(fd)
            fd = -1  # Mark as closed
            os.rename(tmp_path, str(self._vault_path))
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def set(self, key: str, value: str) -> None:
        """Store an encrypted secret."""
        store = self._load_store()
        store[key] = value
        self._save_store(store)

    def get(self, key: str) -> str | None:
        """Retrieve a secret; returns None if not found."""
        store = self._load_store()
        return store.get(key)

    def delete(self, key: str) -> bool:
        """Delete a secret. Returns True if it existed."""
        store = self._load_store()
        if key in store:
            del store[key]
            self._save_store(store)
            return True
        return False

    def list_keys(self) -> list[str]:
        """Return all stored key names (not values)."""
        return list(self._load_store().keys())

    def resolve(self, ref: str) -> str:
        """Resolve a vault:// reference or plain value.

        vault://KEY_NAME → looks up KEY_NAME in vault
        $ENV_VAR → reads from environment
        anything else → returned as-is
        """
        if ref.startswith("vault://"):
            key = ref[len("vault://") :]
            val = self.get(key)
            if val is None:
                raise VaultError(f"vault://{key} not found in vault")
            return val
        if ref.startswith("$"):
            env_key = ref[1:]
            val = os.environ.get(env_key)
            if val is None:
                raise VaultError(f"Environment variable {env_key} is not set")
            return val
        return ref
