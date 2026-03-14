"""Encrypted local secrets vault using ChaCha20-Poly1305.

Secrets are stored encrypted at ~/.missy/secrets/vault.enc.
The encryption key is derived from a key file at ~/.missy/secrets/vault.key.

Usage::

    vault = Vault()
    vault.set("OPENAI_API_KEY", "sk-...")
    key = vault.get("OPENAI_API_KEY")
"""
from __future__ import annotations

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
        if self._key_path.exists():
            key = self._key_path.read_bytes()
            if len(key) != 32:
                raise VaultError("Invalid vault key length; expected 32 bytes.")
            return key
        key = secrets.token_bytes(32)
        self._key_path.write_bytes(key)
        self._key_path.chmod(0o600)
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
        self._vault_path.write_bytes(encrypted)
        self._vault_path.chmod(0o600)

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
            key = ref[len("vault://"):]
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
