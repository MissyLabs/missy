"""Tests for missy.security.identity — Ed25519 agent identity."""

from __future__ import annotations

import os
import stat

from missy.security.identity import AgentIdentity


class TestAgentIdentity:
    """Tests for AgentIdentity keypair operations."""

    def test_generate_and_sign_verify(self):
        """Round-trip: sign a message and verify the signature."""
        identity = AgentIdentity.generate()
        message = b"hello world"
        signature = identity.sign(message)
        assert identity.verify(message, signature) is True

    def test_invalid_signature_rejected(self):
        """Tampered message must fail verification."""
        identity = AgentIdentity.generate()
        message = b"original message"
        signature = identity.sign(message)
        # Tamper with the message
        assert identity.verify(b"tampered message", signature) is False

    def test_save_and_load(self, tmp_path):
        """Persist key to file and reload — signatures must still verify."""
        identity = AgentIdentity.generate()
        key_file = str(tmp_path / "test_identity.pem")
        identity.save(key_file)

        loaded = AgentIdentity.from_key_file(key_file)
        message = b"persistence check"
        signature = identity.sign(message)
        assert loaded.verify(message, signature) is True

    def test_key_file_permissions(self, tmp_path):
        """Key file must be created with 0o600 permissions."""
        identity = AgentIdentity.generate()
        key_file = str(tmp_path / "test_identity.pem")
        identity.save(key_file)

        mode = os.stat(key_file).st_mode
        # Check only the permission bits
        assert stat.S_IMODE(mode) == 0o600

    def test_fingerprint_deterministic(self):
        """Same key must always produce the same fingerprint."""
        identity = AgentIdentity.generate()
        fp1 = identity.public_key_fingerprint()
        fp2 = identity.public_key_fingerprint()
        assert fp1 == fp2
        # SHA-256 hex digest is 64 chars
        assert len(fp1) == 64

    def test_jwk_export(self):
        """JWK export must contain required Ed25519 fields."""
        identity = AgentIdentity.generate()
        jwk = identity.to_jwk()
        assert jwk["kty"] == "OKP"
        assert jwk["crv"] == "Ed25519"
        assert "x" in jwk
        # x should be base64url encoded (no padding)
        assert "=" not in jwk["x"]
