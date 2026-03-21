"""Tests covering security module gaps.


Fills identified coverage gaps in:
- PromptDriftDetector: verify_all, overwrite, empty state
- AgentIdentity: save/load, sign/verify edge cases, JWK, error handling
- TrustScorer: thread safety, boundary conditions, mutation resistance
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# PromptDriftDetector
# ---------------------------------------------------------------------------


class TestPromptDriftDetectorEdgeCases:
    """Fill coverage gaps for PromptDriftDetector."""

    def test_verify_unregistered_returns_true(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        assert d.verify("unknown", "anything") is True

    def test_verify_unchanged_returns_true(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "You are helpful.")
        assert d.verify("sys", "You are helpful.") is True

    def test_verify_changed_returns_false(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "You are helpful.")
        assert d.verify("sys", "You are evil.") is False

    def test_register_overwrite(self) -> None:
        """Re-registering same prompt_id overwrites the hash."""
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "original")
        assert d.verify("sys", "original") is True

        d.register("sys", "updated")
        assert d.verify("sys", "updated") is True
        assert d.verify("sys", "original") is False

    def test_get_drift_report_empty(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        assert d.get_drift_report() == []

    def test_get_drift_report_multiple(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("a", "alpha")
        d.register("b", "bravo")
        report = d.get_drift_report()
        assert len(report) == 2
        ids = {r["prompt_id"] for r in report}
        assert ids == {"a", "b"}
        assert all("expected_hash" in r for r in report)

    def test_verify_all_no_drift(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "hello")
        d.register("tool", "world")
        report = d.verify_all({"sys": "hello", "tool": "world"})
        assert len(report) == 2
        assert all(not r["drifted"] for r in report)

    def test_verify_all_with_drift(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "hello")
        report = d.verify_all({"sys": "tampered"})
        assert len(report) == 1
        assert report[0]["drifted"] is True
        assert report[0]["actual_hash"] != report[0]["expected_hash"]

    def test_verify_all_missing_content(self) -> None:
        """When a registered prompt has no content in verify_all, drifted=False."""
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "hello")
        d.register("tool", "world")
        # Only provide content for 'sys', not 'tool'
        report = d.verify_all({"sys": "hello"})
        assert len(report) == 2
        tool_report = next(r for r in report if r["prompt_id"] == "tool")
        assert tool_report["actual_hash"] is None
        assert tool_report["drifted"] is False

    def test_verify_all_empty_contents(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "hello")
        report = d.verify_all({})
        assert len(report) == 1
        assert report[0]["actual_hash"] is None
        assert report[0]["drifted"] is False

    def test_verify_all_no_registered(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        report = d.verify_all({"sys": "hello"})
        assert report == []

    def test_hash_deterministic(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        h1 = d._hash("test content")
        h2 = d._hash("test content")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_different_for_different_content(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        assert d._hash("alpha") != d._hash("beta")

    def test_unicode_content(self) -> None:
        from missy.security.drift import PromptDriftDetector

        d = PromptDriftDetector()
        d.register("sys", "こんにちは 🌍")
        assert d.verify("sys", "こんにちは 🌍") is True
        assert d.verify("sys", "こんにちは") is False


# ---------------------------------------------------------------------------
# AgentIdentity
# ---------------------------------------------------------------------------


class TestAgentIdentityEdgeCases:
    """Fill coverage gaps for AgentIdentity."""

    def test_generate_creates_valid_identity(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        assert identity is not None

    def test_sign_and_verify(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        msg = b"Hello, world!"
        sig = identity.sign(msg)
        assert identity.verify(msg, sig) is True

    def test_verify_wrong_message(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        sig = identity.sign(b"original")
        assert identity.verify(b"tampered", sig) is False

    def test_verify_wrong_signature(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        identity.sign(b"test")
        assert identity.verify(b"test", b"invalidsig") is False

    def test_verify_empty_message(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        sig = identity.sign(b"")
        assert identity.verify(b"", sig) is True

    def test_verify_empty_signature(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        assert identity.verify(b"test", b"") is False

    def test_sign_large_payload(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        large_msg = b"x" * 100_000
        sig = identity.sign(large_msg)
        assert identity.verify(large_msg, sig) is True

    def test_save_and_load(self, tmp_path: Path) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        key_path = str(tmp_path / "test.pem")
        identity.save(key_path)

        loaded = AgentIdentity.from_key_file(key_path)
        # Same key should produce same fingerprint
        assert loaded.public_key_fingerprint() == identity.public_key_fingerprint()

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        key_path = str(tmp_path / "subdir" / "deep" / "test.pem")
        identity.save(key_path)
        assert Path(key_path).exists()

    def test_save_file_permissions(self, tmp_path: Path) -> None:

        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        key_path = str(tmp_path / "test.pem")
        identity.save(key_path)
        mode = Path(key_path).stat().st_mode
        assert (mode & 0o777) == 0o600

    def test_load_nonexistent_raises(self) -> None:
        from missy.security.identity import AgentIdentity

        with pytest.raises(FileNotFoundError):
            AgentIdentity.from_key_file("/nonexistent/path.pem")

    def test_load_corrupted_raises(self, tmp_path: Path) -> None:
        from missy.security.identity import AgentIdentity

        bad_pem = tmp_path / "bad.pem"
        bad_pem.write_text("not a valid PEM file")
        with pytest.raises(ValueError):
            AgentIdentity.from_key_file(str(bad_pem))

    def test_fingerprint_deterministic(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        fp1 = identity.public_key_fingerprint()
        fp2 = identity.public_key_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_fingerprint_unique_per_key(self) -> None:
        from missy.security.identity import AgentIdentity

        id1 = AgentIdentity.generate()
        id2 = AgentIdentity.generate()
        assert id1.public_key_fingerprint() != id2.public_key_fingerprint()

    def test_to_jwk_structure(self) -> None:
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        jwk = identity.to_jwk()
        assert jwk["kty"] == "OKP"
        assert jwk["crv"] == "Ed25519"
        assert "x" in jwk
        assert isinstance(jwk["x"], str)

    def test_sign_verify_round_trip_after_save_load(self, tmp_path: Path) -> None:
        """Sign with original key, verify with loaded key."""
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.generate()
        msg = b"round trip test"
        sig = identity.sign(msg)

        key_path = str(tmp_path / "rt.pem")
        identity.save(key_path)
        loaded = AgentIdentity.from_key_file(key_path)

        assert loaded.verify(msg, sig) is True


# ---------------------------------------------------------------------------
# TrustScorer
# ---------------------------------------------------------------------------


class TestTrustScorerEdgeCases:
    """Fill coverage gaps for TrustScorer."""

    def test_default_score(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        assert scorer.score("new_entity") == 500

    def test_success_increases_score(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("tool1", weight=10)
        assert scorer.score("tool1") == 510

    def test_failure_decreases_score(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_failure("tool1", weight=50)
        assert scorer.score("tool1") == 450

    def test_violation_major_decrease(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_violation("tool1", weight=200)
        assert scorer.score("tool1") == 300

    def test_score_capped_at_max(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        for _ in range(100):
            scorer.record_success("tool1", weight=100)
        assert scorer.score("tool1") == 1000

    def test_score_floored_at_zero(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        for _ in range(10):
            scorer.record_violation("tool1", weight=200)
        assert scorer.score("tool1") == 0

    def test_is_trusted_above_threshold(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        assert scorer.is_trusted("tool1", threshold=200) is True  # 500 > 200

    def test_is_trusted_below_threshold(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_violation("tool1", weight=400)  # 500 - 400 = 100
        assert scorer.is_trusted("tool1", threshold=200) is False

    def test_is_trusted_at_boundary(self) -> None:
        """At exactly the threshold, should return False (strictly greater)."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        # Score = 200 exactly
        scorer.record_failure("tool1", weight=300)  # 500 - 300 = 200
        assert scorer.is_trusted("tool1", threshold=200) is False
        assert scorer.is_trusted("tool1", threshold=199) is True

    def test_reset_restores_default(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_violation("tool1")
        assert scorer.score("tool1") != 500
        scorer.reset("tool1")
        assert scorer.score("tool1") == 500

    def test_get_scores_returns_copy(self) -> None:
        """get_scores() should return a copy that's mutation-resistant."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("tool1")
        scores = scorer.get_scores()
        scores["tool1"] = 9999  # mutate the copy
        assert scorer.score("tool1") != 9999  # original unchanged

    def test_get_scores_empty(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        assert scorer.get_scores() == {}

    def test_multiple_entities(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("a")
        scorer.record_failure("b")
        scorer.record_violation("c")
        assert scorer.score("a") == 510
        assert scorer.score("b") == 450
        assert scorer.score("c") == 300

    def test_concurrent_access(self) -> None:
        """Multiple threads updating scores shouldn't corrupt state."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        barrier = threading.Barrier(10)
        errors: list[Exception] = []

        def worker(entity: str, success: bool):
            try:
                barrier.wait()
                for _ in range(100):
                    if success:
                        scorer.record_success(entity, weight=1)
                    else:
                        scorer.record_failure(entity, weight=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"e{i}", i % 2 == 0)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All entities should have valid scores (0 <= score <= 1000)
        for i in range(10):
            s = scorer.score(f"e{i}")
            assert 0 <= s <= 1000

    def test_zero_weight_no_change(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("tool1", weight=0)
        assert scorer.score("tool1") == 500

    def test_large_weight_clamps(self) -> None:
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        scorer.record_success("tool1", weight=999999)
        assert scorer.score("tool1") == 1000

        scorer.record_failure("tool2", weight=999999)
        assert scorer.score("tool2") == 0
