"""Tests for TrustScorer persistence (F11)."""

from __future__ import annotations

import json
from pathlib import Path

from missy.security.trust import DEFAULT_SCORE, TrustScorer


class TestInMemoryModeUnchanged:
    def test_no_path_writes_nothing(self, tmp_path: Path) -> None:
        # A scorer with no persist_path must not create any file.
        scorer = TrustScorer()
        scorer.record_success("tool_a")
        scorer.record_failure("tool_b")
        assert list(tmp_path.iterdir()) == []
        assert scorer.score("tool_a") == DEFAULT_SCORE + 10
        assert scorer.score("tool_b") == DEFAULT_SCORE - 50


class TestPersistence:
    def test_scores_survive_new_instance(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        a = TrustScorer(persist_path=p)
        a.record_success("calculator")  # 510
        a.record_violation("shell_exec")  # 300
        # A brand-new instance (simulating a restart / separate process)
        # loads the persisted scores.
        b = TrustScorer(persist_path=p)
        assert b.score("calculator") == 510
        assert b.score("shell_exec") == 300

    def test_file_is_written_on_each_mutation(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        scorer = TrustScorer(persist_path=p)
        scorer.record_success("x")
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["x"] == DEFAULT_SCORE + 10

    def test_reset_is_persisted(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        a = TrustScorer(persist_path=p)
        a.record_failure("t")  # 450
        a.reset("t")  # back to 500
        b = TrustScorer(persist_path=p)
        assert b.score("t") == DEFAULT_SCORE

    def test_parent_dir_is_created(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "dir" / "trust.json"
        scorer = TrustScorer(persist_path=p)
        scorer.record_success("y")
        assert p.exists()

    def test_corrupt_file_starts_empty_not_crash(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        p.write_text("this is not json{{{")
        scorer = TrustScorer(persist_path=p)  # must not raise
        assert scorer.get_scores() == {}
        # And it recovers by overwriting on the next mutation.
        scorer.record_success("z")
        assert json.loads(p.read_text())["z"] == DEFAULT_SCORE + 10

    def test_non_dict_json_is_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        p.write_text("[1, 2, 3]")
        scorer = TrustScorer(persist_path=p)
        assert scorer.get_scores() == {}

    def test_non_numeric_values_filtered_on_load(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        p.write_text(json.dumps({"good": 700, "bad": "nope", "alsobad": True}))
        scorer = TrustScorer(persist_path=p)
        scores = scorer.get_scores()
        assert scores == {"good": 700}

    def test_missing_file_is_fine(self, tmp_path: Path) -> None:
        p = tmp_path / "does-not-exist.json"
        scorer = TrustScorer(persist_path=p)
        assert scorer.get_scores() == {}
        assert scorer.score("anything") == DEFAULT_SCORE

    def test_atomic_write_leaves_no_temp_files(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        scorer = TrustScorer(persist_path=p)
        for i in range(5):
            scorer.record_success(f"tool_{i}")
        leftovers = [f.name for f in tmp_path.iterdir() if f.name.startswith(".trust-")]
        assert leftovers == []
