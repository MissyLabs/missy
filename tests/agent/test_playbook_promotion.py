"""Tests for F20 — Playbook -> SKILL.md proposal materialization."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from missy.agent.playbook import Playbook, PlaybookEntry
from missy.skills.discovery import SkillDiscovery


@pytest.fixture
def playbook(tmp_path: Path) -> Playbook:
    return Playbook(store_path=str(tmp_path / "playbook.json"))


def _seed_promotable(pb: Playbook, n: int = 4) -> None:
    for _ in range(n):
        pb.record("shell", "deploy the app", ["shell_exec", "file_write"], "use rsync")


class TestWriteSkillProposal:
    def test_writes_discoverable_skill_md(self, playbook: Playbook, tmp_path: Path) -> None:
        entry = PlaybookEntry(
            pattern_id="abc123def456",
            task_type="shell",
            description="deploy the app",
            tool_sequence=["shell_exec", "file_write"],
            prompt_template="use rsync",
            success_count=4,
        )
        prop_dir = str(tmp_path / "proposals")
        path = playbook.write_skill_proposal(entry, proposals_dir=prop_dir)
        assert os.path.exists(path)
        # The REAL SkillDiscovery must parse it (valid frontmatter + name).
        manifests = SkillDiscovery().scan_directory(prop_dir)
        assert len(manifests) == 1
        m = manifests[0]
        assert m.name.startswith("playbook-shell-")
        assert m.tools == ["shell_exec", "file_write"]
        assert m.version == "0.1.0"

    def test_description_with_quotes_stays_valid_yaml(
        self, playbook: Playbook, tmp_path: Path
    ) -> None:
        entry = PlaybookEntry(
            pattern_id="quote123",
            task_type="shell",
            description='run "the thing" now',
            tool_sequence=["shell_exec"],
            prompt_template="hint",
            success_count=3,
        )
        prop_dir = str(tmp_path / "proposals")
        playbook.write_skill_proposal(entry, proposals_dir=prop_dir)
        # Must still parse (double-quotes were escaped to single).
        manifests = SkillDiscovery().scan_directory(prop_dir)
        assert len(manifests) == 1

    def test_empty_description_falls_back(self, playbook: Playbook, tmp_path: Path) -> None:
        entry = PlaybookEntry(
            pattern_id="file123",
            task_type="file",
            description="",
            tool_sequence=["file_read"],
            prompt_template="hint",
            success_count=3,
        )
        prop_dir = str(tmp_path / "proposals")
        playbook.write_skill_proposal(entry, proposals_dir=prop_dir)
        manifests = SkillDiscovery().scan_directory(prop_dir)
        assert "file_read" in manifests[0].description


class TestPromoteToSkills:
    def test_promotes_and_marks(self, playbook: Playbook, tmp_path: Path) -> None:
        _seed_promotable(playbook)
        prop_dir = str(tmp_path / "proposals")
        results = playbook.promote_to_skills(threshold=3, proposals_dir=prop_dir)
        assert len(results) == 1
        assert results[0]["task_type"] == "shell"
        assert results[0]["success_count"] == 4
        assert results[0]["path"] and os.path.exists(results[0]["path"])

    def test_idempotent(self, playbook: Playbook, tmp_path: Path) -> None:
        _seed_promotable(playbook)
        prop_dir = str(tmp_path / "proposals")
        first = playbook.promote_to_skills(threshold=3, proposals_dir=prop_dir)
        second = playbook.promote_to_skills(threshold=3, proposals_dir=prop_dir)
        assert len(first) == 1
        assert len(second) == 0  # already marked promoted

    def test_below_threshold_not_promoted(self, playbook: Playbook, tmp_path: Path) -> None:
        playbook.record("shell", "x", ["shell_exec"], "n")  # success_count 1
        results = playbook.promote_to_skills(threshold=3, proposals_dir=str(tmp_path / "p"))
        assert results == []

    def test_dry_run_writes_nothing(self, playbook: Playbook, tmp_path: Path) -> None:
        _seed_promotable(playbook)
        prop_dir = str(tmp_path / "proposals")
        results = playbook.promote_to_skills(threshold=3, proposals_dir=prop_dir, dry_run=True)
        assert len(results) == 1
        assert results[0]["path"] is None
        assert not os.path.exists(prop_dir)  # nothing written
        # And the pattern is NOT marked promoted (a real run still promotes it).
        real = playbook.promote_to_skills(threshold=3, proposals_dir=prop_dir)
        assert len(real) == 1
