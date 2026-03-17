"""Tests for missy.skills.discovery — SKILL.md parsing and scanning."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.skills.discovery import SkillDiscovery, SkillManifest


@pytest.fixture
def discovery() -> SkillDiscovery:
    return SkillDiscovery()


@pytest.fixture
def skill_md_content() -> str:
    return (
        "---\n"
        "name: web-search\n"
        "description: Search the web using DuckDuckGo\n"
        "version: 1.0.0\n"
        "author: MissyLabs\n"
        "tools: [web_fetch]\n"
        "---\n"
        "\n"
        "# Instructions\n"
        "When the user asks to search the web, use web_fetch.\n"
    )


class TestParseSkillMd:
    """Test parse_skill_md with valid SKILL.md content."""

    def test_parse_skill_md(
        self, discovery: SkillDiscovery, skill_md_content: str, tmp_path: Path
    ) -> None:
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(skill_md_content)

        manifest = discovery.parse_skill_md(str(skill_file))

        assert manifest.name == "web-search"
        assert manifest.description == "Search the web using DuckDuckGo"
        assert manifest.version == "1.0.0"
        assert manifest.author == "MissyLabs"
        assert manifest.tools == ["web_fetch"]
        assert "web_fetch" in manifest.instructions
        assert manifest.path == str(skill_file.resolve())

    def test_parse_multiple_tools(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = (
            "---\n"
            "name: multi-tool\n"
            "description: Uses multiple tools\n"
            "version: 0.1.0\n"
            "author: Test\n"
            "tools: [web_fetch, shell_exec, file_read]\n"
            "---\n"
            "\n"
            "# Instructions\n"
            "Use all the tools.\n"
        )
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        manifest = discovery.parse_skill_md(str(skill_file))
        assert manifest.tools == ["web_fetch", "shell_exec", "file_read"]

    def test_parse_minimal_frontmatter(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = "---\nname: minimal\n---\n\nBody text.\n"
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        manifest = discovery.parse_skill_md(str(skill_file))
        assert manifest.name == "minimal"
        assert manifest.description == ""
        assert manifest.version == ""
        assert manifest.tools == []


class TestScanDirectory:
    """Test scan_directory finding SKILL.md files recursively."""

    def test_scan_directory(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        # Create two SKILL.md files in different subdirectories.
        (tmp_path / "skill-a").mkdir()
        (tmp_path / "skill-a" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Alpha skill\nversion: 1.0.0\nauthor: A\n---\n\nAlpha instructions.\n"
        )
        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text(
            "---\nname: beta\ndescription: Beta skill\nversion: 2.0.0\nauthor: B\n---\n\nBeta instructions.\n"
        )

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 2
        names = {m.name for m in manifests}
        assert names == {"alpha", "beta"}

    def test_empty_directory(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        manifests = discovery.scan_directory(str(tmp_path))
        assert manifests == []

    def test_nonexistent_directory(self, discovery: SkillDiscovery) -> None:
        manifests = discovery.scan_directory("/tmp/nonexistent_skill_dir_12345")
        assert manifests == []

    def test_skips_invalid_files(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        # Valid skill
        (tmp_path / "good").mkdir()
        (tmp_path / "good" / "SKILL.md").write_text("---\nname: good-skill\n---\n\nGood.\n")
        # Invalid skill (no frontmatter)
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "SKILL.md").write_text("No frontmatter here.")

        manifests = discovery.scan_directory(str(tmp_path))
        assert len(manifests) == 1
        assert manifests[0].name == "good-skill"


class TestSearch:
    """Test fuzzy search by name and description."""

    @pytest.fixture
    def sample_skills(self) -> list[SkillManifest]:
        return [
            SkillManifest(
                name="web-search", description="Search the web", version="1.0.0", author="A"
            ),
            SkillManifest(
                name="file-manager", description="Manage files on disk", version="1.0.0", author="B"
            ),
            SkillManifest(
                name="calculator",
                description="Perform math calculations",
                version="1.0.0",
                author="C",
            ),
        ]

    def test_search_by_name(
        self, discovery: SkillDiscovery, sample_skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("web", sample_skills)
        # "web-search" matches by name; description "Search the web" also
        # contains "web" but the skill already matched by name so it is
        # not duplicated in desc_matches.
        assert len(results) >= 1
        assert results[0].name == "web-search"

    def test_search_by_description(
        self, discovery: SkillDiscovery, sample_skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("math", sample_skills)
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_search_no_match(
        self, discovery: SkillDiscovery, sample_skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("nonexistent", sample_skills)
        assert results == []

    def test_search_empty_query(
        self, discovery: SkillDiscovery, sample_skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("", sample_skills)
        assert len(results) == len(sample_skills)

    def test_search_case_insensitive(
        self, discovery: SkillDiscovery, sample_skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("WEB", sample_skills)
        assert len(results) >= 1
        assert results[0].name == "web-search"


class TestMissingFrontmatter:
    """Test graceful handling of missing/invalid frontmatter."""

    def test_missing_frontmatter(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Just some markdown without frontmatter.")

        with pytest.raises(ValueError, match="No YAML frontmatter"):
            discovery.parse_skill_md(str(skill_file))

    def test_missing_name_field(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\ndescription: No name\n---\n\nBody.\n")

        with pytest.raises(ValueError, match="Missing required 'name'"):
            discovery.parse_skill_md(str(skill_file))

    def test_file_not_found(self, discovery: SkillDiscovery) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.parse_skill_md("/tmp/nonexistent_skill_file.md")
