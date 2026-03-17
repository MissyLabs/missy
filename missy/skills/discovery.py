"""Dynamic skill discovery via SKILL.md files.

Supports the SKILL.md open standard for cross-agent skill portability.
Skills are markdown files with YAML frontmatter that define capabilities.

Example SKILL.md::

    ---
    name: web-search
    description: Search the web using DuckDuckGo
    version: 1.0.0
    author: MissyLabs
    tools: [web_fetch]
    ---

    # Instructions
    When the user asks to search the web...

Usage::

    from missy.skills.discovery import SkillDiscovery

    discovery = SkillDiscovery()
    skills = discovery.scan_directory("~/.missy/skills")
    results = discovery.search("web", skills)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default directory for user-installed SKILL.md files.
DEFAULT_SKILLS_DIR = os.path.expanduser("~/.missy/skills")


@dataclass
class SkillManifest:
    """Parsed representation of a SKILL.md file.

    Attributes:
        name: Short identifier for the skill.
        description: Human-readable description.
        version: Semantic version string.
        author: Author or organisation name.
        tools: List of tool names the skill requires.
        instructions: Body content (markdown) below the frontmatter.
        path: Absolute filesystem path of the source SKILL.md file.
    """

    name: str
    description: str
    version: str
    author: str
    tools: list[str] = field(default_factory=list)
    instructions: str = ""
    path: str = ""


class SkillDiscovery:
    """Discover and parse SKILL.md files from the filesystem."""

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_directory(self, path: str) -> list[SkillManifest]:
        """Recursively find all SKILL.md files under *path*.

        Args:
            path: Directory to scan.  ``~`` is expanded.

        Returns:
            A list of parsed :class:`SkillManifest` objects.  Files that
            cannot be parsed are skipped with a warning.
        """
        root = Path(os.path.expanduser(path))
        if not root.is_dir():
            logger.debug("Skills directory does not exist: %s", root)
            return []

        manifests: list[SkillManifest] = []
        for skill_path in sorted(root.rglob("SKILL.md")):
            try:
                manifest = self.parse_skill_md(str(skill_path))
                manifests.append(manifest)
            except Exception as exc:
                logger.warning("Skipping %s: %s", skill_path, exc)
        return manifests

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_skill_md(self, path: str) -> SkillManifest:
        """Parse a single SKILL.md file into a :class:`SkillManifest`.

        The file must start with a YAML frontmatter block delimited by
        ``---`` lines.  The ``name`` field is required.

        Args:
            path: Absolute or relative path to the SKILL.md file.

        Returns:
            A populated :class:`SkillManifest`.

        Raises:
            ValueError: When the file has no valid frontmatter or is
                missing the required ``name`` field.
            FileNotFoundError: When the file does not exist.
        """
        resolved = Path(os.path.expanduser(path)).resolve()
        text = resolved.read_text(encoding="utf-8")

        frontmatter, body = self._split_frontmatter(text)
        if frontmatter is None:
            raise ValueError(f"No YAML frontmatter found in {resolved}")

        meta = self._parse_yaml(frontmatter)
        if not isinstance(meta, dict):
            raise ValueError(f"Frontmatter is not a YAML mapping in {resolved}")

        name = meta.get("name")
        if not name:
            raise ValueError(f"Missing required 'name' field in {resolved}")

        tools_raw = meta.get("tools", [])
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        elif isinstance(tools_raw, list):
            tools = [str(t) for t in tools_raw]
        else:
            tools = []

        return SkillManifest(
            name=str(name),
            description=str(meta.get("description", "")),
            version=str(meta.get("version", "")),
            author=str(meta.get("author", "")),
            tools=tools,
            instructions=body.strip(),
            path=str(resolved),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, skills: list[SkillManifest]) -> list[SkillManifest]:
        """Fuzzy-match skills by name and description.

        Performs a case-insensitive substring match of *query* against
        the ``name`` and ``description`` fields.  Name matches are
        ranked higher than description-only matches.

        Args:
            query: Search string.
            skills: Skill manifests to search through.

        Returns:
            Matching manifests sorted by relevance (name match first).
        """
        if not query:
            return list(skills)

        q = query.lower()
        name_matches: list[SkillManifest] = []
        desc_matches: list[SkillManifest] = []

        for skill in skills:
            if q in skill.name.lower():
                name_matches.append(skill)
            elif q in skill.description.lower():
                desc_matches.append(skill)

        return name_matches + desc_matches

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[str | None, str]:
        """Split text into (frontmatter, body).

        Returns ``(None, text)`` if no frontmatter delimiters are found.
        """
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return None, text

        # Find the closing delimiter.
        after_first = stripped[3:]
        # Skip the first line (may contain nothing after ---)
        newline_idx = after_first.find("\n")
        if newline_idx == -1:
            return None, text
        rest = after_first[newline_idx + 1 :]

        end_idx = rest.find("\n---")
        if end_idx == -1:
            return None, text

        frontmatter = rest[:end_idx]
        body = rest[end_idx + 4 :]  # skip "\n---"
        # Strip optional newline after closing ---
        if body.startswith("\n"):
            body = body[1:]
        return frontmatter, body

    @staticmethod
    def _parse_yaml(text: str) -> dict:
        """Minimal YAML-subset parser for frontmatter.

        Handles simple ``key: value`` pairs and inline lists
        (``[a, b, c]``).  Does NOT require PyYAML to be installed.
        """
        result: dict[str, str | list[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Inline list: [a, b, c]
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1]
                items = [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
                result[key] = items
            else:
                # Strip optional quotes
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                result[key] = value
        return result
