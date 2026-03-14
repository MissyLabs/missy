"""Tests for missy.skills.base covering BaseSkill.get_help and SkillResult."""

from __future__ import annotations

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult


class StubSkill(BaseSkill):
    name = "stub"
    description = "A stub skill"
    version = "2.0.0"
    permissions = SkillPermissions()

    def execute(self, **kwargs):
        return SkillResult(success=True, output="ok")


class TestSkillPermissions:
    def test_defaults_all_false(self):
        p = SkillPermissions()
        assert p.network is False
        assert p.filesystem_read is False
        assert p.filesystem_write is False
        assert p.shell is False


class TestSkillResult:
    def test_default_error_empty(self):
        r = SkillResult(success=True, output="hi")
        assert r.error == ""

    def test_error_result(self):
        r = SkillResult(success=False, output=None, error="bad")
        assert r.error == "bad"


class TestBaseSkillGetHelp:
    def test_get_help_format(self):
        skill = StubSkill()
        help_text = skill.get_help()
        assert help_text == "stub v2.0.0: A stub skill"

    def test_default_version(self):
        class MinSkill(BaseSkill):
            name = "min"
            description = "Minimal"
            permissions = SkillPermissions()

            def execute(self, **kwargs):
                return SkillResult(success=True, output="")

        skill = MinSkill()
        assert "v0.1.0" in skill.get_help()
