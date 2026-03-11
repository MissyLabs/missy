"""Tests for missy.skills.base."""

from __future__ import annotations

import pytest

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult


class ConcreteSkill(BaseSkill):
    name = "concrete"
    description = "A concrete test skill"
    permissions = SkillPermissions(network=True)

    def execute(self, *, value: int = 0) -> SkillResult:
        return SkillResult(success=True, output=value * 2)


class TestSkillPermissionsDefaults:
    def test_all_default_to_false(self):
        p = SkillPermissions()
        assert p.network is False
        assert p.filesystem_read is False
        assert p.filesystem_write is False
        assert p.shell is False

    def test_can_set_individual_permissions(self):
        p = SkillPermissions(network=True, shell=True)
        assert p.network is True
        assert p.shell is True
        assert p.filesystem_read is False


class TestSkillResult:
    def test_success_result(self):
        r = SkillResult(success=True, output=42)
        assert r.success is True
        assert r.output == 42
        assert r.error == ""

    def test_failure_result_with_error(self):
        r = SkillResult(success=False, output=None, error="something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"


class TestBaseSkill:
    def test_execute_returns_skill_result(self):
        skill = ConcreteSkill()
        result = skill.execute(value=5)
        assert isinstance(result, SkillResult)
        assert result.output == 10

    def test_get_help_format(self):
        skill = ConcreteSkill()
        help_text = skill.get_help()
        assert help_text == "concrete v0.1.0: A concrete test skill"

    def test_default_version_is_0_1_0(self):
        assert ConcreteSkill.version == "0.1.0"

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseSkill()  # type: ignore[abstract]
