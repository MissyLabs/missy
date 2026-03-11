"""Tests for missy.skills.registry."""

from __future__ import annotations

import pytest

from missy.core.events import AuditEvent, event_bus
from missy.skills.base import BaseSkill, SkillPermissions, SkillResult
from missy.skills.registry import SkillRegistry, get_skill_registry, init_skill_registry


# ---------------------------------------------------------------------------
# Concrete skill implementations for testing
# ---------------------------------------------------------------------------


class EchoSkill(BaseSkill):
    name = "echo"
    description = "Return input text"
    permissions = SkillPermissions()

    def execute(self, *, text: str = "") -> SkillResult:
        return SkillResult(success=True, output=text)


class FailingSkill(BaseSkill):
    name = "failing"
    description = "Always raises"
    permissions = SkillPermissions()

    def execute(self, **kwargs) -> SkillResult:
        raise RuntimeError("deliberate failure")


class UnhappySkill(BaseSkill):
    name = "unhappy"
    description = "Returns success=False"
    permissions = SkillPermissions()

    def execute(self, **kwargs) -> SkillResult:
        return SkillResult(success=False, output=None, error="unhappy path")


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry()


class TestSkillRegistryRegister:
    def test_register_adds_skill(self, registry: SkillRegistry):
        registry.register(EchoSkill())
        assert registry.get("echo") is not None

    def test_register_replaces_existing(self, registry: SkillRegistry):
        s1 = EchoSkill()
        s2 = EchoSkill()
        registry.register(s1)
        registry.register(s2)
        assert registry.get("echo") is s2

    def test_get_returns_none_for_unknown(self, registry: SkillRegistry):
        assert registry.get("not_here") is None


class TestSkillRegistryListSkills:
    def test_empty_registry_returns_empty_list(self, registry: SkillRegistry):
        assert registry.list_skills() == []

    def test_list_skills_sorted_alphabetically(self, registry: SkillRegistry):
        registry.register(UnhappySkill())
        registry.register(EchoSkill())
        registry.register(FailingSkill())
        assert registry.list_skills() == ["echo", "failing", "unhappy"]


class TestSkillRegistryExecute:
    def test_execute_success(self, registry: SkillRegistry):
        registry.register(EchoSkill())
        result = registry.execute("echo", text="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_execute_unknown_returns_failure(self, registry: SkillRegistry):
        result = registry.execute("unknown")
        assert result.success is False
        assert "unknown" in result.error

    def test_execute_raising_skill_returns_failure(self, registry: SkillRegistry):
        registry.register(FailingSkill())
        result = registry.execute("failing")
        assert result.success is False
        assert "deliberate failure" in result.error

    def test_execute_unhappy_result_propagated(self, registry: SkillRegistry):
        registry.register(UnhappySkill())
        result = registry.execute("unhappy")
        assert result.success is False
        assert result.error == "unhappy path"

    def test_execute_emits_audit_event_on_success(self, registry: SkillRegistry):
        event_bus.clear()
        registry.register(EchoSkill())
        registry.execute("echo", session_id="s1", task_id="t1", text="x")
        events = event_bus.get_events(event_type="skill.execute")
        assert len(events) == 1
        assert events[0].result == "allow"
        assert events[0].detail["skill"] == "echo"

    def test_execute_emits_audit_event_on_error(self, registry: SkillRegistry):
        event_bus.clear()
        registry.register(FailingSkill())
        registry.execute("failing", session_id="s1", task_id="t1")
        events = event_bus.get_events(event_type="skill.execute")
        assert len(events) == 1
        assert events[0].result == "error"

    def test_execute_emits_audit_event_when_skill_missing(self, registry: SkillRegistry):
        event_bus.clear()
        registry.execute("ghost", session_id="s", task_id="t")
        events = event_bus.get_events(event_type="skill.execute")
        assert len(events) == 1
        assert events[0].result == "error"

    def test_execute_passes_kwargs_to_skill(self, registry: SkillRegistry):
        registry.register(EchoSkill())
        result = registry.execute("echo", text="specific value")
        assert result.output == "specific value"


class TestSkillRegistrySingleton:
    def test_init_returns_registry(self):
        r = init_skill_registry()
        assert isinstance(r, SkillRegistry)

    def test_get_returns_same_instance(self):
        r = init_skill_registry()
        assert get_skill_registry() is r

    def test_get_before_init_raises(self, monkeypatch):
        import missy.skills.registry as mod
        monkeypatch.setattr(mod, "_registry", None)
        with pytest.raises(RuntimeError, match="SkillRegistry has not been initialised"):
            get_skill_registry()

    def test_second_init_replaces_first(self):
        r1 = init_skill_registry()
        r2 = init_skill_registry()
        assert r1 is not r2
        assert get_skill_registry() is r2
