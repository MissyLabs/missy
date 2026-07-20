"""Run-18 validation regressions for the live skill contract."""

from __future__ import annotations

import threading
from dataclasses import Field, FrozenInstanceError, field
from unittest.mock import patch

import pytest

from missy.core.events import event_bus
from missy.skills.base import BaseSkill, SkillPermissions, SkillResult
from missy.skills.registry import SkillRegistry


class _EchoSkill(BaseSkill):
    name = "validation_echo"
    description = "Validation echo"
    version = "1.0.0"

    def execute(self, **kwargs):
        return SkillResult(success=True, output=kwargs)


def test_skill_041_builtin_catalog_exports_all_six() -> None:
    from missy.skills import builtin

    assert sorted(builtin.__all__) == [
        "ConfigShowSkill",
        "DateTimeSkill",
        "HealthCheckSkill",
        "SummarizeSessionSkill",
        "SystemInfoSkill",
        "WorkspaceListSkill",
    ]
    assert all(getattr(builtin, name) is not None for name in builtin.__all__)


def test_skill_042_omitted_permission_is_immutable_deny_all() -> None:
    skill = _EchoSkill()
    assert skill.permissions == SkillPermissions()
    assert not isinstance(skill.permissions, Field)
    with pytest.raises(FrozenInstanceError):
        skill.permissions.network = True  # type: ignore[misc]


@pytest.mark.parametrize("permissions", [None, field(default_factory=SkillPermissions), object()])
def test_skill_042_invalid_permission_declarations_rejected(permissions) -> None:
    class Invalid(_EchoSkill):
        name = "invalid_permissions"

    Invalid.permissions = permissions
    with pytest.raises(TypeError, match="SkillPermissions"):
        SkillRegistry().register(Invalid())


def test_skill_043_result_contract_is_serializable_and_coherent() -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        SkillResult(success=True, output={"bad": {1, 2}})
    with pytest.raises(ValueError, match="cannot contain an error"):
        SkillResult(success=True, output=None, error="contradiction")

    class FailureOutput(_EchoSkill):
        name = "failure_output"

        def execute(self, **kwargs):
            return SkillResult(success=False, output="must not escape", error="failed")

    registry = SkillRegistry()
    registry.register(FailureOutput())
    result = registry.execute("failure_output")
    assert result.output is None


@pytest.mark.parametrize(
    "name,description,version",
    [
        ("../bad", "ok", "1.0.0"),
        ("BadCase", "ok", "1.0.0"),
        ("safe", "row\nforgery", "1.0.0"),
        ("safe", "ok", "latest"),
    ],
)
def test_skill_044_045_metadata_is_inert_and_canonical(name, description, version) -> None:
    class Invalid(_EchoSkill):
        pass

    Invalid.name = name
    Invalid.description = description
    Invalid.version = version
    with pytest.raises(ValueError):
        SkillRegistry().register(Invalid())


def test_skill_046_audit_uses_skill_taxonomy_and_provenance() -> None:
    event_bus.clear()
    registry = SkillRegistry()
    registry.register(_EchoSkill())
    registry.execute("validation_echo", session_id="s", task_id="t", value=1)
    event = event_bus.get_events(event_type="skill.execute")[-1]
    assert event.category == "skill"
    assert event.detail["subsystem"] == "skill"
    assert event.detail["implementation"].endswith("_EchoSkill")
    assert event.detail["version"] == "1.0.0"


def test_skill_047_censor_failure_never_returns_or_audits_raw_secret() -> None:
    class Failing(_EchoSkill):
        name = "failing_skill"

        def execute(self, **kwargs):
            raise RuntimeError("sk-test-abcdefghijklmnopqrstuvwxyz")

    event_bus.clear()
    registry = SkillRegistry()
    registry.register(Failing())
    with patch("missy.security.censor.censor_response", side_effect=RuntimeError("censor down")):
        result = registry.execute("failing_skill")
    assert "sk-test" not in result.error
    event = event_bus.get_events(event_type="skill.execute")[-1]
    assert "sk-test" not in str(event.detail)


def test_skill_048_arguments_are_snapshotted_from_caller_and_callee_mutation() -> None:
    entered = threading.Event()
    release = threading.Event()
    observed = {}

    class Mutating(_EchoSkill):
        name = "mutating_skill"

        def execute(self, **kwargs):
            entered.set()
            release.wait(2)
            observed.update(kwargs)
            kwargs["nested"]["value"] = "callee"
            return SkillResult(success=True, output=kwargs)

    registry = SkillRegistry()
    registry.register(Mutating())
    original = {"nested": {"value": "initial"}}
    holder = {}
    thread = threading.Thread(
        target=lambda: holder.setdefault("result", registry.execute("mutating_skill", **original))
    )
    thread.start()
    assert entered.wait(2)
    original["nested"]["value"] = "caller"
    release.set()
    thread.join(2)
    assert observed["nested"]["value"] == "callee"
    assert holder["result"].output["nested"]["value"] == "callee"
    assert original["nested"]["value"] == "caller"


def test_skill_049_recursive_cycle_terminates_without_corrupting_registry() -> None:
    registry = SkillRegistry()

    class Recursive(_EchoSkill):
        name = "recursive_skill"

        def execute(self, **kwargs):
            return registry.execute("recursive_skill")

    recursive = Recursive()
    registry.register(recursive)
    result = registry.execute("recursive_skill")
    assert not result.success
    assert "Recursive skill execution denied" in result.error
    assert registry.get("recursive_skill") is recursive
    assert registry.list_skills() == ["recursive_skill"]


def test_skill_050_unregister_refuses_active_then_revokes_name() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Blocking(_EchoSkill):
        name = "blocking_skill"

        def execute(self, **kwargs):
            entered.set()
            release.wait(2)
            return SkillResult(success=True, output="done")

    registry = SkillRegistry()
    registry.register(Blocking())
    thread = threading.Thread(target=lambda: registry.execute("blocking_skill"))
    thread.start()
    assert entered.wait(2)
    with pytest.raises(RuntimeError, match="active"):
        registry.unregister("blocking_skill")
    release.set()
    thread.join(2)
    registry.unregister("blocking_skill")
    assert registry.get("blocking_skill") is None
