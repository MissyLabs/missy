"""Tests for controlled runtime loading of enabled tool candidates."""

from __future__ import annotations

from pathlib import Path

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.intelligence.candidate_loader import (
    CandidateDelegatedTool,
    CandidateRuntimeLoader,
)
from missy.tools.intelligence.candidate_store import (
    CandidateStore,
    ToolCandidate,
    ToolLifecycleState,
)
from missy.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    permissions = ToolPermissions()
    parameters = {
        "text": {
            "type": "string",
            "required": True,
        }
    }

    def execute(self, **kwargs):
        return ToolResult(success=True, output=kwargs["text"])


class TakenTool(EchoTool):
    name = "taken"


def _store(tmp_path: Path) -> CandidateStore:
    return CandidateStore(db_path=tmp_path / "candidates.db")


def _candidate(**overrides) -> ToolCandidate:
    candidate = ToolCandidate.create(
        name=overrides.pop("name", "echo_alias"),
        description="Reviewed echo alias",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        permissions={},
        provenance="unit test review",
        implementation={"type": "delegated_tool", "tool": "echo"},
    )
    candidate.state = ToolLifecycleState.ENABLED
    candidate.provider_enabled = {"mock": True}
    for key, value in overrides.items():
        setattr(candidate, key, value)
    return candidate


def test_loads_enabled_delegated_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(_candidate())
    registry = ToolRegistry()
    registry.register(EchoTool())

    report = CandidateRuntimeLoader(store, registry).load_enabled("mock")

    assert report.loaded == ["echo_alias"]
    loaded = registry.get("echo_alias")
    assert isinstance(loaded, CandidateDelegatedTool)
    assert loaded.get_schema()["parameters"]["required"] == ["text"]
    assert registry.execute("echo_alias", text="hello").output == "hello"


def test_skips_candidate_without_implementation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(_candidate(implementation={}))
    registry = ToolRegistry()
    registry.register(EchoTool())

    report = CandidateRuntimeLoader(store, registry).load_enabled("mock")

    assert report.loaded == []
    assert "implementation is missing" in report.skipped[0].reason
    assert registry.get("echo_alias") is None


def test_skips_candidate_when_provider_not_enabled(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(_candidate(provider_enabled={"mock": False, "openai": True}))
    registry = ToolRegistry()
    registry.register(EchoTool())

    report = CandidateRuntimeLoader(store, registry).load_enabled("mock")

    assert report.loaded == []
    assert "provider 'mock' is not enabled" in report.skipped[0].reason


def test_skips_candidate_with_existing_builtin_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(_candidate(name="taken"))
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(TakenTool())

    report = CandidateRuntimeLoader(store, registry).load_enabled("mock")

    assert report.loaded == []
    assert "already exists" in report.skipped[0].reason


def test_skips_candidate_with_invalid_schema(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add(
        _candidate(
            schema={
                "type": "object",
                "properties": {},
                "required": ["text"],
            }
        )
    )
    registry = ToolRegistry()
    registry.register(EchoTool())

    report = CandidateRuntimeLoader(store, registry).load_enabled("mock")

    assert report.loaded == []
    assert "undefined properties" in report.skipped[0].reason
