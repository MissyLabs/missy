"""Tests for missy.tools.intelligence.candidate_generator."""

from __future__ import annotations

from missy.tools.intelligence.candidate_generator import (
    CandidateGenerator,
    _derive_description,
    _derive_name,
    _derive_permissions,
    _derive_schema,
    _validate,
)
from missy.tools.intelligence.request_tracker import RequestPattern


def _make_pattern(
    representative: str = "summarise the quarterly report for me",
    count: int = 5,
    recent_count: int = 5,
    score: float = 0.9,
    common_tools: list[str] | None = None,
    key: str = "abc123",
) -> RequestPattern:
    return RequestPattern(
        pattern_key=key,
        representative=representative,
        count=count,
        recent_count=recent_count,
        frequency_score=score,
        common_tools=list(common_tools or []),
        first_seen="2026-01-01T00:00:00+00:00",
        last_seen="2026-06-01T00:00:00+00:00",
        example_messages=["sample message"],
    )


class TestDeriveHelpers:
    def test_derive_name_produces_safe_name(self):
        p = _make_pattern("summarise the quarterly report")
        name, err = _derive_name(p)
        assert err == ""
        assert name  # non-empty
        import re

        assert re.match(r"^[a-z][a-z0-9_-]{0,62}$", name)

    def test_derive_name_empty_representative(self):
        p = _make_pattern(representative="")
        name, err = _derive_name(p)
        assert err != ""

    def test_derive_description_includes_representative(self):
        p = _make_pattern("convert pdf to text")
        desc = _derive_description(p)
        assert "convert pdf to text" in desc

    def test_derive_description_truncates_long(self):
        p = _make_pattern("x" * 200)
        desc = _derive_description(p)
        assert len(desc) <= 200

    def test_derive_schema_has_properties(self):
        p = _make_pattern("convert the file format please")
        schema = _derive_schema(p)
        assert "properties" in schema
        assert schema["type"] == "object"

    def test_derive_permissions_read_from_read_keyword(self):
        p = _make_pattern("read and display the file contents")
        perms = _derive_permissions(p)
        assert perms.get("filesystem_read") is True

    def test_derive_permissions_network_from_fetch(self):
        p = _make_pattern("fetch the data from the remote url")
        perms = _derive_permissions(p)
        assert perms.get("network") is True

    def test_derive_permissions_no_shell_by_default(self):
        p = _make_pattern("run the shell command exec this")
        perms = _derive_permissions(p, allow_shell=False)
        assert not perms.get("shell", False)

    def test_derive_permissions_shell_allowed_when_flag_set(self):
        p = _make_pattern("run the shell command exec script bash")
        perms = _derive_permissions(p, allow_shell=True)
        assert perms.get("shell") is True


class TestValidate:
    def test_valid_passes(self):
        assert (
            _validate(
                "my_tool",
                {"type": "object", "properties": {"x": {"type": "string"}}, "required": []},
                {},
            )
            == ""
        )

    def test_bad_name(self):
        assert _validate("Bad Name!", {}, {}) != ""

    def test_too_many_params(self):
        props = {f"p{i}": {"type": "string"} for i in range(9)}
        assert _validate("ok", {"type": "object", "properties": props}, {}) != ""

    def test_unsafe_param_name(self):
        assert (
            _validate(
                "ok", {"type": "object", "properties": {"bad-param!": {}}, "required": []}, {}
            )
            != ""
        )

    def test_unknown_permission(self):
        assert _validate("ok", {}, {"sudo": True}) != ""


class TestCandidateGeneratorDisabled:
    def test_disabled_returns_failure(self):
        gen = CandidateGenerator(tool_creation_enabled=False)
        p = _make_pattern()
        result = gen.generate_from_pattern(p)
        assert not result.ok
        assert result.candidate is None
        assert "disabled" in result.reason

    def test_generate_from_schema_disabled(self):
        gen = CandidateGenerator(tool_creation_enabled=False)
        result = gen.generate_from_schema("my_tool", "desc", {})
        assert not result.ok


class TestCandidateGeneratorEnabled:
    def test_generates_candidate(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        p = _make_pattern()
        result = gen.generate_from_pattern(p)
        assert result.ok
        assert result.candidate is not None
        assert result.candidate.state.value == "proposed"

    def test_candidate_has_auto_generated_tag(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        result = gen.generate_from_pattern(_make_pattern())
        assert "auto_generated" in result.candidate.tags

    def test_candidate_provenance_includes_pattern(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        result = gen.generate_from_pattern(_make_pattern(key="mykey123"))
        assert "mykey123" in result.candidate.provenance

    def test_candidate_includes_examples(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        p = _make_pattern()
        p.example_messages = ["msg1", "msg2"]
        result = gen.generate_from_pattern(p)
        assert len(result.candidate.examples) <= 3

    def test_generate_from_schema_direct(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        result = gen.generate_from_schema(
            name="my_direct_tool",
            description="does things",
            parameters={"input": {"type": "string", "description": "...", "required": True}},
            provenance="manual",
        )
        assert result.ok
        assert result.candidate.name == "my_direct_tool"

    def test_invalid_name_fails(self):
        gen = CandidateGenerator(tool_creation_enabled=True)
        result = gen.generate_from_schema(
            name="BAD NAME!",
            description="d",
            parameters={},
        )
        assert not result.ok

    def test_generate_from_schema_denies_shell_without_allow_shell(self):
        """Regression: generate_from_pattern's _derive_permissions() gates
        "shell" behind allow_shell, but generate_from_schema() took
        caller-supplied permissions verbatim and only ran them through
        _validate() (which merely checks _SAFE_PERMISSIONS membership,
        which itself includes "shell") -- bypassing the class's own
        documented always-denied-without-override contract for this
        permission on the direct-schema path.
        """
        gen = CandidateGenerator(tool_creation_enabled=True, allow_shell=False)
        result = gen.generate_from_schema(
            name="run_shell_thing",
            description="does a shell thing",
            parameters={"command": {"type": "string", "required": True}},
            permissions={"shell": True},
        )
        assert not result.ok
        assert result.candidate is None

    def test_generate_from_schema_allows_shell_with_allow_shell(self):
        gen = CandidateGenerator(tool_creation_enabled=True, allow_shell=True)
        result = gen.generate_from_schema(
            name="run_shell_thing",
            description="does a shell thing",
            parameters={"command": {"type": "string", "required": True}},
            permissions={"shell": True},
        )
        assert result.ok
        assert result.candidate.permissions.get("shell") is True

    def test_owner_propagated(self):
        gen = CandidateGenerator(tool_creation_enabled=True, owner="test_owner")
        result = gen.generate_from_pattern(_make_pattern())
        assert result.candidate.owner == "test_owner"
