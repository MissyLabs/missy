"""Requirement-focused direct checks discovered during validation run 19."""

from __future__ import annotations

import datetime
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.core.events import event_bus
from missy.skills.base import BaseSkill, SkillResult
from missy.skills.builtin.config_show import ConfigShowSkill
from missy.skills.builtin.datetime_info import DateTimeSkill
from missy.skills.builtin.health_check import HealthCheckSkill
from missy.skills.builtin.summarize_session import SummarizeSessionSkill
from missy.skills.builtin.system_info import SystemInfoSkill
from missy.skills.builtin.workspace_list import WorkspaceListSkill
from missy.skills.discovery import SkillDiscovery, SkillManifest
from missy.skills.registry import SkillRegistry, get_skill_registry, init_skill_registry


def _write_skill(tmp_path, frontmatter: str, body: str = "Body"):
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return path


def test_skill_035_builtin_ownership_is_truthfully_library_only() -> None:
    repo = Path(__file__).resolve().parents[2]
    builtin_names = (
        "SystemInfoSkill",
        "DateTimeSkill",
        "ConfigShowSkill",
        "HealthCheckSkill",
        "SummarizeSessionSkill",
        "WorkspaceListSkill",
    )
    documentation = (repo / "docs" / "skills-and-plugins.md").read_text(encoding="utf-8")
    assert "library-only" in documentation
    for name in builtin_names:
        assert f"`{name}`" in documentation

    # A future bootstrap must update the ownership documentation and this
    # tripwire together instead of silently making a library class live.
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo / "missy").rglob("*.py")
        if "skills/builtin" not in path.as_posix()
    )
    for name in builtin_names:
        assert f"{name}(" not in runtime_sources
    assert "init_skill_registry(" not in (repo / "missy" / "cli" / "main.py").read_text(
        encoding="utf-8"
    )


class TestSkill017FrontmatterMatrix:
    def test_crlf_and_delimiter_at_eof_are_accepted(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_bytes(b"---\r\nname: crlf-skill\r\n---")

        manifest = SkillDiscovery().parse_skill_md(str(path))

        assert manifest.name == "crlf-skill"
        assert manifest.instructions == ""

    @pytest.mark.parametrize(
        "content",
        [
            "\ufeff---\nname: bom-skill\n---\nBody",
            "---\nname: missing-close\nBody",
            "---\n---\nBody",
        ],
    )
    def test_malformed_or_unsupported_headers_reject_without_partial_registration(
        self, tmp_path, content
    ):
        bad = tmp_path / "bad" / "SKILL.md"
        good = tmp_path / "good" / "SKILL.md"
        bad.parent.mkdir()
        good.parent.mkdir()
        bad.write_text(content, encoding="utf-8")
        good.write_text("---\nname: good-skill\n---\nBody", encoding="utf-8")

        manifests = SkillDiscovery().scan_directory(str(tmp_path))

        assert [manifest.name for manifest in manifests] == ["good-skill"]

    def test_body_delimiter_is_body_content_after_header_close(self, tmp_path):
        path = _write_skill(tmp_path, "name: delimiter-skill", "First\n---\nSecond")

        manifest = SkillDiscovery().parse_skill_md(str(path))

        assert manifest.instructions == "First\n---\nSecond"


class TestSkill019ToolNameCoercion:
    def test_valid_names_are_trimmed_and_deduplicated_without_registry_mutation(self, tmp_path):
        registry = init_skill_registry()
        path = _write_skill(
            tmp_path,
            "name: tools-skill\ntools: web_fetch, file_read, web_fetch",
        )

        manifest = SkillDiscovery().parse_skill_md(str(path))

        assert manifest.tools == ["web_fetch", "file_read"]
        assert registry.list_skills() == []

    @pytest.mark.parametrize("bad_value", [{"shell_exec": True}, 7, ["file_read", 9]])
    def test_non_string_tool_declarations_are_rejected(self, tmp_path, bad_value):
        path = _write_skill(tmp_path, "name: tools-skill\ntools: placeholder")
        discovery = SkillDiscovery()
        with (
            patch.object(
                discovery,
                "_parse_yaml",
                return_value={"name": "tools-skill", "tools": bad_value},
            ),
            pytest.raises(ValueError, match="tools"),
        ):
            discovery.parse_skill_md(str(path))

    @pytest.mark.parametrize(
        "tools_line",
        [
            'tools: ["shell_exec,file_read"]',
            "tools: shell_exec\\nfile_read",
            "tools: shell_exec file_read",
            "tools: ../shell_exec",
        ],
    )
    def test_ambiguous_or_control_character_names_are_rejected(self, tmp_path, tools_line):
        path = _write_skill(tmp_path, f"name: tools-skill\n{tools_line}")

        with pytest.raises(ValueError, match="tools"):
            SkillDiscovery().parse_skill_md(str(path))


def _manifest(name: str, description: str = "") -> SkillManifest:
    return SkillManifest(name=name, description=description, version="1.0.0", author="test")


class TestSkill022UnicodeSearch:
    def test_composed_and_decomposed_accents_match(self):
        skills = [_manifest("café-search")]

        assert SkillDiscovery().search("cafe\u0301", skills) == skills

    def test_full_width_text_normalizes_but_homoglyph_does_not_alias(self):
        skills = [_manifest("file-reader"), _manifest("pаypal-audit")]
        discovery = SkillDiscovery()

        assert discovery.search("ｆｉｌｅ", skills) == [skills[0]]
        assert discovery.search("paypal", skills) == []

    def test_whitespace_and_separator_variants_are_stable(self):
        skills = [_manifest("web_search"), _manifest("web-search-pro")]
        discovery = SkillDiscovery()

        expected = discovery.search("  web   search  ", skills)
        assert expected == discovery.search("web-search", skills)
        assert expected == discovery.search("web_search", skills)

    def test_empty_and_broad_queries_are_bounded_and_repeatable(self):
        skills = [_manifest(f"skill-{index:03d}") for index in range(250)]
        discovery = SkillDiscovery()

        first = discovery.search("", skills)
        second = discovery.search("   ", skills)

        assert first == second
        assert len(first) == 100


class TestSkill029WorkspaceBoundary:
    def test_absolute_outside_dotdot_and_proc_are_denied(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        skill = WorkspaceListSkill(workspace_root=str(workspace))

        for path in (str(outside), "../outside", "/proc"):
            result = skill.execute(workspace_path=path)
            assert not result.success
            assert "outside" in result.error.lower()

    def test_symlink_does_not_reveal_outside_metadata(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret-name.txt").write_text("secret")
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
        skill = WorkspaceListSkill(workspace_root=str(workspace))

        direct = skill.execute(workspace_path="escape")
        listing = skill.execute()

        assert not direct.success
        assert "symlink" in direct.error.lower()
        assert listing.success
        assert "secret-name.txt" not in listing.output
        assert "[symlink] escape" in listing.output

    def test_allowed_nested_workspace_and_entry_bound(self, tmp_path):
        workspace = tmp_path / "workspace"
        nested = workspace / "allowed" / "nested"
        nested.mkdir(parents=True)
        for index in range(8):
            (nested / f"file-{index}.txt").write_text("ok")
        skill = WorkspaceListSkill(workspace_root=str(workspace), max_entries=3)

        result = skill.execute(workspace_path="allowed/nested")

        assert result.success
        assert result.output.count("[file]") == 3
        assert "[truncated]" in result.output


class _Run19Skill(BaseSkill):
    name = "run19_skill"
    description = "Validation run 19 fixture"
    version = "1.0.0"

    def execute(self, **kwargs):
        return SkillResult(success=True, output=kwargs)


def _authorized_registry() -> SkillRegistry:
    return SkillRegistry(permission_authorizer=lambda _name, _permissions: True)


class TestSkill014PermissionEnforcement:
    def test_default_registry_denies_before_invoking_skill(self):
        called = False

        class Untrusted(_Run19Skill):
            name = "untrusted_skill"

            def execute(self, **kwargs):
                nonlocal called
                called = True
                return SkillResult(success=True, output="should not run")

        event_bus.clear()
        registry = SkillRegistry()
        registry.register(Untrusted())

        result = registry.execute("untrusted_skill", session_id="s", task_id="t")

        assert not result.success
        assert not called
        event = event_bus.get_events(event_type="skill.execute")[-1]
        assert event.result == "deny"

    def test_authorizer_exception_fails_closed(self):
        registry = SkillRegistry(
            permission_authorizer=lambda _name, _permissions: (_ for _ in ()).throw(
                RuntimeError("policy unavailable")
            )
        )
        registry.register(_Run19Skill())

        result = registry.execute("run19_skill")

        assert not result.success
        assert "denied" in result.error.lower()


class TestSkill025SecretSafeExceptions:
    def test_exception_result_log_and_audit_are_redacted(self, caplog):
        raw_secrets = (
            "api_key=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 "
            "Bearer abcdefghijklmnopqrstuvwxyz123456 "
            "token=COOKIESECRETABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "password=filesystem-secret-value"
        )

        class Failing(_Run19Skill):
            name = "secret_failing_skill"

            def execute(self, **kwargs):
                raise RuntimeError(raw_secrets)

        event_bus.clear()
        registry = _authorized_registry()
        registry.register(Failing())

        result = registry.execute("secret_failing_skill")
        event = event_bus.get_events(event_type="skill.execute")[-1]
        published = f"{result.error}\n{caplog.text}\n{event.detail}"

        assert not result.success
        for secret in (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
            "abcdefghijklmnopqrstuvwxyz123456",
            "COOKIESECRETABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "filesystem-secret-value",
        ):
            assert secret not in published

    def test_audit_failure_does_not_repeat_execution_or_expose_secret(self):
        calls = 0

        class Failing(_Run19Skill):
            name = "audit_failing_skill"

            def execute(self, **kwargs):
                nonlocal calls
                calls += 1
                raise RuntimeError("api_key=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")

        registry = _authorized_registry()
        registry.register(Failing())
        with patch.object(event_bus, "publish", side_effect=RuntimeError("audit unavailable")):
            result = registry.execute("audit_failing_skill")

        assert calls == 1
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" not in result.error


class TestSkill026RegistryConcurrency:
    def test_same_name_collision_has_one_winner_and_coherent_execution(self):
        registry = _authorized_registry()
        barrier = threading.Barrier(3)
        winners: list[_Run19Skill] = []
        failures: list[Exception] = []

        class First(_Run19Skill):
            def execute(self, **kwargs):
                return SkillResult(success=True, output="first")

        class Second(_Run19Skill):
            def execute(self, **kwargs):
                return SkillResult(success=True, output="second")

        def register(skill):
            barrier.wait()
            try:
                registry.register(skill)
                winners.append(skill)
            except Exception as exc:  # collision is the expected losing outcome
                failures.append(exc)

        threads = [
            threading.Thread(target=register, args=(skill,)) for skill in (First(), Second())
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)

        assert len(winners) == 1
        assert len(failures) == 1
        result = registry.execute("run19_skill")
        assert result.output == ("first" if isinstance(winners[0], First) else "second")

    def test_different_names_have_one_audit_event_per_execution(self):
        event_bus.clear()
        registry = _authorized_registry()
        skills = []
        for index in range(32):
            skill_type = type(
                f"ConcurrentSkill{index}",
                (_Run19Skill,),
                {"name": f"concurrent_skill_{index}"},
            )
            skill = skill_type()
            skills.append(skill)
            registry.register(skill)

        threads = [
            threading.Thread(target=registry.execute, args=(skill.name,)) for skill in skills
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)

        events = event_bus.get_events(event_type="skill.execute")
        assert len(events) == len(skills)
        assert sorted(registry.list_skills()) == sorted(skill.name for skill in skills)


class TestSkill027SingletonIsolation:
    def test_reinitialization_never_leaks_previous_registry_state(self):
        first = init_skill_registry()
        first.register(_Run19Skill())

        second = init_skill_registry()

        assert second is get_skill_registry()
        assert second is not first
        assert second.list_skills() == []
        assert first.list_skills() == ["run19_skill"]

    def test_discovery_roots_are_isolated(self, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _write_skill(root_a, "name: only-a")
        _write_skill(root_b, "name: only-b")
        discovery = SkillDiscovery()

        assert [item.name for item in discovery.scan_directory(str(root_a))] == ["only-a"]
        assert [item.name for item in discovery.scan_directory(str(root_b))] == ["only-b"]


class TestSkill037BuiltinSignatures:
    @pytest.mark.parametrize(
        "skill,arguments",
        [
            (SystemInfoSkill(), {"unexpected": "x" * 100_000}),
            (DateTimeSkill(), {"unexpected": None}),
            (ConfigShowSkill(), {"unexpected": []}),
            (HealthCheckSkill(), {"unexpected": True}),
            (SummarizeSessionSkill(), {"session_id": "s", "unexpected": {}}),
            (WorkspaceListSkill(), {"unexpected": False}),
        ],
    )
    def test_unknown_arguments_fail_before_implementation(self, skill, arguments):
        result = skill.execute(**arguments)

        assert not result.success
        assert result.output is None
        assert result.error == "Unknown arguments: unexpected"

    @pytest.mark.parametrize(
        "skill,arguments,error_fragment",
        [
            (DateTimeSkill(), {"timezone": None}, "timezone must be a string"),
            (WorkspaceListSkill(), {"workspace_path": None}, "must be a string"),
            (SummarizeSessionSkill(), {"session_id": []}, "session_id is required"),
            (SummarizeSessionSkill(), {"session_id": True}, "session_id is required"),
        ],
    )
    def test_wrong_typed_known_arguments_fail_cleanly(self, skill, arguments, error_fragment):
        result = skill.execute(**arguments)

        assert not result.success
        assert error_fragment in result.error


def _output_fields(result: SkillResult) -> dict[str, str]:
    assert result.success, result.error
    return dict(line.split(": ", 1) for line in result.output.splitlines())


class TestSkill038DateTimeBoundaries:
    def test_clock_is_called_once_and_iana_dst_offsets_are_unambiguous(self):
        calls = 0

        def spring_clock():
            nonlocal calls
            calls += 1
            return datetime.datetime(2026, 3, 8, 7, 30, tzinfo=datetime.UTC)

        fields = _output_fields(
            DateTimeSkill(clock=spring_clock).execute(timezone="America/New_York")
        )

        assert calls == 1
        assert fields["datetime_local"].startswith("2026-03-08T03:30:00-04:00")
        assert fields["timezone"] == "America/New_York"

        fall_fields = _output_fields(
            DateTimeSkill(
                clock=lambda: datetime.datetime(2026, 11, 1, 6, 30, tzinfo=datetime.UTC)
            ).execute(timezone="America/New_York")
        )
        assert fall_fields["datetime_local"].startswith("2026-11-01T01:30:00-05:00")

    def test_leap_day_and_explicit_utc_offset(self):
        fields = _output_fields(
            DateTimeSkill(
                clock=lambda: datetime.datetime(2024, 2, 29, 23, 0, tzinfo=datetime.UTC)
            ).execute(timezone="+05:30")
        )

        assert fields["datetime_utc"].startswith("2024-02-29T23:00:00+00:00")
        assert fields["datetime_local"].startswith("2024-03-01T04:30:00+05:30")
        assert fields["timezone"] == "UTC+05:30"

    @pytest.mark.parametrize(
        "timezone",
        ["CST", "../etc/passwd", "/etc/localtime", "America/New_York\nforged", "+15:00"],
    )
    def test_ambiguous_invalid_and_hostile_timezones_fail_cleanly(self, timezone):
        result = DateTimeSkill(
            clock=lambda: datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        ).execute(timezone=timezone)

        assert not result.success
        assert result.output is None
        assert len(result.error) < 160


class TestSkill039SystemInformation:
    def test_fields_are_independent_bounded_redacted_and_namespace_qualified(self):
        token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        with (
            patch(
                "missy.skills.builtin.system_info.socket.gethostname",
                return_value=f"api_key={token}",
            ),
            patch(
                "missy.skills.builtin.system_info.platform.system", side_effect=OSError("missing")
            ),
            patch("missy.skills.builtin.system_info.platform.release", return_value="r" * 10_000),
            patch("missy.skills.builtin.system_info.platform.machine", return_value="x86_64"),
        ):
            fields = _output_fields(SystemInfoSkill().execute())

        assert fields["scope"] == "process-visible namespace; not physical-host attestation"
        assert token not in fields["hostname"]
        assert "[REDACTED]" in fields["hostname"]
        assert fields["os"] == "unavailable"
        assert len(fields["os_release"]) == 256
        assert fields["machine"] == "x86_64"

    def test_censor_failure_is_per_field_and_never_returns_raw_value(self):
        token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        with (
            patch(
                "missy.skills.builtin.system_info.socket.gethostname",
                return_value=f"api_key={token}",
            ),
            patch("missy.security.censor.censor_response", side_effect=RuntimeError("down")),
        ):
            fields = _output_fields(SystemInfoSkill().execute())

        assert token not in str(fields)
        assert fields["hostname"] == "unavailable (redaction failed)"
        assert fields["scope"].startswith("process-visible namespace")


class TestSkill040SessionSummaryBoundary:
    def test_default_denies_and_bound_skill_cannot_read_another_session(self):
        default = SummarizeSessionSkill()
        bound = SummarizeSessionSkill(authorized_session_id="allowed-session")

        assert not default.execute(session_id="allowed-session").success
        denied = bound.execute(session_id="other-session")
        assert not denied.success
        assert "denied" in denied.error.lower()

    def test_configured_database_symlink_is_refused(self, tmp_path):
        target = tmp_path / "target.db"
        target.write_text("not a database")
        link = tmp_path / "memory.db"
        link.symlink_to(target)
        skill = SummarizeSessionSkill(
            authorized_session_id="allowed-session",
            db_path=str(link),
        )

        result = skill.execute(session_id="allowed-session")

        assert not result.success
        assert "symlink" in result.error.lower()

    def test_corrupt_database_fails_without_leaking_path_or_exception(self, tmp_path):
        db_path = tmp_path / "private-secret-name.db"
        db_path.write_bytes(b"not sqlite")
        skill = SummarizeSessionSkill(
            authorized_session_id="allowed-session",
            db_path=str(db_path),
        )

        result = skill.execute(session_id="allowed-session")

        assert not result.success
        assert result.error == "Memory store unavailable or unreadable."
        assert str(db_path) not in result.error

    def test_turn_content_is_redacted_and_truncation_is_labeled(self, tmp_path):
        token = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        turn = MagicMock(
            timestamp="2026-01-01T00:00:00+00:00",
            role="user",
            content=f"api_key={token} " + "x" * 500,
        )
        store = MagicMock()
        store.get_session_turns.return_value = [turn]
        skill = SummarizeSessionSkill(
            authorized_session_id="allowed-session",
            db_path=str(tmp_path / "memory.db"),
        )

        with patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=store):
            result = skill.execute(session_id="allowed-session")

        assert result.success
        assert token not in result.output
        assert "[REDACTED]" in result.output
        assert "…" in result.output
        assert "max 20" in result.output
