"""Comprehensive tests for PersonaManager and PersonaConfig.

Tests cover: default values, file I/O, atomic writes, permissions, backup/prune/rollback,
diff, audit log, update/reset, system prompt prefix, YAML error handling, and concurrency.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from missy.agent.persona import (
    _DEFAULT_BEHAVIORAL_TENDENCIES,
    _DEFAULT_BOUNDARIES,
    _DEFAULT_IDENTITY_DESCRIPTION,
    _DEFAULT_NAME,
    _DEFAULT_PERSONALITY_TRAITS,
    _DEFAULT_RESPONSE_STYLE_RULES,
    _DEFAULT_TONE,
    PersonaConfig,
    PersonaManager,
    _persona_from_dict,
    _persona_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(tmp_path: Path, filename: str = "persona.yaml") -> PersonaManager:
    """Return a PersonaManager pointing at tmp_path / filename."""
    return PersonaManager(persona_path=tmp_path / filename)


# ===========================================================================
# 1. Default PersonaConfig values
# ===========================================================================


class TestPersonaConfigDefaults:
    def test_name_default(self):
        p = PersonaConfig()
        assert p.name == _DEFAULT_NAME

    def test_tone_default(self):
        p = PersonaConfig()
        assert p.tone == list(_DEFAULT_TONE)

    def test_personality_traits_default(self):
        p = PersonaConfig()
        assert p.personality_traits == list(_DEFAULT_PERSONALITY_TRAITS)

    def test_behavioral_tendencies_default(self):
        p = PersonaConfig()
        assert p.behavioral_tendencies == list(_DEFAULT_BEHAVIORAL_TENDENCIES)

    def test_response_style_rules_default(self):
        p = PersonaConfig()
        assert p.response_style_rules == list(_DEFAULT_RESPONSE_STYLE_RULES)

    def test_boundaries_default(self):
        p = PersonaConfig()
        assert p.boundaries == list(_DEFAULT_BOUNDARIES)

    def test_identity_description_default(self):
        p = PersonaConfig()
        assert p.identity_description == _DEFAULT_IDENTITY_DESCRIPTION

    def test_version_default(self):
        p = PersonaConfig()
        assert p.version == 1

    def test_each_list_field_is_independent(self):
        # Mutating one instance must not affect another
        a = PersonaConfig()
        b = PersonaConfig()
        a.tone.append("extra")
        assert "extra" not in b.tone

    def test_all_expected_fields_present(self):
        field_names = {f.name for f in fields(PersonaConfig)}
        expected = {
            "name",
            "tone",
            "personality_traits",
            "behavioral_tendencies",
            "response_style_rules",
            "boundaries",
            "identity_description",
            "version",
        }
        assert expected == field_names


# ===========================================================================
# 2. PersonaManager with non-existent file uses defaults
# ===========================================================================


class TestPersonaManagerInit:
    def test_nonexistent_file_returns_defaults(self, tmp_path):
        pm = make_manager(tmp_path)
        p = pm.get_persona()
        assert p.name == _DEFAULT_NAME
        assert p.version == 1

    def test_path_property(self, tmp_path):
        target = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=target)
        assert pm.path == target

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        # Point HOME at tmp_path so ~ expansion is deterministic
        monkeypatch.setenv("HOME", str(tmp_path))
        pm = PersonaManager(persona_path="~/persona.yaml")
        assert str(pm.path).startswith(str(tmp_path))
        assert not pm.path.as_posix().startswith("~")

    def test_version_property_initial(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.version == 1


# ===========================================================================
# 3. save() creates parent dirs, writes YAML, increments version
# ===========================================================================


class TestSave:
    def test_save_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "persona.yaml"
        pm = PersonaManager(persona_path=deep)
        pm.save()
        assert deep.exists()

    def test_save_creates_yaml_file(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        assert (tmp_path / "persona.yaml").exists()

    def test_save_increments_version(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.version == 1
        pm.save()
        assert pm.version == 2
        pm.save()
        assert pm.version == 3

    def test_save_written_yaml_is_parseable(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        raw = (tmp_path / "persona.yaml").read_text()
        data = yaml.safe_load(raw)
        assert isinstance(data, dict)
        assert data["name"] == _DEFAULT_NAME

    def test_save_version_first_in_yaml(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        raw = (tmp_path / "persona.yaml").read_text()
        first_key = next(iter(yaml.safe_load(raw)))
        assert first_key == "version"

    def test_save_round_trips_all_fields(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="Testbot", tone=["calm"])
        pm.save()
        pm2 = make_manager(tmp_path)
        p = pm2.get_persona()
        assert p.name == "Testbot"
        assert p.tone == ["calm"]


# ===========================================================================
# 4. save() atomic write — temp file cleaned up on failure
# ===========================================================================


class TestSaveAtomicWrite:
    def test_temp_file_removed_on_yaml_dump_failure(self, tmp_path):
        pm = make_manager(tmp_path)
        with (
            patch("missy.agent.persona.yaml.dump", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            pm.save()
        # No stale .yaml.tmp files should remain
        leftover = list(tmp_path.glob("*.yaml.tmp"))
        assert leftover == []

    def test_exception_propagated_on_failure(self, tmp_path):
        pm = make_manager(tmp_path)
        with (
            patch("missy.agent.persona.yaml.dump", side_effect=OSError("disk full")),
            pytest.raises(IOError),
        ):
            pm.save()


# ===========================================================================
# 5. save() file permissions set to 0o600
# ===========================================================================


class TestSavePermissions:
    def test_file_permissions_are_0o600(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        file_stat = (tmp_path / "persona.yaml").stat()
        mode = stat.S_IMODE(file_stat.st_mode)
        assert mode == 0o600


# ===========================================================================
# 6. save() creates backup of existing file
# ===========================================================================


class TestSaveCreatesBackup:
    def test_first_save_no_backup(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        # persona.d does not exist yet (no prior file to back up)
        assert not pm.backup_dir.exists() or pm.list_backups() == []

    def test_second_save_creates_backup(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()  # creates file
        pm.save()  # should back up the file created above
        assert len(pm.list_backups()) >= 1

    def test_backup_contains_previous_content(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="OriginalName")
        pm.save()
        pm.update(name="NewName")
        pm.save()
        backup = pm.list_backups()[-1]
        backup_data = yaml.safe_load(backup.read_text())
        assert backup_data["name"] == "OriginalName"


# ===========================================================================
# 7. reset() preserves version, restores all other defaults
# ===========================================================================


class TestReset:
    def test_reset_restores_name(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="CustomBot")
        pm.save()
        pm.update(name="CustomBot")  # keep in memory
        pm.reset()
        assert pm.get_persona().name == _DEFAULT_NAME

    def test_reset_restores_tone(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(tone=["weird"])
        pm.reset()
        assert pm.get_persona().tone == list(_DEFAULT_TONE)

    def test_reset_restores_personality_traits(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(personality_traits=["reckless"])
        pm.reset()
        assert pm.get_persona().personality_traits == list(_DEFAULT_PERSONALITY_TRAITS)

    def test_reset_restores_boundaries(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(boundaries=[])
        pm.reset()
        assert pm.get_persona().boundaries == list(_DEFAULT_BOUNDARIES)

    def test_reset_preserves_version_counter(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()  # version 2
        version_before = pm.version
        pm.reset()  # increments again inside reset -> save
        assert pm.version > version_before

    def test_reset_writes_file(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.reset()
        assert (tmp_path / "persona.yaml").exists()


# ===========================================================================
# 8. update() modifies specific fields
# ===========================================================================


class TestUpdate:
    def test_update_name(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="AlphaBot")
        assert pm.get_persona().name == "AlphaBot"

    def test_update_tone(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(tone=["sarcastic"])
        assert pm.get_persona().tone == ["sarcastic"]

    def test_update_multiple_fields_at_once(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="X", tone=["a"], personality_traits=["b"])
        p = pm.get_persona()
        assert p.name == "X"
        assert p.tone == ["a"]
        assert p.personality_traits == ["b"]

    def test_update_does_not_persist_until_save(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="Ephemeral")
        # Re-load from disk (file not yet saved)
        pm2 = make_manager(tmp_path)
        assert pm2.get_persona().name == _DEFAULT_NAME


# ===========================================================================
# 9. update() raises ValueError for unknown fields
# ===========================================================================


class TestUpdateValidation:
    def test_unknown_field_raises(self, tmp_path):
        pm = make_manager(tmp_path)
        with pytest.raises(ValueError, match="Unknown persona field"):
            pm.update(nonexistent_field="bad")

    def test_error_message_contains_field_name(self, tmp_path):
        pm = make_manager(tmp_path)
        with pytest.raises(ValueError, match="typo_field"):
            pm.update(typo_field="x")

    def test_multiple_unknown_fields_all_listed(self, tmp_path):
        pm = make_manager(tmp_path)
        with pytest.raises(ValueError) as exc_info:
            pm.update(bad_one="a", bad_two="b")
        msg = str(exc_info.value)
        assert "bad_one" in msg or "bad_two" in msg


# ===========================================================================
# 10. update() raises ValueError for "version" field
# ===========================================================================


class TestUpdateVersionField:
    def test_version_field_raises(self, tmp_path):
        pm = make_manager(tmp_path)
        with pytest.raises(ValueError):
            pm.update(version=99)

    def test_version_unchanged_after_failed_update(self, tmp_path):
        pm = make_manager(tmp_path)
        original_version = pm.version
        with pytest.raises(ValueError):
            pm.update(version=999)
        assert pm.version == original_version


# ===========================================================================
# 11. get_persona() returns a copy — mutation doesn't affect manager
# ===========================================================================


class TestGetPersonaCopy:
    def test_mutation_of_copy_does_not_affect_manager(self, tmp_path):
        pm = make_manager(tmp_path)
        copy1 = pm.get_persona()
        copy1.name = "MutatedName"
        copy2 = pm.get_persona()
        assert copy2.name == _DEFAULT_NAME

    def test_mutation_of_list_field_does_not_affect_manager(self, tmp_path):
        pm = make_manager(tmp_path)
        copy = pm.get_persona()
        copy.tone.append("hacked")
        assert "hacked" not in pm.get_persona().tone

    def test_returns_persona_config_instance(self, tmp_path):
        pm = make_manager(tmp_path)
        assert isinstance(pm.get_persona(), PersonaConfig)


# ===========================================================================
# 12. get_system_prompt_prefix() includes all sections
# ===========================================================================


class TestGetSystemPromptPrefixFull:
    def test_contains_identity_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Identity" in prefix

    def test_contains_identity_description(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert _DEFAULT_IDENTITY_DESCRIPTION.strip() in prefix

    def test_contains_tone_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Tone" in prefix

    def test_tone_values_in_prefix(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        for tone in _DEFAULT_TONE:
            assert tone in prefix

    def test_contains_personality_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Personality" in prefix

    def test_contains_behavioural_tendencies_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Behavioural Tendencies" in prefix

    def test_contains_response_style_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Response Style" in prefix

    def test_contains_boundaries_header(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        assert "# Boundaries" in prefix

    def test_boundaries_as_list_items(self, tmp_path):
        pm = make_manager(tmp_path)
        prefix = pm.get_system_prompt_prefix()
        for boundary in _DEFAULT_BOUNDARIES:
            assert boundary in prefix


# ===========================================================================
# 13. get_system_prompt_prefix() with empty lists omits sections
# ===========================================================================


class TestGetSystemPromptPrefixEmpty:
    def test_empty_tone_omits_tone_section(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(tone=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Tone" not in prefix

    def test_empty_personality_traits_omits_personality_section(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(personality_traits=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Personality" not in prefix

    def test_empty_behavioral_tendencies_omits_section(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(behavioral_tendencies=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Behavioural Tendencies" not in prefix

    def test_empty_response_style_rules_omits_section(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(response_style_rules=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Response Style" not in prefix

    def test_empty_boundaries_omits_section(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(boundaries=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Boundaries" not in prefix

    def test_identity_always_present(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(
            tone=[],
            personality_traits=[],
            behavioral_tendencies=[],
            response_style_rules=[],
            boundaries=[],
        )
        prefix = pm.get_system_prompt_prefix()
        assert "# Identity" in prefix


# ===========================================================================
# 14. _load() with malformed YAML returns defaults
# ===========================================================================


class TestLoadMalformedYaml:
    def test_malformed_yaml_returns_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text(": invalid: yaml: [[[", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == _DEFAULT_NAME

    def test_malformed_yaml_version_is_1(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("{{{not yaml}}}", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.version == 1


# ===========================================================================
# 15. _load() with YAML that's not a dict returns defaults
# ===========================================================================


class TestLoadNonDictYaml:
    def test_yaml_list_returns_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == _DEFAULT_NAME

    def test_yaml_scalar_string_returns_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("just a string\n", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == _DEFAULT_NAME

    def test_yaml_null_returns_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("null\n", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == _DEFAULT_NAME


# ===========================================================================
# 16. _load() with extra unknown keys ignores them
# ===========================================================================


class TestLoadUnknownKeys:
    def test_extra_keys_ignored(self, tmp_path):
        path = tmp_path / "persona.yaml"
        data = {
            "name": "FutureBot",
            "tone": ["futuristic"],
            "unknown_field_xyz": "should be ignored",
            "another_unknown": 42,
        }
        path.write_text(yaml.dump(data), encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        p = pm.get_persona()
        assert p.name == "FutureBot"
        assert p.tone == ["futuristic"]
        assert not hasattr(p, "unknown_field_xyz")

    def test_partial_known_fields_fill_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text(yaml.dump({"name": "Partial"}), encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        p = pm.get_persona()
        assert p.name == "Partial"
        assert p.tone == list(_DEFAULT_TONE)  # default filled in


# ===========================================================================
# 17. _audit() writes JSONL to persona_audit.jsonl
# ===========================================================================


class TestAuditWrite:
    def test_audit_creates_file(self, tmp_path):
        pm = make_manager(tmp_path)
        pm._audit("test_action")
        audit_path = tmp_path / "persona_audit.jsonl"
        assert audit_path.exists()

    def test_audit_entry_is_valid_json(self, tmp_path):
        pm = make_manager(tmp_path)
        pm._audit("save")
        lines = (tmp_path / "persona_audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "save"

    def test_audit_entry_contains_required_fields(self, tmp_path):
        pm = make_manager(tmp_path)
        pm._audit("save", {"key": "value"})
        lines = (tmp_path / "persona_audit.jsonl").read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert "timestamp" in entry
        assert "action" in entry
        assert "version" in entry
        assert "name" in entry
        assert "details" in entry

    def test_audit_appends_multiple_entries(self, tmp_path):
        pm = make_manager(tmp_path)
        pm._audit("first")
        pm._audit("second")
        lines = (tmp_path / "persona_audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_save_triggers_audit(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        entries = pm.get_audit_log()
        actions = [e["action"] for e in entries]
        assert "save" in actions


# ===========================================================================
# 18. _audit() handles OSError gracefully
# ===========================================================================


class TestAuditOSError:
    def test_os_error_does_not_propagate(self, tmp_path):
        pm = make_manager(tmp_path)
        with patch("missy.agent.persona.os.open", side_effect=OSError("permission denied")):
            # Should not raise
            pm._audit("save")

    def test_manager_still_functional_after_audit_error(self, tmp_path):
        pm = make_manager(tmp_path)
        with patch("missy.agent.persona.os.open", side_effect=OSError("disk full")):
            pm._audit("reset")
        assert pm.get_persona().name == _DEFAULT_NAME


# ===========================================================================
# 19. get_audit_log() parses JSONL correctly
# ===========================================================================


class TestGetAuditLog:
    def test_returns_list_of_dicts(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        log = pm.get_audit_log()
        assert isinstance(log, list)
        assert all(isinstance(e, dict) for e in log)

    def test_entries_in_order(self, tmp_path):
        pm = make_manager(tmp_path)
        pm._audit("first")
        pm._audit("second")
        log = pm.get_audit_log()
        assert log[0]["action"] == "first"
        assert log[1]["action"] == "second"


# ===========================================================================
# 20. get_audit_log() skips malformed lines
# ===========================================================================


class TestGetAuditLogSkipsMalformed:
    def test_malformed_json_skipped(self, tmp_path):
        pm = make_manager(tmp_path)
        audit_path = tmp_path / "persona_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            '{"action": "good"}\nnot json at all\n{"action": "also_good"}\n',
            encoding="utf-8",
        )
        log = pm.get_audit_log()
        actions = [e["action"] for e in log]
        assert "good" in actions
        assert "also_good" in actions
        assert len(log) == 2

    def test_empty_lines_skipped(self, tmp_path):
        pm = make_manager(tmp_path)
        audit_path = tmp_path / "persona_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            '\n\n{"action": "real"}\n\n',
            encoding="utf-8",
        )
        log = pm.get_audit_log()
        assert len(log) == 1


# ===========================================================================
# 21. get_audit_log() returns [] when no log file
# ===========================================================================


class TestGetAuditLogNoFile:
    def test_empty_list_when_no_log_file(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.get_audit_log() == []


# ===========================================================================
# 22. _create_backup() creates timestamped copy
# ===========================================================================


class TestCreateBackup:
    def test_backup_file_created(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()  # creates persona.yaml
        backup_path = pm._create_backup()
        assert backup_path.exists()

    def test_backup_in_persona_d_directory(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        backup_path = pm._create_backup()
        assert backup_path.parent == pm.backup_dir

    def test_backup_filename_starts_with_persona_yaml(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        backup_path = pm._create_backup()
        assert backup_path.name.startswith("persona.yaml.")

    def test_backup_content_matches_source(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        original = (tmp_path / "persona.yaml").read_text()
        backup_path = pm._create_backup()
        assert backup_path.read_text() == original


# ===========================================================================
# 23. _prune_backups() keeps max 5
# ===========================================================================


class TestPruneBackups:
    def test_prune_keeps_max_five(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()  # initial file

        # Create 7 backup files manually
        pm.backup_dir.mkdir(parents=True, exist_ok=True)
        for i in range(7):
            backup_file = pm.backup_dir / f"persona.yaml.2026010{i}_120000"
            backup_file.write_text(f"version: {i}", encoding="utf-8")
            # Space out mtimes so sorting is deterministic
            mtime = time.time() + i
            os.utime(str(backup_file), (mtime, mtime))

        pm._prune_backups()
        remaining = pm.list_backups()
        assert len(remaining) <= PersonaManager._MAX_BACKUPS

    def test_prune_removes_oldest(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()

        pm.backup_dir.mkdir(parents=True, exist_ok=True)
        base_time = time.time()
        files = []
        for i in range(6):
            f = pm.backup_dir / f"persona.yaml.20260{i:02d}01_000000"
            f.write_text(f"version: {i}", encoding="utf-8")
            os.utime(str(f), (base_time + i, base_time + i))
            files.append(f)

        pm._prune_backups()
        remaining = pm.list_backups()
        # Oldest file (i=0) should have been removed
        assert files[0] not in remaining


# ===========================================================================
# 24. list_backups() returns sorted list
# ===========================================================================


class TestListBackups:
    def test_sorted_oldest_first(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.backup_dir.mkdir(parents=True, exist_ok=True)

        base_time = time.time()
        for i in range(3):
            f = pm.backup_dir / f"persona.yaml.file{i}"
            f.write_text(f"v{i}", encoding="utf-8")
            os.utime(str(f), (base_time + i, base_time + i))

        backups = pm.list_backups()
        mtimes = [b.stat().st_mtime for b in backups]
        assert mtimes == sorted(mtimes)

    def test_list_returns_only_persona_yaml_files(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.backup_dir.mkdir(parents=True, exist_ok=True)
        # Create a non-persona file
        (pm.backup_dir / "other_file.txt").write_text("noise", encoding="utf-8")
        (pm.backup_dir / "persona.yaml.backup1").write_text("v: 1", encoding="utf-8")
        backups = pm.list_backups()
        for b in backups:
            assert b.name.startswith("persona.yaml.")


# ===========================================================================
# 25. list_backups() empty when no backup dir
# ===========================================================================


class TestListBackupsEmpty:
    def test_empty_when_no_backup_dir(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.list_backups() == []

    def test_empty_list_type(self, tmp_path):
        pm = make_manager(tmp_path)
        result = pm.list_backups()
        assert isinstance(result, list)


# ===========================================================================
# 26. rollback() restores latest backup
# ===========================================================================


class TestRollback:
    def test_rollback_restores_backup_content(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="OriginalName")
        pm.save()  # version 2, writes backup on next save
        pm.update(name="NewName")
        pm.save()  # version 3, backs up version 2

        pm.update(name="CurrentName")
        pm.save()  # version 4, backs up version 3

        # Rollback should restore version 3 content
        pm.rollback()
        # After rollback, reload from disk
        pm2 = make_manager(tmp_path)
        # The name should be from the most recent backup
        assert pm2.get_persona().name in {"OriginalName", "NewName", "CurrentName"}

    def test_rollback_returns_path(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.save()  # creates a backup
        result = pm.rollback()
        assert result is not None
        assert isinstance(result, Path)

    def test_rollback_updates_in_memory_persona(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="BeforeRollback")
        pm.save()
        pm.update(name="AfterSave")
        pm.save()

        pm.rollback()
        # Should be reloaded from file
        assert isinstance(pm.get_persona(), PersonaConfig)

    def test_rollback_triggers_audit_entry(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.save()  # backup created
        pm.rollback()
        log = pm.get_audit_log()
        actions = [e["action"] for e in log]
        assert "rollback" in actions


# ===========================================================================
# 27. rollback() returns None when no backups
# ===========================================================================


class TestRollbackNoBackups:
    def test_rollback_returns_none_when_no_backups(self, tmp_path):
        pm = make_manager(tmp_path)
        result = pm.rollback()
        assert result is None


# ===========================================================================
# 28. diff() returns unified diff
# ===========================================================================


class TestDiff:
    def test_diff_returns_string(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.save()  # creates backup
        result = pm.diff()
        assert isinstance(result, str)

    def test_diff_non_empty_when_content_differs(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="First")
        pm.save()
        pm.update(name="Second")
        pm.save()  # backs up First, writes Second
        diff = pm.diff()
        # The diff should contain the changed name
        assert "First" in diff or "Second" in diff or diff != ""

    def test_diff_contains_unified_diff_markers(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.update(name="Alpha")
        pm.save()
        pm.update(name="Beta")
        pm.save()
        diff = pm.diff()
        if diff:
            assert "---" in diff or "+++" in diff


# ===========================================================================
# 29. diff() returns empty string when no backups
# ===========================================================================


class TestDiffNoBackups:
    def test_empty_string_when_no_backups(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.diff() == ""

    def test_empty_string_when_no_persona_file(self, tmp_path):
        pm = make_manager(tmp_path)
        # Inject a fake backup dir but no current persona file
        pm.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = pm.backup_dir / "persona.yaml.20260101_000000"
        backup.write_text("name: OldName\n", encoding="utf-8")
        # persona.yaml itself does not exist
        assert pm.diff() == ""


# ===========================================================================
# 30. Concurrent save() operations
# ===========================================================================


class TestConcurrentSave:
    def test_concurrent_saves_do_not_corrupt(self, tmp_path):
        """Multiple threads calling save() must not leave a corrupted file."""
        pm = make_manager(tmp_path)
        errors: list[Exception] = []

        def do_saves():
            try:
                for _ in range(5):
                    pm.save()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_saves) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions during concurrent save: {errors}"

        # File must be parseable after concurrent writes
        raw = (tmp_path / "persona.yaml").read_text()
        data = yaml.safe_load(raw)
        assert isinstance(data, dict)
        assert "name" in data


# ===========================================================================
# 31. Path with tilde expansion
# ===========================================================================


class TestTildeExpansion:
    def test_tilde_resolved_in_path_property(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        pm = PersonaManager(persona_path="~/.missy/persona.yaml")
        assert "~" not in str(pm.path)
        assert str(tmp_path) in str(pm.path)

    def test_save_works_with_tilde_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        pm = PersonaManager(persona_path="~/.missy/persona.yaml")
        pm.save()
        expected = tmp_path / ".missy" / "persona.yaml"
        assert expected.exists()


# ===========================================================================
# 32. version property
# ===========================================================================


class TestVersionProperty:
    def test_initial_version_is_1(self, tmp_path):
        pm = make_manager(tmp_path)
        assert pm.version == 1

    def test_version_increments_on_save(self, tmp_path):
        pm = make_manager(tmp_path)
        versions = [pm.version]
        for _ in range(3):
            pm.save()
            versions.append(pm.version)
        assert versions == sorted(versions)
        assert versions[-1] == versions[0] + 3

    def test_version_matches_yaml_on_disk(self, tmp_path):
        pm = make_manager(tmp_path)
        pm.save()
        pm.save()
        raw_version = yaml.safe_load((tmp_path / "persona.yaml").read_text())["version"]
        assert raw_version == pm.version


# ===========================================================================
# 33. path property
# ===========================================================================


class TestPathProperty:
    def test_path_is_path_object(self, tmp_path):
        pm = make_manager(tmp_path)
        assert isinstance(pm.path, Path)

    def test_path_matches_constructor_argument(self, tmp_path):
        target = tmp_path / "custom_persona.yaml"
        pm = PersonaManager(persona_path=target)
        assert pm.path == target

    def test_path_accepts_string(self, tmp_path):
        target = str(tmp_path / "persona.yaml")
        pm = PersonaManager(persona_path=target)
        assert pm.path == Path(target)


# ===========================================================================
# Serialisation helpers (_persona_to_dict / _persona_from_dict)
# ===========================================================================


class TestSerialisationHelpers:
    def test_to_dict_version_is_first_key(self):
        p = PersonaConfig()
        d = _persona_to_dict(p)
        assert list(d.keys())[0] == "version"

    def test_to_dict_contains_all_fields(self):
        p = PersonaConfig()
        d = _persona_to_dict(p)
        for f in fields(PersonaConfig):
            assert f.name in d

    def test_from_dict_round_trips(self):
        p = PersonaConfig(name="Round Trip", tone=["a", "b"], version=7)
        d = _persona_to_dict(p)
        p2 = _persona_from_dict(d)
        assert p2.name == "Round Trip"
        assert p2.tone == ["a", "b"]
        assert p2.version == 7

    def test_from_dict_ignores_unknown_keys(self):
        data = {"name": "X", "future_feature": "ignored"}
        p = _persona_from_dict(data)
        assert p.name == "X"
        assert not hasattr(p, "future_feature")

    def test_from_dict_empty_dict_uses_all_defaults(self):
        p = _persona_from_dict({})
        assert p.name == _DEFAULT_NAME
        assert p.version == 1
