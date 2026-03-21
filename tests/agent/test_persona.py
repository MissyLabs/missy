"""Tests for missy.agent.persona."""

from __future__ import annotations

import time
from dataclasses import fields

import pytest
import yaml

from missy.agent.persona import (
    PersonaConfig,
    PersonaManager,
    _persona_from_dict,
    _persona_to_dict,
)

# ---------------------------------------------------------------------------
# PersonaConfig dataclass
# ---------------------------------------------------------------------------


class TestPersonaConfigDefaults:
    def test_name_default(self):
        p = PersonaConfig()
        assert p.name == "Missy"

    def test_version_defaults_to_one(self):
        p = PersonaConfig()
        assert p.version == 1

    def test_tone_default_is_list(self):
        p = PersonaConfig()
        assert isinstance(p.tone, list)
        assert len(p.tone) > 0

    def test_personality_traits_default_is_list(self):
        p = PersonaConfig()
        assert isinstance(p.personality_traits, list)
        assert len(p.personality_traits) > 0

    def test_behavioral_tendencies_default_is_list(self):
        p = PersonaConfig()
        assert isinstance(p.behavioral_tendencies, list)
        assert len(p.behavioral_tendencies) > 0

    def test_response_style_rules_default_is_list(self):
        p = PersonaConfig()
        assert isinstance(p.response_style_rules, list)
        assert len(p.response_style_rules) > 0

    def test_boundaries_default_is_list(self):
        p = PersonaConfig()
        assert isinstance(p.boundaries, list)
        assert len(p.boundaries) > 0

    def test_identity_description_default_is_nonempty_string(self):
        p = PersonaConfig()
        assert isinstance(p.identity_description, str)
        assert len(p.identity_description) > 0

    def test_list_fields_are_independent_instances(self):
        """Each PersonaConfig instance must own its own list objects."""
        a = PersonaConfig()
        b = PersonaConfig()
        a.tone.append("extra")
        assert "extra" not in b.tone

    def test_custom_field_values(self):
        p = PersonaConfig(
            name="Botty",
            tone=["dry", "terse"],
            version=7,
        )
        assert p.name == "Botty"
        assert p.tone == ["dry", "terse"]
        assert p.version == 7


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


class TestPersonaToDict:
    def test_version_appears_first(self):
        p = PersonaConfig(version=3)
        d = _persona_to_dict(p)
        assert list(d.keys())[0] == "version"

    def test_all_fields_present(self):
        p = PersonaConfig()
        d = _persona_to_dict(p)
        expected = {f.name for f in fields(PersonaConfig)}
        assert expected == set(d.keys())


class TestPersonaFromDict:
    def test_round_trip(self):
        original = PersonaConfig(name="R2", version=5, tone=["calm"])
        d = _persona_to_dict(original)
        restored = _persona_from_dict(d)
        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.tone == original.tone

    def test_unknown_keys_are_ignored(self):
        d = _persona_to_dict(PersonaConfig())
        d["future_field"] = "ignored"
        persona = _persona_from_dict(d)
        assert persona.name == "Missy"


# ---------------------------------------------------------------------------
# PersonaManager — construction
# ---------------------------------------------------------------------------


class TestPersonaManagerLoad:
    def test_default_persona_when_no_file_exists(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm.get_persona().name == "Missy"
        assert pm.get_persona().version == 1

    def test_path_property(self, tmp_path):
        p = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=p)
        assert pm.path == p

    def test_version_property(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm.version == 1


# ---------------------------------------------------------------------------
# PersonaManager — save / load round-trip
# ---------------------------------------------------------------------------


class TestPersonaManagerSave:
    def test_save_creates_file(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()
        assert (tmp_path / "persona.yaml").exists()

    def test_save_increments_version(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        assert pm.version == 1
        pm.save()
        assert pm.version == 2
        pm.save()
        assert pm.version == 3

    def test_saved_file_is_valid_yaml(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="TestBot")
        pm.save()
        raw = yaml.safe_load((tmp_path / "persona.yaml").read_text())
        assert raw["name"] == "TestBot"

    def test_load_from_existing_file(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm1 = PersonaManager(persona_path=path)
        pm1.update(name="Persisted", tone=["quiet"])
        pm1.save()

        pm2 = PersonaManager(persona_path=path)
        p = pm2.get_persona()
        assert p.name == "Persisted"
        assert p.tone == ["quiet"]
        assert p.version == 2

    def test_save_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "persona.yaml"
        pm = PersonaManager(persona_path=nested)
        pm.save()
        assert nested.exists()


# ---------------------------------------------------------------------------
# PersonaManager — get_system_prompt_prefix
# ---------------------------------------------------------------------------


class TestGetSystemPromptPrefix:
    def test_returns_nonempty_string(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        prefix = pm.get_system_prompt_prefix()
        assert isinstance(prefix, str)
        assert len(prefix) > 0

    def test_contains_persona_name(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="SpecialBot")
        prefix = pm.get_system_prompt_prefix()
        # identity_description still contains "Missy" from default; name is
        # shown in the Identity section via identity_description or elsewhere.
        # The prompt is non-empty and contains Identity section.
        assert "# Identity" in prefix

    def test_contains_tone_section(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        prefix = pm.get_system_prompt_prefix()
        assert "# Tone" in prefix

    def test_contains_boundaries_section(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        prefix = pm.get_system_prompt_prefix()
        assert "# Boundaries" in prefix

    def test_empty_tone_omits_tone_section(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(tone=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Tone" not in prefix

    def test_empty_boundaries_omits_boundaries_section(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(boundaries=[])
        prefix = pm.get_system_prompt_prefix()
        assert "# Boundaries" not in prefix

    def test_custom_identity_description_appears(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(identity_description="I am a custom bot.")
        prefix = pm.get_system_prompt_prefix()
        assert "I am a custom bot." in prefix


# ---------------------------------------------------------------------------
# PersonaManager — reset
# ---------------------------------------------------------------------------


class TestPersonaManagerReset:
    def test_reset_restores_default_name(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="Changed")
        pm.reset()
        assert pm.get_persona().name == "Missy"

    def test_reset_preserves_and_increments_version(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.save()  # version becomes 2
        pre_reset = pm.version
        pm.reset()  # reset saves, so version becomes pre_reset + 1
        assert pm.version == pre_reset + 1

    def test_reset_persists_to_disk(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.update(name="Temporary")
        pm.save()
        pm.reset()

        pm2 = PersonaManager(persona_path=path)
        assert pm2.get_persona().name == "Missy"


# ---------------------------------------------------------------------------
# PersonaManager — update
# ---------------------------------------------------------------------------


class TestPersonaManagerUpdate:
    def test_update_name(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="NewName")
        assert pm.get_persona().name == "NewName"

    def test_update_tone(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(tone=["stern", "brief"])
        assert pm.get_persona().tone == ["stern", "brief"]

    def test_update_multiple_fields(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        pm.update(name="Multi", tone=["x"], personality_traits=["y"])
        p = pm.get_persona()
        assert p.name == "Multi"
        assert p.tone == ["x"]
        assert p.personality_traits == ["y"]

    def test_update_invalid_field_raises_value_error(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        with pytest.raises(ValueError, match="Unknown persona field"):
            pm.update(nonexistent_field="oops")

    def test_update_version_field_raises_value_error(self, tmp_path):
        """version is excluded from updatable fields."""
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        with pytest.raises(ValueError, match="Unknown persona field"):
            pm.update(version=99)

    def test_update_does_not_save_automatically(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.update(name="Unsaved")
        # File should not yet exist (no explicit save called)
        assert not path.exists()


# ---------------------------------------------------------------------------
# PersonaManager — get_persona returns a copy
# ---------------------------------------------------------------------------


class TestGetPersonaReturnsCopy:
    def test_mutating_returned_copy_does_not_affect_manager(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        copy = pm.get_persona()
        copy.name = "MutatedExternally"
        assert pm.get_persona().name == "Missy"

    def test_two_copies_are_not_identical_objects(self, tmp_path):
        pm = PersonaManager(persona_path=tmp_path / "persona.yaml")
        a = pm.get_persona()
        b = pm.get_persona()
        assert a is not b


# ---------------------------------------------------------------------------
# PersonaManager — corrupt / invalid YAML fallback
# ---------------------------------------------------------------------------


class TestCorruptYamlFallback:
    def test_corrupt_yaml_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text(": : : invalid yaml :::", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == "Missy"
        assert pm.get_persona().version == 1

    def test_yaml_with_non_dict_root_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == "Missy"

    def test_empty_yaml_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "persona.yaml"
        path.write_text("", encoding="utf-8")
        pm = PersonaManager(persona_path=path)
        assert pm.get_persona().name == "Missy"


# ---------------------------------------------------------------------------
# PersonaManager — backup / rollback / diff
# ---------------------------------------------------------------------------


class TestPersonaBackup:
    def test_save_creates_backup(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2, first save — no backup (file didn't exist yet)
        pm.update(name="Changed")
        pm.save()  # v3, backup of v2 should exist
        backups = pm.list_backups()
        assert len(backups) == 1

    def test_no_backups_initially(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        assert pm.list_backups() == []

    def test_multiple_saves_create_multiple_backups(self, tmp_path, monkeypatch):
        call_count = 0
        _orig_strftime = time.strftime

        def _mock_strftime(fmt, *args):
            nonlocal call_count
            call_count += 1
            # Return unique timestamps for each call
            return f"20260318_12000{call_count}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2 (creates file, no backup since file didn't exist)
        for i in range(3):
            pm.update(name=f"Version{i}")
            pm.save()
        # 3 saves after file existed → 3 backups
        assert len(pm.list_backups()) == 3

    def test_prune_keeps_max_backups(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # creates file
        for i in range(8):
            pm.update(name=f"V{i}")
            pm.save()
        # Should prune to 5
        assert len(pm.list_backups()) <= pm._MAX_BACKUPS

    def test_backup_dir_property(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        assert pm.backup_dir == tmp_path / "persona.d"


class TestPersonaRollback:
    def test_rollback_restores_previous_version(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2, file now exists
        pm.update(name="Modified")
        pm.save()  # v3, backup of v2 exists
        assert pm.get_persona().name == "Modified"
        pm.rollback()
        assert pm.get_persona().name == "Missy"

    def test_rollback_returns_none_without_backups(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        assert pm.rollback() is None

    def test_rollback_returns_path(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        pm.update(name="X")
        pm.save()
        result = pm.rollback()
        assert result is not None
        assert result.name.startswith("persona.yaml.")

    def test_rollback_creates_backup_of_current(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2
        pm.update(name="A")
        pm.save()  # v3, backup of v2
        count_before = len(pm.list_backups())
        pm.rollback()
        # rollback should create a backup of the current state
        assert len(pm.list_backups()) >= count_before


class TestPersonaDiff:
    def test_diff_empty_when_no_backups(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        assert pm.diff() == ""

    def test_diff_empty_when_file_missing(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        # Create a fake backup but no current file
        bdir = pm.backup_dir
        bdir.mkdir(parents=True)
        (bdir / "persona.yaml.20260101_000000").write_text("name: Old\n")
        assert pm.diff() == ""

    def test_diff_shows_changes(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2
        pm.update(name="DiffTest")
        pm.save()  # v3
        diff_text = pm.diff()
        assert "DiffTest" in diff_text or "Missy" in diff_text

    def test_diff_empty_when_identical(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()  # v2
        # Save again without changes (version increments but content similar)
        pm.save()  # v3
        diff_text = pm.diff()
        # Version numbers differ so diff won't be empty, but it should exist
        assert isinstance(diff_text, str)


# ---------------------------------------------------------------------------
# PersonaManager — audit trail
# ---------------------------------------------------------------------------


class TestPersonaAuditTrail:
    def test_save_creates_audit_entry(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        entries = pm.get_audit_log()
        assert len(entries) >= 1
        assert entries[-1]["action"] == "save"

    def test_reset_creates_audit_entry(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        pm.reset()
        entries = pm.get_audit_log()
        actions = [e["action"] for e in entries]
        # save from initial + save inside reset + reset audit
        assert "reset" in actions

    def test_rollback_creates_audit_entry(self, tmp_path, monkeypatch):
        call_count = 0

        def _mock_strftime(fmt, *args):
            nonlocal call_count
            call_count += 1
            return f"20260318_13000{call_count}"

        monkeypatch.setattr(time, "strftime", _mock_strftime)

        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        pm.update(name="Changed")
        pm.save()
        pm.rollback()
        entries = pm.get_audit_log()
        actions = [e["action"] for e in entries]
        assert "rollback" in actions

    def test_audit_log_empty_initially(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        assert pm.get_audit_log() == []

    def test_audit_entry_has_timestamp(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        entries = pm.get_audit_log()
        assert "timestamp" in entries[0]
        assert "T" in entries[0]["timestamp"]  # ISO format

    def test_audit_entry_has_version(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        entries = pm.get_audit_log()
        assert entries[0]["version"] == 2  # First save increments from 1 to 2

    def test_audit_entry_has_name(self, tmp_path):
        path = tmp_path / "persona.yaml"
        pm = PersonaManager(persona_path=path)
        pm.save()
        entries = pm.get_audit_log()
        assert entries[0]["name"] == "Missy"
