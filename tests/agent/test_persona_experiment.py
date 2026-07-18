"""Tests for persona A/B experiments (F24)."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.agent.persona import PersonaConfig
from missy.agent.persona_experiment import PersonaExperiment, variant_persona_from_current


def _exp(tmp_path: Path) -> PersonaExperiment:
    return PersonaExperiment(path=str(tmp_path / "pexp.json"))


def _persona(name: str, tone: list[str]) -> PersonaConfig:
    return PersonaConfig(name=name, tone=tone)


class TestVariants:
    def test_add_list_get(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("Missy", ["direct"]))
        e.add_variant("B", _persona("Missy", ["playful"]))
        assert e.list_variants() == ["A", "B"]
        assert e.get_variant_persona("A").tone == ["direct"]
        assert e.get_variant_persona("B").tone == ["playful"]

    def test_add_empty_name_rejected(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        with pytest.raises(ValueError):
            e.add_variant("  ", _persona("X", []))

    def test_replace_variant(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("Missy", ["direct"]))
        e.add_variant("A", _persona("Missy", ["warm"]))
        assert e.get_variant_persona("A").tone == ["warm"]

    def test_remove(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("Missy", ["direct"]))
        assert e.remove_variant("A") is True
        assert e.remove_variant("A") is False
        assert e.list_variants() == []

    def test_unknown_variant_persona_is_none(self, tmp_path: Path) -> None:
        assert _exp(tmp_path).get_variant_persona("nope") is None


class TestAssignment:
    def test_deterministic_and_stable(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("M", ["a"]))
        e.add_variant("B", _persona("M", ["b"]))
        first = e.assign("session-123")
        # Same key -> same variant every time (across fresh instances too).
        for _ in range(5):
            assert e.assign("session-123") == first
        e2 = PersonaExperiment(path=str(tmp_path / "pexp.json"))
        assert e2.assign("session-123") == first

    def test_distributes_across_variants(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        for v in ("A", "B", "C"):
            e.add_variant(v, _persona("M", [v]))
        seen = {e.assign(f"s{i}") for i in range(200)}
        assert seen == {"A", "B", "C"}  # all variants get traffic

    def test_no_variants_returns_none(self, tmp_path: Path) -> None:
        assert _exp(tmp_path).assign("x") is None

    def test_persona_for_returns_assigned(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("Missy", ["direct"]))
        e.add_variant("B", _persona("Missy", ["playful"]))
        key = "sess-xyz"
        assigned = e.assign(key)
        persona = e.persona_for(key)
        assert persona is not None
        assert persona.tone == (["direct"] if assigned == "A" else ["playful"])


class TestEnabled:
    def test_toggle_persists(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        assert e.enabled is False
        e.set_enabled(True)
        assert PersonaExperiment(path=str(tmp_path / "pexp.json")).enabled is True


class TestOutcomes:
    def test_record_and_results(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("M", ["a"]))
        e.add_variant("B", _persona("M", ["b"]))
        e.record_outcome("A", success=True)
        e.record_outcome("A", success=True)
        e.record_outcome("A", success=False, refused=True)
        e.record_outcome("B", success=True)
        res = e.results()
        assert res["A"]["n"] == 3
        assert res["A"]["success"] == 2
        assert res["A"]["success_rate"] == pytest.approx(2 / 3)
        assert res["A"]["refusal_rate"] == pytest.approx(1 / 3)
        assert res["B"]["n"] == 1
        assert res["B"]["success_rate"] == 1.0

    def test_record_unknown_variant_ignored(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.record_outcome("ghost", success=True)  # no-op, no crash
        assert e.results() == {}

    def test_results_empty_variant_zero_rates(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("M", ["a"]))
        assert e.results()["A"] == {
            "n": 0,
            "success": 0,
            "failure": 0,
            "refused": 0,
            "success_rate": 0.0,
            "refusal_rate": 0.0,
        }


class TestClearAndCorrupt:
    def test_clear(self, tmp_path: Path) -> None:
        e = _exp(tmp_path)
        e.add_variant("A", _persona("M", ["a"]))
        e.set_enabled(True)
        e.clear()
        assert e.list_variants() == []
        assert e.enabled is False

    def test_corrupt_file_recovers(self, tmp_path: Path) -> None:
        p = tmp_path / "pexp.json"
        p.write_text("not json{{{")
        e = PersonaExperiment(path=str(p))
        assert e.list_variants() == []
        e.add_variant("A", _persona("M", ["a"]))
        assert e.list_variants() == ["A"]


class TestRuntimeWiring:
    def _runtime(self):
        from unittest.mock import MagicMock, patch

        from missy.agent.runtime import AgentConfig, AgentRuntime
        from missy.providers.base import CompletionResponse

        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        provider.complete.return_value = CompletionResponse(
            content="ok", model="m", provider="fake", usage={}, raw={}, finish_reason="stop"
        )
        reg = MagicMock()
        reg.get.return_value = provider
        reg.get_available.return_value = [provider]
        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
            patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
        ):
            return AgentRuntime(AgentConfig(provider="fake"))

    def test_uses_base_persona_when_no_experiment(self, tmp_path: Path, monkeypatch) -> None:
        from missy.agent import persona_experiment as mod

        monkeypatch.setattr(mod, "DEFAULT_EXPERIMENT_PATH", str(tmp_path / "pexp.json"))
        runtime = self._runtime()
        base = PersonaConfig(name="BasePersona")
        runtime._persona_manager = type("PM", (), {"get_persona": lambda self: base})()
        assert runtime._persona_for_session("s1").name == "BasePersona"

    def test_uses_variant_when_experiment_enabled(self, tmp_path: Path, monkeypatch) -> None:
        from missy.agent import persona_experiment as mod

        monkeypatch.setattr(mod, "DEFAULT_EXPERIMENT_PATH", str(tmp_path / "pexp.json"))
        exp = PersonaExperiment(path=str(tmp_path / "pexp.json"))
        exp.add_variant("A", PersonaConfig(name="VariantA"))
        exp.add_variant("B", PersonaConfig(name="VariantB"))
        exp.set_enabled(True)

        runtime = self._runtime()
        runtime._persona_manager = type(
            "PM", (), {"get_persona": lambda self: PersonaConfig(name="BasePersona")}
        )()
        result = runtime._persona_for_session("sess-xyz")
        # It's one of the variants, not the base.
        assert result.name in {"VariantA", "VariantB"}
        # Deterministic: same session always the same variant.
        assert runtime._persona_for_session("sess-xyz").name == result.name


class TestVariantFromCurrent:
    def test_override_builds_variant(self, tmp_path: Path, monkeypatch) -> None:
        # Point PersonaManager at a temp persona so the snapshot is deterministic.
        from missy.agent import persona_experiment as mod

        monkeypatch.setattr(
            mod, "_snapshot_current_persona", lambda: PersonaConfig(name="Base", tone=["neutral"])
        )
        v = variant_persona_from_current(tone=["playful"])
        assert v.name == "Base"  # copied
        assert v.tone == ["playful"]  # overridden
