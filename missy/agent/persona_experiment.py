"""Persona A/B experiments (F24).

The agent has a single static persona (``persona.yaml``). ``PersonaExperiment``
adds lightweight A/B testing on top: register several named persona *variants*,
deterministically assign a variant to each session/channel (stable hashing, so
the same key always maps to the same variant — no flapping mid-conversation),
record per-variant outcomes (task success and refusals), and read back
success/refusal rates to compare variants on real behaviour.

Everything persists to a single JSON file and is pure logic (no infra), so it is
fully deterministic and testable. A variant's persona is a full
:class:`~missy.agent.persona.PersonaConfig` snapshot, so switching a session to a
variant is just handing that config to the persona-prefix builder; rollback is
"stop the experiment," leaving the base ``persona.yaml`` untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from missy.agent.persona import PersonaConfig, _persona_from_dict, _persona_to_dict

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_PATH = "~/.missy/persona_experiment.json"


class PersonaExperiment:
    """Register persona variants, assign them deterministically, tally outcomes.

    Args:
        path: JSON persistence file. Defaults to
            ``~/.missy/persona_experiment.json``.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        # Resolve the default at call-time (not as a bound default argument) so
        # patching DEFAULT_EXPERIMENT_PATH in tests / at runtime takes effect.
        self._path = Path(path if path is not None else DEFAULT_EXPERIMENT_PATH).expanduser()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("variants", {})
                data.setdefault("outcomes", {})
                data.setdefault("enabled", False)
                return data
        except (OSError, ValueError, TypeError):
            pass
        return {"variants": {}, "outcomes": {}, "enabled": False}

    def _save(self, data: dict) -> None:
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=dir_path or ".", prefix=".pexp-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            logger.debug("PersonaExperiment: could not persist to %s", self._path, exc_info=True)

    # ------------------------------------------------------------------
    # Variant management
    # ------------------------------------------------------------------

    def add_variant(self, name: str, persona: PersonaConfig) -> None:
        """Register (or replace) a variant's persona snapshot."""
        name = name.strip()
        if not name:
            raise ValueError("variant name must be non-empty")
        with self._lock:
            data = self._load()
            data["variants"][name] = _persona_to_dict(persona)
            data["outcomes"].setdefault(name, {"success": 0, "failure": 0, "refused": 0})
            self._save(data)

    def remove_variant(self, name: str) -> bool:
        with self._lock:
            data = self._load()
            existed = data["variants"].pop(name, None) is not None
            data["outcomes"].pop(name, None)
            if existed:
                self._save(data)
            return existed

    def list_variants(self) -> list[str]:
        return sorted(self._load()["variants"].keys())

    def get_variant_persona(self, name: str) -> PersonaConfig | None:
        raw = self._load()["variants"].get(name)
        return _persona_from_dict(raw) if raw is not None else None

    # ------------------------------------------------------------------
    # Enable / assignment
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            data = self._load()
            data["enabled"] = bool(enabled)
            self._save(data)

    @property
    def enabled(self) -> bool:
        return bool(self._load().get("enabled", False))

    def assign(self, key: str) -> str | None:
        """Return the variant deterministically assigned to *key*.

        Stable across calls/processes (hash of *key* modulo the sorted variant
        list), so a session/channel never flaps between variants. Returns
        ``None`` when no variants are registered.
        """
        variants = self.list_variants()
        if not variants:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(variants)
        return variants[idx]

    def persona_for(self, key: str) -> PersonaConfig | None:
        """Return the assigned variant's persona for *key*, or ``None``."""
        variant = self.assign(key)
        return self.get_variant_persona(variant) if variant else None

    # ------------------------------------------------------------------
    # Outcome measurement
    # ------------------------------------------------------------------

    def record_outcome(self, variant: str, *, success: bool = True, refused: bool = False) -> None:
        """Tally an outcome for *variant* (success/failure and refusal count)."""
        with self._lock:
            data = self._load()
            if variant not in data["variants"]:
                return
            tally = data["outcomes"].setdefault(variant, {"success": 0, "failure": 0, "refused": 0})
            if success:
                tally["success"] += 1
            else:
                tally["failure"] += 1
            if refused:
                tally["refused"] += 1
            self._save(data)

    def results(self) -> dict[str, dict[str, Any]]:
        """Return per-variant ``{n, success_rate, refusal_rate}`` metrics."""
        data = self._load()
        out: dict[str, dict[str, Any]] = {}
        for name in sorted(data["variants"].keys()):
            tally = data["outcomes"].get(name, {"success": 0, "failure": 0, "refused": 0})
            total = tally["success"] + tally["failure"]
            out[name] = {
                "n": total,
                "success": tally["success"],
                "failure": tally["failure"],
                "refused": tally["refused"],
                "success_rate": (tally["success"] / total) if total else 0.0,
                "refusal_rate": (tally["refused"] / total) if total else 0.0,
            }
        return out

    def clear(self) -> None:
        """Remove all variants, outcomes, and disable the experiment."""
        with self._lock:
            self._save({"variants": {}, "outcomes": {}, "enabled": False})


def _snapshot_current_persona() -> PersonaConfig:
    """Return a copy of the active on-disk persona as a variant baseline."""
    from missy.agent.persona import PersonaManager

    return PersonaManager().get_persona()


def variant_persona_from_current(**overrides: Any) -> PersonaConfig:
    """Build a variant persona by copying the current persona with *overrides*.

    Convenience for creating an A/B variant that differs from the live persona
    in only a few fields (e.g. ``tone=["playful"]``).
    """
    base = asdict(_snapshot_current_persona())
    base.update(overrides)
    return _persona_from_dict(base)
