"""Persona system for the Missy AI assistant.

Manages the agent's identity, tone, personality traits, and response style.
A :class:`PersonaConfig` holds the structured persona definition, and
:class:`PersonaManager` handles loading/saving from ``~/.missy/persona.yaml``,
version tracking, and building system-prompt prefix strings.

Example::

    from missy.agent.persona import PersonaManager

    pm = PersonaManager()
    prefix = pm.get_system_prompt_prefix()
    print(prefix)

    pm.update(name="Missy v2", tone=["playful", "technical"])
    pm.save()
"""

from __future__ import annotations

import contextlib
import difflib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_NAME = "Missy"

_DEFAULT_TONE: list[str] = [
    "helpful",
    "direct",
    "technical",
]

_DEFAULT_PERSONALITY_TRAITS: list[str] = [
    "curious",
    "thorough",
    "security-conscious",
    "pragmatic",
]

_DEFAULT_BEHAVIORAL_TENDENCIES: list[str] = [
    "prefers action over narration",
    "adapts formality to context",
    "asks clarifying questions when needed",
]

_DEFAULT_RESPONSE_STYLE_RULES: list[str] = [
    "Be concise unless detail is requested",
    "Use technical terms when appropriate",
    "Show reasoning for non-obvious decisions",
    "Acknowledge uncertainty rather than guessing",
]

_DEFAULT_BOUNDARIES: list[str] = [
    "Never execute destructive operations without confirmation",
    "Never expose secrets or credentials",
    "Always respect policy engine decisions",
    "Flag security concerns proactively",
]

_DEFAULT_IDENTITY_DESCRIPTION: str = (
    "Missy is a security-first local AI assistant for Linux systems. "
    "She is knowledgeable, practical, and focused on getting things done "
    "safely and efficiently."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PersonaConfig:
    """Structured representation of the agent's persona.

    Attributes:
        name: The agent's display name.
        tone: Adjectives describing communication style (e.g. "direct").
        personality_traits: Core character traits (e.g. "curious").
        behavioral_tendencies: Habitual behavioural patterns expressed as
            short phrases.
        response_style_rules: Explicit rules that govern how the agent
            formulates responses.
        boundaries: Hard constraints the agent must never violate.
        identity_description: A paragraph-length narrative description of
            the agent, suitable for injection at the top of a system prompt.
        version: Monotonically increasing integer, incremented on each save.
    """

    name: str = _DEFAULT_NAME
    tone: list[str] = field(default_factory=lambda: list(_DEFAULT_TONE))
    personality_traits: list[str] = field(default_factory=lambda: list(_DEFAULT_PERSONALITY_TRAITS))
    behavioral_tendencies: list[str] = field(
        default_factory=lambda: list(_DEFAULT_BEHAVIORAL_TENDENCIES)
    )
    response_style_rules: list[str] = field(
        default_factory=lambda: list(_DEFAULT_RESPONSE_STYLE_RULES)
    )
    boundaries: list[str] = field(default_factory=lambda: list(_DEFAULT_BOUNDARIES))
    identity_description: str = _DEFAULT_IDENTITY_DESCRIPTION
    version: int = 1


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _persona_to_dict(persona: PersonaConfig) -> dict[str, Any]:
    """Convert a :class:`PersonaConfig` to a YAML-serialisable dict.

    The ``version`` field is placed at the top of the mapping for readability.
    """
    data = asdict(persona)
    # Re-order so version comes first
    ordered: dict[str, Any] = {"version": data.pop("version")}
    ordered.update(data)
    return ordered


def _persona_from_dict(data: dict[str, Any]) -> PersonaConfig:
    """Build a :class:`PersonaConfig` from a raw YAML-loaded mapping.

    Unknown keys are silently ignored so that future schema additions do not
    break older installs reading a newer file.
    """
    known = {f.name for f in fields(PersonaConfig)}
    filtered = {k: v for k, v in data.items() if k in known}
    return PersonaConfig(**filtered)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PersonaManager:
    """Load, persist, and expose the Missy persona configuration.

    Args:
        persona_path: Path to ``persona.yaml``.  Defaults to
            ``~/.missy/persona.yaml``.

    On construction the file is read if it exists; a default
    :class:`PersonaConfig` is used otherwise.
    """

    def __init__(
        self,
        persona_path: str | Path = "~/.missy/persona.yaml",
    ) -> None:
        self._path = Path(os.path.expanduser(str(persona_path)))
        self._persona: PersonaConfig = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_persona(self) -> PersonaConfig:
        """Return the current :class:`PersonaConfig` (a shallow copy).

        Returns:
            A copy of the in-memory persona so callers cannot mutate state
            accidentally.
        """
        # Return a new instance with the same field values
        return PersonaConfig(**asdict(self._persona))

    def get_system_prompt_prefix(self) -> str:
        """Build a persona description string for injection into system prompts.

        The resulting text is structured so that an LLM can parse the intent
        without extra prompt engineering.  It covers identity, tone,
        personality, behavioural tendencies, response style, and hard
        boundaries.

        Returns:
            A multi-line string ready for use as the opening section of a
            system prompt.
        """
        p = self._persona
        lines: list[str] = []

        # Identity
        lines.append(f"# Identity\n{p.identity_description.strip()}")

        # Tone
        if p.tone:
            tone_str = ", ".join(p.tone)
            lines.append(f"# Tone\nYour communication style is {tone_str}.")

        # Personality
        if p.personality_traits:
            traits_str = ", ".join(p.personality_traits)
            lines.append(f"# Personality\nYour core character traits are: {traits_str}.")

        # Behavioural tendencies
        if p.behavioral_tendencies:
            lines.append("# Behavioural Tendencies")
            lines.extend(f"- {t}" for t in p.behavioral_tendencies)

        # Response style
        if p.response_style_rules:
            lines.append("# Response Style")
            lines.extend(f"- {r}" for r in p.response_style_rules)

        # Boundaries
        if p.boundaries:
            lines.append("# Boundaries")
            lines.extend(f"- {b}" for b in p.boundaries)

        return "\n\n".join(lines)

    def save(self) -> None:
        """Persist the current persona to disk, incrementing the version.

        Creates a timestamped backup of the previous persona file before
        overwriting.  The write itself is atomic: data is written to a temp
        file in the same directory and then renamed into place, so a crash
        mid-write cannot corrupt the existing file.

        Raises:
            OSError: If the directory cannot be created or the file cannot
                be written.
        """
        # Back up existing file before overwriting
        if self._path.exists():
            self._create_backup()

        self._persona.version += 1
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = _persona_to_dict(self._persona)
        dir_ = str(self._path.parent)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.dump(
                    data,
                    fh,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            os.replace(tmp_path, self._path)
            # Restrict file permissions — persona may contain sensitive identity info
            with contextlib.suppress(OSError):
                self._path.chmod(0o600)
            self._audit("save")
            logger.debug(
                "Persona saved to %s (version %d)",
                self._path,
                self._persona.version,
            )
        except Exception:
            # Clean up temp file on failure; ignore errors during cleanup
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def reset(self) -> None:
        """Restore the persona to factory defaults and save.

        The version counter is preserved and incremented rather than reset,
        so the history of saves is not ambiguous.
        """
        current_version = self._persona.version
        self._persona = PersonaConfig(version=current_version)
        self.save()
        self._audit("reset")
        logger.info("Persona reset to defaults (version %d)", self._persona.version)

    def update(self, **kwargs: Any) -> None:
        """Modify specific persona fields without replacing the whole object.

        Only fields defined on :class:`PersonaConfig` are accepted.  Unknown
        keys raise :class:`ValueError` to prevent silent typos.

        Args:
            **kwargs: Field names and their new values.

        Raises:
            ValueError: If an unknown field name is provided.

        Example::

            pm.update(name="Missy v2", tone=["playful", "technical"])
        """
        valid_fields = {f.name for f in fields(PersonaConfig)} - {"version"}
        unknown = set(kwargs) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown persona field(s): {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(valid_fields))}."
            )
        for key, value in kwargs.items():
            setattr(self._persona, key, value)
        logger.debug("Persona fields updated: %s", list(kwargs.keys()))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Current persona version number."""
        return self._persona.version

    @property
    def path(self) -> Path:
        """Resolved path to the persona YAML file."""
        return self._path

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def _audit(self, action: str, details: dict[str, Any] | None = None) -> None:
        """Append a structured entry to the persona audit log.

        Args:
            action: Short label such as ``"save"``, ``"reset"``, ``"rollback"``.
            details: Optional extra structured data.
        """
        audit_path = self._path.parent / "persona_audit.jsonl"
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "version": self._persona.version,
            "name": self._persona.name,
            "details": details or {},
        }
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(str(audit_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            logger.debug("Could not write to persona audit log at %s", audit_path)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return all persona audit log entries in chronological order.

        Returns:
            A list of entry dictionaries. Returns an empty list when the
            audit log does not exist.
        """
        audit_path = self._path.parent / "persona_audit.jsonl"
        if not audit_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with audit_path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        entries.append(json.loads(line))
        except OSError:
            pass
        return entries

    # ------------------------------------------------------------------
    # Backup / rollback / diff
    # ------------------------------------------------------------------

    _MAX_BACKUPS = 5

    @property
    def backup_dir(self) -> Path:
        """Directory where persona backups are stored."""
        return self._path.parent / "persona.d"

    def _create_backup(self) -> Path:
        """Create a timestamped backup of the current persona file.

        Returns:
            Path to the newly created backup file.
        """
        bdir = self.backup_dir
        bdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = bdir / f"persona.yaml.{timestamp}"
        shutil.copy2(str(self._path), str(backup_path))
        self._prune_backups()
        logger.debug("Persona backup created: %s", backup_path)
        return backup_path

    def _prune_backups(self) -> None:
        """Remove oldest backups so at most ``_MAX_BACKUPS`` remain."""
        backups = self.list_backups()
        while len(backups) > self._MAX_BACKUPS:
            oldest = backups.pop(0)
            oldest.unlink()

    def list_backups(self) -> list[Path]:
        """Return all persona backup files sorted oldest-first.

        Returns:
            List of backup :class:`Path` objects.
        """
        bdir = self.backup_dir
        if not bdir.exists():
            return []
        return sorted(
            [p for p in bdir.iterdir() if p.name.startswith("persona.yaml.")],
            key=lambda p: p.stat().st_mtime,
        )

    def rollback(self) -> Path | None:
        """Restore the latest backup, backing up the current persona first.

        Returns:
            Path to the restored backup, or ``None`` if no backups exist.
        """
        backups = self.list_backups()
        if not backups:
            return None

        latest = backups[-1]
        restore_content = latest.read_text(encoding="utf-8")

        # Back up current before overwriting (without incrementing version)
        if self._path.exists():
            self._create_backup()

        self._path.write_text(restore_content, encoding="utf-8")
        self._persona = self._load()
        self._audit("rollback", {"from_backup": latest.name})
        logger.info(
            "Persona rolled back to backup %s (version %d)",
            latest.name,
            self._persona.version,
        )
        return latest

    def diff(self) -> str:
        """Return a unified diff between the current persona and the latest backup.

        Returns:
            A unified diff string, or an empty string if no backups exist or
            files are identical.
        """
        backups = self.list_backups()
        if not backups or not self._path.exists():
            return ""

        latest = backups[-1]
        a_lines = latest.read_text(encoding="utf-8").splitlines(keepends=True)
        b_lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                a_lines,
                b_lines,
                fromfile=f"backup ({latest.name})",
                tofile="current (persona.yaml)",
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> PersonaConfig:
        """Read persona from disk, falling back to defaults on any error.

        Returns:
            A :class:`PersonaConfig` populated from the file, or the default
            config if the file does not exist or cannot be parsed.
        """
        if not self._path.exists():
            logger.debug("No persona file at %s; using defaults.", self._path)
            return PersonaConfig()

        try:
            raw = self._path.read_text(encoding="utf-8")
            data: Any = yaml.safe_load(raw)
            if not isinstance(data, dict):
                raise ValueError("Persona YAML root must be a mapping.")
            persona = _persona_from_dict(data)
            logger.debug(
                "Loaded persona '%s' (version %d) from %s",
                persona.name,
                persona.version,
                self._path,
            )
            return persona
        except (yaml.YAMLError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse persona file %s (%s); using defaults.",
                self._path,
                exc,
            )
            return PersonaConfig()
