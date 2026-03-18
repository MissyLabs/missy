"""Hatching — first-run bootstrapping for the Missy AI assistant.

"Hatching" is the initial bootstrapping experience where the agent establishes
identity, configures behaviour, initialises memory, and becomes usable.  Think
of it as a first-run wizard that makes the agent feel alive.

The process is split into discrete, idempotent steps so that a failed or
interrupted hatching can resume from where it left off.  State is persisted to
``~/.missy/hatching.yaml`` after each step, and a structured log is written to
``~/.missy/hatching_log.jsonl``.

Example::

    from missy.agent.hatching import HatchingManager

    manager = HatchingManager()
    if manager.needs_hatching():
        state = manager.run_hatching(interactive=False)
        if state.status.name == "HATCHED":
            print("Missy is ready.")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from missy.agent.persona import PersonaManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_MISSY_DIR = Path.home() / ".missy"
_HATCHING_STATE_PATH = _MISSY_DIR / "hatching.yaml"
_HATCHING_LOG_PATH = _MISSY_DIR / "hatching_log.jsonl"
_CONFIG_PATH = _MISSY_DIR / "config.yaml"
_IDENTITY_PATH = _MISSY_DIR / "identity.pem"
_SECRETS_DIR = _MISSY_DIR / "secrets"
_PERSONA_PATH = _MISSY_DIR / "persona.yaml"
_MEMORY_DB_PATH = _MISSY_DIR / "memory.db"

# Minimum required free disk space in bytes (50 MiB).
_MIN_FREE_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class HatchingStatus(Enum):
    """Lifecycle state of the hatching process."""

    UNHATCHED = "unhatched"
    IN_PROGRESS = "in_progress"
    HATCHED = "hatched"
    FAILED = "failed"


@dataclass
class HatchingState:
    """Snapshot of the hatching process at any point in time.

    Attributes:
        status: Current lifecycle phase.
        started_at: ISO-8601 UTC timestamp when hatching began, or ``None``.
        completed_at: ISO-8601 UTC timestamp when hatching finished, or ``None``.
        steps_completed: Names of individual steps that have succeeded.
        persona_generated: ``True`` once the persona file has been created.
        environment_validated: ``True`` once environment checks passed.
        provider_verified: ``True`` once at least one provider was confirmed.
        security_initialized: ``True`` once the security subsystem is ready.
        memory_seeded: ``True`` once the memory store contains the welcome entry.
        error: Human-readable description of the last error, or ``None``.
    """

    status: HatchingStatus = HatchingStatus.UNHATCHED
    started_at: str | None = None
    completed_at: str | None = None
    steps_completed: list[str] = field(default_factory=list)
    persona_generated: bool = False
    environment_validated: bool = False
    provider_verified: bool = False
    security_initialized: bool = False
    memory_seeded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a YAML-compatible dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HatchingState:
        """Deserialise from a dictionary loaded from YAML."""
        raw_status = data.get("status", HatchingStatus.UNHATCHED.value)
        try:
            status = HatchingStatus(raw_status)
        except ValueError:
            status = HatchingStatus.UNHATCHED
        return cls(
            status=status,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            steps_completed=list(data.get("steps_completed") or []),
            persona_generated=bool(data.get("persona_generated", False)),
            environment_validated=bool(data.get("environment_validated", False)),
            provider_verified=bool(data.get("provider_verified", False)),
            security_initialized=bool(data.get("security_initialized", False)),
            memory_seeded=bool(data.get("memory_seeded", False)),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Hatching log
# ---------------------------------------------------------------------------


class HatchingLog:
    """Append-only structured log for hatching events.

    Each entry is a JSON object written as a single line to
    ``~/.missy/hatching_log.jsonl``.  The log survives across re-hatch
    attempts so the full history is preserved.

    Args:
        log_path: Override the default log file location.

    Example::

        log = HatchingLog()
        log.log("validate_environment", "ok", "Python version check passed")
        entries = log.get_entries()
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or _HATCHING_LOG_PATH

    def log(
        self,
        step: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append a structured entry to the log file.

        Args:
            step: Identifier of the hatching step (e.g. ``"validate_environment"``).
            status: Short status label such as ``"ok"``, ``"warn"``, or ``"error"``.
            message: Human-readable description of what happened.
            details: Optional extra structured data to include in the entry.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "status": status,
            "message": message,
            "details": details or {},
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Could not write to hatching log at %s", self._path)

    def get_entries(self) -> list[dict[str, Any]]:
        """Return all log entries in chronological order.

        Returns:
            A list of entry dictionaries.  Returns an empty list when the log
            file does not exist or cannot be read.
        """
        if not self._path.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with self._path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed hatching log line: %r", line)
        except OSError:
            logger.warning("Could not read hatching log at %s", self._path)
        return entries


# ---------------------------------------------------------------------------
# Hatching manager
# ---------------------------------------------------------------------------

# Default minimal config written when no config.yaml exists.
_DEFAULT_CONFIG_YAML = """\
# Missy configuration — created by hatching bootstrapper.
# Edit this file to configure providers, policy, and features.
# See https://missylabs.github.io/ for full documentation.

config_version: 2

network:
  default_deny: true
  presets:
    - anthropic

filesystem:
  allowed_write_paths: []
  allowed_read_paths: []

shell:
  enabled: false
  allowed_commands: []

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
    api_key: null    # set ANTHROPIC_API_KEY or add key here

workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
max_spend_usd: 0.0
"""


class HatchingManager:
    """Orchestrate the first-run hatching experience.

    :class:`HatchingManager` persists state to ``~/.missy/hatching.yaml``
    and writes a structured log to ``~/.missy/hatching_log.jsonl``.  Each
    step is idempotent; a partially-completed hatching can be resumed by
    calling :meth:`run_hatching` again.

    Args:
        state_path: Override the default state file location.
        log_path: Override the default log file location.

    Example::

        manager = HatchingManager()
        if manager.needs_hatching():
            state = manager.run_hatching(interactive=False)
    """

    def __init__(
        self,
        state_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._state_path = state_path or _HATCHING_STATE_PATH
        self._log = HatchingLog(log_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_hatched(self) -> bool:
        """Return ``True`` if hatching completed successfully.

        Returns:
            ``True`` when the persisted status is :attr:`HatchingStatus.HATCHED`.
        """
        return self.get_state().status is HatchingStatus.HATCHED

    def needs_hatching(self) -> bool:
        """Return ``True`` if hatching has not been completed.

        Hatching is required when:

        * No state file exists, or
        * The persisted status is :attr:`HatchingStatus.UNHATCHED` or
          :attr:`HatchingStatus.FAILED`.

        Returns:
            ``True`` when the agent should run the hatching flow.
        """
        if not self._state_path.exists():
            return True
        status = self.get_state().status
        return status in (HatchingStatus.UNHATCHED, HatchingStatus.FAILED)

    def get_state(self) -> HatchingState:
        """Load and return the current hatching state from disk.

        Returns:
            A :class:`HatchingState` instance.  Returns a default
            ``UNHATCHED`` state when no state file exists or the file cannot
            be parsed.
        """
        if not self._state_path.exists():
            return HatchingState()
        try:
            with self._state_path.open(encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                logger.warning("Hatching state file is not a mapping; using defaults.")
                return HatchingState()
            return HatchingState.from_dict(raw)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Could not load hatching state from %s: %s", self._state_path, exc)
            return HatchingState()

    def run_hatching(self, interactive: bool = True) -> HatchingState:
        """Execute the full hatching flow and return the final state.

        Steps are executed in order.  Already-completed steps (tracked via
        ``steps_completed``) are skipped so the method is safe to call
        multiple times after a partial failure.

        Args:
            interactive: When ``True``, steps may print progress to stdout.
                         When ``False``, all output is suppressed and defaults
                         are used without prompting.

        Returns:
            The final :class:`HatchingState` after all steps have been
            attempted.  Check ``state.status`` to determine whether hatching
            succeeded.
        """
        state = self.get_state()

        # Short-circuit: already hatched — nothing to do.
        if state.status is HatchingStatus.HATCHED:
            return state

        # Allow re-running after a previous failure; reset error and status.
        if state.status is HatchingStatus.FAILED:
            state.error = None

        state.status = HatchingStatus.IN_PROGRESS
        if state.started_at is None:
            state.started_at = datetime.now(UTC).isoformat()

        self._save_state(state)
        self._log.log("hatching", "started", "Hatching flow initiated")

        steps = [
            ("validate_environment", self._validate_environment),
            ("initialize_config", self._initialize_config),
            ("verify_providers", self._verify_providers),
            ("initialize_security", self._initialize_security),
            ("generate_persona", self._generate_persona),
            ("seed_memory", self._seed_memory),
            ("finalize", self._finalize),
        ]

        for step_name, step_fn in steps:
            if step_name in state.steps_completed:
                if interactive:
                    print(f"  [skip] {step_name} (already completed)")
                continue

            if interactive:
                print(f"  [....] {step_name}", end="", flush=True)

            try:
                step_fn(state, interactive=interactive)
                state.steps_completed.append(step_name)
                self._save_state(state)
                if interactive:
                    print(f"\r  [ ok ] {step_name}")
            except _HatchingStepWarning as warn:
                # Non-fatal — log and continue.
                self._log.log(step_name, "warn", str(warn))
                state.steps_completed.append(step_name)
                self._save_state(state)
                if interactive:
                    print(f"\r  [warn] {step_name}: {warn}")
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Hatching step %r failed", step_name)
                self._log.log(step_name, "error", msg)
                state.status = HatchingStatus.FAILED
                state.error = f"Step '{step_name}' failed: {msg}"
                self._save_state(state)
                if interactive:
                    print(f"\r  [FAIL] {step_name}: {exc}")
                return state

        return state

    def reset(self) -> None:
        """Remove the hatching state file so the agent can be re-hatched.

        The hatching log is preserved so the history of all hatching attempts
        survives the reset.
        """
        try:
            self._state_path.unlink(missing_ok=True)
            self._log.log("reset", "ok", "Hatching state cleared; re-hatching required")
            logger.info("Hatching state reset; state file removed: %s", self._state_path)
        except OSError as exc:
            logger.warning("Could not remove hatching state file: %s", exc)

    def get_hatching_log(self) -> list[dict[str, Any]]:
        """Return all hatching log entries in chronological order.

        Returns:
            A list of structured log entry dictionaries.
        """
        return self._log.get_entries()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_state(self, state: HatchingState) -> None:
        """Persist *state* to disk atomically."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp_path = self._state_path.with_suffix(".yaml.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(state.to_dict(), fh, default_flow_style=False, allow_unicode=True)
            tmp_path.replace(self._state_path)
        except OSError as exc:
            logger.warning("Could not save hatching state: %s", exc)

    # ------------------------------------------------------------------
    # Individual hatching steps
    # ------------------------------------------------------------------

    def _validate_environment(self, state: HatchingState, *, interactive: bool) -> None:
        """Check Python version, required directories, and disk space.

        Raises:
            RuntimeError: When a hard requirement is not met.
            _HatchingStepWarning: For non-fatal issues such as low disk space.
        """
        # Python version check.
        major, minor = sys.version_info.major, sys.version_info.minor
        if (major, minor) < (3, 11):
            raise RuntimeError(
                f"Python 3.11+ is required; running {major}.{minor}.{sys.version_info.micro}"
            )
        self._log.log(
            "validate_environment",
            "ok",
            f"Python {major}.{minor}.{sys.version_info.micro} — OK",
        )

        # Ensure ~/.missy/ can be created or already exists.
        try:
            _MISSY_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create Missy data directory {_MISSY_DIR}: {exc}"
            ) from exc

        # Basic write permission check.
        if not os.access(_MISSY_DIR, os.W_OK):
            raise RuntimeError(
                f"Missy data directory {_MISSY_DIR} is not writable by the current user"
            )

        # Disk space check (non-fatal).
        try:
            stat = os.statvfs(_MISSY_DIR)
            free_bytes = stat.f_bavail * stat.f_frsize
            if free_bytes < _MIN_FREE_BYTES:
                free_mib = free_bytes / (1024 * 1024)
                raise _HatchingStepWarning(
                    f"Low disk space: only {free_mib:.1f} MiB free in {_MISSY_DIR}"
                )
        except OSError:
            pass  # statvfs not available on all platforms — skip gracefully.

        state.environment_validated = True
        self._log.log(
            "validate_environment",
            "ok",
            f"Environment validated; data dir: {_MISSY_DIR}",
        )

    def _initialize_config(self, state: HatchingState, *, interactive: bool) -> None:
        """Ensure ``~/.missy/config.yaml`` exists, creating a default if absent."""
        if _CONFIG_PATH.exists():
            self._log.log("initialize_config", "ok", f"Config already exists at {_CONFIG_PATH}")
            return

        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _CONFIG_PATH.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
            self._log.log(
                "initialize_config",
                "ok",
                f"Default config written to {_CONFIG_PATH}",
            )
            if interactive:
                print(f"\n        Created default config at {_CONFIG_PATH}")
        except OSError as exc:
            raise RuntimeError(f"Cannot write default config to {_CONFIG_PATH}: {exc}") from exc

    def _verify_providers(self, state: HatchingState, *, interactive: bool) -> None:
        """Check that at least one provider has an API key available.

        Inspects environment variables and the config file.  Emits a warning
        (rather than failing) when no provider is found so that the hatching
        can complete and the user can add credentials later.

        Raises:
            _HatchingStepWarning: When no provider API key is detected.
        """
        env_vars = {
            "ANTHROPIC_API_KEY": "anthropic",
            "OPENAI_API_KEY": "openai",
        }

        found_provider: str | None = None
        for env_var, provider_name in env_vars.items():
            if os.environ.get(env_var):
                found_provider = provider_name
                break

        if not found_provider:
            # Try reading the config file for api_key values.
            found_provider = self._check_config_for_provider_key()

        if found_provider:
            state.provider_verified = True
            self._log.log(
                "verify_providers",
                "ok",
                f"Provider available: {found_provider}",
                {"provider": found_provider},
            )
        else:
            state.provider_verified = False
            raise _HatchingStepWarning(
                "No provider API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                f"or add an api_key to {_CONFIG_PATH}"
            )

    def _check_config_for_provider_key(self) -> str | None:
        """Return the first provider name with a non-empty api_key in config."""
        if not _CONFIG_PATH.exists():
            return None
        try:
            with _CONFIG_PATH.open(encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            return None

        providers_data: dict[str, Any] = config_data.get("providers", {})
        if not isinstance(providers_data, dict):
            return None

        for provider_name, provider_cfg in providers_data.items():
            if not isinstance(provider_cfg, dict):
                continue
            api_key = provider_cfg.get("api_key")
            api_keys = provider_cfg.get("api_keys", [])
            if (api_key and str(api_key).strip()) or api_keys:
                return str(provider_name)
        return None

    def _initialize_security(self, state: HatchingState, *, interactive: bool) -> None:
        """Ensure the security subsystem directories exist and note key status.

        Creates ``~/.missy/secrets/`` with mode 700.  Does not generate a new
        Ed25519 identity key — that is performed by :class:`AgentIdentity`
        on first use.

        Raises:
            _HatchingStepWarning: When the identity key is absent (informational).
        """
        # Ensure secrets directory exists with restricted permissions.
        try:
            _SECRETS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            _SECRETS_DIR.chmod(0o700)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create secrets directory {_SECRETS_DIR}: {exc}"
            ) from exc

        self._log.log(
            "initialize_security",
            "ok",
            f"Secrets directory ready at {_SECRETS_DIR} (mode 700)",
        )

        identity_present = _IDENTITY_PATH.exists()
        if not identity_present:
            # Note the absence — AgentIdentity generates it lazily on first use.
            self._log.log(
                "initialize_security",
                "warn",
                f"Agent identity key not found at {_IDENTITY_PATH}; "
                "it will be generated on first agent run",
            )
            if interactive:
                print(
                    f"\n        Note: identity key will be generated at {_IDENTITY_PATH} "
                    "on first run"
                )

        state.security_initialized = True
        self._log.log(
            "initialize_security",
            "ok",
            "Security subsystem initialized",
            {"identity_present": identity_present},
        )

    def _generate_persona(self, state: HatchingState, *, interactive: bool) -> None:
        """Create the default persona file if it does not already exist.

        Uses :class:`~missy.agent.persona.PersonaManager` to write a default
        ``~/.missy/persona.yaml``.

        Raises:
            _HatchingStepWarning: When PersonaManager cannot be imported or
                the persona file cannot be written (non-fatal).
        """
        if _PERSONA_PATH.exists():
            state.persona_generated = True
            self._log.log("generate_persona", "ok", f"Persona already exists at {_PERSONA_PATH}")
            return

        try:
            from missy.agent.persona import PersonaManager  # noqa: PLC0415

            pm: PersonaManager = PersonaManager(persona_path=_PERSONA_PATH)
            pm.save()
            state.persona_generated = True
            self._log.log(
                "generate_persona",
                "ok",
                f"Default persona written to {_PERSONA_PATH}",
            )
            if interactive:
                print(f"\n        Persona initialised at {_PERSONA_PATH}")
        except ImportError as exc:
            raise _HatchingStepWarning(
                f"Could not import PersonaManager; skipping persona generation: {exc}"
            ) from exc
        except OSError as exc:
            raise _HatchingStepWarning(
                f"Could not write persona file to {_PERSONA_PATH}: {exc}"
            ) from exc

    def _seed_memory(self, state: HatchingState, *, interactive: bool) -> None:
        """Initialise the SQLite memory DB and write a welcome entry.

        Raises:
            _HatchingStepWarning: When the memory store cannot be initialised
                (non-fatal — hatching should still succeed).
        """
        try:
            from missy.memory.sqlite_store import (  # noqa: PLC0415
                ConversationTurn,
                SQLiteMemoryStore,
            )

            store = SQLiteMemoryStore(db_path=_MEMORY_DB_PATH)
            welcome_turn = ConversationTurn.new(
                session_id="hatching",
                role="system",
                content="Missy hatching completed. Ready to assist.",
                provider="system",
            )
            store.add_turn(welcome_turn)
            state.memory_seeded = True
            self._log.log(
                "seed_memory",
                "ok",
                f"Memory store seeded at {_MEMORY_DB_PATH}",
                {"turn_id": welcome_turn.id},
            )
        except ImportError as exc:
            raise _HatchingStepWarning(
                f"Could not import SQLiteMemoryStore; skipping memory seed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise _HatchingStepWarning(
                f"Memory seeding failed; continuing without welcome entry: {exc}"
            ) from exc

    def _finalize(self, state: HatchingState, *, interactive: bool) -> None:
        """Mark hatching as complete and record the completion timestamp."""
        state.status = HatchingStatus.HATCHED
        state.completed_at = datetime.now(UTC).isoformat()
        state.error = None
        self._save_state(state)
        self._log.log(
            "finalize",
            "ok",
            "Hatching complete",
            {"completed_at": state.completed_at},
        )
        if interactive:
            print("\n  Missy is hatched and ready to assist.")


# ---------------------------------------------------------------------------
# Internal sentinel exception
# ---------------------------------------------------------------------------


class _HatchingStepWarning(Exception):
    """Raised inside a step to signal a non-fatal issue.

    :class:`HatchingManager.run_hatching` catches this, logs the warning,
    marks the step as completed, and continues to the next step — unlike a
    bare ``Exception``, which causes the whole hatching flow to fail.
    """
