"""Targeted, backed-up writes to a subset of ``config.yaml`` fields.

Unlike :mod:`missy.config.migrate` (whole-file structural rewrite, run
automatically on startup), this module makes small, explicit, operator- or
API-triggered edits to individual fields -- the provider-preference
hierarchy's ``default_provider``, and a single provider's ``weight`` /
``account_weights`` -- so a choice made via ``missy providers switch``/
``missy providers weight`` or the Web TUI's provider controls survives a
restart, the same way :mod:`missy.config.migrate` already makes its own
edits durable. Every write backs up the previous file first
(:func:`missy.config.plan.backup_config`) and writes atomically via a temp
file + ``os.replace``, mirroring
:func:`missy.config.migrate._atomic_write_yaml` exactly.

Round-trips through ``yaml.safe_load``/``yaml.dump`` like the rest of this
codebase's config writers (:mod:`missy.config.migrate`,
``missy/cli/main.py``'s persona editor) -- comments and key ordering in a
hand-edited ``config.yaml`` are not preserved. Callers needing that should
edit the file directly instead of going through this module.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import yaml

from missy.config.plan import backup_config

logger = logging.getLogger(__name__)


class ConfigWriteError(Exception):
    """Raised when a targeted config.yaml field write cannot be completed."""


def _load_raw(path: Path) -> dict:
    if not path.exists():
        raise ConfigWriteError(f"Config file not found: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigWriteError(f"Cannot read config file '{path}': {exc}") from exc
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigWriteError(f"Invalid YAML in '{path}': {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigWriteError(f"Top-level YAML value in '{path}' must be a mapping.")
    return data


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write *data* as YAML to *path* atomically via a temp file (0600)."""
    content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config_write_")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def set_default_provider(config_path: str, name: str) -> None:
    """Persist *name* as ``default_provider`` in ``config.yaml``.

    Args:
        config_path: Path to ``config.yaml``.
        name: Registry key of the provider to make the persisted default.
            Not validated against the live registry here -- callers (the
            CLI, the Web TUI operator control) already check the name is
            registered and available before calling this.

    Raises:
        ConfigWriteError: If the file cannot be read, parsed, or written.
    """
    path = Path(config_path).expanduser()
    data = _load_raw(path)
    if data.get("default_provider") == name:
        return
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    data["default_provider"] = name
    _atomic_write_yaml(path, data)
    logger.info("Persisted default_provider=%r to %s", name, path)


def set_provider_weight(config_path: str, name: str, weight: float) -> None:
    """Persist *weight* for provider *name* under ``providers.<name>.weight``.

    Args:
        config_path: Path to ``config.yaml``.
        name: Provider key under the ``providers:`` section. Must already
            exist there (a weight for a provider config.yaml doesn't
            define yet has nothing to attach to).
        weight: The new weight. Must be >= 0.

    Raises:
        ConfigWriteError: If the file cannot be read/parsed/written, or
            *name* is not a configured provider, or *weight* is negative.
    """
    if weight < 0:
        raise ConfigWriteError(f"weight must be >= 0, got {weight!r}.")
    path = Path(config_path).expanduser()
    data = _load_raw(path)
    providers = data.get("providers")
    if not isinstance(providers, dict) or name not in providers:
        raise ConfigWriteError(f"Provider {name!r} is not configured in {path}.")
    provider_entry = providers[name]
    if not isinstance(provider_entry, dict):
        raise ConfigWriteError(f"Provider {name!r}'s config in {path} is not a mapping.")
    if provider_entry.get("weight") == weight:
        return
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    provider_entry["weight"] = weight
    _atomic_write_yaml(path, data)
    logger.info("Persisted providers.%s.weight=%r to %s", name, weight, path)


def set_account_weights(config_path: str, name: str, weights: list[float]) -> None:
    """Persist per-account *weights* for provider *name*.

    Args:
        config_path: Path to ``config.yaml``.
        name: Provider key under the ``providers:`` section.
        weights: New ``account_weights`` list (parallel to that provider's
            ``api_keys``/``oauth_accounts``). An empty list resets to
            equal weighting.

    Raises:
        ConfigWriteError: If the file cannot be read/parsed/written, *name*
            is not a configured provider, or a weight is not positive.
    """
    if any(w <= 0 for w in weights):
        raise ConfigWriteError("Every account weight must be > 0.")
    path = Path(config_path).expanduser()
    data = _load_raw(path)
    providers = data.get("providers")
    if not isinstance(providers, dict) or name not in providers:
        raise ConfigWriteError(f"Provider {name!r} is not configured in {path}.")
    provider_entry = providers[name]
    if not isinstance(provider_entry, dict):
        raise ConfigWriteError(f"Provider {name!r}'s config in {path} is not a mapping.")
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    provider_entry["account_weights"] = list(weights)
    _atomic_write_yaml(path, data)
    logger.info("Persisted providers.%s.account_weights=%r to %s", name, weights, path)
