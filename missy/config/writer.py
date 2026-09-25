"""Targeted, backed-up writes to a subset of ``config.yaml`` fields.

Unlike :mod:`missy.config.migrate` (whole-file structural rewrite, run
automatically on startup), this module makes small, explicit, operator- or
API-triggered edits to individual fields -- the provider-preference
hierarchy, non-secret provider tuning, and allow-listed agent orchestration
limits -- so a choice made via the CLI or Web TUI survives a restart, the
same way :mod:`missy.config.migrate` already makes its own edits durable.
Every write backs up the previous file first
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
import math
import os
import tempfile
from pathlib import Path
from typing import Any

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
    if not math.isfinite(weight) or weight < 0:
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


#: Provider config fields the Web TUI's provider inspector may edit
#: directly, and how to coerce/validate a submitted value for each.
#: Deliberately excludes anything credential-shaped (api_key, api_keys,
#: oauth_accounts) -- those are set via `missy providers auth`/vault
#: references, never round-tripped through this plain-YAML writer.
EDITABLE_PROVIDER_FIELDS: dict[str, str] = {
    "model": "str",
    "fast_model": "str",
    "premium_model": "str",
    "context_worker_provider": "str",
    "context_worker_model": "str",
    "base_url": "str",
    "timeout": "int",
    "requests_per_minute": "int",
    "tokens_per_minute": "int",
    "max_wait_seconds": "float",
    "circuit_breaker_threshold": "positive_int",
    "circuit_breaker_cooldown_seconds": "float",
    "key_rotation_strategy": "rotation_strategy",
}

# Root runtime fields intentionally exposed to the operator console.  This
# allow-list keeps arbitrary config paths, credentials, and policy surfaces
# out of the generic control.
EDITABLE_AGENT_FIELDS: dict[str, str] = {
    "max_iterations": "positive_int",
    "temperature": "temperature",
    "max_sub_agents": "max_sub_agents",
    "max_concurrent_agents": "max_concurrent_agents",
    "max_sub_agent_depth": "max_sub_agent_depth",
    "max_spend_usd": "nonnegative_float",
    "global_max_spend_usd": "nonnegative_float",
    "global_budget_period": "budget_period",
}


def set_provider_field(config_path: str, name: str, field: str, value: Any) -> Any:
    """Persist a single editable field for provider *name* in ``config.yaml``.

    Args:
        config_path: Path to ``config.yaml``.
        name: Provider key under the ``providers:`` section. Must already
            exist there.
        field: One of :data:`EDITABLE_PROVIDER_FIELDS`'s keys.
        value: The new value. Coerced to that field's expected type
            (``str`` fields accept an empty string to clear the override;
            ``int`` fields must be a non-negative integer, with ``0``
            meaning "unlimited" for the rate-limit fields and "use the
            provider's own default" for ``timeout``).

    Returns:
        The coerced value that was actually written.

    Raises:
        ConfigWriteError: If *field* isn't editable, the value fails
            coercion/validation, *name* isn't a configured provider, or the
            file cannot be read/parsed/written.
    """
    kind = EDITABLE_PROVIDER_FIELDS.get(field)
    if kind is None:
        raise ConfigWriteError(
            f"Field {field!r} is not editable. Allowed: "
            f"{', '.join(sorted(EDITABLE_PROVIDER_FIELDS))}."
        )
    if kind == "str":
        coerced: Any = str(value).strip()
    elif kind in {"int", "positive_int"}:
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigWriteError(f"{field} must be an integer, got {value!r}.") from exc
        if not math.isfinite(coerced) or coerced < 0:
            raise ConfigWriteError(f"{field} must be >= 0, got {coerced!r}.")
        if kind == "positive_int" and coerced < 1:
            raise ConfigWriteError(f"{field} must be >= 1, got {coerced!r}.")
    elif kind == "float":
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigWriteError(f"{field} must be a number, got {value!r}.") from exc
        if not math.isfinite(coerced) or coerced < 0:
            raise ConfigWriteError(f"{field} must be >= 0, got {coerced!r}.")
    else:
        coerced = str(value).strip().lower()
        if coerced not in {"failover", "round_robin"}:
            raise ConfigWriteError("key_rotation_strategy must be 'failover' or 'round_robin'.")

    path = Path(config_path).expanduser()
    data = _load_raw(path)
    providers = data.get("providers")
    if not isinstance(providers, dict) or name not in providers:
        raise ConfigWriteError(f"Provider {name!r} is not configured in {path}.")
    provider_entry = providers[name]
    if not isinstance(provider_entry, dict):
        raise ConfigWriteError(f"Provider {name!r}'s config in {path} is not a mapping.")
    if provider_entry.get(field) == coerced:
        return coerced
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    provider_entry[field] = coerced
    _atomic_write_yaml(path, data)
    logger.info("Persisted providers.%s.%s=%r to %s", name, field, coerced, path)
    return coerced


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
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ConfigWriteError("Every account weight must be > 0.")
    path = Path(config_path).expanduser()
    data = _load_raw(path)
    providers = data.get("providers")
    if not isinstance(providers, dict) or name not in providers:
        raise ConfigWriteError(f"Provider {name!r} is not configured in {path}.")
    provider_entry = providers[name]
    if not isinstance(provider_entry, dict):
        raise ConfigWriteError(f"Provider {name!r}'s config in {path} is not a mapping.")
    account_count = len(
        provider_entry.get("oauth_accounts") or provider_entry.get("api_keys") or []
    )
    if weights and account_count and len(weights) != account_count:
        raise ConfigWriteError(
            f"Expected {account_count} account weights for provider {name!r}, got {len(weights)}."
        )
    if provider_entry.get("account_weights", []) == list(weights):
        return
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    provider_entry["account_weights"] = list(weights)
    _atomic_write_yaml(path, data)
    logger.info("Persisted providers.%s.account_weights=%r to %s", name, weights, path)


def set_agent_field(config_path: str, field: str, value: Any) -> Any:
    """Persist one allow-listed root agent/orchestration setting."""
    kind = EDITABLE_AGENT_FIELDS.get(field)
    if kind is None:
        raise ConfigWriteError(
            f"Field {field!r} is not editable. Allowed: {', '.join(sorted(EDITABLE_AGENT_FIELDS))}."
        )
    try:
        if kind == "positive_int":
            coerced: Any = int(value)
            if coerced < 1:
                raise ValueError
        elif kind in {"max_sub_agents", "max_concurrent_agents"}:
            coerced = int(value)
            if not 1 <= coerced <= 50:
                raise ValueError
        elif kind == "max_sub_agent_depth":
            coerced = int(value)
            if not 0 <= coerced <= 5:
                raise ValueError
        elif kind == "temperature":
            coerced = float(value)
            if not math.isfinite(coerced) or not 0 <= coerced <= 2:
                raise ValueError
        elif kind == "nonnegative_float":
            coerced = float(value)
            if not math.isfinite(coerced) or coerced < 0:
                raise ValueError
        else:
            coerced = str(value).strip().lower()
            if coerced not in {"total", "daily", "monthly"}:
                raise ValueError
    except (TypeError, ValueError) as exc:
        limits = {
            "max_sub_agents": "an integer between 1 and 50",
            "max_concurrent_agents": "an integer between 1 and 50",
            "max_sub_agent_depth": "an integer between 0 and 5",
            "temperature": "a number between 0 and 2",
            "budget_period": "one of: total, daily, monthly",
            "nonnegative_float": "a number >= 0",
            "positive_int": "an integer >= 1",
        }
        raise ConfigWriteError(f"{field} must be {limits[kind]}.") from exc

    path = Path(config_path).expanduser()
    data = _load_raw(path)
    if field == "max_concurrent_agents":
        max_agents = int(data.get("max_sub_agents", 10))
        if coerced > max_agents:
            raise ConfigWriteError("max_concurrent_agents cannot exceed max_sub_agents.")
    if field == "max_sub_agents":
        concurrency = int(data.get("max_concurrent_agents", 3))
        if coerced < concurrency:
            raise ConfigWriteError("max_sub_agents cannot be lower than max_concurrent_agents.")
    if data.get(field) == coerced:
        return coerced
    try:
        backup_config(path)
    except Exception as exc:
        logger.warning("Could not back up config before write: %s", exc)
    data[field] = coerced
    _atomic_write_yaml(path, data)
    logger.info("Persisted %s=%r to %s", field, coerced, path)
    return coerced
