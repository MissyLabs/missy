"""Daily data-retention maintenance for a running gateway (SCHED-04).

Several stores have a working prune/cleanup method that nothing ever called,
so ``~/.missy`` grew without bound -- including the Discord inbound
attachment/zip directories that hold *untrusted* extracted content. This
module runs every enabled pruner from the ``retention:`` config section once a
day as an internal APScheduler job on the gateway's
:class:`~missy.scheduler.manager.SchedulerManager`. It is not a
``jobs.json`` job, makes no LLM/provider calls, and emits one
``maintenance.retention`` audit event per run with per-store counts.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: APScheduler id of the internal maintenance job.
MAINTENANCE_JOB_ID = "__missy_maintenance_retention__"

CAPTURES_DIR = "~/.missy/captures"
INBOUND_DIR_NAMES = ("discord_inbound", "discord_inbound_zips")


def prune_directory(
    base: str | os.PathLike[str],
    older_than_days: int,
    *,
    exclude_top_level: Iterable[str] = (),
) -> int:
    """Delete regular files under *base* older than *older_than_days* (by mtime).

    Symlinks are never followed or deleted, and nothing outside *base* is
    touched. Directories emptied by the prune are removed (never *base*
    itself). Returns the number of files deleted.
    """
    if older_than_days <= 0:
        return 0
    root = Path(base).expanduser()
    try:
        root_stat = root.lstat()
    except OSError:
        return 0
    if not stat.S_ISDIR(root_stat.st_mode):
        return 0
    excluded = set(exclude_top_level)
    cutoff = time.time() - older_than_days * 86400
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        current = Path(dirpath)
        rel_parts = current.relative_to(root).parts
        if rel_parts and rel_parts[0] in excluded:
            continue
        for name in filenames:
            if not rel_parts and name in excluded:
                continue
            path = current / name
            try:
                st = path.lstat()
                if stat.S_ISREG(st.st_mode) and st.st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.debug("retention: could not prune %s", path, exc_info=True)
        if current != root:
            try:
                if not any(current.iterdir()) and not current.is_symlink():
                    current.rmdir()
            except OSError:
                pass
    return removed


def _step(results: dict[str, Any], name: str, fn: Callable[[], int]) -> None:
    try:
        results[name] = int(fn())
    except Exception as exc:
        logger.warning("retention: %s failed: %s", name, exc)
        results[name] = f"error: {exc}"


DEVICES_PATH = "~/.missy/devices.json"


def run_retention(
    retention: Any,
    *,
    captures_dir: str = CAPTURES_DIR,
    devices_path: str = DEVICES_PATH,
) -> dict[str, Any]:
    """Run every enabled pruner in *retention* and return per-store counts."""
    results: dict[str, Any] = {}
    if not getattr(retention, "enabled", False):
        return results

    memory_days = int(getattr(retention, "memory_days", 0) or 0)
    if memory_days > 0:

        def _memory() -> int:
            from missy.memory.sqlite_store import SQLiteMemoryStore

            return SQLiteMemoryStore().cleanup(older_than_days=memory_days, include_summaries=True)

        _step(results, "memory_turns", _memory)

    checkpoints_days = int(getattr(retention, "checkpoints_days", 0) or 0)
    if checkpoints_days > 0:

        def _checkpoints() -> int:
            from missy.agent.checkpoint import CheckpointManager

            return CheckpointManager().cleanup(older_than_days=checkpoints_days)

        _step(results, "checkpoints", _checkpoints)

    captures_days = int(getattr(retention, "captures_days", 0) or 0)
    if captures_days > 0:
        _step(
            results,
            "capture_files",
            lambda: prune_directory(
                captures_dir, captures_days, exclude_top_level=INBOUND_DIR_NAMES
            ),
        )

    inbound_days = int(getattr(retention, "inbound_attachments_days", 0) or 0)
    if inbound_days > 0:
        _step(
            results,
            "inbound_attachment_files",
            lambda: sum(
                prune_directory(Path(captures_dir).expanduser() / name, inbound_days)
                for name in INBOUND_DIR_NAMES
            ),
        )

    tracker_days = int(getattr(retention, "request_tracker_days", 0) or 0)
    if tracker_days > 0:

        def _tracker() -> int:
            from datetime import UTC, datetime, timedelta

            from missy.tools.intelligence.request_tracker import RequestTracker

            before = (datetime.now(UTC) - timedelta(days=tracker_days)).isoformat()
            return RequestTracker().purge_before(before)

        _step(results, "request_tracker_events", _tracker)

    graph_days = int(getattr(retention, "graph_memory_days", 0) or 0)
    if graph_days > 0:

        def _graph() -> int:
            from missy.memory.graph_store import GraphMemoryStore

            # Only rarely-mentioned entities go; recurring ones are kept.
            return GraphMemoryStore().prune(min_mentions=2, older_than_days=graph_days)

        _step(results, "graph_entities", _graph)

    def _voice() -> int:
        from missy.channels.voice.pairing import PairingManager
        from missy.channels.voice.registry import DeviceRegistry

        registry = DeviceRegistry(devices_path)
        registry.load()
        removed = registry.purge_audio_logs()
        pairing_hours = int(getattr(retention, "pending_pairing_hours", 0) or 0)
        if pairing_hours > 0:
            removed += PairingManager(registry).expire_pending(pairing_hours)
        return removed

    if Path(devices_path).expanduser().exists():
        _step(results, "voice_audio_and_pending_pairings", _voice)

    try:
        from missy.core.events import AuditEvent, event_bus

        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type="maintenance.retention",
                category="scheduler",
                result="allow",
                detail=results,
            )
        )
    except Exception:
        logger.debug("retention: could not publish audit event", exc_info=True)
    logger.info("Retention maintenance complete: %s", results)
    return results


def register_maintenance_job(
    scheduler: Any,
    config_loader: Callable[[], Any],
    *,
    hour: int = 3,
    minute: int = 17,
) -> bool:
    """Register the daily retention job on a started APScheduler instance.

    *config_loader* is called on every run so a hot-reloaded ``retention:``
    section takes effect without a restart. Returns ``True`` when registered.
    """

    def _job() -> None:
        try:
            cfg = config_loader()
        except Exception as exc:
            logger.warning("retention: could not load config: %s", exc)
            return
        run_retention(getattr(cfg, "retention", None))

    try:
        scheduler.add_job(
            func=_job,
            trigger="cron",
            hour=hour,
            minute=minute,
            id=MAINTENANCE_JOB_ID,
            name="Data retention maintenance",
            replace_existing=True,
        )
    except Exception:
        logger.warning("Could not register retention maintenance job", exc_info=True)
        return False
    return True
