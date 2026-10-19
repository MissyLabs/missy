"""Per-conversation serial, cross-conversation concurrent message dispatch.

The gateway used to drain each Discord account's single inbound
``asyncio.Queue`` with one loop that awaited every agent run in turn, so a
ten-minute task from one user in one channel blocked every other channel
behind it.  :class:`KeyedSerialDispatcher` keeps the single queue (and its
``queue_max`` backpressure) but hands each message to a per-key worker:

* messages with the same key (a Discord channel or thread) run one at a
  time, in arrival order;
* different keys run concurrently, bounded by a global ``max_concurrency``;
* at most ``max_pending`` messages are taken off the queue but not yet
  finished, so when the agent falls behind the inbound queue fills and
  ``queue_max`` backpressure applies exactly as before.

Cancelling :meth:`KeyedSerialDispatcher.run` cancels every in-flight worker
and waits for them, so gateway shutdown never leaves orphaned handler tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Hashable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_CONCURRENCY = 3


class KeyedSerialDispatcher(Generic[T]):
    """Dispatch queue items to *handler*, serialized per key.

    Args:
        handler: Coroutine function run once per item. Exceptions are
            logged and do not stop the dispatcher or the key's worker.
        key_fn: Returns the serialization key of an item.
        max_concurrency: Maximum number of handlers running at once
            across all keys (>= 1).
        max_pending: Maximum items taken from the queue but not yet
            finished (>= *max_concurrency*); ``None`` uses the queue's
            ``maxsize`` (or 256 for an unbounded queue).
    """

    def __init__(
        self,
        handler: Callable[[T], Awaitable[None]],
        key_fn: Callable[[T], Hashable],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_pending: int | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._handler = handler
        self._key_fn = key_fn
        self._max_concurrency = max_concurrency
        self._max_pending = max_pending
        self._pending: dict[Hashable, deque[T]] = {}
        self._workers: dict[Hashable, asyncio.Task[None]] = {}
        self._slots: asyncio.Semaphore | None = None
        self._capacity: asyncio.Semaphore | None = None

    @property
    def active_keys(self) -> set[Hashable]:
        """Keys that currently have a running or queued item."""
        return set(self._workers)

    async def run(self, queue: asyncio.Queue[T], should_stop: Callable[[], bool]) -> None:
        """Consume *queue* until *should_stop* returns true or the task is cancelled."""
        max_pending = self._max_pending
        if max_pending is None:
            max_pending = queue.maxsize or 256
        max_pending = max(max_pending, self._max_concurrency)
        self._slots = asyncio.Semaphore(self._max_concurrency)
        self._capacity = asyncio.Semaphore(max_pending)
        try:
            while not should_stop():
                await self._capacity.acquire()
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    self._capacity.release()
                    continue
                except BaseException:
                    self._capacity.release()
                    raise
                self._submit(item)
        finally:
            await self._shutdown()

    def _submit(self, item: T) -> None:
        try:
            key = self._key_fn(item)
        except Exception:
            logger.debug("dispatch key_fn failed; using shared key", exc_info=True)
            key = None
        self._pending.setdefault(key, deque()).append(item)
        if key not in self._workers:
            self._workers[key] = asyncio.create_task(
                self._drain(key), name=f"missy-dispatch-{str(key)[:40]}"
            )

    async def _drain(self, key: Hashable) -> None:
        assert self._slots is not None and self._capacity is not None
        queue = self._pending[key]
        try:
            while queue:
                item = queue.popleft()
                try:
                    async with self._slots:
                        await self._handler(item)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Discord message handler failed (key=%s)", key)
                finally:
                    self._capacity.release()
        finally:
            # Only reached with an empty deque (normal exit) or on cancel.
            if self._workers.get(key) is asyncio.current_task():
                del self._workers[key]
                self._pending.pop(key, None)

    async def _shutdown(self) -> None:
        workers = list(self._workers.values())
        for task in workers:
            task.cancel()
        for task in workers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._workers.clear()
        self._pending.clear()
