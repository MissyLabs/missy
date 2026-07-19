"""Last-Writer-Wins CRDT map for mesh-shared memory (F01).

A :class:`LWWMap` lets every mesh node hold a replica of shared key/value
memory and merge peers' replicas **deterministically** — the merge is
commutative, associative, and idempotent, so nodes that see the same set of
updates in any order (or more than once) converge to the same state without a
coordinator. Each entry is stamped with a Lamport-ish ``(timestamp, peer_id)``
so concurrent writes to the same key resolve the same way on every replica
(higher timestamp wins; peer_id breaks ties). Tombstones handle deletes so a
delete can't be silently resurrected by a stale re-delivered write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Stamp:
    """A version stamp giving each write a total order across replicas."""

    timestamp: float
    peer_id: str

    def dominates(self, other: Stamp) -> bool:
        """True if this stamp wins over ``other`` (later ts; peer_id tiebreak)."""
        if self.timestamp != other.timestamp:
            return self.timestamp > other.timestamp
        return self.peer_id > other.peer_id


@dataclass
class _Entry:
    stamp: Stamp
    value: Any = None
    deleted: bool = False


@dataclass
class LWWMap:
    """A last-writer-wins map CRDT."""

    _entries: dict[str, _Entry] = field(default_factory=dict)

    # -- local mutations --------------------------------------------------
    def set(self, key: str, value: Any, *, timestamp: float, peer_id: str) -> None:
        self._apply(key, _Entry(Stamp(timestamp, peer_id), value, deleted=False))

    def delete(self, key: str, *, timestamp: float, peer_id: str) -> None:
        self._apply(key, _Entry(Stamp(timestamp, peer_id), None, deleted=True))

    def _apply(self, key: str, entry: _Entry) -> None:
        cur = self._entries.get(key)
        if cur is None or entry.stamp.dominates(cur.stamp):
            self._entries[key] = entry

    # -- reads ------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        entry = self._entries.get(key)
        if entry is None or entry.deleted:
            return default
        return entry.value

    def __contains__(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and not entry.deleted

    def items(self) -> dict[str, Any]:
        return {k: e.value for k, e in self._entries.items() if not e.deleted}

    def keys(self) -> list[str]:
        return [k for k, e in self._entries.items() if not e.deleted]

    # -- merge ------------------------------------------------------------
    def merge(self, other: LWWMap) -> LWWMap:
        """Merge ``other`` into this map in place; returns self.

        The operation is commutative, associative and idempotent: for each
        key the dominating stamp wins regardless of which side it came from.
        """
        for key, entry in other._entries.items():
            self._apply(key, entry)
        return self

    def merged(self, other: LWWMap) -> LWWMap:
        """Return a new map that is the merge of ``self`` and ``other``."""
        result = LWWMap(dict(self._entries))
        result.merge(other)
        return result

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {
                k: {
                    "timestamp": e.stamp.timestamp,
                    "peer_id": e.stamp.peer_id,
                    "value": e.value,
                    "deleted": e.deleted,
                }
                for k, e in self._entries.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LWWMap:
        entries: dict[str, _Entry] = {}
        for key, e in data.get("entries", {}).items():
            entries[key] = _Entry(
                stamp=Stamp(float(e["timestamp"]), str(e["peer_id"])),
                value=e.get("value"),
                deleted=bool(e.get("deleted", False)),
            )
        return cls(entries)
