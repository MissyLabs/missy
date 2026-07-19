"""Tests for the LWW-map CRDT (F01)."""

from __future__ import annotations

from missy.mesh.crdt import LWWMap, Stamp


class TestBasicOps:
    def test_set_and_get(self) -> None:
        m = LWWMap()
        m.set("k", "v", timestamp=1.0, peer_id="p")
        assert m.get("k") == "v"
        assert "k" in m

    def test_later_write_wins(self) -> None:
        m = LWWMap()
        m.set("k", "old", timestamp=1.0, peer_id="p")
        m.set("k", "new", timestamp=2.0, peer_id="p")
        assert m.get("k") == "new"

    def test_earlier_write_ignored(self) -> None:
        m = LWWMap()
        m.set("k", "new", timestamp=2.0, peer_id="p")
        m.set("k", "old", timestamp=1.0, peer_id="p")
        assert m.get("k") == "new"

    def test_delete_tombstone(self) -> None:
        m = LWWMap()
        m.set("k", "v", timestamp=1.0, peer_id="p")
        m.delete("k", timestamp=2.0, peer_id="p")
        assert m.get("k") is None
        assert "k" not in m

    def test_stale_write_cannot_resurrect_delete(self) -> None:
        m = LWWMap()
        m.delete("k", timestamp=5.0, peer_id="p")
        m.set("k", "late", timestamp=3.0, peer_id="p")  # older than the delete
        assert "k" not in m

    def test_items_and_keys_exclude_tombstones(self) -> None:
        m = LWWMap()
        m.set("a", 1, timestamp=1.0, peer_id="p")
        m.set("b", 2, timestamp=1.0, peer_id="p")
        m.delete("a", timestamp=2.0, peer_id="p")
        assert m.items() == {"b": 2}
        assert m.keys() == ["b"]


class TestTieBreaking:
    def test_equal_timestamp_higher_peer_id_wins(self) -> None:
        m = LWWMap()
        m.set("k", "from-a", timestamp=1.0, peer_id="aaa")
        m.set("k", "from-z", timestamp=1.0, peer_id="zzz")
        assert m.get("k") == "from-z"

    def test_stamp_dominance(self) -> None:
        assert Stamp(2.0, "a").dominates(Stamp(1.0, "z"))
        assert Stamp(1.0, "z").dominates(Stamp(1.0, "a"))
        assert not Stamp(1.0, "a").dominates(Stamp(1.0, "z"))


class TestMergeLaws:
    def _a(self):
        m = LWWMap()
        m.set("k", "a", timestamp=1.0, peer_id="p1")
        return m

    def _b(self):
        m = LWWMap()
        m.set("k", "b", timestamp=2.0, peer_id="p2")
        return m

    def test_commutative(self) -> None:
        assert self._a().merged(self._b()).get("k") == self._b().merged(self._a()).get("k")

    def test_idempotent(self) -> None:
        merged = self._a().merged(self._b())
        assert merged.merged(self._b()).get("k") == merged.get("k") == "b"

    def test_associative(self) -> None:
        c = LWWMap()
        c.set("k", "c", timestamp=3.0, peer_id="p3")
        left = self._a().merged(self._b()).merged(c)
        right = self._a().merged(self._b().merged(c))
        assert left.get("k") == right.get("k") == "c"

    def test_merge_combines_disjoint_keys(self) -> None:
        a = LWWMap()
        a.set("x", 1, timestamp=1.0, peer_id="p")
        b = LWWMap()
        b.set("y", 2, timestamp=1.0, peer_id="p")
        merged = a.merged(b)
        assert merged.items() == {"x": 1, "y": 2}

    def test_merge_in_place_returns_self(self) -> None:
        a = self._a()
        assert a.merge(self._b()) is a


class TestSerialization:
    def test_to_from_dict_round_trip(self) -> None:
        m = LWWMap()
        m.set("k", {"nested": [1, 2]}, timestamp=1.5, peer_id="p")
        m.delete("gone", timestamp=2.0, peer_id="p")
        restored = LWWMap.from_dict(m.to_dict())
        assert restored.get("k") == {"nested": [1, 2]}
        assert "gone" not in restored
        # merging restored with original is idempotent
        assert restored.merged(m).items() == m.items()
