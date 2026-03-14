"""Tests for missy.channels.voice.presence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.channels.voice.presence import PresenceData, PresenceStore


def _make_mock_registry(nodes=None):
    """Create a mock DeviceRegistry with optional pre-registered nodes."""
    registry = MagicMock()
    if nodes is None:
        nodes = []

    mock_nodes = []
    for n in nodes:
        node = MagicMock()
        node.node_id = n["node_id"]
        node.room = n.get("room", "Test Room")
        node.sensor_data = n.get("sensor_data", {})
        mock_nodes.append(node)

    registry.list_nodes.return_value = mock_nodes

    def get_node(nid):
        for mn in mock_nodes:
            if mn.node_id == nid:
                return mn
        return None

    registry.get_node = get_node
    return registry


class TestPresenceData:
    def test_default_fields(self):
        pd = PresenceData(node_id="n1", room="Kitchen")
        assert pd.node_id == "n1"
        assert pd.room == "Kitchen"
        assert pd.occupancy is None
        assert pd.noise_level is None
        assert pd.wake_word_false_positives == 0
        assert pd.updated_at > 0

    def test_all_fields(self):
        pd = PresenceData(
            node_id="n2",
            room="Bedroom",
            occupancy=True,
            noise_level=0.5,
            wake_word_false_positives=3,
            updated_at=12345.0,
        )
        assert pd.occupancy is True
        assert pd.noise_level == 0.5
        assert pd.wake_word_false_positives == 3


class TestPresenceStoreInit:
    def test_empty_registry(self):
        reg = _make_mock_registry()
        store = PresenceStore(reg)
        assert store.get_all() == []

    def test_seeded_from_registry(self):
        reg = _make_mock_registry(
            [
                {"node_id": "a", "room": "Living Room", "sensor_data": {"occupancy": True}},
                {"node_id": "b", "room": "Kitchen", "sensor_data": {"noise_level": 0.3}},
            ]
        )
        store = PresenceStore(reg)
        assert len(store.get_all()) == 2
        assert store.get("a").occupancy is True
        assert store.get("b").noise_level == 0.3


class TestPresenceStoreUpdate:
    def test_update_existing_occupancy(self):
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        store = PresenceStore(reg)
        store.update("n1", occupancy=True)
        assert store.get("n1").occupancy is True

    def test_update_noise_level(self):
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        store = PresenceStore(reg)
        store.update("n1", noise_level=0.7)
        assert store.get("n1").noise_level == 0.7

    def test_update_wake_word_fp(self):
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        store = PresenceStore(reg)
        store.update("n1", wake_word_fp=True)
        store.update("n1", wake_word_fp=True)
        assert store.get("n1").wake_word_false_positives == 2

    def test_update_unknown_node_from_registry(self):
        """If node isn't in store but is in registry, create on-the-fly."""
        reg = _make_mock_registry([{"node_id": "new", "room": "Garage"}])
        store = PresenceStore(reg)
        # Clear in-memory data to simulate late registration
        store._data.clear()
        store.update("new", occupancy=False)
        assert store.get("new").occupancy is False
        assert store.get("new").room == "Garage"

    def test_update_unknown_node_not_in_registry_raises(self):
        reg = _make_mock_registry()
        reg.get_node = MagicMock(return_value=None)
        store = PresenceStore(reg)
        with pytest.raises(KeyError, match="Node not found"):
            store.update("ghost", occupancy=True)

    def test_update_syncs_to_registry(self):
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        store = PresenceStore(reg)
        store.update("n1", occupancy=True, noise_level=0.5)
        reg.update_sensor_data.assert_called_once_with("n1", occupancy=True, noise_level=0.5)

    def test_update_registry_sync_failure_logged(self):
        """If registry.update_sensor_data raises KeyError, it's handled."""
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        reg.update_sensor_data.side_effect = KeyError("gone")
        store = PresenceStore(reg)
        store.update("n1", occupancy=True)  # should not raise


class TestPresenceStoreReset:
    def test_reset_false_positives(self):
        reg = _make_mock_registry([{"node_id": "n1", "room": "Room"}])
        store = PresenceStore(reg)
        store.update("n1", wake_word_fp=True)
        store.update("n1", wake_word_fp=True)
        assert store.get("n1").wake_word_false_positives == 2
        store.reset_false_positives("n1")
        assert store.get("n1").wake_word_false_positives == 0

    def test_reset_unknown_node_noop(self):
        reg = _make_mock_registry()
        store = PresenceStore(reg)
        store.reset_false_positives("nonexistent")  # should not raise


class TestPresenceStoreQueries:
    def test_get_unknown_returns_none(self):
        reg = _make_mock_registry()
        store = PresenceStore(reg)
        assert store.get("nope") is None

    def test_get_occupied_rooms(self):
        reg = _make_mock_registry(
            [
                {"node_id": "a", "room": "Living Room"},
                {"node_id": "b", "room": "Kitchen"},
                {"node_id": "c", "room": "Bedroom"},
            ]
        )
        store = PresenceStore(reg)
        store.update("a", occupancy=True)
        store.update("b", occupancy=False)
        # c has no occupancy update (None)
        assert store.get_occupied_rooms() == ["Living Room"]

    def test_get_context_summary_empty(self):
        reg = _make_mock_registry()
        store = PresenceStore(reg)
        assert store.get_context_summary() == "(no nodes registered)"

    def test_get_context_summary_mixed(self):
        reg = _make_mock_registry(
            [
                {"node_id": "a", "room": "Living Room"},
                {"node_id": "b", "room": "Kitchen"},
                {"node_id": "c", "room": "Bedroom"},
            ]
        )
        store = PresenceStore(reg)
        store.update("a", occupancy=True)
        store.update("b", occupancy=False)
        summary = store.get_context_summary()
        assert "Living Room: occupied" in summary
        assert "Kitchen: empty" in summary
        assert "Bedroom: unknown" in summary
        assert " | " in summary
