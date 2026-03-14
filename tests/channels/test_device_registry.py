"""Tests for missy.channels.voice.registry."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from missy.channels.voice.registry import (
    DeviceRegistry,
    EdgeNode,
    _node_from_dict,
    _node_to_dict,
)


@pytest.fixture
def registry(tmp_path):
    """Fresh DeviceRegistry backed by a temp file."""
    path = str(tmp_path / "devices.json")
    reg = DeviceRegistry(registry_path=path)
    return reg


@pytest.fixture
def sample_node():
    return EdgeNode(
        node_id="node-1",
        friendly_name="Living Room",
        room="Living Room",
        ip_address="192.168.1.100",
    )


class TestEdgeNode:
    def test_defaults(self):
        n = EdgeNode(node_id="n", friendly_name="N", room="R", ip_address="1.2.3.4")
        assert n.status == "offline"
        assert n.policy_mode == "full"
        assert n.paired is False
        assert n.audio_logging is False
        assert n.sensor_data["occupancy"] is None

    def test_serialization_round_trip(self):
        n = EdgeNode(
            node_id="n1", friendly_name="Test", room="Room",
            ip_address="10.0.0.1", policy_mode="safe-chat", paired=True,
        )
        d = _node_to_dict(n)
        restored = _node_from_dict(d)
        assert restored.node_id == n.node_id
        assert restored.policy_mode == "safe-chat"
        assert restored.paired is True

    def test_from_dict_ignores_unknown_fields(self):
        d = {
            "node_id": "n2", "friendly_name": "X", "room": "Y",
            "ip_address": "0.0.0.0", "unknown_field": "ignored",
        }
        n = _node_from_dict(d)
        assert n.node_id == "n2"
        assert not hasattr(n, "unknown_field")


class TestDeviceRegistryPersistence:
    def test_load_empty_file(self, registry):
        registry.load()
        assert registry.list_nodes() == []

    def test_save_and_load(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        # Create a new registry instance pointing to the same file
        reg2 = DeviceRegistry(registry_path=str(registry._path))
        reg2.load()
        nodes = reg2.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "node-1"

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "devices.json"
        path.write_text("NOT VALID JSON", encoding="utf-8")
        reg = DeviceRegistry(registry_path=str(path))
        reg.load()
        assert reg.list_nodes() == []

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "devices.json")
        reg = DeviceRegistry(registry_path=nested)
        reg.load()
        reg.add_node(EdgeNode(node_id="n", friendly_name="N", room="R", ip_address="1.2.3.4"))
        assert Path(nested).exists()


class TestDeviceRegistryCRUD:
    def test_add_and_get_node(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        assert registry.get_node("node-1") is sample_node

    def test_get_nonexistent_returns_none(self, registry):
        registry.load()
        assert registry.get_node("nope") is None

    def test_update_node(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.update_node("node-1", status="online", ip_address="10.0.0.2")
        n = registry.get_node("node-1")
        assert n.status == "online"
        assert n.ip_address == "10.0.0.2"

    def test_update_nonexistent_raises(self, registry):
        registry.load()
        with pytest.raises(KeyError):
            registry.update_node("ghost", status="online")

    def test_update_ignores_unknown_fields(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.update_node("node-1", bogus_field="ignored")  # should not raise

    def test_remove_node(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.remove_node("node-1")
        assert registry.get_node("node-1") is None

    def test_remove_nonexistent_noop(self, registry):
        registry.load()
        registry.remove_node("ghost")  # should not raise


class TestDeviceRegistryQueries:
    def test_list_nodes(self, registry):
        registry.load()
        registry.add_node(EdgeNode(node_id="a", friendly_name="A", room="R", ip_address="1.1.1.1"))
        registry.add_node(EdgeNode(node_id="b", friendly_name="B", room="R", ip_address="2.2.2.2"))
        assert len(registry.list_nodes()) == 2

    def test_list_paired(self, registry):
        registry.load()
        n1 = EdgeNode(node_id="a", friendly_name="A", room="R", ip_address="1.1.1.1", paired=True)
        n2 = EdgeNode(node_id="b", friendly_name="B", room="R", ip_address="2.2.2.2", paired=False)
        registry.add_node(n1)
        registry.add_node(n2)
        paired = registry.list_paired()
        assert len(paired) == 1
        assert paired[0].node_id == "a"

    def test_list_pending(self, registry):
        registry.load()
        n1 = EdgeNode(node_id="a", friendly_name="A", room="R", ip_address="1.1.1.1", paired=True)
        n2 = EdgeNode(node_id="b", friendly_name="B", room="R", ip_address="2.2.2.2", paired=False)
        registry.add_node(n1)
        registry.add_node(n2)
        pending = registry.list_pending()
        assert len(pending) == 1
        assert pending[0].node_id == "b"


class TestDeviceRegistryPairing:
    def test_approve_node(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        assert sample_node.paired is False
        registry.approve_node("node-1")
        assert registry.get_node("node-1").paired is True

    def test_approve_nonexistent_raises(self, registry):
        registry.load()
        with pytest.raises(KeyError):
            registry.approve_node("ghost")


class TestDeviceRegistryTokens:
    def test_generate_and_verify_token(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        token = registry.generate_token("node-1")
        assert isinstance(token, str)
        assert len(token) > 20
        assert registry.verify_token("node-1", token) is True

    def test_wrong_token_fails(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.generate_token("node-1")
        assert registry.verify_token("node-1", "wrong-token") is False

    def test_verify_nonexistent_node(self, registry):
        registry.load()
        assert registry.verify_token("ghost", "token") is False

    def test_verify_no_stored_hash(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        # No token generated yet
        assert registry.verify_token("node-1", "anything") is False

    def test_regenerate_invalidates_old(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        old_token = registry.generate_token("node-1")
        new_token = registry.generate_token("node-1")
        assert registry.verify_token("node-1", new_token) is True
        assert registry.verify_token("node-1", old_token) is False


class TestDeviceRegistrySensorData:
    def test_update_sensor_data(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.update_sensor_data("node-1", occupancy=True, noise_level=0.5)
        n = registry.get_node("node-1")
        assert n.sensor_data["occupancy"] is True
        assert n.sensor_data["noise_level"] == 0.5

    def test_mark_online(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.mark_online("node-1", "10.0.0.5")
        n = registry.get_node("node-1")
        assert n.status == "online"
        assert n.ip_address == "10.0.0.5"

    def test_mark_offline(self, registry, sample_node):
        registry.load()
        registry.add_node(sample_node)
        registry.mark_online("node-1", "10.0.0.5")
        registry.mark_offline("node-1")
        assert registry.get_node("node-1").status == "offline"


class TestAudioLogPurge:
    def test_purge_no_logging_nodes(self, registry):
        registry.load()
        assert registry.purge_audio_logs() == 0

    def test_purge_deletes_old_files(self, registry, tmp_path):
        registry.load()
        log_dir = tmp_path / "audio_logs"
        log_dir.mkdir()
        # Create an old file
        old_file = log_dir / "old.wav"
        old_file.write_text("audio")
        import os
        old_time = time.time() - 86400 * 30  # 30 days ago
        os.utime(old_file, (old_time, old_time))
        # Create a recent file
        new_file = log_dir / "new.wav"
        new_file.write_text("audio")

        node = EdgeNode(
            node_id="n1", friendly_name="N", room="R", ip_address="1.1.1.1",
            audio_logging=True, audio_log_dir=str(log_dir),
            audio_log_retention_days=7,
        )
        registry.add_node(node)
        deleted = registry.purge_audio_logs()
        assert deleted == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_purge_missing_dir_skipped(self, registry):
        registry.load()
        node = EdgeNode(
            node_id="n1", friendly_name="N", room="R", ip_address="1.1.1.1",
            audio_logging=True, audio_log_dir="/nonexistent/path",
            audio_log_retention_days=7,
        )
        registry.add_node(node)
        assert registry.purge_audio_logs() == 0
