"""Comprehensive tests for voice channel modules.

Covers:
- missy/channels/voice/registry.py   — DeviceRegistry, EdgeNode, CRUD, token management
- missy/channels/voice/pairing.py    — PairingManager full lifecycle
- missy/channels/voice/presence.py   — PresenceStore seeding, updates, queries
- missy/channels/voice/stt/base.py   — TranscriptionResult, STTEngine contract
- missy/channels/voice/tts/base.py   — AudioBuffer duration math, TTSEngine contract
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.presence import PresenceData, PresenceStore
from missy.channels.voice.registry import DeviceRegistry, EdgeNode, _node_from_dict, _node_to_dict
from missy.channels.voice.stt.base import STTEngine, TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer, TTSEngine
from missy.core.events import event_bus

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "node-1",
    friendly_name: str = "Living Room",
    room: str = "living_room",
    ip_address: str = "192.168.1.10",
    *,
    paired: bool = False,
    status: str = "offline",
    policy_mode: str = "full",
) -> EdgeNode:
    return EdgeNode(
        node_id=node_id,
        friendly_name=friendly_name,
        room=room,
        ip_address=ip_address,
        paired=paired,
        status=status,
        policy_mode=policy_mode,
    )


@pytest.fixture(autouse=True)
def clear_event_bus():
    """Clear the module-level event_bus before every test."""
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "devices.json"


@pytest.fixture()
def registry(registry_path: Path) -> DeviceRegistry:
    reg = DeviceRegistry(registry_path=str(registry_path))
    reg.load()
    return reg


# ===========================================================================
# EdgeNode data model
# ===========================================================================


class TestEdgeNode:
    def test_defaults(self):
        node = EdgeNode(
            node_id="n1",
            friendly_name="Bedroom",
            room="bedroom",
            ip_address="10.0.0.1",
        )
        assert node.paired is False
        assert node.status == "offline"
        assert node.policy_mode == "full"
        assert node.token_hash == ""
        assert node.audio_logging is False
        assert node.audio_log_retention_days == 7
        assert node.sensor_data["occupancy"] is None
        assert node.sensor_data["noise_level"] is None

    def test_sensor_data_default_factory_is_independent(self):
        # Each node gets its own sensor_data dict.
        a = EdgeNode(node_id="a", friendly_name="A", room="a", ip_address="1.1.1.1")
        b = EdgeNode(node_id="b", friendly_name="B", room="b", ip_address="2.2.2.2")
        a.sensor_data["occupancy"] = True
        assert b.sensor_data["occupancy"] is None


# ===========================================================================
# Serialisation helpers
# ===========================================================================


class TestNodeSerialisation:
    def test_round_trip(self):
        node = _make_node(paired=True, status="online")
        node.token_hash = "abc123"
        d = _node_to_dict(node)
        restored = _node_from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.friendly_name == node.friendly_name
        assert restored.paired is True
        assert restored.token_hash == "abc123"

    def test_unknown_keys_discarded(self):
        node = _make_node()
        d = _node_to_dict(node)
        d["future_field"] = "should be ignored"
        restored = _node_from_dict(d)
        assert not hasattr(restored, "future_field")

    def test_partial_dict_uses_defaults(self):
        # Minimal dict — only required fields.
        d = {
            "node_id": "x",
            "friendly_name": "X",
            "room": "x_room",
            "ip_address": "0.0.0.0",
        }
        node = _node_from_dict(d)
        assert node.paired is False
        assert node.status == "offline"


# ===========================================================================
# DeviceRegistry — persistence
# ===========================================================================


class TestDeviceRegistryPersistence:
    def test_load_missing_file_starts_empty(self, registry: DeviceRegistry):
        assert registry.list_nodes() == []

    def test_save_and_reload(self, registry_path: Path, registry: DeviceRegistry):
        node = _make_node()
        registry.add_node(node)

        # Reload from same path.
        reg2 = DeviceRegistry(registry_path=str(registry_path))
        reg2.load()
        assert len(reg2.list_nodes()) == 1
        assert reg2.get_node("node-1").friendly_name == "Living Room"

    def test_load_corrupt_file_starts_empty(self, registry_path: Path):
        registry_path.write_text("not valid json", encoding="utf-8")
        reg = DeviceRegistry(registry_path=str(registry_path))
        reg.load()
        assert reg.list_nodes() == []

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "devices.json"
        reg = DeviceRegistry(registry_path=str(deep_path))
        reg.load()
        reg.add_node(_make_node())
        assert deep_path.exists()

    def test_save_is_valid_json(self, registry_path: Path, registry: DeviceRegistry):
        registry.add_node(_make_node())
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert data[0]["node_id"] == "node-1"

    def test_atomic_write_uses_rename(self, registry_path: Path, registry: DeviceRegistry):
        """Save should not leave .tmp files behind."""
        registry.add_node(_make_node())
        tmp_files = list(registry_path.parent.glob(".devices_*.json.tmp"))
        assert tmp_files == []


# ===========================================================================
# DeviceRegistry — CRUD
# ===========================================================================


class TestDeviceRegistryCRUD:
    def test_add_node_stores_and_emits_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.device.registered", events.append)

        node = _make_node()
        registry.add_node(node)

        assert registry.get_node("node-1") is node
        assert len(events) == 1
        assert events[0].detail["node_id"] == "node-1"

    def test_add_node_overwrites_existing(self, registry: DeviceRegistry):
        registry.add_node(_make_node(friendly_name="Old Name"))
        registry.add_node(_make_node(friendly_name="New Name"))
        assert registry.get_node("node-1").friendly_name == "New Name"

    def test_get_node_returns_none_when_missing(self, registry: DeviceRegistry):
        assert registry.get_node("nonexistent") is None

    def test_update_node_applies_changes(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        registry.update_node("node-1", status="online", ip_address="192.168.1.99")
        node = registry.get_node("node-1")
        assert node.status == "online"
        assert node.ip_address == "192.168.1.99"

    def test_update_node_ignores_unknown_fields(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        # Should not raise even with junk keys.
        registry.update_node("node-1", nonexistent_field="value")

    def test_update_node_raises_for_missing(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.update_node("ghost", status="online")

    def test_remove_node_removes_and_emits_event(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        events = []
        event_bus.subscribe("voice.device.removed", events.append)

        registry.remove_node("node-1")

        assert registry.get_node("node-1") is None
        assert len(events) == 1
        assert events[0].detail["node_id"] == "node-1"

    def test_remove_node_noop_when_missing(self, registry: DeviceRegistry):
        # Should not raise.
        registry.remove_node("ghost")

    def test_list_nodes_empty(self, registry: DeviceRegistry):
        assert registry.list_nodes() == []

    def test_list_nodes_returns_all(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a"))
        registry.add_node(_make_node(node_id="b"))
        ids = {n.node_id for n in registry.list_nodes()}
        assert ids == {"a", "b"}

    def test_list_paired(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="unpaired", paired=False))
        registry.add_node(_make_node(node_id="paired", paired=True))
        paired = registry.list_paired()
        assert len(paired) == 1
        assert paired[0].node_id == "paired"

    def test_list_pending(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="unpaired", paired=False))
        registry.add_node(_make_node(node_id="paired", paired=True))
        pending = registry.list_pending()
        assert len(pending) == 1
        assert pending[0].node_id == "unpaired"


# ===========================================================================
# DeviceRegistry — pairing / approval
# ===========================================================================


class TestDeviceRegistryApproval:
    def test_approve_node_sets_paired_true(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        registry.approve_node("node-1")
        assert registry.get_node("node-1").paired is True

    def test_approve_node_emits_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.device.approved", events.append)
        registry.add_node(_make_node())
        registry.approve_node("node-1")
        assert len(events) == 1

    def test_approve_node_raises_for_missing(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.approve_node("ghost")


# ===========================================================================
# DeviceRegistry — token management
# ===========================================================================


class TestDeviceRegistryTokens:
    def test_generate_token_returns_nonempty_string(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        token = registry.generate_token("node-1")
        assert isinstance(token, str) and len(token) > 0

    def test_verify_token_accepts_correct_token(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        token = registry.generate_token("node-1")
        assert registry.verify_token("node-1", token) is True

    def test_verify_token_rejects_wrong_token(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        registry.generate_token("node-1")
        assert registry.verify_token("node-1", "wrong-token") is False

    def test_verify_token_returns_false_for_missing_node(self, registry: DeviceRegistry):
        assert registry.verify_token("ghost", "any-token") is False

    def test_verify_token_returns_false_when_no_hash(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        assert registry.verify_token("node-1", "any-token") is False

    def test_generate_token_rotates_old_token(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        token1 = registry.generate_token("node-1")
        token2 = registry.generate_token("node-1")
        # Old token must no longer work.
        assert registry.verify_token("node-1", token1) is False
        assert registry.verify_token("node-1", token2) is True

    def test_generate_token_raises_for_missing_node(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.generate_token("ghost")

    def test_tokens_are_unique_across_nodes(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a"))
        registry.add_node(_make_node(node_id="b"))
        t_a = registry.generate_token("a")
        t_b = registry.generate_token("b")
        # Token for a does not work for b.
        assert registry.verify_token("b", t_a) is False
        assert registry.verify_token("a", t_b) is False


# ===========================================================================
# DeviceRegistry — presence / sensor data
# ===========================================================================


class TestDeviceRegistryPresence:
    def test_update_sensor_data_persists(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        registry.update_sensor_data("node-1", occupancy=True, noise_level=0.4)
        node = registry.get_node("node-1")
        assert node.sensor_data["occupancy"] is True
        assert node.sensor_data["noise_level"] == pytest.approx(0.4)

    def test_update_sensor_data_raises_for_missing(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.update_sensor_data("ghost", occupancy=False, noise_level=None)

    def test_mark_online(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        before = time.time()
        registry.mark_online("node-1", ip_address="10.0.0.5")
        node = registry.get_node("node-1")
        assert node.status == "online"
        assert node.ip_address == "10.0.0.5"
        assert node.last_seen >= before

    def test_mark_offline(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        registry.mark_online("node-1", ip_address="10.0.0.5")
        registry.mark_offline("node-1")
        assert registry.get_node("node-1").status == "offline"

    def test_mark_online_raises_for_missing(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.mark_online("ghost", ip_address="1.1.1.1")

    def test_mark_offline_raises_for_missing(self, registry: DeviceRegistry):
        with pytest.raises(KeyError):
            registry.mark_offline("ghost")


# ===========================================================================
# DeviceRegistry — audio log purge
# ===========================================================================


class TestDeviceRegistryPurgeAudioLogs:
    def test_purge_returns_zero_when_no_audio_logging(self, registry: DeviceRegistry):
        registry.add_node(_make_node())
        assert registry.purge_audio_logs() == 0

    def test_purge_deletes_old_files(self, registry: DeviceRegistry, tmp_path: Path):
        log_dir = tmp_path / "audio_logs"
        log_dir.mkdir()

        # Create one old file and one recent file.
        old_file = log_dir / "old.wav"
        old_file.write_bytes(b"x")
        recent_file = log_dir / "recent.wav"
        recent_file.write_bytes(b"y")

        # Make old_file appear 10 days old.
        old_mtime = time.time() - (10 * 86400)
        import os
        os.utime(old_file, (old_mtime, old_mtime))

        node = EdgeNode(
            node_id="node-1",
            friendly_name="Test",
            room="test",
            ip_address="1.1.1.1",
            audio_logging=True,
            audio_log_dir=str(log_dir),
            audio_log_retention_days=7,
        )
        registry.add_node(node)
        count = registry.purge_audio_logs()

        assert count == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_purge_skips_missing_directory(self, registry: DeviceRegistry, tmp_path: Path):
        node = EdgeNode(
            node_id="node-1",
            friendly_name="Test",
            room="test",
            ip_address="1.1.1.1",
            audio_logging=True,
            audio_log_dir=str(tmp_path / "nonexistent"),
            audio_log_retention_days=7,
        )
        registry.add_node(node)
        # Should not raise.
        count = registry.purge_audio_logs()
        assert count == 0

    def test_purge_ignores_subdirectories(self, registry: DeviceRegistry, tmp_path: Path):
        log_dir = tmp_path / "audio"
        log_dir.mkdir()
        subdir = log_dir / "subdir"
        subdir.mkdir()

        node = EdgeNode(
            node_id="node-1",
            friendly_name="T",
            room="t",
            ip_address="1.1.1.1",
            audio_logging=True,
            audio_log_dir=str(log_dir),
            audio_log_retention_days=0,  # zero days = delete everything older than now
        )
        registry.add_node(node)
        registry.purge_audio_logs()
        # The subdirectory should never be deleted.
        assert subdir.exists()


# ===========================================================================
# PairingManager
# ===========================================================================


class TestPairingManagerInitiate:
    def test_creates_pending_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        node_id = mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Kitchen",
            room="kitchen",
            ip_address="192.168.1.20",
            hardware_profile={"mic": "respeaker"},
        )
        assert node_id == "n1"
        node = registry.get_node("n1")
        assert node is not None
        assert node.paired is False
        assert node.status == "offline"
        assert node.friendly_name == "Kitchen"

    def test_generates_uuid_when_node_id_empty(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        node_id = mgr.initiate_pairing(
            node_id="",
            friendly_name="Hallway",
            room="hallway",
            ip_address="10.0.0.2",
            hardware_profile={},
        )
        assert len(node_id) > 0
        assert registry.get_node(node_id) is not None

    def test_idempotent_for_existing_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="First",
            room="room",
            ip_address="1.1.1.1",
            hardware_profile={},
        )
        # Second call with same node_id should not overwrite.
        returned_id = mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Second",
            room="room2",
            ip_address="2.2.2.2",
            hardware_profile={},
        )
        assert returned_id == "n1"
        assert registry.get_node("n1").friendly_name == "First"

    def test_emits_initiated_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.pairing.initiated", events.append)
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Office",
            room="office",
            ip_address="10.0.0.3",
            hardware_profile={},
        )
        assert len(events) == 1
        assert events[0].event_type == "voice.pairing.initiated"
        assert events[0].detail["node_id"] == "n1"

    def test_policy_mode_is_stored(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Safe Node",
            room="room",
            ip_address="1.1.1.1",
            hardware_profile={},
            policy_mode="safe-chat",
        )
        assert registry.get_node("n1").policy_mode == "safe-chat"


class TestPairingManagerApprove:
    def test_approve_sets_paired_and_returns_token(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Bedroom",
            room="bedroom",
            ip_address="10.0.0.1",
            hardware_profile={},
        )
        token = mgr.approve_pairing("n1")
        assert isinstance(token, str) and len(token) > 0
        assert registry.get_node("n1").paired is True

    def test_token_is_verifiable(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Garage",
            room="garage",
            ip_address="10.0.0.4",
            hardware_profile={},
        )
        token = mgr.approve_pairing("n1")
        assert registry.verify_token("n1", token) is True

    def test_approve_emits_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.pairing.approved", events.append)
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Lab",
            room="lab",
            ip_address="10.0.0.5",
            hardware_profile={},
        )
        mgr.approve_pairing("n1")
        assert len(events) == 1

    def test_approve_raises_for_missing_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        with pytest.raises(ValueError, match="Node not found"):
            mgr.approve_pairing("ghost")

    def test_approve_raises_for_already_paired(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="TV Room",
            room="tv_room",
            ip_address="10.0.0.6",
            hardware_profile={},
        )
        mgr.approve_pairing("n1")
        with pytest.raises(ValueError, match="already paired"):
            mgr.approve_pairing("n1")


class TestPairingManagerReject:
    def test_reject_removes_pending_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Porch",
            room="porch",
            ip_address="10.0.0.7",
            hardware_profile={},
        )
        mgr.reject_pairing("n1")
        assert registry.get_node("n1") is None

    def test_reject_emits_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.pairing.rejected", events.append)
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Attic",
            room="attic",
            ip_address="10.0.0.8",
            hardware_profile={},
        )
        mgr.reject_pairing("n1")
        assert len(events) == 1

    def test_reject_noop_for_missing_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        # Should not raise.
        mgr.reject_pairing("ghost")

    def test_reject_raises_for_already_paired(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Study",
            room="study",
            ip_address="10.0.0.9",
            hardware_profile={},
        )
        mgr.approve_pairing("n1")
        with pytest.raises(ValueError, match="already paired"):
            mgr.reject_pairing("n1")


class TestPairingManagerListAndUnpair:
    def test_list_pending_returns_only_unpaired(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="pending",
            friendly_name="Pending",
            room="p",
            ip_address="1.1.1.1",
            hardware_profile={},
        )
        mgr.initiate_pairing(
            node_id="approved",
            friendly_name="Approved",
            room="a",
            ip_address="2.2.2.2",
            hardware_profile={},
        )
        mgr.approve_pairing("approved")
        pending = mgr.list_pending()
        assert len(pending) == 1
        assert pending[0].node_id == "pending"

    def test_unpair_removes_paired_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Nursery",
            room="nursery",
            ip_address="10.0.0.10",
            hardware_profile={},
        )
        mgr.approve_pairing("n1")
        mgr.unpair_node("n1")
        assert registry.get_node("n1") is None

    def test_unpair_emits_event(self, registry: DeviceRegistry):
        events = []
        event_bus.subscribe("voice.pairing.unpaired", events.append)
        mgr = PairingManager(registry)
        mgr.initiate_pairing(
            node_id="n1",
            friendly_name="Gym",
            room="gym",
            ip_address="10.0.0.11",
            hardware_profile={},
        )
        mgr.approve_pairing("n1")
        mgr.unpair_node("n1")
        assert len(events) == 1
        assert events[0].detail["node_id"] == "n1"

    def test_unpair_noop_for_missing_node(self, registry: DeviceRegistry):
        mgr = PairingManager(registry)
        # Should not raise.
        mgr.unpair_node("ghost")


# ===========================================================================
# PresenceData dataclass
# ===========================================================================


class TestPresenceData:
    def test_defaults(self):
        before = time.time()
        pd = PresenceData(node_id="n1", room="lounge")
        assert pd.occupancy is None
        assert pd.noise_level is None
        assert pd.wake_word_false_positives == 0
        assert pd.updated_at >= before

    def test_explicit_values(self):
        pd = PresenceData(
            node_id="n1",
            room="lounge",
            occupancy=True,
            noise_level=0.7,
            wake_word_false_positives=3,
            updated_at=1_000_000.0,
        )
        assert pd.occupancy is True
        assert pd.noise_level == pytest.approx(0.7)
        assert pd.wake_word_false_positives == 3
        assert pd.updated_at == pytest.approx(1_000_000.0)


# ===========================================================================
# PresenceStore — seeding
# ===========================================================================


class TestPresenceStoreSeeding:
    def test_empty_registry_produces_empty_store(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        assert store.get_all() == []

    def test_seeds_from_registry_nodes(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1", room="living_room"))
        store = PresenceStore(registry)
        pd = store.get("n1")
        assert pd is not None
        assert pd.room == "living_room"

    def test_seeds_occupancy_from_sensor_data(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        registry.update_sensor_data("n1", occupancy=True, noise_level=0.3)
        store = PresenceStore(registry)
        pd = store.get("n1")
        assert pd.occupancy is True
        assert pd.noise_level == pytest.approx(0.3)

    def test_seeds_multiple_nodes(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a"))
        registry.add_node(_make_node(node_id="b"))
        store = PresenceStore(registry)
        assert len(store.get_all()) == 2


# ===========================================================================
# PresenceStore — update
# ===========================================================================


class TestPresenceStoreUpdate:
    def test_update_sets_occupancy(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        store.update("n1", occupancy=True)
        assert store.get("n1").occupancy is True

    def test_update_sets_noise_level(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        store.update("n1", noise_level=0.5)
        assert store.get("n1").noise_level == pytest.approx(0.5)

    def test_update_increments_false_positives(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        store.update("n1", wake_word_fp=True)
        store.update("n1", wake_word_fp=True)
        assert store.get("n1").wake_word_false_positives == 2

    def test_update_none_fields_unchanged(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        registry.update_sensor_data("n1", occupancy=True, noise_level=0.6)
        store = PresenceStore(registry)
        # Update with no occupancy/noise_level should not clear existing values.
        store.update("n1", wake_word_fp=True)
        pd = store.get("n1")
        assert pd.occupancy is True
        assert pd.noise_level == pytest.approx(0.6)

    def test_update_syncs_to_registry(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        store.update("n1", occupancy=False, noise_level=0.1)
        sd = registry.get_node("n1").sensor_data
        assert sd["occupancy"] is False
        assert sd["noise_level"] == pytest.approx(0.1)

    def test_update_unknown_node_not_in_store(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        # Node not in registry — should raise KeyError.
        with pytest.raises(KeyError):
            store.update("ghost", occupancy=True)

    def test_update_creates_entry_for_node_added_after_init(
        self, registry: DeviceRegistry
    ):
        store = PresenceStore(registry)
        # Add a node AFTER the store was seeded.
        registry.add_node(_make_node(node_id="late"))
        store.update("late", occupancy=True)
        assert store.get("late").occupancy is True

    def test_update_refreshes_updated_at(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        before = time.time()
        store.update("n1", occupancy=True)
        assert store.get("n1").updated_at >= before


# ===========================================================================
# PresenceStore — reset_false_positives
# ===========================================================================


class TestPresenceStoreResetFalsePositives:
    def test_reset_zeros_counter(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1"))
        store = PresenceStore(registry)
        store.update("n1", wake_word_fp=True)
        store.update("n1", wake_word_fp=True)
        store.reset_false_positives("n1")
        assert store.get("n1").wake_word_false_positives == 0

    def test_reset_noop_for_missing(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        # Should not raise.
        store.reset_false_positives("ghost")


# ===========================================================================
# PresenceStore — queries
# ===========================================================================


class TestPresenceStoreQueries:
    def test_get_returns_none_for_unknown(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        assert store.get("ghost") is None

    def test_get_all_snapshot(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a"))
        registry.add_node(_make_node(node_id="b"))
        store = PresenceStore(registry)
        snapshot = store.get_all()
        assert {pd.node_id for pd in snapshot} == {"a", "b"}

    def test_get_occupied_rooms(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a", room="lounge"))
        registry.add_node(_make_node(node_id="b", room="kitchen"))
        registry.add_node(_make_node(node_id="c", room="bedroom"))
        store = PresenceStore(registry)
        store.update("a", occupancy=True)
        store.update("b", occupancy=False)
        # c remains None occupancy.
        rooms = store.get_occupied_rooms()
        assert rooms == ["lounge"]

    def test_get_occupied_rooms_empty(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        assert store.get_occupied_rooms() == []


class TestPresenceStoreContextSummary:
    def test_empty_store(self, registry: DeviceRegistry):
        store = PresenceStore(registry)
        assert store.get_context_summary() == "(no nodes registered)"

    def test_occupied_room(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1", room="Lounge"))
        store = PresenceStore(registry)
        store.update("n1", occupancy=True)
        summary = store.get_context_summary()
        assert "Lounge: occupied" in summary

    def test_empty_room(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1", room="Kitchen"))
        store = PresenceStore(registry)
        store.update("n1", occupancy=False)
        assert "Kitchen: empty" in store.get_context_summary()

    def test_unknown_occupancy(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="n1", room="Bedroom"))
        store = PresenceStore(registry)
        # No occupancy update — stays None.
        assert "Bedroom: unknown" in store.get_context_summary()

    def test_multiple_rooms_pipe_delimited(self, registry: DeviceRegistry):
        registry.add_node(_make_node(node_id="a", room="A"))
        registry.add_node(_make_node(node_id="b", room="B"))
        store = PresenceStore(registry)
        store.update("a", occupancy=True)
        store.update("b", occupancy=False)
        summary = store.get_context_summary()
        assert " | " in summary
        assert "A: occupied" in summary
        assert "B: empty" in summary


# ===========================================================================
# TranscriptionResult
# ===========================================================================


class TestTranscriptionResult:
    def test_required_fields(self):
        r = TranscriptionResult(text="hello", confidence=0.95, processing_ms=120)
        assert r.text == "hello"
        assert r.confidence == pytest.approx(0.95)
        assert r.processing_ms == 120
        assert r.language == ""

    def test_language_field(self):
        r = TranscriptionResult(text="hola", confidence=0.8, processing_ms=200, language="es")
        assert r.language == "es"

    def test_confidence_minus_one_sentinel(self):
        r = TranscriptionResult(text="?", confidence=-1.0, processing_ms=50)
        assert r.confidence == pytest.approx(-1.0)

    def test_empty_text(self):
        r = TranscriptionResult(text="", confidence=0.0, processing_ms=0)
        assert r.text == ""


# ===========================================================================
# STTEngine — abstract contract
# ===========================================================================


class TestSTTEngineAbstractContract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            STTEngine()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        """A subclass that omits any abstract method still can't be instantiated."""
        class IncompleteSTT(STTEngine):
            name = "incomplete"

            def load(self): ...
            def unload(self): ...
            # Missing is_loaded and transcribe.

        with pytest.raises(TypeError):
            IncompleteSTT()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class DummySTT(STTEngine):
            name = "dummy"

            def load(self): pass
            def unload(self): pass
            def is_loaded(self): return True
            async def transcribe(self, audio, sample_rate=16000, channels=1):
                return TranscriptionResult(text="ok", confidence=1.0, processing_ms=1)

        engine = DummySTT()
        engine.load()
        assert engine.is_loaded() is True
        engine.unload()

    def test_transcribe_is_async(self):
        class DummySTT(STTEngine):
            name = "dummy"

            def load(self): pass
            def unload(self): pass
            def is_loaded(self): return True
            async def transcribe(self, audio, sample_rate=16000, channels=1):
                return TranscriptionResult(text="hello", confidence=0.9, processing_ms=10)

        engine = DummySTT()
        result = asyncio.run(engine.transcribe(b"\x00\x00"))
        assert result.text == "hello"


# ===========================================================================
# AudioBuffer
# ===========================================================================


class TestAudioBuffer:
    def test_duration_ms_mono_16khz(self):
        # 16000 samples/sec * 2 bytes each = 32000 bytes = 1000 ms.
        data = bytes(32000)
        buf = AudioBuffer(data=data, sample_rate=16000, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 1000

    def test_duration_ms_stereo_22khz(self):
        # 22050 samples/sec * 2 channels * 2 bytes = 88200 bytes/sec.
        # 88200 bytes => exactly 1000 ms.
        data = bytes(88200)
        buf = AudioBuffer(data=data, sample_rate=22050, channels=2, format="wav")
        assert buf.duration_ms == 1000

    def test_duration_ms_empty_data(self):
        buf = AudioBuffer(data=b"", sample_rate=16000, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 0

    def test_duration_ms_half_second(self):
        # 0.5 s at 16 kHz mono = 16000 samples = 32000 bytes.
        data = bytes(16000)
        buf = AudioBuffer(data=data, sample_rate=16000, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 500

    def test_format_stored(self):
        buf = AudioBuffer(data=b"\x00" * 100, sample_rate=44100, channels=1, format="wav")
        assert buf.format == "wav"

    def test_duration_ms_is_int(self):
        buf = AudioBuffer(data=bytes(100), sample_rate=16000, channels=1, format="pcm_s16le")
        assert isinstance(buf.duration_ms, int)

    def test_duration_ms_not_in_init(self):
        """duration_ms is derived, not accepted as a constructor argument."""
        import dataclasses
        init_fields = {f.name for f in dataclasses.fields(AudioBuffer) if f.init}
        assert "duration_ms" not in init_fields


# ===========================================================================
# TTSEngine — abstract contract
# ===========================================================================


class TestTTSEngineAbstractContract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TTSEngine()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        class IncompleteTTS(TTSEngine):
            name = "incomplete"

            def load(self): ...
            def unload(self): ...
            # Missing is_loaded, list_voices, synthesize.

        with pytest.raises(TypeError):
            IncompleteTTS()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class DummyTTS(TTSEngine):
            name = "dummy"

            def load(self): pass
            def unload(self): pass
            def is_loaded(self): return True
            def list_voices(self): return ["en_US-lessac-medium"]
            async def synthesize(self, text, voice=None):
                data = bytes(3200)  # 100 ms at 16 kHz mono.
                return AudioBuffer(data=data, sample_rate=16000, channels=1, format="pcm_s16le")

        engine = DummyTTS()
        engine.load()
        assert engine.is_loaded() is True
        assert "en_US-lessac-medium" in engine.list_voices()
        engine.unload()

    def test_synthesize_is_async(self):
        class DummyTTS(TTSEngine):
            name = "dummy"

            def load(self): pass
            def unload(self): pass
            def is_loaded(self): return True
            def list_voices(self): return []
            async def synthesize(self, text, voice=None):
                return AudioBuffer(data=bytes(3200), sample_rate=16000, channels=1, format="pcm_s16le")

        engine = DummyTTS()
        buf = asyncio.run(engine.synthesize("hi"))
        assert buf.duration_ms == 100

    def test_list_voices_returns_list(self):
        class DummyTTS(TTSEngine):
            name = "dummy"

            def load(self): pass
            def unload(self): pass
            def is_loaded(self): return True
            def list_voices(self): return ["voice-a", "voice-b"]
            async def synthesize(self, text, voice=None):
                return AudioBuffer(data=b"", sample_rate=16000, channels=1, format="wav")

        engine = DummyTTS()
        voices = engine.list_voices()
        assert isinstance(voices, list)
        assert len(voices) == 2


# ===========================================================================
# Voice channel __init__ public API surface
# ===========================================================================


class TestVoicePackagePublicAPI:
    def test_public_names_importable(self):
        from missy.channels.voice import (
            DeviceRegistry,
            EdgeNode,
            PairingManager,
            PresenceStore,
            VoiceChannel,
            VoiceServer,
        )
        # Simply verifying imports succeed and are the expected types.
        assert DeviceRegistry is not None
        assert EdgeNode is not None
        assert PairingManager is not None
        assert PresenceStore is not None
        assert VoiceChannel is not None
        assert VoiceServer is not None
