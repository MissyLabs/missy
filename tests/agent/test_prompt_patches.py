"""Tests for missy.agent.prompt_patches — prompt self-tuning patch system."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.agent.prompt_patches import (
    PatchStatus,
    PatchType,
    PromptPatch,
    PromptPatchManager,
)

# ---- PromptPatch dataclass ----


class TestPromptPatch:
    def test_default_created_at(self):
        p = PromptPatch(id="abc", patch_type=PatchType.TOOL_USAGE_HINT, content="test", confidence=0.5)
        assert p.created_at  # auto-populated
        assert p.status == PatchStatus.PROPOSED

    def test_explicit_created_at(self):
        p = PromptPatch(
            id="abc",
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="test",
            confidence=0.5,
            created_at="2026-01-01T00:00:00",
        )
        assert p.created_at == "2026-01-01T00:00:00"

    def test_success_rate_no_applications(self):
        p = PromptPatch(id="x", patch_type=PatchType.TOOL_USAGE_HINT, content="c", confidence=0.5)
        assert p.success_rate == 0.0

    def test_success_rate_with_data(self):
        p = PromptPatch(
            id="x",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="c",
            confidence=0.5,
            applications=10,
            successes=7,
        )
        assert p.success_rate == pytest.approx(0.7)

    def test_is_expired_few_applications(self):
        p = PromptPatch(
            id="x",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="c",
            confidence=0.5,
            applications=3,
            successes=0,
        )
        assert p.is_expired is False  # not enough applications

    def test_is_expired_low_success_rate(self):
        p = PromptPatch(
            id="x",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="c",
            confidence=0.5,
            applications=10,
            successes=2,
        )
        assert p.is_expired is True  # 20% < 40%

    def test_not_expired_decent_rate(self):
        p = PromptPatch(
            id="x",
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="c",
            confidence=0.5,
            applications=10,
            successes=5,
        )
        assert p.is_expired is False  # 50% >= 40%


# ---- PatchType and PatchStatus enums ----


class TestEnums:
    def test_patch_types(self):
        assert PatchType.TOOL_USAGE_HINT == "tool_usage_hint"
        assert PatchType.ERROR_AVOIDANCE == "error_avoidance"
        assert PatchType.WORKFLOW_PATTERN == "workflow_pattern"
        assert PatchType.DOMAIN_KNOWLEDGE == "domain_knowledge"
        assert PatchType.STYLE_PREFERENCE == "style_preference"

    def test_patch_statuses(self):
        assert PatchStatus.PROPOSED == "proposed"
        assert PatchStatus.APPROVED == "approved"
        assert PatchStatus.REJECTED == "rejected"
        assert PatchStatus.EXPIRED == "expired"


# ---- PromptPatchManager ----


@pytest.fixture
def mgr(tmp_path):
    return PromptPatchManager(store_path=str(tmp_path / "patches.json"))


class TestManagerPropose:
    def test_propose_creates_patch(self, mgr):
        p = mgr.propose(PatchType.ERROR_AVOIDANCE, "Avoid X", confidence=0.6)
        assert p is not None
        assert p.status == PatchStatus.PROPOSED
        assert p.content == "Avoid X"
        assert p.confidence == 0.6

    def test_propose_auto_approves_low_risk_high_confidence(self, mgr):
        p = mgr.propose(PatchType.TOOL_USAGE_HINT, "Use paths", confidence=0.9)
        assert p.status == PatchStatus.APPROVED

    def test_propose_auto_approves_domain_knowledge(self, mgr):
        p = mgr.propose(PatchType.DOMAIN_KNOWLEDGE, "Python uses PEP8", confidence=0.85)
        assert p.status == PatchStatus.APPROVED

    def test_propose_auto_approves_style_preference(self, mgr):
        p = mgr.propose(PatchType.STYLE_PREFERENCE, "Use snake_case", confidence=0.8)
        assert p.status == PatchStatus.APPROVED

    def test_propose_does_not_auto_approve_low_confidence(self, mgr):
        p = mgr.propose(PatchType.TOOL_USAGE_HINT, "Maybe this", confidence=0.5)
        assert p.status == PatchStatus.PROPOSED

    def test_propose_does_not_auto_approve_high_risk(self, mgr):
        p = mgr.propose(PatchType.ERROR_AVOIDANCE, "Critical thing", confidence=0.95)
        assert p.status == PatchStatus.PROPOSED

    def test_propose_does_not_auto_approve_workflow(self, mgr):
        p = mgr.propose(PatchType.WORKFLOW_PATTERN, "Do X then Y", confidence=0.9)
        assert p.status == PatchStatus.PROPOSED

    def test_propose_max_patches_returns_none(self, mgr):
        for i in range(PromptPatchManager.MAX_PATCHES):
            mgr.propose(PatchType.TOOL_USAGE_HINT, f"Patch {i}")
        result = mgr.propose(PatchType.TOOL_USAGE_HINT, "overflow")
        assert result is None

    def test_propose_persists(self, tmp_path):
        path = str(tmp_path / "patches.json")
        mgr1 = PromptPatchManager(store_path=path)
        mgr1.propose(PatchType.ERROR_AVOIDANCE, "persisted", confidence=0.7)
        # Reload from disk
        mgr2 = PromptPatchManager(store_path=path)
        patches = mgr2.list_all()
        assert len(patches) == 1
        assert patches[0].content == "persisted"


class TestManagerApproveReject:
    def test_approve_patch(self, mgr):
        p = mgr.propose(PatchType.ERROR_AVOIDANCE, "test", confidence=0.5)
        assert mgr.approve(p.id)
        patches = mgr.list_all()
        assert patches[0].status == PatchStatus.APPROVED

    def test_approve_nonexistent_returns_false(self, mgr):
        assert mgr.approve("nonexistent") is False

    def test_reject_patch(self, mgr):
        p = mgr.propose(PatchType.ERROR_AVOIDANCE, "test", confidence=0.5)
        assert mgr.reject(p.id)
        patches = mgr.list_all()
        assert patches[0].status == PatchStatus.REJECTED

    def test_reject_nonexistent_returns_false(self, mgr):
        assert mgr.reject("nonexistent") is False


class TestManagerGetActivePatches:
    def test_returns_approved_only(self, mgr):
        p1 = mgr.propose(PatchType.TOOL_USAGE_HINT, "approved", confidence=0.9)
        mgr.propose(PatchType.ERROR_AVOIDANCE, "proposed", confidence=0.5)
        p3 = mgr.propose(PatchType.WORKFLOW_PATTERN, "rejected", confidence=0.5)
        mgr.reject(p3.id)
        active = mgr.get_active_patches()
        assert len(active) == 1
        assert active[0].id == p1.id

    def test_expires_poor_performers(self, mgr):
        p = mgr.propose(PatchType.TOOL_USAGE_HINT, "will expire", confidence=0.9)
        # Manually set poor stats
        p.applications = 10
        p.successes = 1
        mgr._save()
        active = mgr.get_active_patches()
        assert len(active) == 0
        # Verify it was marked expired
        all_patches = mgr.list_all()
        assert all_patches[0].status == PatchStatus.EXPIRED


class TestManagerListMethods:
    def test_list_proposed(self, mgr):
        mgr.propose(PatchType.ERROR_AVOIDANCE, "proposed1", confidence=0.5)
        mgr.propose(PatchType.TOOL_USAGE_HINT, "approved1", confidence=0.9)
        proposed = mgr.list_proposed()
        assert len(proposed) == 1
        assert proposed[0].content == "proposed1"

    def test_list_all(self, mgr):
        mgr.propose(PatchType.ERROR_AVOIDANCE, "a", confidence=0.5)
        mgr.propose(PatchType.TOOL_USAGE_HINT, "b", confidence=0.9)
        assert len(mgr.list_all()) == 2


class TestManagerRecordOutcome:
    def test_record_success(self, mgr):
        mgr.propose(PatchType.TOOL_USAGE_HINT, "test", confidence=0.9)
        mgr.record_outcome(success=True)
        patches = mgr.list_all()
        assert patches[0].applications == 1
        assert patches[0].successes == 1

    def test_record_failure(self, mgr):
        mgr.propose(PatchType.TOOL_USAGE_HINT, "test", confidence=0.9)
        mgr.record_outcome(success=False)
        patches = mgr.list_all()
        assert patches[0].applications == 1
        assert patches[0].successes == 0

    def test_only_records_for_approved(self, mgr):
        mgr.propose(PatchType.ERROR_AVOIDANCE, "proposed", confidence=0.5)
        mgr.record_outcome(success=True)
        patches = mgr.list_all()
        assert patches[0].applications == 0  # not approved, not counted


class TestManagerBuildPatchPrompt:
    def test_empty_when_no_active(self, mgr):
        assert mgr.build_patch_prompt() == ""

    def test_builds_prompt_from_active(self, mgr):
        mgr.propose(PatchType.TOOL_USAGE_HINT, "Always verify paths", confidence=0.9)
        mgr.propose(PatchType.DOMAIN_KNOWLEDGE, "Python uses PEP8", confidence=0.85)
        prompt = mgr.build_patch_prompt()
        assert "Active Prompt Guidance" in prompt
        assert "[tool_usage_hint] Always verify paths" in prompt
        assert "[domain_knowledge] Python uses PEP8" in prompt


class TestManagerPersistence:
    def test_load_from_malformed_file(self, tmp_path):
        path = tmp_path / "patches.json"
        path.write_text("NOT VALID JSON")
        mgr = PromptPatchManager(store_path=str(path))
        assert mgr.list_all() == []

    def test_load_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "patches.json")
        mgr = PromptPatchManager(store_path=path)
        mgr.propose(PatchType.TOOL_USAGE_HINT, "test", confidence=0.9)
        assert Path(path).exists()

    def test_round_trip_all_statuses(self, tmp_path):
        path = str(tmp_path / "patches.json")
        mgr = PromptPatchManager(store_path=path)
        mgr.propose(PatchType.TOOL_USAGE_HINT, "approved", confidence=0.9)
        mgr.propose(PatchType.ERROR_AVOIDANCE, "proposed", confidence=0.5)
        rejected = mgr.propose(PatchType.WORKFLOW_PATTERN, "rejected", confidence=0.5)
        mgr.reject(rejected.id)

        mgr2 = PromptPatchManager(store_path=path)
        patches = mgr2.list_all()
        assert len(patches) == 3
        statuses = {p.content: p.status for p in patches}
        assert statuses["approved"] == PatchStatus.APPROVED
        assert statuses["proposed"] == PatchStatus.PROPOSED
        assert statuses["rejected"] == PatchStatus.REJECTED
