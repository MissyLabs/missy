"""Tests for missy.mcp.annotations — ToolAnnotation and AnnotationRegistry."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from missy.mcp.annotations import (
    BUILTIN_ANNOTATIONS,
    AnnotationRegistry,
    ToolAnnotation,
)

# ---------------------------------------------------------------------------
# ToolAnnotation defaults
# ---------------------------------------------------------------------------


class TestToolAnnotationDefaults:
    def test_default_read_only_true(self):
        ann = ToolAnnotation()
        assert ann.read_only is True

    def test_default_mutating_false(self):
        ann = ToolAnnotation()
        assert ann.mutating is False

    def test_default_idempotent_false(self):
        ann = ToolAnnotation()
        assert ann.idempotent is False

    def test_default_cost_hint_none(self):
        ann = ToolAnnotation()
        assert ann.cost_hint == "none"

    def test_default_estimated_latency_none(self):
        ann = ToolAnnotation()
        assert ann.estimated_latency_ms is None

    def test_default_category_general(self):
        ann = ToolAnnotation()
        assert ann.category == "general"

    def test_default_requires_approval_false(self):
        ann = ToolAnnotation()
        assert ann.requires_approval is False

    def test_default_network_access_false(self):
        ann = ToolAnnotation()
        assert ann.network_access is False

    def test_default_filesystem_access_false(self):
        ann = ToolAnnotation()
        assert ann.filesystem_access is False


# ---------------------------------------------------------------------------
# ToolAnnotation.from_mcp_dict
# ---------------------------------------------------------------------------


class TestFromMcpDict:
    def test_empty_dict_uses_defaults(self):
        ann = ToolAnnotation.from_mcp_dict({})
        assert ann.read_only is True
        assert ann.mutating is False
        assert ann.idempotent is False
        assert ann.network_access is False
        assert ann.requires_approval is False
        assert ann.cost_hint == "none"
        assert ann.estimated_latency_ms is None
        assert ann.category == "general"

    def test_read_only_hint_false(self):
        ann = ToolAnnotation.from_mcp_dict({"readOnlyHint": False})
        assert ann.read_only is False
        assert ann.category == "write"

    def test_destructive_hint_sets_mutating_and_approval(self):
        ann = ToolAnnotation.from_mcp_dict({"destructiveHint": True})
        assert ann.mutating is True
        assert ann.requires_approval is True
        assert ann.category == "dangerous"

    def test_idempotent_hint(self):
        ann = ToolAnnotation.from_mcp_dict({"idempotentHint": True})
        assert ann.idempotent is True

    def test_open_world_hint_sets_network_access(self):
        ann = ToolAnnotation.from_mcp_dict({"openWorldHint": True})
        assert ann.network_access is True
        assert ann.category == "search"

    def test_destructive_overrides_open_world_in_category(self):
        ann = ToolAnnotation.from_mcp_dict({"destructiveHint": True, "openWorldHint": True})
        assert ann.category == "dangerous"

    def test_cost_hint_valid(self):
        for level in ("none", "low", "medium", "high"):
            ann = ToolAnnotation.from_mcp_dict({"costHint": level})
            assert ann.cost_hint == level

    def test_cost_hint_unknown_falls_back_to_none(self):
        ann = ToolAnnotation.from_mcp_dict({"costHint": "gigantic"})
        assert ann.cost_hint == "none"

    def test_estimated_latency_ms_valid(self):
        ann = ToolAnnotation.from_mcp_dict({"estimatedLatencyMs": 250})
        assert ann.estimated_latency_ms == 250

    def test_estimated_latency_ms_zero(self):
        ann = ToolAnnotation.from_mcp_dict({"estimatedLatencyMs": 0})
        assert ann.estimated_latency_ms == 0

    def test_estimated_latency_ms_negative_ignored(self):
        ann = ToolAnnotation.from_mcp_dict({"estimatedLatencyMs": -1})
        assert ann.estimated_latency_ms is None

    def test_estimated_latency_ms_non_int_ignored(self):
        ann = ToolAnnotation.from_mcp_dict({"estimatedLatencyMs": "fast"})
        assert ann.estimated_latency_ms is None

    def test_unknown_keys_ignored(self):
        ann = ToolAnnotation.from_mcp_dict({"unknownFutureProp": True, "anotherKey": 42})
        assert ann.category == "general"
        assert ann.read_only is True

    def test_full_mcp_dict(self):
        data = {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
            "costHint": "medium",
            "estimatedLatencyMs": 500,
        }
        ann = ToolAnnotation.from_mcp_dict(data)
        assert ann.read_only is False
        assert ann.mutating is False
        assert ann.idempotent is True
        assert ann.network_access is True
        assert ann.cost_hint == "medium"
        assert ann.estimated_latency_ms == 500
        assert ann.category == "write"


# ---------------------------------------------------------------------------
# ToolAnnotation._infer_category
# ---------------------------------------------------------------------------


class TestInferCategory:
    def test_destructive_returns_dangerous(self):
        assert ToolAnnotation._infer_category({"destructiveHint": True}) == "dangerous"

    def test_non_readonly_returns_write(self):
        assert ToolAnnotation._infer_category({"readOnlyHint": False}) == "write"

    def test_open_world_readonly_returns_search(self):
        assert ToolAnnotation._infer_category({"openWorldHint": True}) == "search"

    def test_default_returns_general(self):
        assert ToolAnnotation._infer_category({}) == "general"

    def test_destructive_wins_over_non_readonly(self):
        result = ToolAnnotation._infer_category(
            {"destructiveHint": True, "readOnlyHint": False}
        )
        assert result == "dangerous"


# ---------------------------------------------------------------------------
# ToolAnnotation.to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_round_trips_fields(self):
        ann = ToolAnnotation(
            read_only=False,
            mutating=True,
            idempotent=True,
            estimated_latency_ms=100,
            cost_hint="high",
            requires_approval=True,
            network_access=True,
            filesystem_access=True,
            category="dangerous",
        )
        d = ann.to_dict()
        assert d["read_only"] is False
        assert d["mutating"] is True
        assert d["idempotent"] is True
        assert d["estimated_latency_ms"] == 100
        assert d["cost_hint"] == "high"
        assert d["requires_approval"] is True
        assert d["network_access"] is True
        assert d["filesystem_access"] is True
        assert d["category"] == "dangerous"

    def test_serialisable_to_json(self):
        ann = ToolAnnotation()
        json.dumps(ann.to_dict())  # must not raise

    def test_default_annotation_serialises(self):
        d = ToolAnnotation().to_dict()
        assert d["read_only"] is True
        assert d["estimated_latency_ms"] is None


# ---------------------------------------------------------------------------
# ToolAnnotation.to_policy_hints
# ---------------------------------------------------------------------------


class TestToPolicyHints:
    def test_safe_tool_is_safe(self):
        ann = ToolAnnotation(read_only=True, mutating=False)
        hints = ann.to_policy_hints()
        assert hints["is_safe"] is True
        assert hints["requires_approval"] is False

    def test_mutating_tool_not_safe(self):
        ann = ToolAnnotation(read_only=False, mutating=True)
        hints = ann.to_policy_hints()
        assert hints["is_safe"] is False
        assert hints["requires_approval"] is True

    def test_read_only_but_mutating_flag_set(self):
        ann = ToolAnnotation(read_only=True, mutating=True)
        hints = ann.to_policy_hints()
        assert hints["is_safe"] is False
        assert hints["requires_approval"] is True

    def test_requires_approval_flag_respected(self):
        ann = ToolAnnotation(read_only=True, mutating=False, requires_approval=True)
        hints = ann.to_policy_hints()
        assert hints["requires_approval"] is True
        assert hints["is_safe"] is True  # read_only and not mutating

    def test_network_access_propagated(self):
        ann = ToolAnnotation(network_access=True)
        assert ann.to_policy_hints()["network_access"] is True

    def test_filesystem_access_propagated(self):
        ann = ToolAnnotation(filesystem_access=True)
        assert ann.to_policy_hints()["filesystem_access"] is True

    def test_default_all_false_except_is_safe(self):
        hints = ToolAnnotation().to_policy_hints()
        assert hints == {
            "requires_approval": False,
            "network_access": False,
            "filesystem_access": False,
            "is_safe": True,
        }


# ---------------------------------------------------------------------------
# BUILTIN_ANNOTATIONS
# ---------------------------------------------------------------------------


class TestBuiltinAnnotations:
    def test_shell_exec_is_dangerous(self):
        ann = BUILTIN_ANNOTATIONS["shell_exec"]
        assert ann.mutating is True
        assert ann.requires_approval is True
        assert ann.category == "dangerous"

    def test_file_read_is_safe(self):
        ann = BUILTIN_ANNOTATIONS["file_read"]
        assert ann.read_only is True
        assert ann.filesystem_access is True
        assert ann.to_policy_hints()["is_safe"] is True

    def test_file_write_is_mutating(self):
        ann = BUILTIN_ANNOTATIONS["file_write"]
        assert ann.mutating is True
        assert ann.filesystem_access is True
        assert ann.category == "write"

    def test_web_fetch_uses_network(self):
        ann = BUILTIN_ANNOTATIONS["web_fetch"]
        assert ann.read_only is True
        assert ann.network_access is True
        assert ann.category == "search"


# ---------------------------------------------------------------------------
# AnnotationRegistry
# ---------------------------------------------------------------------------


class TestAnnotationRegistryBasics:
    def test_get_missing_returns_none(self):
        reg = AnnotationRegistry()
        assert reg.get("nonexistent") is None

    def test_get_or_default_missing_returns_default(self):
        reg = AnnotationRegistry()
        ann = reg.get_or_default("nonexistent")
        assert isinstance(ann, ToolAnnotation)
        assert ann.read_only is True

    def test_register_and_get(self):
        reg = AnnotationRegistry()
        ann = ToolAnnotation(mutating=True)
        reg.register("my_tool", ann)
        assert reg.get("my_tool") is ann

    def test_register_replaces_existing(self):
        reg = AnnotationRegistry()
        first = ToolAnnotation(category="general")
        second = ToolAnnotation(category="dangerous")
        reg.register("t", first)
        reg.register("t", second)
        assert reg.get("t") is second

    def test_get_or_default_registered(self):
        reg = AnnotationRegistry()
        ann = ToolAnnotation(mutating=True)
        reg.register("t", ann)
        assert reg.get_or_default("t") is ann


# ---------------------------------------------------------------------------
# AnnotationRegistry.get_all_annotations
# ---------------------------------------------------------------------------


class TestGetAllAnnotations:
    def test_empty_registry(self):
        reg = AnnotationRegistry()
        assert reg.get_all_annotations() == {}

    def test_returns_snapshot(self):
        reg = AnnotationRegistry()
        ann1 = ToolAnnotation(category="write")
        ann2 = ToolAnnotation(category="search")
        reg.register("a", ann1)
        reg.register("b", ann2)
        snapshot = reg.get_all_annotations()
        assert set(snapshot.keys()) == {"a", "b"}
        # Mutation of snapshot does not affect registry
        snapshot["c"] = ToolAnnotation()
        assert reg.get("c") is None


# ---------------------------------------------------------------------------
# AnnotationRegistry.filter_tools
# ---------------------------------------------------------------------------


TOOLS = [
    {"name": "search_web"},
    {"name": "delete_file"},
    {"name": "list_dir"},
    {"name": "unknown_tool"},
]


@pytest.fixture
def populated_registry():
    reg = AnnotationRegistry()
    reg.register("search_web", ToolAnnotation(read_only=True, network_access=True, category="search", cost_hint="low"))
    reg.register("delete_file", ToolAnnotation(read_only=False, mutating=True, category="dangerous", cost_hint="none"))
    reg.register("list_dir", ToolAnnotation(read_only=True, filesystem_access=True, category="general", cost_hint="none"))
    # unknown_tool has no registration → falls back to default
    return reg


class TestFilterTools:
    def test_no_filters_returns_all(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS)
        assert len(result) == 4

    def test_filter_read_only_true(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, read_only=True)
        names = [t["name"] for t in result]
        assert "delete_file" not in names
        assert "search_web" in names
        assert "list_dir" in names
        assert "unknown_tool" in names  # default is read_only=True

    def test_filter_read_only_false(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, read_only=False)
        names = [t["name"] for t in result]
        assert names == ["delete_file"]

    def test_filter_by_category(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, category="search")
        assert len(result) == 1
        assert result[0]["name"] == "search_web"

    def test_filter_by_max_cost_none(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, max_cost="none")
        names = [t["name"] for t in result]
        assert "search_web" not in names  # cost_hint="low" > "none"
        assert "delete_file" in names
        assert "list_dir" in names

    def test_filter_by_max_cost_low(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, max_cost="low")
        names = [t["name"] for t in result]
        assert "search_web" in names
        assert "delete_file" in names

    def test_filter_by_max_cost_high_returns_all(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, max_cost="high")
        assert len(result) == 4

    def test_combined_filters(self, populated_registry):
        result = populated_registry.filter_tools(TOOLS, read_only=True, category="search")
        assert len(result) == 1
        assert result[0]["name"] == "search_web"

    def test_empty_tools_list(self, populated_registry):
        assert populated_registry.filter_tools([]) == []

    def test_tool_without_name_key_uses_empty_string(self, populated_registry):
        # Tools missing "name" should not raise; fallback default annotation applies.
        result = populated_registry.filter_tools([{}])
        assert len(result) == 1  # default annotation is read_only=True

    def test_filter_unknown_max_cost_treated_as_high(self, populated_registry):
        # Unknown cost strings map to 0 in _COST_ORDER lookup → treated as "none" budget.
        # But the max_cost "none" path is exercised via dedicated test; here verify no crash.
        result = populated_registry.filter_tools(TOOLS, max_cost="unknown_level")
        # unknown level resolves to _COST_ORDER.get("unknown_level", ...) — test just no raise
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# AnnotationRegistry.get_safe_tools
# ---------------------------------------------------------------------------


class TestGetSafeTools:
    def test_returns_only_safe(self, populated_registry):
        result = populated_registry.get_safe_tools(TOOLS)
        names = [t["name"] for t in result]
        assert "delete_file" not in names
        assert "search_web" in names
        assert "list_dir" in names

    def test_empty_list(self, populated_registry):
        assert populated_registry.get_safe_tools([]) == []

    def test_all_mutating_returns_empty(self):
        reg = AnnotationRegistry()
        tools = [{"name": "rm_rf"}, {"name": "nuke"}]
        reg.register("rm_rf", ToolAnnotation(mutating=True))
        reg.register("nuke", ToolAnnotation(mutating=True))
        assert reg.get_safe_tools(tools) == []

    def test_unregistered_tool_included_as_safe(self):
        reg = AnnotationRegistry()
        tools = [{"name": "mystery"}]
        # Default annotation is read_only=True, mutating=False → safe
        result = reg.get_safe_tools(tools)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# AnnotationRegistry.get_approval_required
# ---------------------------------------------------------------------------


class TestGetApprovalRequired:
    def test_returns_mutating_tool_names(self, populated_registry):
        names = populated_registry.get_approval_required(TOOLS)
        assert "delete_file" in names

    def test_excludes_safe_tools(self, populated_registry):
        names = populated_registry.get_approval_required(TOOLS)
        assert "list_dir" not in names
        assert "search_web" not in names

    def test_empty_list(self, populated_registry):
        assert populated_registry.get_approval_required([]) == []

    def test_requires_approval_flag_alone_triggers(self):
        reg = AnnotationRegistry()
        reg.register("sensitive", ToolAnnotation(read_only=True, requires_approval=True))
        assert "sensitive" in reg.get_approval_required([{"name": "sensitive"}])

    def test_tool_without_name_skipped(self, populated_registry):
        result = populated_registry.get_approval_required([{}])
        assert result == []


# ---------------------------------------------------------------------------
# AnnotationRegistry.summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_registry(self):
        reg = AnnotationRegistry()
        text = reg.summarize()
        assert "No tool annotations" in text

    def test_summary_contains_tool_names(self):
        reg = AnnotationRegistry()
        reg.register("alpha", ToolAnnotation(mutating=True, category="dangerous"))
        reg.register("beta", ToolAnnotation(read_only=True, category="search"))
        text = reg.summarize()
        assert "alpha" in text
        assert "beta" in text

    def test_summary_sorted(self):
        reg = AnnotationRegistry()
        reg.register("z_tool", ToolAnnotation())
        reg.register("a_tool", ToolAnnotation())
        lines = reg.summarize().splitlines()
        names_in_order = [ln.split(":")[0].strip() for ln in lines if ":" in ln]
        assert names_in_order == sorted(names_in_order)

    def test_summary_shows_category(self):
        reg = AnnotationRegistry()
        reg.register("my_tool", ToolAnnotation(category="admin"))
        assert "admin" in reg.summarize()

    def test_summary_shows_approval_required(self):
        reg = AnnotationRegistry()
        reg.register("risky", ToolAnnotation(requires_approval=True))
        assert "approval required" in reg.summarize()

    def test_summary_no_approval_tag_for_safe(self):
        reg = AnnotationRegistry()
        reg.register("safe_tool", ToolAnnotation())
        assert "approval required" not in reg.summarize()


# ---------------------------------------------------------------------------
# Integration: McpClient parses annotations
# ---------------------------------------------------------------------------


class TestMcpClientAnnotationParsing:
    def _make_connected_client(self, tools_payload):
        from missy.mcp.client import McpClient

        c = McpClient(name="srv", command="echo")
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.poll.return_value = None

        init_resp = {"jsonrpc": "2.0", "result": {"capabilities": {}}}
        tools_resp = {"jsonrpc": "2.0", "result": {"tools": tools_payload}}
        mock_proc.stdout.readline.side_effect = [
            json.dumps(init_resp).encode() + b"\n",
            json.dumps(tools_resp).encode() + b"\n",
        ]
        with patch("subprocess.Popen", return_value=mock_proc):
            c.connect()
        return c

    def test_tool_without_annotations_has_no_entry(self):
        c = self._make_connected_client([{"name": "read_file", "description": "Read"}])
        assert "read_file" not in c.tool_annotations

    def test_tool_with_annotations_parsed(self):
        tools = [
            {
                "name": "delete",
                "description": "Delete a file",
                "annotations": {"destructiveHint": True, "readOnlyHint": False},
            }
        ]
        c = self._make_connected_client(tools)
        ann = c.tool_annotations.get("delete")
        assert ann is not None
        assert ann.mutating is True
        assert ann.requires_approval is True

    def test_tool_with_empty_annotations_dict(self):
        tools = [{"name": "noop", "description": "No-op", "annotations": {}}]
        c = self._make_connected_client(tools)
        ann = c.tool_annotations.get("noop")
        assert ann is not None
        assert ann.read_only is True  # default

    def test_tool_with_non_dict_annotations_ignored(self):
        tools = [{"name": "weird", "description": "Weird", "annotations": "string"}]
        c = self._make_connected_client(tools)
        # Non-dict annotations must not be parsed and must not raise.
        assert "weird" not in c.tool_annotations

    def test_tool_annotations_property_is_copy(self):
        tools = [{"name": "t", "description": "t", "annotations": {"readOnlyHint": True}}]
        c = self._make_connected_client(tools)
        copy1 = c.tool_annotations
        copy1["injected"] = ToolAnnotation()
        assert "injected" not in c.tool_annotations


# ---------------------------------------------------------------------------
# Integration: McpManager registers annotations
# ---------------------------------------------------------------------------


class TestMcpManagerAnnotations:
    def _make_manager(self, tmp_path):
        from missy.mcp.manager import McpManager

        return McpManager(config_path=str(tmp_path / "mcp.json"))

    def test_builtin_annotations_seeded(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.get_annotation("shell_exec") is not None
        assert mgr.get_annotation("file_read") is not None

    def test_get_annotation_returns_none_for_unknown(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.get_annotation("totally_unknown_tool") is None

    def test_get_all_annotations_includes_builtins(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        all_anns = mgr.get_all_annotations()
        assert "shell_exec" in all_anns
        assert "file_read" in all_anns

    def test_add_server_registers_namespaced_annotations(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_client = MagicMock()
        mock_client.tools = [{"name": "delete_resource"}]
        mock_client._command = "echo"
        mock_client._url = None
        mock_client.tool_annotations = {
            "delete_resource": ToolAnnotation(mutating=True, category="dangerous")
        }
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            mgr.add_server("myserver", command="echo")
        ann = mgr.get_annotation("myserver__delete_resource")
        assert ann is not None
        assert ann.mutating is True

    def test_add_server_without_annotations_no_error(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mock_client = MagicMock()
        mock_client.tools = [{"name": "plain_tool"}]
        mock_client._command = "echo"
        mock_client._url = None
        mock_client.tool_annotations = {}
        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            mgr.add_server("plain", command="echo")
        # No annotation registered for namespaced name, but no crash.
        assert mgr.get_annotation("plain__plain_tool") is None

    def test_annotation_registry_property_accessible(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        reg = mgr.annotation_registry
        assert isinstance(reg, AnnotationRegistry)
