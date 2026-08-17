"""Tests for the Web TUI's per-agent activity projection."""

from __future__ import annotations

from missy.api.agent_activity import query_agent_activity
from missy.core.events import AuditEvent, EventBus


def _event(event_type: str, task_id: str, detail: dict, result: str = "allow") -> AuditEvent:
    return AuditEvent.now(
        session_id="session-1",
        task_id=task_id,
        event_type=event_type,
        category="provider" if event_type.startswith("agent.") else "tool",
        result=result,  # type: ignore[arg-type]
        detail=detail,
    )


def test_reconstructs_parent_and_child_agent_lifecycle() -> None:
    bus = EventBus()
    bus.publish(
        _event(
            "agent.run.start",
            "root-task",
            {"agent_id": "main", "agent_name": "Missy", "role": "primary", "goal": "plan"},
        )
    )
    bus.publish(
        _event(
            "agent.run.start",
            "child-task",
            {
                "agent_id": "subagent-abc",
                "agent_name": "Researcher",
                "role": "subagent",
                "delegation_depth": 1,
                "parent_agent_id": "main",
                "parent_task_id": "root-task",
                "goal": "research options",
            },
        )
    )
    bus.publish(
        _event(
            "tool_execute",
            "child-task",
            {"tool": "web_fetch", "message": "Tool executed successfully."},
        )
    )
    bus.publish(
        _event(
            "agent.run.complete",
            "child-task",
            {
                "agent_id": "subagent-abc",
                "agent_name": "Researcher",
                "role": "subagent",
                "delegation_depth": 1,
                "parent_agent_id": "main",
                "parent_task_id": "root-task",
                "provider": "mock",
                "tools_used": ["web_fetch"],
            },
        )
    )

    result = query_agent_activity({}, bus=bus)

    assert result["count"] == 2
    assert result["active_count"] == 1
    child = next(agent for agent in result["agents"] if agent["agent_id"] == "subagent-abc")
    assert child["parent_agent_id"] == "main"
    assert child["depth"] == 1
    assert child["status"] == "complete"
    assert child["tools"] == ["web_fetch"]
    assert any(action["label"].startswith("Tool web_fetch") for action in child["actions"])


def test_filters_status_and_redacts_agent_goal() -> None:
    bus = EventBus()
    bus.publish(
        _event(
            "agent.run.start",
            "task-1",
            {
                "agent_id": "main",
                "agent_name": "Missy",
                "goal": "use sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            },
        )
    )

    result = query_agent_activity({"status": "running"}, bus=bus)

    assert result["active_count"] == 1
    assert "sk-ant-api03" not in result["agents"][0]["goal"]
