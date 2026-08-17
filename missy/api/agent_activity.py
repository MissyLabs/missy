"""Build redacted per-agent activity snapshots for the operator console."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from missy.api.audit_browser import redact_audit_value
from missy.core.events import EventBus, event_bus


def query_agent_activity(params: dict[str, Any], *, bus: EventBus = event_bus) -> dict[str, Any]:
    """Return recent primary/sub-agent state reconstructed from audit events.

    The audit bus sees runs from every channel, unlike the Web run registry,
    so this endpoint also exposes CLI, scheduler, Discord, and voice activity.
    """
    try:
        limit = max(1, min(int(params.get("limit", 100)), 500))
    except (TypeError, ValueError):
        limit = 100
    wanted_session = str(params.get("session_id") or "")
    wanted_status = str(params.get("status") or "")

    raw_events = bus.get_events()
    task_agents: dict[tuple[str, str], str] = {}
    agents: dict[tuple[str, str], dict[str, Any]] = {}

    for event in raw_events:
        if not event.event_type.startswith("agent.run."):
            continue
        detail = event.detail if isinstance(event.detail, dict) else {}
        agent_id = str(detail.get("agent_id") or "default")
        task_agents[(event.session_id, event.task_id)] = agent_id

    for event in raw_events:
        detail = event.detail if isinstance(event.detail, dict) else {}
        agent_id = str(
            detail.get("agent_id") or task_agents.get((event.session_id, event.task_id)) or ""
        )
        if not agent_id:
            continue
        key = (event.session_id, agent_id)
        timestamp = event.timestamp.isoformat()
        agent = agents.setdefault(
            key,
            {
                "agent_id": agent_id,
                "name": str(detail.get("agent_name") or agent_id),
                "role": str(detail.get("role") or "agent"),
                "depth": int(detail.get("delegation_depth") or 0),
                "parent_agent_id": str(detail.get("parent_agent_id") or ""),
                "parent_task_id": str(detail.get("parent_task_id") or ""),
                "session_id": event.session_id,
                "task_id": event.task_id,
                "goal": "",
                "status": "unknown",
                "current_action": "",
                "started_at": None,
                "updated_at": timestamp,
                "finished_at": None,
                "provider": "",
                "tools": [],
                "actions": [],
            },
        )
        agent["updated_at"] = timestamp
        agent["task_id"] = event.task_id or agent["task_id"]

        action = event.event_type
        if event.event_type == "agent.run.start":
            agent["status"] = "running"
            agent["current_action"] = "Thinking"
            agent["goal"] = str(detail.get("goal") or "")
            agent["started_at"] = timestamp
            action = "Started"
        elif event.event_type == "agent.run.complete":
            agent["status"] = "complete"
            agent["current_action"] = "Complete"
            agent["finished_at"] = timestamp
            agent["provider"] = str(detail.get("provider") or "")
            for tool in detail.get("tools_used") or []:
                if str(tool) not in agent["tools"]:
                    agent["tools"].append(str(tool))
            action = "Completed"
        elif event.event_type == "agent.run.error":
            agent["status"] = "error"
            agent["current_action"] = str(detail.get("stage") or "Failed")
            agent["finished_at"] = timestamp
            action = f"Error: {detail.get('stage') or 'run'}"
        elif event.event_type == "agent.tool.start":
            tool = str(detail.get("tool") or "tool")
            if tool not in agent["tools"]:
                agent["tools"].append(tool)
            agent["current_action"] = f"Running {tool}"
            action = f"Started tool {tool}"
        elif event.event_type == "agent.tool.complete":
            tool = str(detail.get("tool") or "tool")
            if tool not in agent["tools"]:
                agent["tools"].append(tool)
            duration = detail.get("duration_ms")
            agent["current_action"] = (
                f"{tool} failed" if detail.get("is_error") else f"Finished {tool}; thinking"
            )
            action = f"Finished tool {tool}" + (f" in {duration}ms" if duration is not None else "")
        elif event.event_type == "tool_execute":
            tool = str(detail.get("tool") or "tool")
            if tool not in agent["tools"]:
                agent["tools"].append(tool)
            agent["current_action"] = f"{tool}: {event.result}"
            action = f"Tool {tool}: {event.result}"
        else:
            agent["current_action"] = event.event_type

        agent["actions"].append(
            {
                "timestamp": timestamp,
                "event_type": event.event_type,
                "result": event.result,
                "label": action,
                "detail": redact_audit_value(detail),
            }
        )
        agent["actions"] = agent["actions"][-20:]

    snapshots = [redact_audit_value(agent) for agent in agents.values()]
    if wanted_session:
        snapshots = [agent for agent in snapshots if agent["session_id"] == wanted_session]
    if wanted_status:
        snapshots = [agent for agent in snapshots if agent["status"] == wanted_status]
    snapshots.sort(
        key=lambda agent: (
            agent["status"] == "running",
            str(agent.get("updated_at") or ""),
        ),
        reverse=True,
    )
    snapshots = snapshots[:limit]
    return {
        "agents": snapshots,
        "count": len(snapshots),
        "active_count": sum(agent["status"] == "running" for agent in snapshots),
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "in-memory audit bus",
        "event_count": len(raw_events),
    }


__all__ = ["query_agent_activity"]
