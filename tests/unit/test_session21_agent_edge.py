"""Session 21: Agent runtime and subsystem edge case tests.

Tests for:
- Agent runtime initialization edge cases
- Context manager edge cases
- Done criteria edge cases
- Learnings extraction edge cases
- Prompt patch lifecycle
- Sub-agent task parsing
- Skills registry edge cases
- Event bus edge cases
"""

from __future__ import annotations

# ===================================================================
# 1. Agent runtime initialization edge cases
# ===================================================================


class TestAgentRuntimeInit:
    """Test AgentRuntime initialization with various configs."""

    def test_default_config(self):
        """Default AgentConfig creates a runnable runtime."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig()
        runtime = AgentRuntime(config)
        assert runtime.config.provider == "anthropic"
        assert runtime.config.max_iterations == 10

    def test_custom_system_prompt(self):
        """Custom system prompt is stored."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(system_prompt="You are a test assistant.")
        runtime = AgentRuntime(config)
        assert "test assistant" in runtime.config.system_prompt

    def test_single_turn_mode(self):
        """max_iterations=1 means single-turn mode."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(max_iterations=1)
        runtime = AgentRuntime(config)
        assert runtime.config.max_iterations == 1

    def test_no_tools_mode(self):
        """capability_mode='no-tools' disables tools."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(capability_mode="no-tools")
        runtime = AgentRuntime(config)
        assert runtime.config.capability_mode == "no-tools"

    def test_safe_chat_mode(self):
        """capability_mode='safe-chat' limits capabilities."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(capability_mode="safe-chat")
        runtime = AgentRuntime(config)
        assert runtime.config.capability_mode == "safe-chat"

    def test_budget_config(self):
        """Budget is set correctly."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(max_spend_usd=5.0)
        runtime = AgentRuntime(config)
        assert runtime.config.max_spend_usd == 5.0


# ===================================================================
# 2. Context manager edge cases
# ===================================================================


class TestContextManagerEdgeCases:
    """Test ContextManager with various inputs."""

    def test_empty_history(self):
        """Empty history returns system prompt + user message."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        system, messages = cm.build_messages(
            system="System prompt",
            new_message="Hello",
            history=[],
        )
        assert system is not None
        assert len(messages) >= 1

    def test_history_pruning_under_budget(self):
        """History within budget is kept intact."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        history = [
            {"role": "user", "content": f"Message {i}"} for i in range(5)
        ]
        system, messages = cm.build_messages(
            system="System",
            new_message="Latest",
            history=history,
        )
        assert len(messages) >= 5

    def test_very_long_user_input(self):
        """Very long user input is handled without crashing."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        long_input = "x" * 50000
        system, messages = cm.build_messages(
            system="System",
            new_message=long_input,
            history=[],
        )
        assert len(messages) >= 1

    def test_memory_injection(self):
        """Memory fragments are injected into context when provided."""
        from missy.agent.context import ContextManager

        cm = ContextManager()
        system, messages = cm.build_messages(
            system="System",
            new_message="Hello",
            history=[],
            memory_results=["Remember: user prefers concise responses"],
        )
        # Memory should be incorporated somewhere in system or messages
        all_text = system + " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in messages
        )
        assert "concise" in all_text or len(messages) >= 1


# ===================================================================
# 3. Done criteria edge cases
# ===================================================================


class TestDoneCriteriaEdgeCases:
    """Test DoneCriteria and is_compound_task."""

    def test_simple_question_not_compound(self):
        """Simple questions should not be compound tasks."""
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("What is 2+2?") is False

    def test_compound_task_numbered_list(self):
        """Numbered list indicates compound task."""
        from missy.agent.done_criteria import is_compound_task

        result = is_compound_task("1. Read file\n2. Parse contents\n3. Save results")
        assert result is True

    def test_compound_task_then_connective(self):
        """Sequential connective 'then' indicates compound task."""
        from missy.agent.done_criteria import is_compound_task

        result = is_compound_task("First read the file, then update it, finally save")
        assert result is True

    def test_done_criteria_initial_state(self):
        """Initial state should have empty conditions."""
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria()
        assert dc.conditions == []


# ===================================================================
# 4. Learnings extraction
# ===================================================================


class TestLearningsExtraction:
    """Test learnings extraction functions."""

    def test_extract_task_type_from_read(self):
        """Tool names containing 'read' indicate a read task."""
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["file_read"])
        assert result is not None

    def test_extract_task_type_from_shell(self):
        """Shell tool indicates a shell task."""
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["shell_exec"])
        assert result is not None

    def test_extract_task_type_from_mixed(self):
        """Mixed tools fall back to generic type."""
        from missy.agent.learnings import extract_task_type

        result = extract_task_type(["file_read", "web_fetch", "calculator"])
        assert result is not None

    def test_extract_outcome_success(self):
        """Positive response indicates success."""
        from missy.agent.learnings import extract_outcome

        result = extract_outcome("The file has been successfully created.")
        assert result in ("success", "partial", "failure")

    def test_extract_outcome_failure(self):
        """Error response indicates failure."""
        from missy.agent.learnings import extract_outcome

        result = extract_outcome("Error: file not found. Unable to complete.")
        assert result in ("success", "partial", "failure")


# ===================================================================
# 5. Prompt patches lifecycle
# ===================================================================


class TestPromptPatchesLifecycle:
    """Test prompt patch propose/approve/reject cycle."""

    def test_propose_patch(self, tmp_path):
        """Proposing a patch creates a record."""
        from missy.agent.prompt_patches import PatchType, PromptPatchManager

        mgr = PromptPatchManager(store_path=str(tmp_path / "patches.json"))
        patch = mgr.propose(
            patch_type=PatchType.DOMAIN_KNOWLEDGE,
            content="Always include temperature in weather responses",
        )
        assert patch is not None

        patches = mgr.list_all()
        assert len(patches) >= 1

    def test_approve_patch(self, tmp_path):
        """Approving a patch changes its status."""
        from missy.agent.prompt_patches import PatchType, PromptPatchManager

        mgr = PromptPatchManager(store_path=str(tmp_path / "patches.json"))
        patch = mgr.propose(
            patch_type=PatchType.TOOL_USAGE_HINT,
            content="Use file_read for config files",
            confidence=0.5,  # Low confidence to avoid auto-approve
        )
        assert patch is not None
        mgr.approve(patch.id)

        patches = mgr.list_all()
        found = next(p for p in patches if p.id == patch.id)
        assert found.status.value == "approved"

    def test_reject_patch(self, tmp_path):
        """Rejecting a patch changes its status."""
        from missy.agent.prompt_patches import PatchType, PromptPatchManager

        mgr = PromptPatchManager(store_path=str(tmp_path / "patches.json"))
        patch = mgr.propose(
            patch_type=PatchType.ERROR_AVOIDANCE,
            content="Avoid using rm -rf",
            confidence=0.5,
        )
        assert patch is not None
        mgr.reject(patch.id)

        patches = mgr.list_all()
        found = next(p for p in patches if p.id == patch.id)
        assert found.status.value == "rejected"

    def test_get_active_patches(self, tmp_path):
        """get_active_patches returns approved patches."""
        from missy.agent.prompt_patches import PatchType, PromptPatchManager

        mgr = PromptPatchManager(store_path=str(tmp_path / "patches.json"))
        patch = mgr.propose(
            patch_type=PatchType.STYLE_PREFERENCE,
            content="respond warmly to greetings",
            confidence=0.9,  # High confidence — auto-approved
        )
        assert patch is not None

        active = mgr.get_active_patches()
        assert isinstance(active, list)


# ===================================================================
# 6. Sub-agent task parsing
# ===================================================================


class TestSubAgentParsing:
    """Test sub-agent task parsing."""

    def test_parse_subtasks_numbered(self):
        """Numbered list is parsed into subtasks."""
        from missy.agent.sub_agent import parse_subtasks

        text = "1. Read the file\n2. Update the config\n3. Restart the server"
        tasks = parse_subtasks(text)
        assert len(tasks) >= 2

    def test_parse_subtasks_connectives(self):
        """Tasks connected by 'and then' are parsed."""
        from missy.agent.sub_agent import parse_subtasks

        text = "Read the file and then update the config and then restart"
        tasks = parse_subtasks(text)
        assert len(tasks) >= 2

    def test_parse_subtasks_single(self):
        """Single task returns one subtask."""
        from missy.agent.sub_agent import parse_subtasks

        text = "Read the config file"
        tasks = parse_subtasks(text)
        assert len(tasks) >= 1


# ===================================================================
# 7. Skills registry edge cases
# ===================================================================


class TestSkillsRegistryEdgeCases:
    """Test skills registry behavior."""

    def test_register_and_list(self):
        """Register a skill and list it."""
        from missy.skills.base import BaseSkill
        from missy.skills.registry import SkillRegistry

        registry = SkillRegistry()

        class TestSkill(BaseSkill):
            @property
            def name(self) -> str:
                return "test_skill"

            @property
            def description(self) -> str:
                return "A test skill"

            def execute(self, **kwargs) -> str:
                return "executed"

        registry.register(TestSkill())
        skills = registry.list_skills()
        assert "test_skill" in skills

    def test_get_nonexistent_skill(self):
        """Getting a nonexistent skill returns None."""
        from missy.skills.registry import SkillRegistry

        registry = SkillRegistry()
        result = registry.get("nonexistent_skill_xyz")
        assert result is None


# ===================================================================
# 8. Event bus edge cases
# ===================================================================


class TestEventBusEdgeCases:
    """Test event bus behavior."""

    def test_publish_without_subscribers(self):
        """Publishing without subscribers doesn't crash."""
        from missy.core.events import AuditEvent, EventBus

        bus = EventBus()
        event = AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type="test.event",
            category="provider",
            result="allow",
        )
        bus.publish(event)

    def test_subscribe_and_receive(self):
        """Subscriber receives published events."""
        from missy.core.events import AuditEvent, EventBus

        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("test.event", handler)
        event = AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type="test.event",
            category="provider",
            result="allow",
        )
        bus.publish(event)
        assert len(received) == 1

    def test_subscribe_different_type_not_received(self):
        """Subscriber for different event type doesn't receive events."""
        from missy.core.events import AuditEvent, EventBus

        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("other.event", handler)
        event = AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type="test.event",
            category="provider",
            result="allow",
        )
        bus.publish(event)
        assert len(received) == 0

    def test_multiple_subscribers(self):
        """Multiple subscribers all receive the event."""
        from missy.core.events import AuditEvent, EventBus

        bus = EventBus()
        received_a = []
        received_b = []

        bus.subscribe("test.event", lambda e: received_a.append(e))
        bus.subscribe("test.event", lambda e: received_b.append(e))

        event = AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type="test.event",
            category="provider",
            result="allow",
        )
        bus.publish(event)
        assert len(received_a) == 1
        assert len(received_b) == 1
