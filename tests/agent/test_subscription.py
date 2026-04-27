"""Tests for the agent subscription state machine."""

from __future__ import annotations

from missy.agent.subscription import AgentSubscription, BlockState, strip_block_tags


def test_delta_updates_build_monotonic_buffer_and_finalize_text():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})

    first = sub.handle_event({"type": "message_update", "delta": "Hel"})
    second = sub.handle_event({"type": "message_update", "delta": "lo"})
    end = sub.handle_event({"type": "message_end"})

    assert first.visible_delta == "Hel"
    assert second.visible_delta == "lo"
    assert end.full_visible_text == "Hello"
    assert sub.delta_buffer == "Hello"
    assert sub.assistant_texts == ["Hello"]


def test_full_content_resends_only_append_new_tail():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})

    assert sub.handle_event({"type": "message_update", "content": "Hel"}).raw_delta == "Hel"
    assert sub.handle_event({"type": "message_update", "content": "Hello"}).raw_delta == "lo"
    assert sub.handle_event({"type": "message_update", "content": "Hello"}).raw_delta == ""
    assert sub.handle_event({"type": "message_update", "content": "Hello!"}).raw_delta == "!"

    assert sub.delta_buffer == "Hello!"
    assert sub.visible_text == "Hello!"


def test_divergent_full_content_is_fail_open_append_only():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})

    sub.handle_event({"type": "message_update", "content": "abc"})
    update = sub.handle_event({"type": "message_update", "content": "xyz"})

    assert update.raw_delta == "xyz"
    assert sub.delta_buffer == "abcxyz"


def test_thinking_tags_are_stripped_across_chunks():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})

    assert sub.handle_event({"type": "message_update", "delta": "Hi <thi"}).visible_delta == "Hi "
    assert (
        sub.handle_event({"type": "message_update", "delta": "nk>secret</thi"}).visible_delta == ""
    )
    assert (
        sub.handle_event({"type": "message_update", "delta": "nk> there"}).visible_delta == " there"
    )

    sub.handle_event({"type": "message_end"})
    assert sub.visible_text == "Hi  there"


def test_tags_inside_inline_code_are_left_literal():
    state = BlockState()

    visible, reasoning = strip_block_tags("Use `<think>` literally", state)

    assert visible == "Use `<think>` literally"
    assert reasoning == ""
    assert state.thinking is False


def test_enforced_final_tag_only_keeps_final_content():
    sub = AgentSubscription(enforce_final_tag=True)
    sub.handle_event({"type": "message_start"})

    update = sub.handle_event(
        {"type": "message_update", "delta": "draft <final>answer</final> trailing"}
    )

    assert update.visible_delta == "answer"
    assert sub.visible_text == "answer"


def test_stream_reasoning_emits_hidden_thinking():
    reasoning: list[str] = []
    sub = AgentSubscription(reasoning_mode="stream", on_reasoning_delta=reasoning.append)
    sub.handle_event({"type": "message_start"})

    sub.handle_event({"type": "message_update", "delta": "<think>hidden</think>Shown"})

    assert reasoning == ["hidden"]
    assert sub.visible_text == "Shown"


def test_reply_directives_are_parsed_and_removed_from_visible_text():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})

    update = sub.handle_event(
        {
            "type": "message_update",
            "delta": "See this [[reply_to_current]][[media:/tmp/a.png]]now",
        }
    )

    assert update.full_visible_text == "See this now"
    assert update.directives.reply_to_current is True
    assert update.directives.media_urls == ["/tmp/a.png"]


def test_block_reply_flushes_before_tool_start():
    flushed: list[str] = []
    sub = AgentSubscription(block_reply_break="message_end", on_block_reply=flushed.append)
    sub.handle_event({"type": "message_start"})
    sub.handle_event({"type": "message_update", "delta": "Before tool."})

    update = sub.handle_event(
        {"type": "tool_execution_start", "tool_name": "shell_exec", "tool_call_id": "call-1"}
    )

    assert update.flushed_blocks == ["Before tool."]
    assert flushed == ["Before tool."]
    assert not sub.has_buffered()


def test_text_end_boundary_flushes_when_configured():
    sub = AgentSubscription(block_reply_break="text_end")
    sub.handle_event({"type": "message_start"})

    update = sub.handle_event(
        {
            "type": "message_update",
            "delta": "Block one.",
            "stream_event": "text_end",
        }
    )

    assert update.flushed_blocks == ["Block one."]
    assert not sub.has_buffered()


def test_compaction_retry_state_tracks_pending_retry():
    sub = AgentSubscription()
    sub.handle_event({"type": "message_start"})
    sub.handle_event({"type": "message_update", "delta": "stale"})

    sub.handle_event({"type": "auto_compaction_start"})
    assert sub.is_compacting()

    sub.handle_event({"type": "auto_compaction_end", "willRetry": True})
    assert sub.pending_compaction_retry == 1
    assert not sub.compaction_in_flight
    assert sub.delta_buffer == ""

    sub.handle_event({"type": "auto_compaction_end", "willRetry": False})
    assert sub.pending_compaction_retry == 0
    assert not sub.is_compacting()
