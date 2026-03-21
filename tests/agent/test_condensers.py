"""Tests for missy.agent.condensers — Memory Condenser Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.agent.condensers import (
    AmortizedForgettingCondenser,
    BaseCondenser,
    CondenserResult,
    LLMAttentionCondenser,
    ObservationMaskingCondenser,
    PipelineCondenser,
    SummarizingCondenser,
    WindowCondenser,
    create_default_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msgs(*contents: str, role: str = "user") -> list[dict]:
    """Build a simple message list with the given contents."""
    return [{"role": role, "content": c} for c in contents]


def _tool_msg(content: str, name: str = "shell_exec") -> dict:
    return {"role": "tool", "name": name, "content": content}


def _make_mock_provider(response_text: str = "- fact A\n- fact B") -> MagicMock:
    """Return a mock BaseProvider whose complete() returns response_text."""
    provider = MagicMock()
    response = MagicMock()
    response.content = response_text
    provider.complete.return_value = response
    return provider


# ---------------------------------------------------------------------------
# CondenserResult
# ---------------------------------------------------------------------------


class TestCondenserResult:
    def test_default_metadata_is_empty_dict(self):
        result = CondenserResult(messages=[])
        assert result.metadata == {}

    def test_stores_messages_and_metadata(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = CondenserResult(messages=msgs, metadata={"dropped": 2})
        assert result.messages is msgs
        assert result.metadata["dropped"] == 2


# ---------------------------------------------------------------------------
# WindowCondenser
# ---------------------------------------------------------------------------


class TestWindowCondenserBasic:
    def test_keeps_last_n_messages(self):
        messages = _msgs("a", "b", "c", "d", "e")
        result = WindowCondenser(max_messages=3).condense(messages)
        assert [m["content"] for m in result.messages] == ["c", "d", "e"]

    def test_no_drop_when_under_limit(self):
        messages = _msgs("a", "b")
        result = WindowCondenser(max_messages=5).condense(messages)
        assert len(result.messages) == 2
        assert result.metadata["dropped"] == 0

    def test_exact_limit_no_drop(self):
        messages = _msgs(*[str(i) for i in range(10)])
        result = WindowCondenser(max_messages=10).condense(messages)
        assert result.metadata["dropped"] == 0

    def test_metadata_dropped_count(self):
        messages = _msgs(*[str(i) for i in range(8)])
        result = WindowCondenser(max_messages=5).condense(messages)
        assert result.metadata["dropped"] == 3
        assert result.metadata["kept"] == 5

    def test_empty_messages(self):
        result = WindowCondenser(max_messages=5).condense([])
        assert result.messages == []
        assert result.metadata["dropped"] == 0

    def test_single_message_under_limit(self):
        result = WindowCondenser(max_messages=5).condense([{"role": "user", "content": "hi"}])
        assert len(result.messages) == 1

    def test_returns_copies_not_same_objects(self):
        messages = [{"role": "user", "content": "x"}]
        result = WindowCondenser(max_messages=10).condense(messages)
        assert result.messages is not messages


# ---------------------------------------------------------------------------
# ObservationMaskingCondenser
# ---------------------------------------------------------------------------


class TestObservationMaskingCondenser:
    def test_masks_large_tool_output(self):
        big = "x" * 5000
        messages = [_tool_msg(big)]
        result = ObservationMaskingCondenser(max_output_chars=2000).condense(messages)
        content = result.messages[0]["content"]
        assert content.startswith("[Tool output masked:")
        assert "5000 chars" in content
        assert result.metadata["masked"] == 1

    def test_small_tool_output_not_masked(self):
        messages = [_tool_msg("short output")]
        result = ObservationMaskingCondenser(max_output_chars=2000).condense(messages)
        assert result.messages[0]["content"] == "short output"
        assert result.metadata["masked"] == 0

    def test_preview_included_in_mask(self):
        big = "ABCDE" * 1000
        messages = [_tool_msg(big)]
        result = ObservationMaskingCondenser(max_output_chars=100).condense(messages)
        assert "ABCDE" in result.messages[0]["content"]

    def test_non_tool_messages_pass_through_unchanged(self):
        big = "x" * 5000
        messages = [{"role": "assistant", "content": big}]
        result = ObservationMaskingCondenser(max_output_chars=2000).condense(messages)
        assert result.messages[0]["content"] == big
        assert result.metadata["masked"] == 0

    def test_preserves_other_message_keys(self):
        messages = [{"role": "tool", "name": "my_tool", "content": "x" * 3000}]
        result = ObservationMaskingCondenser(max_output_chars=100).condense(messages)
        assert result.messages[0]["name"] == "my_tool"
        assert result.messages[0]["role"] == "tool"

    def test_multiple_messages_mixed(self):
        messages = [
            _tool_msg("x" * 3000, name="big_tool"),
            _tool_msg("small", name="small_tool"),
            {"role": "user", "content": "question"},
        ]
        result = ObservationMaskingCondenser(max_output_chars=2000).condense(messages)
        assert result.metadata["masked"] == 1
        # Second tool and user message are untouched.
        assert result.messages[1]["content"] == "small"
        assert result.messages[2]["content"] == "question"

    def test_empty_messages(self):
        result = ObservationMaskingCondenser().condense([])
        assert result.messages == []
        assert result.metadata["masked"] == 0

    def test_exactly_at_threshold_not_masked(self):
        # Content at exactly max_output_chars should NOT be masked.
        content = "x" * 2000
        messages = [_tool_msg(content)]
        result = ObservationMaskingCondenser(max_output_chars=2000).condense(messages)
        assert result.metadata["masked"] == 0


# ---------------------------------------------------------------------------
# AmortizedForgettingCondenser
# ---------------------------------------------------------------------------


class TestAmortizedForgettingCondenser:
    def test_always_preserves_first_message(self):
        messages = [{"role": "user", "content": "first"}] + _msgs(*["old"] * 30)
        result = AmortizedForgettingCondenser(forget_threshold=0.9, decay_rate=0.05).condense(
            messages
        )
        assert result.messages[0]["content"] == "first"

    def test_always_preserves_last_four(self):
        messages = _msgs(*[str(i) for i in range(20)])
        result = AmortizedForgettingCondenser(forget_threshold=0.99, always_keep=4).condense(
            messages
        )
        tail = [m["content"] for m in result.messages[-4:]]
        assert tail == ["16", "17", "18", "19"]

    def test_drops_low_scoring_middle_messages(self):
        # With high threshold and slow decay, middle messages should be dropped.
        messages = (
            [{"role": "user", "content": "first"}]
            + [{"role": "user", "content": f"middle {i}"} for i in range(15)]
            + [{"role": "user", "content": f"recent {i}"} for i in range(4)]
        )
        result = AmortizedForgettingCondenser(
            forget_threshold=0.7, decay_rate=0.08, always_keep=4
        ).condense(messages)
        # Should have fewer messages than input.
        assert len(result.messages) < len(messages)
        assert result.metadata["dropped"] > 0

    def test_tool_messages_get_bonus(self):
        # A tool message at a low position should score higher than a user message.
        condenser = AmortizedForgettingCondenser(decay_rate=0.05)
        tool_msg = {"role": "tool", "content": "output"}
        user_msg = {"role": "user", "content": "text"}
        # position_from_end=10 for both.
        tool_score = condenser._score(tool_msg, 10)
        user_score = condenser._score(user_msg, 10)
        assert tool_score > user_score

    def test_keyword_messages_get_bonus(self):
        condenser = AmortizedForgettingCondenser(decay_rate=0.05)
        keyword_msg = {"role": "assistant", "content": "The result was successful"}
        plain_msg = {"role": "assistant", "content": "Okay, I can do that"}
        keyword_score = condenser._score(keyword_msg, 5)
        plain_score = condenser._score(plain_msg, 5)
        assert keyword_score > plain_score

    def test_empty_messages(self):
        result = AmortizedForgettingCondenser().condense([])
        assert result.messages == []
        assert result.metadata["dropped"] == 0

    def test_single_message_always_kept(self):
        result = AmortizedForgettingCondenser(forget_threshold=0.99).condense(
            [{"role": "user", "content": "only"}]
        )
        assert len(result.messages) == 1

    def test_metadata_kept_plus_dropped_equals_input(self):
        messages = _msgs(*[str(i) for i in range(20)])
        result = AmortizedForgettingCondenser(forget_threshold=0.5, decay_rate=0.04).condense(
            messages
        )
        assert result.metadata["kept"] + result.metadata["dropped"] == len(messages)

    def test_no_drop_when_threshold_zero(self):
        messages = _msgs(*[str(i) for i in range(10)])
        result = AmortizedForgettingCondenser(forget_threshold=0.0).condense(messages)
        assert result.metadata["dropped"] == 0
        assert len(result.messages) == 10


# ---------------------------------------------------------------------------
# SummarizingCondenser
# ---------------------------------------------------------------------------


class TestSummarizingCondenserWithoutProvider:
    def test_falls_back_to_keyword_extraction(self):
        messages = [
            {"role": "user", "content": "Deploy the app"},
            {"role": "assistant", "content": "Result: deployed successfully"},
            {"role": "user", "content": "recent 1"},
            {"role": "user", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "user", "content": "recent 4"},
            {"role": "user", "content": "recent 5"},
            {"role": "user", "content": "recent 6"},
        ]
        result = SummarizingCondenser(provider=None, preserve_recent=6).condense(messages)
        # Old 2 messages become 1 summary chunk; recent 6 are kept.
        assert len(result.messages) == 7
        summary_content = result.messages[0]["content"]
        assert summary_content.startswith("[Conversation Summary]")
        assert result.metadata["used_llm"] is False

    def test_preserves_recent_messages_intact(self):
        recent = [{"role": "user", "content": f"recent {i}"} for i in range(6)]
        old = [{"role": "user", "content": "old one"}]
        result = SummarizingCondenser(provider=None, preserve_recent=6).condense(old + recent)
        kept_recent = result.messages[-6:]
        for i, msg in enumerate(kept_recent):
            assert msg["content"] == f"recent {i}"

    def test_empty_messages(self):
        result = SummarizingCondenser().condense([])
        assert result.messages == []

    def test_all_messages_within_preserve_recent(self):
        messages = _msgs("a", "b", "c")
        result = SummarizingCondenser(provider=None, preserve_recent=6).condense(messages)
        # Nothing old to summarise.
        assert len(result.messages) == 3
        assert result.metadata["chunks_summarised"] == 0

    def test_multiple_chunks(self):
        messages = [{"role": "user", "content": f"old {i}"} for i in range(25)] + [
            {"role": "user", "content": f"recent {i}"} for i in range(6)
        ]
        result = SummarizingCondenser(provider=None, preserve_recent=6, chunk_size=10).condense(
            messages
        )
        # 25 old messages in chunks of 10 -> 3 chunks + 6 recent.
        assert result.metadata["chunks_summarised"] == 3
        assert len(result.messages) == 9  # 3 summaries + 6 recent

    def test_keyword_fallback_extracts_facts(self):
        messages = [
            {"role": "assistant", "content": "Result: build passed"},
            {"role": "user", "content": "recent 1"},
            {"role": "user", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "user", "content": "recent 4"},
            {"role": "user", "content": "recent 5"},
            {"role": "user", "content": "recent 6"},
        ]
        result = SummarizingCondenser(provider=None, preserve_recent=6).condense(messages)
        summary = result.messages[0]["content"]
        assert "Result: build passed" in summary

    def test_tool_role_messages_in_keyword_fallback(self):
        messages = [
            _tool_msg("exit code 0", name="run_tests"),
            {"role": "user", "content": "recent 1"},
            {"role": "user", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "user", "content": "recent 4"},
            {"role": "user", "content": "recent 5"},
            {"role": "user", "content": "recent 6"},
        ]
        result = SummarizingCondenser(provider=None, preserve_recent=6).condense(messages)
        summary = result.messages[0]["content"]
        assert "run_tests" in summary


class TestSummarizingCondenserWithProvider:
    def test_uses_llm_when_provider_available(self):
        provider = _make_mock_provider("- deployed OK")
        messages = [
            {"role": "user", "content": "old 1"},
            {"role": "user", "content": "old 2"},
        ] + [{"role": "user", "content": f"recent {i}"} for i in range(6)]
        result = SummarizingCondenser(provider=provider, preserve_recent=6).condense(messages)
        assert result.metadata["used_llm"] is True
        assert provider.complete.called

    def test_falls_back_on_provider_exception(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("api down")
        messages = [{"role": "user", "content": "old"}] + [
            {"role": "user", "content": f"recent {i}"} for i in range(6)
        ]
        # Should not raise; fallback produces a summary.
        result = SummarizingCondenser(provider=provider, preserve_recent=6).condense(messages)
        assert result.metadata["used_llm"] is False
        assert len(result.messages) > 0

    def test_model_kwarg_forwarded(self):
        provider = _make_mock_provider()
        messages = [{"role": "user", "content": "old"}] + [
            {"role": "user", "content": f"r{i}"} for i in range(6)
        ]
        SummarizingCondenser(provider=provider, model="claude-haiku", preserve_recent=6).condense(
            messages
        )
        _, kwargs = provider.complete.call_args
        assert kwargs.get("model") == "claude-haiku"


# ---------------------------------------------------------------------------
# LLMAttentionCondenser
# ---------------------------------------------------------------------------


class TestLLMAttentionCondenserWithoutProvider:
    def test_fallback_keeps_every_other_plus_tail(self):
        messages = [{"role": "user", "content": str(i)} for i in range(12)]
        result = LLMAttentionCondenser(provider=None, preserve_recent=4).condense(messages)
        # Tail (last 4) always kept.
        tail_contents = {m["content"] for m in result.messages[-4:]}
        assert tail_contents == {"8", "9", "10", "11"}

    def test_empty_messages(self):
        result = LLMAttentionCondenser(provider=None).condense([])
        assert result.messages == []

    def test_all_tail_no_body(self):
        messages = _msgs("a", "b", "c", "d")
        result = LLMAttentionCondenser(provider=None, preserve_recent=4).condense(messages)
        assert len(result.messages) == 4

    def test_metadata_kept_plus_dropped_equals_input(self):
        messages = _msgs(*[str(i) for i in range(10)])
        result = LLMAttentionCondenser(provider=None, preserve_recent=4).condense(messages)
        assert result.metadata["kept"] + result.metadata["dropped"] == len(messages)


class TestLLMAttentionCondenserWithProvider:
    def test_uses_llm_indices(self):
        provider = _make_mock_provider("[0, 2]")
        messages = [{"role": "user", "content": str(i)} for i in range(8)]
        result = LLMAttentionCondenser(provider=provider, preserve_recent=4).condense(messages)
        assert result.metadata["used_llm"] is True
        # Body is messages[0:4]; LLM selects indices 0 and 2 from body.
        body_kept = [m["content"] for m in result.messages[:-4]]
        assert "0" in body_kept
        assert "2" in body_kept

    def test_fallback_on_provider_error(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("fail")
        messages = _msgs(*[str(i) for i in range(10)])
        result = LLMAttentionCondenser(provider=provider, preserve_recent=4).condense(messages)
        # Falls back gracefully; tail still present.
        assert len(result.messages) > 0

    def test_fallback_on_unparseable_response(self):
        provider = _make_mock_provider("I cannot determine which to keep.")
        messages = _msgs(*[str(i) for i in range(10)])
        result = LLMAttentionCondenser(provider=provider, preserve_recent=4).condense(messages)
        assert len(result.messages) > 0

    def test_parse_indices_rejects_out_of_range(self):
        condenser = LLMAttentionCondenser()
        indices = condenser._parse_indices("[0, 99, 3]", max_index=5)
        # 99 > max_index=5 should be excluded.
        assert 99 not in (indices or [])

    def test_parse_indices_returns_none_on_invalid_json(self):
        condenser = LLMAttentionCondenser()
        assert condenser._parse_indices("not json", max_index=5) is None

    def test_parse_indices_finds_array_in_prose(self):
        condenser = LLMAttentionCondenser()
        indices = condenser._parse_indices(
            "Keep these: [1, 3, 5] because they matter.", max_index=10
        )
        assert indices == [1, 3, 5]


# ---------------------------------------------------------------------------
# PipelineCondenser
# ---------------------------------------------------------------------------


class TestPipelineCondenser:
    def test_chains_condensers_in_order(self):
        # Two window condensers: first keeps 6, second keeps 3.
        pipeline = PipelineCondenser(
            steps=[WindowCondenser(max_messages=6), WindowCondenser(max_messages=3)]
        )
        messages = _msgs(*[str(i) for i in range(10)])
        result = pipeline.condense(messages)
        assert len(result.messages) == 3

    def test_metadata_keyed_by_class_name(self):
        pipeline = PipelineCondenser(
            steps=[WindowCondenser(max_messages=5), ObservationMaskingCondenser()]
        )
        messages = _msgs(*[str(i) for i in range(10)])
        result = pipeline.condense(messages)
        assert "WindowCondenser" in result.metadata
        assert "ObservationMaskingCondenser" in result.metadata

    def test_empty_steps_returns_copy(self):
        pipeline = PipelineCondenser(steps=[])
        messages = _msgs("a", "b")
        result = pipeline.condense(messages)
        assert result.messages == messages
        assert result.messages is not messages

    def test_system_prompt_forwarded_to_each_step(self):
        spy = MagicMock(spec=BaseCondenser)
        spy.condense.return_value = CondenserResult(messages=[{"role": "user", "content": "x"}])
        pipeline = PipelineCondenser(steps=[spy])
        pipeline.condense([{"role": "user", "content": "hello"}], system_prompt="sys")
        spy.condense.assert_called_once()
        _, kwargs = spy.condense.call_args
        # system_prompt can be positional or keyword depending on the call.
        call_args = spy.condense.call_args
        assert "sys" in call_args.args or call_args.kwargs.get("system_prompt") == "sys"

    def test_intermediate_result_feeds_next_step(self):
        # First step drops to 5 messages, second step drops to 3.
        messages = _msgs(*[str(i) for i in range(20)])
        pipeline = PipelineCondenser(
            steps=[
                WindowCondenser(max_messages=5),
                WindowCondenser(max_messages=3),
            ]
        )
        result = pipeline.condense(messages)
        assert len(result.messages) == 3
        assert result.metadata["WindowCondenser"]["kept"] == 3


# ---------------------------------------------------------------------------
# create_default_pipeline
# ---------------------------------------------------------------------------


class TestCreateDefaultPipeline:
    def test_returns_pipeline_condenser(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline, PipelineCondenser)

    def test_pipeline_has_four_steps(self):
        pipeline = create_default_pipeline()
        assert len(pipeline.steps) == 4

    def test_first_step_is_observation_masking(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline.steps[0], ObservationMaskingCondenser)

    def test_second_step_is_amortized_forgetting(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline.steps[1], AmortizedForgettingCondenser)

    def test_third_step_is_summarizing(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline.steps[2], SummarizingCondenser)

    def test_fourth_step_is_window(self):
        pipeline = create_default_pipeline()
        assert isinstance(pipeline.steps[3], WindowCondenser)

    def test_window_cap_derived_from_max_tokens(self):
        pipeline = create_default_pipeline(max_tokens=30_000)
        window: WindowCondenser = pipeline.steps[3]  # type: ignore[assignment]
        assert window.max_messages == 30_000 // 150

    def test_small_max_tokens_minimum_window(self):
        pipeline = create_default_pipeline(max_tokens=100)
        window: WindowCondenser = pipeline.steps[3]  # type: ignore[assignment]
        assert window.max_messages >= 10

    def test_summarizing_step_receives_provider(self):
        provider = _make_mock_provider()
        pipeline = create_default_pipeline(provider=provider)
        summarizer: SummarizingCondenser = pipeline.steps[2]  # type: ignore[assignment]
        assert summarizer.provider is provider

    def test_runs_without_error_on_typical_input(self):
        messages = [_tool_msg("x" * 5000)] + [
            {"role": "user", "content": f"msg {i}"} for i in range(30)
        ]
        pipeline = create_default_pipeline()
        result = pipeline.condense(messages)
        assert isinstance(result.messages, list)
        assert len(result.messages) > 0


# ---------------------------------------------------------------------------
# MemoryConsolidator integration (uses pipeline under the hood)
# ---------------------------------------------------------------------------


class TestMemoryConsolidatorUsesPipeline:
    def test_consolidate_returns_tuple(self):
        from missy.agent.consolidation import MemoryConsolidator

        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result, summary = mc.consolidate(messages, "sys")
        assert isinstance(result, list)
        assert isinstance(summary, str)

    def test_pipeline_property_returns_pipeline_condenser(self):
        from missy.agent.condensers import PipelineCondenser
        from missy.agent.consolidation import MemoryConsolidator

        mc = MemoryConsolidator()
        assert isinstance(mc.pipeline, PipelineCondenser)

    def test_custom_pipeline_is_used(self):
        from missy.agent.condensers import PipelineCondenser
        from missy.agent.consolidation import MemoryConsolidator

        custom = PipelineCondenser(steps=[WindowCondenser(max_messages=3)])
        mc = MemoryConsolidator(pipeline=custom)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result, _ = mc.consolidate(messages, "sys")
        # Custom pipeline caps at 3.
        assert len(result) == 3

    def test_empty_messages_returns_empty(self):
        from missy.agent.consolidation import MemoryConsolidator

        mc = MemoryConsolidator()
        result, summary = mc.consolidate([], "sys")
        assert result == []
        assert summary == ""

    def test_few_messages_returned_as_is(self):
        from missy.agent.consolidation import MemoryConsolidator

        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(3)]
        result, summary = mc.consolidate(messages, "sys")
        assert result == messages
        assert summary == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_tool_messages_observation_masking(self):
        messages = [_tool_msg("x" * 3000) for _ in range(5)]
        result = ObservationMaskingCondenser(max_output_chars=100).condense(messages)
        assert result.metadata["masked"] == 5

    def test_pipeline_on_single_message(self):
        messages = [{"role": "user", "content": "only message"}]
        pipeline = create_default_pipeline()
        result = pipeline.condense(messages)
        assert len(result.messages) >= 1

    def test_amortized_forgetting_all_messages_identical_role(self):
        messages = [{"role": "assistant", "content": "blah"} for _ in range(20)]
        result = AmortizedForgettingCondenser(forget_threshold=0.5, decay_rate=0.1).condense(
            messages
        )
        # First and last 4 always preserved; some middle ones may be dropped.
        assert result.messages[0]["content"] == "blah"

    def test_window_condenser_max_messages_one(self):
        messages = _msgs("a", "b", "c")
        result = WindowCondenser(max_messages=1).condense(messages)
        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "c"

    def test_summarizing_condenser_no_old_messages(self):
        # All messages fit within preserve_recent.
        messages = _msgs("a", "b")
        result = SummarizingCondenser(preserve_recent=10).condense(messages)
        assert len(result.messages) == 2
        assert result.metadata["chunks_summarised"] == 0

    def test_pipeline_metadata_accumulates_all_steps(self):
        pipeline = create_default_pipeline()
        messages = [_tool_msg("x" * 5000)] + _msgs(*[str(i) for i in range(10)])
        result = pipeline.condense(messages)
        step_names = set(result.metadata.keys())
        assert "ObservationMaskingCondenser" in step_names
        assert "AmortizedForgettingCondenser" in step_names
        assert "SummarizingCondenser" in step_names
        assert "WindowCondenser" in step_names
