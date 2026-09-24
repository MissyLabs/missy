"""Regression tests for the September 2026 SleeptimeWorker incident."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from missy.agent.sleeptime import SleeptimeConfig, SleeptimeWorker
from missy.core.exceptions import ProviderError


def test_background_processing_defaults_to_no_thread_or_provider_calls():
    provider = MagicMock()
    registry = MagicMock()
    registry.get.return_value = provider
    worker = SleeptimeWorker(provider_registry=registry)

    assert worker.start() is False
    assert worker._thread is None
    assert worker._llm_summarize("content") is None
    provider.complete.assert_not_called()


def test_only_one_worker_can_own_a_shared_store():
    store = MagicMock()
    first = SleeptimeWorker(
        config=SleeptimeConfig(enabled=True, check_interval_seconds=9999),
        memory_store=store,
    )
    second = SleeptimeWorker(
        config=SleeptimeConfig(enabled=True, check_interval_seconds=9999),
        memory_store=store,
    )
    try:
        assert first.start() is True
        assert second.start() is False
        assert sum(t.name == "missy-sleeptime" for t in threading.enumerate()) == 1
    finally:
        first.stop(timeout=1)
        second.stop(timeout=1)


def test_configured_provider_is_used_without_alphabetical_fallback():
    acpx = MagicMock()
    local = MagicMock()
    local.complete.return_value = MagicMock(content="local summary")
    registry = MagicMock()
    registry.get.side_effect = lambda name: {"acpx": acpx, "ollama": local}.get(name)
    registry.list_providers.return_value = ["acpx", "ollama"]
    worker = SleeptimeWorker(
        config=SleeptimeConfig(use_llm_summarization=True, provider="ollama"),
        provider_registry=registry,
    )

    assert worker._llm_summarize("content") == "local summary"
    acpx.complete.assert_not_called()


def test_usage_limit_opens_provider_wide_backoff_and_stops_calls():
    provider = MagicMock()
    provider.complete.side_effect = ProviderError("Claude usage limit reached")
    registry = MagicMock()
    registry.get.return_value = provider
    worker = SleeptimeWorker(
        config=SleeptimeConfig(
            use_llm_summarization=True,
            provider="incident-acpx",
            backoff_base_seconds=120,
        ),
        provider_registry=registry,
    )

    assert worker._llm_summarize("first") is None
    assert worker._llm_summarize("second") is None
    assert provider.complete.call_count == 1
    assert worker.stats.next_retry_at is not None
    assert worker._abort_cycle is True
