"""DATA-02: trust scores are shared across runtimes and never lost across
instances/processes sharing one trust.json."""

from __future__ import annotations

import json

from missy.security.trust import DEFAULT_SCORE, TrustScorer, get_trust_scorer


def test_interleaved_instances_do_not_lose_updates(tmp_path):
    p = tmp_path / "trust.json"
    a = TrustScorer(persist_path=p, flush_interval=0)
    b = TrustScorer(persist_path=p, flush_interval=0)
    a.record_success("tool_a")  # a's cache: {tool_a: 510}
    b.record_violation("tool_b")  # b never saw tool_a -- must not erase it
    a.record_success("tool_a")
    data = json.loads(p.read_text())
    assert data == {"tool_a": DEFAULT_SCORE + 20, "tool_b": DEFAULT_SCORE - 200}


def test_flush_refreshes_view_with_other_writers(tmp_path):
    p = tmp_path / "trust.json"
    a = TrustScorer(persist_path=p, flush_interval=0)
    b = TrustScorer(persist_path=p, flush_interval=0)
    b.record_violation("shell_exec")
    a.record_success("other")
    assert a.score("shell_exec") == DEFAULT_SCORE - 200


def test_batching_defers_routine_writes_but_not_violations(tmp_path):
    p = tmp_path / "trust.json"
    s = TrustScorer(persist_path=p, flush_interval=3600)
    s.record_success("x")  # first write flushes immediately
    s.record_success("x")  # batched
    assert json.loads(p.read_text())["x"] == DEFAULT_SCORE + 10
    s.record_violation("y")  # forces flush of everything pending
    data = json.loads(p.read_text())
    assert data["x"] == DEFAULT_SCORE + 20 and data["y"] == DEFAULT_SCORE - 200
    s.record_success("x")
    s.flush()
    assert json.loads(p.read_text())["x"] == DEFAULT_SCORE + 30


def test_get_trust_scorer_is_shared_per_path(tmp_path):
    p = tmp_path / "trust.json"
    assert get_trust_scorer(p) is get_trust_scorer(str(p))
    assert get_trust_scorer(p) is not get_trust_scorer(tmp_path / "other.json")


def test_runtimes_share_one_scorer():
    from missy.agent.runtime import AgentConfig, AgentRuntime

    a = AgentRuntime(AgentConfig(provider="anthropic"))
    b = AgentRuntime(AgentConfig(provider="anthropic"))
    try:
        assert a._trust_scorer is b._trust_scorer
        a._trust_scorer.record_violation("danger")
        assert not b._trust_scorer.is_trusted("danger", threshold=400)
    finally:
        a.shutdown()
        b.shutdown()


def test_file_mode_is_0600(tmp_path):
    import stat

    p = tmp_path / "trust.json"
    TrustScorer(persist_path=p).record_success("x")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_first_write_flushes_even_right_after_boot(tmp_path, monkeypatch):
    """Regression (CI): time.monotonic() is seconds since boot, so a fresh
    host has a small value; the first write must still persist at once."""
    import missy.security.trust as trust_mod

    monkeypatch.setattr(trust_mod.time, "monotonic", lambda: 5.0)
    p = tmp_path / "trust.json"
    TrustScorer(persist_path=p, flush_interval=3600).record_success("x")
    assert json.loads(p.read_text())["x"] == DEFAULT_SCORE + 10
