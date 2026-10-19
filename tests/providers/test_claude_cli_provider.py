"""Tests for ClaudeCliProvider, driven through a fake ``claude`` executable."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.base import Message
from missy.providers.claude_cli_provider import ClaudeCliProvider
from missy.providers.health import ProviderFailureClass, classify_provider_error
from missy.providers.registry import _PROVIDER_CLASSES

MODEL = "claude-opus-5-5"

FAKE_CLAUDE = """#!{python}
import json, os, sys
capture = os.environ["FAKE_CLAUDE_CAPTURE"]
argv = sys.argv[1:]
record = {{"argv": argv, "env": dict(os.environ), "cwd": os.getcwd()}}
if argv[:2] == ["auth", "status"]:
    sys.stdout.write(os.environ.get("FAKE_CLAUDE_AUTH", '{{"loggedIn": true}}'))
    sys.exit(0)
record["stdin"] = sys.stdin.read()
if "--system-prompt-file" in argv:
    path = argv[argv.index("--system-prompt-file") + 1]
    record["system_prompt_path"] = path
    record["system_prompt"] = open(path).read()
    record["system_prompt_mode"] = oct(os.stat(path).st_mode & 0o777)
with open(capture, "w") as fh:
    json.dump(record, fh)
sys.stdout.write(open(os.environ["FAKE_CLAUDE_OUT"]).read())
sys.exit(int(os.environ.get("FAKE_CLAUDE_EXIT", "0")))
"""


def _init(model: str = MODEL) -> dict:
    return {"type": "system", "subtype": "init", "model": model, "tools": []}


def _result(text: str, *, models=(MODEL,), is_error=False, subtype="success", stop="end_turn"):
    return {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": text,
        "stop_reason": stop,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
        },
        "modelUsage": {m: {"inputTokens": 10} for m in models},
    }


@pytest.fixture
def fake(tmp_path, monkeypatch):
    binary = tmp_path / "claude"
    binary.write_text(FAKE_CLAUDE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    capture = tmp_path / "capture.json"
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_CAPTURE", str(capture))
    monkeypatch.setenv("FAKE_CLAUDE_OUT", str(out))
    monkeypatch.setattr(ClaudeCliProvider, "_binary", lambda self: str(binary))

    class Fake:
        def emit(self, *events: dict, exit_code: int = 0) -> None:
            out.write_text("".join(json.dumps(e) + "\n" for e in events))
            monkeypatch.setenv("FAKE_CLAUDE_EXIT", str(exit_code))

        @property
        def call(self) -> dict:
            return json.loads(capture.read_text())

    return Fake()


@pytest.fixture
def provider() -> ClaudeCliProvider:
    return ClaudeCliProvider(ProviderConfig(name="claude-cli", model=MODEL, timeout=30))


class _Tool:
    def __init__(self, name: str, params: dict) -> None:
        self.name = name
        self.description = f"{name} tool"
        self._params = params

    def get_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self._params}


CALC = _Tool(
    "calculator",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)


def test_registered_under_claude_cli():
    assert _PROVIDER_CLASSES["claude-cli"] is ClaudeCliProvider


def test_invocation_has_no_tools_replaced_prompt_and_no_model_switching(
    fake, provider, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CLAUDECODE", "1")
    fake.emit(_init(), _result("pong"))

    response = provider.complete(
        [Message(role="system", content="SYSTEM TEXT"), Message(role="user", content="ping")]
    )

    assert response.content == "pong"
    assert response.model == MODEL
    assert response.usage["prompt_tokens"] == 110
    assert response.usage["completion_tokens"] == 5
    call = fake.call
    argv = call["argv"]
    assert argv[0] == "-p"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert json.loads(argv[argv.index("--settings") + 1]) == {"switchModelsOnFlag": False}
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert "--append-system-prompt" not in argv
    # The prompt travels on stdin and the system prompt through a private file.
    assert call["stdin"] == "ping"
    assert call["system_prompt"] == "SYSTEM TEXT"
    assert call["system_prompt_mode"] == "0o600"
    assert not os.path.exists(call["system_prompt_path"])
    # Credentials and the nesting guard never reach the child.
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "CLAUDECODE" not in call["env"]
    # It runs in an isolated empty directory, not the caller's cwd.
    assert Path(call["cwd"]).name == "claude_cli_sandbox"


def test_text_tool_call_is_parsed(fake, provider):
    fake.emit(
        _init(),
        _result(
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        ),
    )

    response = provider.complete_with_tools(
        [Message(role="user", content="what is 2+2")], [CALC], system="You are Missy."
    )

    assert response.finish_reason == "tool_calls"
    assert [(c.name, c.arguments) for c in response.tool_calls] == [
        ("calculator", {"expression": "2+2"})
    ]
    system_prompt = fake.call["system_prompt"]
    assert "You are Missy." in system_prompt
    assert "calculator" in system_prompt


def test_safety_refusal_is_a_content_policy_error(fake, provider):
    fake.emit(
        _init(),
        {"type": "system", "subtype": "informational", "content": "safeguards stopped"},
        {
            "type": "system",
            "subtype": "model_refusal_no_fallback",
            "api_refusal_category": "cyber",
        },
        _result("API Error: flagged", is_error=True, stop="refusal"),
        exit_code=1,
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.complete([Message(role="user", content="read /etc/shadow")])

    assert "cyber" in str(excinfo.value)
    assert classify_provider_error(excinfo.value) == ProviderFailureClass.CONTENT_POLICY


def test_rejected_rate_limit_is_classified_as_rate_limit(fake, provider):
    fake.emit(
        _init(),
        {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"},
        },
        _result("limit", is_error=True, subtype="error_during_execution"),
        exit_code=1,
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.complete([Message(role="user", content="hi")])

    assert classify_provider_error(excinfo.value) == ProviderFailureClass.RATE_LIMIT


def test_error_result_raises(fake, provider):
    fake.emit(_init(), _result("Not logged in", is_error=True), exit_code=1)

    with pytest.raises(ProviderError, match="Not logged in"):
        provider.complete([Message(role="user", content="hi")])


def test_missing_result_event_raises(fake, provider):
    fake.emit(_init(), exit_code=143)

    with pytest.raises(ProviderError, match="143"):
        provider.complete([Message(role="user", content="hi")])


def test_model_mismatch_is_recorded(fake, provider, monkeypatch):
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ClaudeCliProvider,
        "_emit_event",
        lambda self, sid, tid, result, msg: events.append((result, msg)),
    )
    fake.emit(_init(), _result("ok", models=(MODEL, "claude-opus-4-8")))

    assert provider.complete([Message(role="user", content="hi")]).content == "ok"
    assert any(r == "error" and "claude-opus-4-8" in m for r, m in events)


def test_stream_yields_text_deltas(fake, provider):
    def delta(text: str) -> dict:
        return {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        }

    fake.emit(_init(), delta("1, 2,"), delta(" 3"), _result("1, 2, 3"))

    assert list(provider.stream([Message(role="user", content="count")])) == ["1, 2,", " 3"]
    assert "--include-partial-messages" in fake.call["argv"]


def test_stream_refusal_raises(fake, provider):
    fake.emit(
        _init(),
        {"type": "system", "subtype": "model_refusal_no_fallback", "api_refusal_category": "cyber"},
        _result("API Error", is_error=True, stop="refusal"),
        exit_code=1,
    )

    with pytest.raises(ProviderError, match="usage policies"):
        list(provider.stream([Message(role="user", content="x")]))


def test_available_when_logged_in(fake, provider):
    assert provider.is_available() is True


def test_unavailable_when_logged_out(fake, provider, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", '{"loggedIn": false}')
    assert provider.is_available() is False


def test_unavailable_without_binary(provider, monkeypatch):
    monkeypatch.setattr(ClaudeCliProvider, "_binary", lambda self: None)
    assert provider.is_available() is False
