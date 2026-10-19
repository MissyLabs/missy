"""Claude Code CLI provider: runs ``claude -p`` directly as a one-shot delegate.

This replaces the ACP route (``acpx_provider.py`` + ``acp_bridge.mjs``). The
2026-10-19 acpx-only validation run showed two failures that came from the
ACP adapter rather than from Missy:

* The adapter only *appended* Missy's envelope to Claude Code's own agentic
  system prompt. With native tools disabled, the model still reached for
  them and got "No such tool available" back from the SDK, so Missy never
  saw a ``<tool_call>`` (43 of 201 cases).
* Safety-classifier notices and the Opus 5.5 -> Opus 4.8 refusal fallback
  were streamed as ordinary text chunks, so they leaked into Discord replies
  and the turn was silently answered by a different model (39 cases).

``claude -p`` with ``--system-prompt-file`` *replaces* the default prompt,
``--tools ""`` removes every built-in tool, and ``--output-format
stream-json`` reports notices, refusals, rate-limit state and per-model
usage as separate structured events, never as reply text.

Everything above the transport (the delegation envelope, text tool-call
protocol, validation, history flattening, transcript-leak stripping) is
inherited unchanged from :class:`AcpxProvider`.

Model switching on a safety flag is disabled (``switchModelsOnFlag: false``):
a flagged turn ends as a refusal, which is raised as a content-policy
:class:`ProviderError` so the runtime renders its sanitized refusal and does
not fall back to another provider.

Configure in ``config.yaml``::

    providers:
      claude-cli:
        name: claude-cli
        model: "claude-opus-5-5"
        timeout: 120
        enabled: true

Requires the ``claude`` CLI logged in with a Claude subscription. Anthropic
API-key variables are removed from the child environment so a stray key can
never silently move billing off the subscription.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError

from .acpx_provider import (
    _ACTIVE_PROCESSES,
    _ACTIVE_PROCESSES_LOCK,
    AcpxProvider,
    _kill_process_group,
    _run_subprocess_with_group_kill,
    _split_system_messages,
)
from .base import Message

logger = logging.getLogger(__name__)

_FALLBACK_BINARY = Path.home() / ".local" / "bin" / "claude"

#: Child-environment variables that must not leak into the CLI: the nesting
#: guard (a gateway started from a Claude Code shell would otherwise refuse to
#: run) and every Anthropic credential/endpoint override, so the call always
#: bills the logged-in subscription.
_SCRUBBED_ENV_VARS = frozenset(
    {
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SSE_PORT",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_OLD",
        "ANTHROPIC_API_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
    }
)

#: End a flagged turn with a refusal instead of re-running it on another model.
_CLI_SETTINGS = json.dumps({"switchModelsOnFlag": False})

_AUTH_PROBE_TIMEOUT_SECONDS = 20


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV_VARS}
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def _parse_events(stdout: str) -> list[dict]:
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


class ClaudeCliProvider(AcpxProvider):
    """Delegate provider that shells out to ``claude -p`` for each turn."""

    name = "claude-cli"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._agent = "claude-cli"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _binary(self) -> str | None:
        found = shutil.which("claude")
        if found:
            return found
        if _FALLBACK_BINARY.is_file() and os.access(_FALLBACK_BINARY, os.X_OK):
            return str(_FALLBACK_BINARY)
        return None

    def is_available(self) -> bool:
        if self._binary() is None:
            logger.debug("claude binary not found on PATH or at %s", _FALLBACK_BINARY)
            return False
        return self._cached_readiness()

    def _probe_bridge_readiness(self) -> bool:
        """Check ``claude auth status`` reports a login, without running inference."""
        binary = self._binary()
        if binary is None:
            return False
        try:
            completed = _run_subprocess_with_group_kill(
                [binary, "auth", "status"],
                self._isolated_cwd(),
                _AUTH_PROBE_TIMEOUT_SECONDS,
                env=_child_env(),
            )
        except Exception:
            logger.warning("claude-cli readiness probe failed; details withheld.")
            return False
        try:
            status = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError):
            logger.warning("claude-cli readiness probe returned unparseable output.")
            return False
        if not (isinstance(status, dict) and status.get("loggedIn") is True):
            logger.warning("claude CLI is not logged in; run `claude auth login`.")
            return False
        return True

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _isolated_cwd(self) -> str:
        """An empty private directory, so the CLI never loads Missy's CLAUDE.md,
        ``.mcp.json`` or hooks from wherever the gateway was started."""
        if self._sandbox_cwd is not None:
            return self._sandbox_cwd
        sandbox = Path(os.path.expanduser("~/.missy/claude_cli_sandbox"))
        sandbox.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(sandbox, 0o700)
        except OSError:
            logger.debug("Could not chmod claude-cli sandbox %s", sandbox, exc_info=True)
        self._sandbox_cwd = str(sandbox)
        return self._sandbox_cwd

    def _build_argv(self, binary: str, system_prompt_file: str, model: str, stream: bool) -> list:
        argv = [
            binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--max-turns",
            "1",
            "--settings",
            _CLI_SETTINGS,
            "--system-prompt-file",
            system_prompt_file,
        ]
        if model:
            argv += ["--model", model]
        if stream:
            argv.append("--include-partial-messages")
        return argv

    def _write_system_prompt(self, system_prompt: str) -> str:
        fd, path = tempfile.mkstemp(prefix="missy-claude-sys-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(system_prompt)
        return path

    def _run_acpx(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str = "",
        task_id: str = "",
        cwd: str | None = None,
        model: str = "",
    ) -> tuple[str, str]:
        """Run one ``claude -p`` turn and return ``(reply text, raw stdout)``."""
        binary = self._binary()
        if binary is None:
            self._emit_event(session_id, task_id, "error", "claude binary not found")
            raise ProviderError("claude CLI not found on PATH; install Claude Code.")
        model = (model or self._model or "").strip()
        resolved_cwd = cwd or self._isolated_cwd()
        sys_path = self._write_system_prompt(system_prompt)
        try:
            result = _run_subprocess_with_group_kill(
                self._build_argv(binary, sys_path, model, stream=False),
                resolved_cwd,
                self._timeout,
                input_text=prompt,
                env=_child_env(),
            )
        except subprocess.TimeoutExpired as exc:
            self._emit_event(session_id, task_id, "error", "subprocess timed out")
            raise ProviderError(
                f"claude-cli subprocess timed out after {self._timeout}s. The outcome of "
                "this call is UNKNOWN -- it was not confirmed to succeed or fail. "
                "Do not assume it happened or that it didn't; perform a fresh "
                "read-only state check before retrying or reporting status, and "
                "make any mutating retry idempotent."
            ) from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"claude-cli subprocess failed: {exc}") from exc
        finally:
            with contextlib.suppress(OSError):
                os.unlink(sys_path)

        events = _parse_events(result.stdout)
        text = self._interpret_result(
            events, result.returncode, result.stderr, model, session_id, task_id
        )
        return text, result.stdout

    def _interpret_result(
        self,
        events: list[dict],
        returncode: int,
        stderr: str,
        model: str,
        session_id: str,
        task_id: str,
    ) -> str:
        """Turn the CLI's event stream into reply text, or raise a classified error."""
        final = next((e for e in reversed(events) if e.get("type") == "result"), None)

        refusal = next(
            (
                e
                for e in events
                if e.get("type") == "system"
                and e.get("subtype") in ("model_refusal_no_fallback", "model_refusal_fallback")
            ),
            None,
        )
        if refusal is not None or (final is not None and final.get("stop_reason") == "refusal"):
            category = (refusal or {}).get("api_refusal_category") or "unspecified"
            self._emit_event(session_id, task_id, "deny", f"model refusal ({category})")
            # "usage policies" is a content-policy marker (providers/health.py):
            # the runtime renders its sanitized refusal and suppresses fallback.
            raise ProviderError(
                f"claude-cli: {model or 'model'} declined the request under its usage "
                f"policies (refusal category: {category})"
            )

        for event in events:
            if event.get("type") != "rate_limit_event":
                continue
            info = event.get("rate_limit_info") or {}
            if info.get("status") == "rejected":
                window = info.get("rateLimitType") or "unknown"
                self._emit_event(session_id, task_id, "error", f"usage limit reached ({window})")
                raise ProviderError(
                    f"claude-cli usage limit reached ({window} window); resets at "
                    f"{info.get('resetsAt')}"
                )

        if final is None:
            detail = (stderr or "").strip()[:500] or "no result event"
            self._emit_event(session_id, task_id, "error", f"exit {returncode}: {detail}")
            raise ProviderError(f"claude-cli exited with code {returncode}: {detail}")

        if final.get("is_error") or final.get("subtype") != "success" or returncode != 0:
            detail = str(final.get("result") or final.get("subtype") or "")[:500]
            self._emit_event(session_id, task_id, "error", f"exit {returncode}: {detail}")
            raise ProviderError(f"claude-cli exited with code {returncode}: {detail}")

        used = sorted((final.get("modelUsage") or {}).keys())
        if model and used and any(m != model for m in used):
            # Switching is disabled, so this should never happen; record the
            # models that actually answered rather than claiming the pin.
            logger.warning("claude-cli answered with %s, not the pinned %s", used, model)
            self._emit_event(session_id, task_id, "error", f"model mismatch: used {used}")

        return str(final.get("result") or "")

    # ------------------------------------------------------------------
    # Output parsing (overrides of the ACP NDJSON helpers)
    # ------------------------------------------------------------------

    def _parse_ndjson_output(self, stdout: str) -> str:
        final = next((e for e in reversed(_parse_events(stdout)) if e.get("type") == "result"), {})
        return str(final.get("result") or "")

    @staticmethod
    def _parse_usage_details(stdout: str) -> dict[str, int]:
        final = next((e for e in reversed(_parse_events(stdout)) if e.get("type") == "result"), {})
        usage = final.get("usage") or {}
        keys = (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
        return {
            k: usage[k]
            for k in keys
            if isinstance(usage.get(k), int) and not isinstance(usage[k], bool)
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Yield text deltas from ``--include-partial-messages`` stream events."""
        binary = self._binary()
        if binary is None:
            raise ProviderError("claude CLI not found on PATH; install Claude Code.")
        system_text, non_system_messages = _split_system_messages(messages)
        prompt = self._build_prompt(non_system_messages)
        sys_path = self._write_system_prompt(system or system_text)
        model = self._model
        try:
            proc = subprocess.Popen(
                self._build_argv(binary, sys_path, model, stream=True),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._isolated_cwd(),
                env=_child_env(),
                start_new_session=True,
            )
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.unlink(sys_path)
            raise ProviderError(f"claude-cli stream failed to start: {exc}") from exc
        with _ACTIVE_PROCESSES_LOCK:
            _ACTIVE_PROCESSES[proc] = threading.current_thread().name

        def _write_stdin() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception:
                logger.debug("Failed writing prompt to claude stdin", exc_info=True)

        writer = threading.Thread(target=_write_stdin, daemon=True)
        writer.start()
        events: list[dict] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                parsed = _parse_events(line)
                if not parsed:
                    continue
                event = parsed[0]
                events.append(event)
                if event.get("type") != "stream_event":
                    continue
                inner = event.get("event") or {}
                delta = inner.get("delta") or {}
                if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        yield text
            proc.wait(timeout=30)
            writer.join(timeout=5)
            stderr = proc.stderr.read() if proc.stderr else ""
            self._interpret_result(events, proc.returncode or 0, stderr, model, "", "")
        except ProviderError:
            raise
        except Exception as exc:
            _kill_process_group(proc)
            raise ProviderError(f"claude-cli stream failed: {exc}") from exc
        finally:
            if proc.poll() is None:
                _kill_process_group(proc, force=False)
            with _ACTIVE_PROCESSES_LOCK:
                _ACTIVE_PROCESSES.pop(proc, None)
            with contextlib.suppress(OSError):
                os.unlink(sys_path)
