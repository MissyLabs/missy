"""ACPX provider for the Missy framework.

Wraps the `acpx <https://github.com/openclaw/acpx>`_ CLI — a headless
Agent Client Protocol client that can talk to Claude, Codex, Gemini,
Cursor, and many other coding agents over a structured protocol instead
of PTY scraping.

Prompts are passed to the ``acpx`` binary via ``exec`` (one-shot, no
saved session) with ``--format json`` so the output is machine-readable
NDJSON.  Persistent session support is also available: when a
``session`` name is configured the provider uses the session lifecycle
(``sessions ensure`` / ``prompt``) instead of ``exec``.

Install ACPX::

    npm install -g acpx@latest

Configure in ``config.yaml``::

    providers:
      acpx:
        name: acpx
        model: "claude"          # agent name: claude, codex, gemini, cursor, …
        timeout: 120
        enabled: true

    # Optional per-provider overrides via base_url:
    #   base_url: "--approve-all --cwd /my/project"
    # (extra CLI flags appended to every invocation)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message

logger = logging.getLogger(__name__)

_DEFAULT_AGENT = "claude"
_DEFAULT_TIMEOUT = 120


class AcpxProvider(BaseProvider):
    """Provider that delegates to the ``acpx`` CLI binary.

    Each completion call spawns ``acpx <agent> exec "<prompt>"`` as a
    subprocess with ``--format json`` output.  The NDJSON events are
    parsed and the final assistant text is extracted.

    Args:
        config: Provider config.

            * ``model`` — the ACPX agent name (``claude``, ``codex``,
              ``gemini``, ``cursor``, etc.).  Defaults to ``"claude"``.
            * ``base_url`` — extra CLI flags to append to every
              invocation (e.g. ``"--approve-all --cwd /my/project"``).
            * ``api_key`` — unused by ACPX itself (agents use their
              own env-var credentials), but stored for consistency.
            * ``timeout`` — subprocess timeout in seconds (default 120).
    """

    name = "acpx"

    def __init__(self, config: ProviderConfig) -> None:
        self._agent: str = config.model or _DEFAULT_AGENT
        self._timeout: int = config.timeout or _DEFAULT_TIMEOUT
        self._extra_flags: list[str] = (
            config.base_url.split() if config.base_url else []
        )
        self._binary: str = shutil.which("acpx") or "acpx"

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the ``acpx`` binary is on ``$PATH``.

        Also runs ``acpx --version`` to verify it executes successfully.
        """
        binary = shutil.which("acpx")
        if not binary:
            logger.debug("acpx binary not found on PATH")
            return False
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.debug("acpx availability check failed: %s", exc)
            return False

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Run a one-shot completion via ``acpx <agent> exec``.

        Conversation history is flattened into a single prompt string
        with role prefixes.  The ``--format json`` flag produces NDJSON
        events from which the assistant text is extracted.

        Args:
            messages: Ordered conversation turns.
            **kwargs: Optional overrides.  Recognised keys:

                * ``cwd`` (str) — working directory for the subprocess.
                * ``approve_all`` (bool) — pass ``--approve-all``.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On subprocess failure or unexpected output.
        """
        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        cwd = kwargs.pop("cwd", None)
        approve_all = kwargs.pop("approve_all", False)

        prompt = self._build_prompt(messages)

        cmd = [self._binary, self._agent, "exec", prompt]
        cmd.extend(["--format", "json"])
        if approve_all:
            cmd.append("--approve-all")
        cmd.extend(self._extra_flags)
        if cwd:
            cmd.extend(["--cwd", str(cwd)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            self._emit_event(session_id, task_id, "error", "subprocess timed out")
            raise ProviderError(
                f"acpx subprocess timed out after {self._timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            self._emit_event(session_id, task_id, "error", "acpx binary not found")
            raise ProviderError(
                "acpx binary not found. Install with: npm install -g acpx@latest"
            ) from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"acpx subprocess failed: {exc}") from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            self._emit_event(session_id, task_id, "error", f"exit {result.returncode}: {stderr}")
            raise ProviderError(
                f"acpx exited with code {result.returncode}: {stderr}"
            )

        content = self._parse_ndjson_output(result.stdout)

        self._emit_event(session_id, task_id, "allow", "completion successful")

        return CompletionResponse(
            content=content,
            model=self._agent,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw={"stdout": result.stdout, "stderr": result.stderr},
        )

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream tokens from ``acpx`` by reading NDJSON events line-by-line.

        Uses ``--format json`` and streams stdout in real time via
        ``Popen`` instead of waiting for the process to finish.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt (prepended if non-empty).

        Yields:
            Text chunks as they arrive.

        Raises:
            ProviderError: On subprocess failure.
        """
        if system:
            messages = [Message(role="system", content=system), *messages]

        prompt = self._build_prompt(messages)
        cmd = [self._binary, self._agent, "exec", prompt, "--format", "json"]
        cmd.extend(self._extra_flags)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                "acpx binary not found. Install with: npm install -g acpx@latest"
            ) from exc

        try:
            assert proc.stdout is not None  # for type checker
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON output — could be plain text, yield as-is
                    yield line
                    continue

                text = self._extract_text_from_event(event)
                if text:
                    yield text

            proc.wait(timeout=30)
            if proc.returncode and proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise ProviderError(
                    f"acpx stream exited with code {proc.returncode}: {stderr[:500]}"
                )
        except ProviderError:
            raise
        except Exception as exc:
            proc.kill()
            raise ProviderError(f"acpx stream failed: {exc}") from exc
        finally:
            if proc.poll() is None:
                proc.terminate()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: list[Message]) -> str:
        """Flatten a conversation into a single prompt string.

        System messages are prefixed with ``[System]:``, user messages
        with ``[User]:``, and assistant messages with ``[Assistant]:``.
        For a single user message the prefix is omitted.
        """
        if len(messages) == 1 and messages[0].role == "user":
            return messages[0].content

        parts: list[str] = []
        for msg in messages:
            prefix = {
                "system": "[System]",
                "user": "[User]",
                "assistant": "[Assistant]",
            }.get(msg.role, f"[{msg.role}]")
            parts.append(f"{prefix}: {msg.content}")
        return "\n".join(parts)

    def _parse_ndjson_output(self, stdout: str) -> str:
        """Parse NDJSON output and extract the final assistant text.

        ACPX ``--format json`` emits one JSON object per line.  We look
        for text delta events and concatenate them.  If the output is
        not valid NDJSON, we fall back to returning the raw stdout.
        """
        text_parts: list[str] = []
        has_json = False

        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                has_json = True
            except json.JSONDecodeError:
                continue

            text = self._extract_text_from_event(event)
            if text:
                text_parts.append(text)

        if text_parts:
            return "".join(text_parts)

        # If no structured text events found, return raw stdout
        if not has_json:
            return stdout.strip()

        return ""

    @staticmethod
    def _extract_text_from_event(event: dict) -> str:
        """Pull text content from a single NDJSON event.

        Handles several event shapes that ACPX may emit:

        * ``{"type": "text_delta", "delta": "..."}``
        * ``{"type": "message", "content": "..."}``
        * ``{"type": "result", "text": "..."}``
        * ``{"content": "..."}`` (generic fallback)
        """
        etype = event.get("type", "")

        # Text delta events (streaming)
        if etype in ("text_delta", "response.output_text.delta"):
            return event.get("delta", "")

        # Final message or result
        if etype in ("message", "result"):
            return event.get("text", "") or event.get("content", "")

        # Generic content field
        if "content" in event and isinstance(event["content"], str) and not etype:
            return event["content"]

        return ""

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a provider audit event to the global event bus."""
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={
                    "provider": self.name,
                    "agent": self._agent,
                    "message": detail_msg,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
