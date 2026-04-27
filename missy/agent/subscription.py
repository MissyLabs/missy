"""Streaming subscription state machine for agent runs.

The runtime-facing :class:`AgentSubscription` keeps provider stream handling
deterministic without knowing the provider's native event shape.  It accepts
simple dict events, reconciles delta and full-content updates into a monotonic
raw buffer, strips assistant-only tags across chunk boundaries, tracks block
reply flush points, and records compaction retry state.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

ReasoningMode = Literal["off", "on", "stream"]
BlockReplyBreak = Literal["text_end", "message_end"]

_TAG_NAMES = ("think", "thinking", "thought", "antthinking", "final")
_TAG_RE = re.compile(
    r"^<\s*(/?)\s*(think(?:ing)?|thought|antthinking|final)\s*>$",
    re.IGNORECASE,
)
_REPLY_DIRECTIVE_RE = re.compile(
    r"\[\[(reply_to_current|media:([^\]]+)|audio:([^\]]+))\]\]",
    re.IGNORECASE,
)


@dataclass
class InlineCodeState:
    """Tracks inline and fenced code spans across chunks."""

    in_inline_code: bool = False
    in_fence: bool = False
    backtick_run: int = 0

    @property
    def in_code(self) -> bool:
        """Return ``True`` when the parser is inside code."""
        return self.in_inline_code or self.in_fence


@dataclass
class BlockState:
    """State used while stripping assistant-only block tags."""

    thinking: bool = False
    final: bool = False
    inline_code: InlineCodeState = field(default_factory=InlineCodeState)
    tag_buffer: str = ""
    seen_final_tag: bool = False


@dataclass
class ReplyDirectives:
    """Reply directives parsed from assistant text."""

    reply_to_current: bool = False
    media_urls: list[str] = field(default_factory=list)
    audio_urls: list[str] = field(default_factory=list)

    def merge(self, other: ReplyDirectives) -> None:
        """Merge another directive set into this one."""
        self.reply_to_current = self.reply_to_current or other.reply_to_current
        self.media_urls.extend(other.media_urls)
        self.audio_urls.extend(other.audio_urls)


@dataclass
class SubscriptionUpdate:
    """Result returned by message and lifecycle handlers."""

    raw_delta: str = ""
    visible_delta: str = ""
    full_visible_text: str = ""
    reasoning_delta: str = ""
    flushed_blocks: list[str] = field(default_factory=list)
    directives: ReplyDirectives = field(default_factory=ReplyDirectives)


RawStreamCallback = Callable[[Mapping[str, Any], str, str], None]
PartialReplyCallback = Callable[[str, str], None]
BlockReplyCallback = Callable[[str], None]
ReasoningCallback = Callable[[str], None]


def _event_type(event: Mapping[str, Any] | str) -> str:
    if isinstance(event, str):
        return event
    return str(event.get("type") or event.get("event") or "")


def _finalize_backticks(state: BlockState) -> None:
    run = state.inline_code.backtick_run
    if run <= 0:
        return
    if run >= 3:
        state.inline_code.in_fence = not state.inline_code.in_fence
    elif run == 1 and not state.inline_code.in_fence:
        state.inline_code.in_inline_code = not state.inline_code.in_inline_code
    state.inline_code.backtick_run = 0


def _is_possible_tag_prefix(value: str) -> bool:
    if not value.startswith("<"):
        return False
    body = value[1:].lstrip()
    if body in {"", "/"}:
        return True
    if body.startswith("/"):
        body = body[1:].lstrip()
    name = re.split(r"\s", body, maxsplit=1)[0].lower()
    if not name:
        return True
    return any(tag.startswith(name) for tag in _TAG_NAMES)


def _is_visible(state: BlockState, enforce_final: bool) -> bool:
    if state.thinking:
        return False
    return not (enforce_final and not state.final)


def _process_tag(raw: str, state: BlockState) -> bool:
    match = _TAG_RE.match(raw)
    if not match:
        return False

    closing = bool(match.group(1))
    name = match.group(2).lower()
    if name in {"think", "thinking", "thought", "antthinking"}:
        state.thinking = not closing
    elif name == "final":
        state.seen_final_tag = True
        state.final = not closing
    return True


def strip_block_tags(
    text: str,
    state: BlockState,
    *,
    reasoning_mode: ReasoningMode = "off",
    enforce_final: bool = False,
) -> tuple[str, str]:
    """Strip ``think`` and ``final`` block tags from a streamed chunk.

    Tags may span chunks.  Tags inside inline code or fenced code are treated
    as literal text.  Thinking content is never returned as visible text; in
    ``reasoning_mode="stream"`` it is returned separately as reasoning text.
    """
    visible: list[str] = []
    reasoning: list[str] = []

    for char in text:
        if state.tag_buffer:
            state.tag_buffer += char
            if char == ">":
                raw_tag = state.tag_buffer
                state.tag_buffer = ""
                if not _process_tag(raw_tag, state) and _is_visible(state, enforce_final):
                    visible.append(raw_tag)
                continue
            if len(state.tag_buffer) > 64 or not _is_possible_tag_prefix(state.tag_buffer):
                raw = state.tag_buffer
                state.tag_buffer = ""
                if _is_visible(state, enforce_final):
                    visible.append(raw)
                elif state.thinking and reasoning_mode == "stream":
                    reasoning.append(raw)
                continue
            continue

        if char == "`":
            state.inline_code.backtick_run += 1
            if _is_visible(state, enforce_final):
                visible.append(char)
            elif state.thinking and reasoning_mode == "stream":
                reasoning.append(char)
            continue

        _finalize_backticks(state)

        if char == "<" and not state.inline_code.in_code:
            state.tag_buffer = char
            continue

        if _is_visible(state, enforce_final):
            visible.append(char)
        elif state.thinking and reasoning_mode == "stream":
            reasoning.append(char)

    return "".join(visible), "".join(reasoning)


def flush_pending_tag_text(
    state: BlockState,
    *,
    reasoning_mode: ReasoningMode = "off",
    enforce_final: bool = False,
) -> tuple[str, str]:
    """Flush any non-closed tag candidate at a boundary as literal text."""
    _finalize_backticks(state)
    if not state.tag_buffer:
        return "", ""
    raw = state.tag_buffer
    state.tag_buffer = ""
    if _is_visible(state, enforce_final):
        return raw, ""
    if state.thinking and reasoning_mode == "stream":
        return "", raw
    return "", ""


def parse_reply_directives(text: str) -> tuple[str, ReplyDirectives]:
    """Remove reply directives from text and return the parsed metadata."""
    directives = ReplyDirectives()

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1).lower()
        if token == "reply_to_current":
            directives.reply_to_current = True
        elif token.startswith("media:"):
            value = match.group(2)
            if value:
                directives.media_urls.append(value.strip())
        elif token.startswith("audio:"):
            value = match.group(3)
            if value:
                directives.audio_urls.append(value.strip())
        return ""

    return _REPLY_DIRECTIVE_RE.sub(_replace, text), directives


@dataclass
class AgentSubscription:
    """State machine for assistant stream, tool, and compaction events."""

    reasoning_mode: ReasoningMode = "off"
    block_reply_break: BlockReplyBreak = "text_end"
    enforce_final_tag: bool = False
    on_partial_reply: PartialReplyCallback | None = None
    on_block_reply: BlockReplyCallback | None = None
    on_reasoning_delta: ReasoningCallback | None = None
    raw_stream_callback: RawStreamCallback | None = None

    assistant_texts: list[str] = field(default_factory=list)
    tool_metas: list[dict[str, Any]] = field(default_factory=list)
    delta_buffer: str = ""
    block_buffer: str = ""
    block_state: BlockState = field(default_factory=BlockState)
    compaction_in_flight: bool = False
    pending_compaction_retry: int = 0
    messaging_pending: bool = False
    reply_directives: ReplyDirectives = field(default_factory=ReplyDirectives)

    _visible_text: str = ""
    _public_text: str = ""

    def handle_event(self, event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Dispatch a provider/session event to the matching handler."""
        event_type = _event_type(event)
        if event_type == "message_start":
            self.handle_message_start(event)
            return SubscriptionUpdate(full_visible_text=self._public_text)
        if event_type == "message_update":
            return self.handle_message_update(event)
        if event_type == "message_end":
            return self.handle_message_end(event)
        if event_type == "tool_execution_start":
            return self.handle_tool_execution_start(event)
        if event_type == "tool_execution_update":
            return self.handle_tool_execution_update(event)
        if event_type == "tool_execution_end":
            return self.handle_tool_execution_end(event)
        if event_type == "agent_start":
            return self.handle_agent_start(event)
        if event_type == "agent_end":
            return self.handle_agent_end(event)
        if event_type == "auto_compaction_start":
            return self.handle_auto_compaction_start(event)
        if event_type == "auto_compaction_end":
            return self.handle_auto_compaction_end(event)
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def handle_agent_start(self, _event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Begin an agent run."""
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def handle_agent_end(self, _event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """End an agent run and force any buffered block reply out."""
        flushed = self.drain_block(force=True)
        return SubscriptionUpdate(full_visible_text=self._public_text, flushed_blocks=flushed)

    def handle_message_start(self, _event: Mapping[str, Any] | str) -> None:
        """Reset per-message text state."""
        self.delta_buffer = ""
        self.block_buffer = ""
        self.block_state = BlockState()
        self.reply_directives = ReplyDirectives()
        self._visible_text = ""
        self._public_text = ""

    def handle_message_update(self, event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Process a streamed assistant text update."""
        raw_delta, update_kind = self._reconcile_update(event)
        if raw_delta:
            self.delta_buffer += raw_delta
            self._write_raw_stream(event, raw_delta)

        visible_delta, reasoning_delta = strip_block_tags(
            raw_delta,
            self.block_state,
            reasoning_mode=self.reasoning_mode,
            enforce_final=self.enforce_final_tag,
        )
        return self._commit_visible_delta(
            visible_delta,
            reasoning_delta=reasoning_delta,
            flush_boundary=update_kind == "text_end",
            raw_delta=raw_delta,
        )

    def handle_message_end(self, _event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Finalize the current assistant message."""
        visible_delta, reasoning_delta = flush_pending_tag_text(
            self.block_state,
            reasoning_mode=self.reasoning_mode,
            enforce_final=self.enforce_final_tag,
        )
        update = self._commit_visible_delta(
            visible_delta,
            reasoning_delta=reasoning_delta,
            flush_boundary=self.block_reply_break == "message_end",
        )
        if self._public_text:
            self.assistant_texts.append(self._public_text)
        return update

    def handle_tool_execution_start(self, event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Flush visible assistant text before a tool starts."""
        if isinstance(event, Mapping):
            name = str(event.get("tool_name") or event.get("name") or "")
            tool_id = str(event.get("tool_call_id") or event.get("id") or "")
            self.tool_metas.append({"id": tool_id, "name": name, "phase": "start"})
            if name in {"message", "sessions_send", "discord_upload_file"}:
                self.messaging_pending = True
        flushed = self.drain_block(force=True)
        return SubscriptionUpdate(full_visible_text=self._public_text, flushed_blocks=flushed)

    def handle_tool_execution_update(self, _event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Accept a tool progress update."""
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def handle_tool_execution_end(self, event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Record tool completion and clear pending messaging state."""
        if isinstance(event, Mapping):
            name = str(event.get("tool_name") or event.get("name") or "")
            tool_id = str(event.get("tool_call_id") or event.get("id") or "")
            is_error = bool(event.get("is_error") or event.get("error"))
            self.tool_metas.append(
                {"id": tool_id, "name": name, "phase": "end", "is_error": is_error}
            )
        self.messaging_pending = False
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def handle_auto_compaction_start(self, _event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Mark compaction as active."""
        self.compaction_in_flight = True
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def handle_auto_compaction_end(self, event: Mapping[str, Any] | str) -> SubscriptionUpdate:
        """Handle compaction completion and retry bookkeeping."""
        will_retry = isinstance(event, Mapping) and bool(
            event.get("will_retry") or event.get("willRetry")
        )
        self.compaction_in_flight = False
        if will_retry:
            self.pending_compaction_retry += 1
            self.handle_message_start({"type": "message_start"})
        elif self.pending_compaction_retry > 0:
            self.pending_compaction_retry -= 1
        return SubscriptionUpdate(full_visible_text=self._public_text)

    def drain_block(self, *, force: bool = False) -> list[str]:
        """Drain buffered block-reply text."""
        if not self.block_buffer:
            return []
        if not force:
            return []
        text = self.block_buffer
        self.block_buffer = ""
        if self.on_block_reply is not None:
            self._safe_callback(self.on_block_reply, text)
        return [text]

    def has_buffered(self) -> bool:
        """Return whether block text is waiting to be flushed."""
        return bool(self.block_buffer)

    def is_compacting(self) -> bool:
        """Return whether compaction is active or retry work is pending."""
        return self.compaction_in_flight or self.pending_compaction_retry > 0

    @property
    def visible_text(self) -> str:
        """Return user-visible assistant text with directives removed."""
        return self._public_text

    def _commit_visible_delta(
        self,
        visible_delta: str,
        *,
        reasoning_delta: str = "",
        flush_boundary: bool = False,
        raw_delta: str = "",
    ) -> SubscriptionUpdate:
        self._visible_text += visible_delta
        clean_text, directives = parse_reply_directives(self._visible_text)
        self.reply_directives.merge(directives)

        if clean_text.startswith(self._public_text):
            public_delta = clean_text[len(self._public_text) :]
        elif clean_text == self._public_text:
            public_delta = ""
        else:
            public_delta = clean_text

        self._public_text = clean_text

        if public_delta:
            self.block_buffer += public_delta
            if self.on_partial_reply is not None:
                self._safe_callback(self.on_partial_reply, public_delta, self._public_text)

        if reasoning_delta and self.reasoning_mode == "stream" and self.on_reasoning_delta:
            self._safe_callback(self.on_reasoning_delta, reasoning_delta)

        flushed: list[str] = []
        if flush_boundary and self.block_reply_break == "text_end":
            flushed = self.drain_block(force=True)

        return SubscriptionUpdate(
            raw_delta=raw_delta,
            visible_delta=public_delta,
            full_visible_text=self._public_text,
            reasoning_delta=reasoning_delta,
            flushed_blocks=flushed,
            directives=self.reply_directives,
        )

    def _reconcile_update(self, event: Mapping[str, Any] | str) -> tuple[str, str]:
        if isinstance(event, str):
            return event, "text_delta"

        update_kind = str(
            event.get("stream_event")
            or event.get("delta_type")
            or event.get("kind")
            or event.get("phase")
            or "text_delta"
        )

        for key in ("delta", "text_delta", "chunk"):
            if key in event and event[key] is not None:
                return str(event[key]), update_kind

        full_content = self._extract_full_content(event)
        if full_content is None:
            return "", update_kind

        if full_content.startswith(self.delta_buffer):
            return full_content[len(self.delta_buffer) :], update_kind
        if self.delta_buffer.startswith(full_content) or full_content in self.delta_buffer:
            return "", update_kind
        return full_content, update_kind

    @staticmethod
    def _extract_full_content(event: Mapping[str, Any]) -> str | None:
        for key in ("content", "text", "full_content", "fullContent"):
            if key in event and event[key] is not None:
                return str(event[key])

        message = event.get("message")
        if isinstance(message, Mapping):
            for key in ("content", "text"):
                if key in message and message[key] is not None:
                    return str(message[key])
        return None

    def _write_raw_stream(self, event: Mapping[str, Any] | str, raw_delta: str) -> None:
        if self.raw_stream_callback is None or not isinstance(event, Mapping):
            return
        try:
            self.raw_stream_callback(event, raw_delta, self.delta_buffer)
        except Exception:
            logger.debug("Raw stream callback failed", exc_info=True)

    @staticmethod
    def _safe_callback(callback: Callable[..., None], *args: Any) -> None:
        try:
            callback(*args)
        except Exception:
            logger.debug("Agent subscription callback failed", exc_info=True)
