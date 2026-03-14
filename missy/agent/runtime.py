"""Agent runtime for the Missy framework.

:class:`AgentRuntime` is the top-level orchestrator that binds a provider,
a session, and a tool registry into a single synchronous run loop.  Each
call to :meth:`AgentRuntime.run` creates (or reuses) a session, resolves
the configured provider, builds the message list, calls the provider, and
returns the model's reply as a plain string.

When tools are registered and ``max_iterations > 1``, the runtime enters a
multi-step agentic loop: tool calls requested by the model are executed and
their results fed back as messages until the model produces a final text
response or the iteration limit is reached.

Example::

    from missy.agent.runtime import AgentRuntime, AgentConfig
    from missy.config.settings import load_config
    from missy.policy.engine import init_policy_engine
    from missy.providers.registry import init_registry

    config = load_config("missy.yaml")
    init_policy_engine(config)
    init_registry(config)

    agent = AgentRuntime(AgentConfig(provider="anthropic"))
    reply = agent.run("What is 2 + 2?")
    print(reply)
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError
from missy.core.session import Session, SessionManager
from missy.providers.base import CompletionResponse, Message, ToolCall, ToolResult
from missy.providers.registry import get_registry
from missy.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a single :class:`AgentRuntime` instance.

    Attributes:
        provider: Registry key of the provider to use.  Defaults to
            ``"anthropic"``.  When the named provider is unavailable the
            runtime falls back to the first available provider.
        model: Optional model override forwarded to the provider's
            :meth:`~missy.providers.base.BaseProvider.complete` call.
            When ``None`` the provider's own default is used.
        system_prompt: System-level instruction prepended to every
            conversation.
        max_iterations: Maximum number of provider calls per
            :meth:`~AgentRuntime.run` invocation.  Set to ``1`` for
            single-turn mode (no tool loop).
        temperature: Sampling temperature forwarded to the provider.
    """

    provider: str = "anthropic"
    model: str | None = None
    system_prompt: str = (
        "You are Missy, a helpful local agentic assistant running on Linux. "
        "You have access to tools and MUST use them to complete tasks. "
        "CRITICAL: Never say 'I will now read...' or 'I am going to...' — "
        "just call the tool immediately. Do not narrate what you plan to do. "
        "Do not describe actions you intend to take. Take the action. "
        "If a task requires reading a file, call file_read now. "
        "If it requires running a command, call shell_exec now. "
        "Available tools: file_read, file_write, file_delete, list_files, "
        "shell_exec, web_fetch, calculator, self_create_tool, code_evolve. "
        "CODE SELF-EVOLUTION: When you encounter a bug or error in your own "
        "source code (missy/), you can fix yourself. Workflow: "
        "1) file_read the source to understand the code "
        "2) code_evolve(action='propose', file_path=..., original_code=..., proposed_code=...) "
        "3) code_evolve(action='approve', proposal_id=...) "
        "4) code_evolve(action='apply', proposal_id=...) — runs tests, commits, restarts. "
        "Use code_evolve(action='rollback', proposal_id=...) to undo. "
        "BROWSER: Use browser_navigate(url=...) for ALL web tasks — never use x11 tools for the browser. "
        "browser_navigate launches Firefox automatically and navigates in one call. "
        "To search: browser_navigate(url='https://duckduckgo.com/?q=query'). "
        "browser_click/browser_fill use page selectors, not screen coordinates. "
        "Tools: browser_navigate, browser_click, browser_fill, browser_screenshot, "
        "browser_get_content, browser_evaluate, browser_wait, browser_get_url, browser_close. "
        "GTK app tools (more reliable than coordinate clicking): "
        "atspi_get_tree (inspect app UI), atspi_click, atspi_get_text, atspi_set_value. "
        "X11 desktop tools available: x11_screenshot (capture screen), "
        "x11_read_screen (screenshot + AI vision interpretation), "
        "x11_click (click at coordinates), x11_type (type text), "
        "x11_key (send keypress), x11_window_list (list open windows). "
        "Use x11_read_screen to verify UI state after actions. "
        "Always use tools to get real information rather than guessing or describing."
    )
    max_iterations: int = 10
    temperature: float = 0.7
    capability_mode: str = "full"  # "full" | "safe-chat" | "no-tools"
    max_spend_usd: float = 0.0  # 0 = unlimited; per-session cost cap


class AgentRuntime:
    """Synchronous agent runtime that resolves a provider and runs completions.

    A :class:`~missy.core.session.SessionManager` is owned by each runtime
    instance.  Sessions are created per :meth:`run` call unless a
    *session_id* is passed in (in which case the existing session is reused
    if it still exists, or a new one is created).

    When the tool registry is initialised and ``config.max_iterations > 1``,
    the runtime executes a multi-step agentic loop:

    1. Call the provider with tool schemas attached.
    2. If the model requests tool calls, execute them via the registry and
       feed results back as messages.
    3. Inject a verification prompt after each round of tool results.
    4. Repeat until the model produces a final text response or
       ``max_iterations`` is reached.

    All subsystems (memory store, context manager, circuit breaker, learnings)
    are loaded lazily and fail gracefully so that the existing test suite is
    not disrupted.

    Args:
        config: Runtime configuration.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._session_mgr = SessionManager()
        # Circuit breaker per runtime instance (keyed to provider name)
        self._circuit_breaker = self._make_circuit_breaker(config.provider)
        # Rate limiter for API calls
        self._rate_limiter = self._make_rate_limiter()
        # Lazy-loaded subsystems
        self._context_manager = self._make_context_manager()
        self._memory_store = self._make_memory_store()
        # Cost tracking (graceful degradation)
        self._cost_tracker = self._make_cost_tracker()
        # Scan for incomplete checkpoints from previous runs
        self._pending_recovery: list = self._scan_checkpoints()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str, session_id: str | None = None) -> str:
        """Run the agent with *user_input* and return the response string.

        Execution steps:

        1. Resolve or create the active :class:`~missy.core.session.Session`.
        2. Emit a ``"agent.run.start"`` audit event.
        3. Resolve the configured provider (with fallback to any available
           provider).
        4. Load session history from the memory store (if available).
        5. Build the message list via :class:`~missy.agent.context.ContextManager`.
        6. If tools are registered and ``max_iterations > 1``: enter the
           multi-step agentic loop.
        7. Otherwise: single-turn :meth:`~missy.providers.base.BaseProvider.complete`.
        8. Persist turn to the memory store (if available).
        9. Extract learnings (if tool calls were made).
        10. Emit a ``"agent.run.complete"`` or ``"agent.run.error"`` audit event.
        11. Return the final response text.

        Args:
            user_input: The user's message text.
            session_id: Optional existing session ID.  When provided and a
                session with that ID is not already active on the current
                thread a new session is created (the ID is stored in its
                metadata for traceability).

        Returns:
            The model's reply as a plain string.

        Raises:
            ProviderError: When no provider is available or the provider
                call fails.
        """
        session = self._resolve_session(session_id)
        sid = str(session.id)
        task_id = str(self._session_mgr.generate_task_id())

        self._emit_event(
            session_id=sid,
            task_id=task_id,
            event_type="agent.run.start",
            result="allow",
            detail={"user_input_length": len(user_input)},
        )

        try:
            provider = self._get_provider()
        except ProviderError as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={"error": str(exc), "stage": "provider_resolution"},
            )
            raise

        # Load history from memory store
        history = self._load_history(sid)

        # Build context-managed messages
        system_prompt, messages = self._build_context_messages(user_input, history)

        # Attempt agentic tool loop; fall back to single-turn on any issue
        all_tool_names_used: list[str] = []
        try:
            final_response, all_tool_names_used = self._run_loop(
                provider=provider,
                system_prompt=system_prompt,
                messages=messages,
                session_id=sid,
                task_id=task_id,
                user_input=user_input,
            )
        except ProviderError as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={
                    "error": str(exc),
                    "stage": "completion",
                    "provider": provider.name,
                },
            )
            raise
        except Exception as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={
                    "error": str(exc),
                    "stage": "completion",
                    "provider": provider.name,
                },
            )
            raise ProviderError(f"Unexpected error during completion: {exc}") from exc

        # Persist turn
        self._save_turn(sid, "user", user_input)
        self._save_turn(sid, "assistant", final_response, provider=provider.name)

        # Extract learnings from tool-augmented runs
        if all_tool_names_used:
            self._record_learnings(all_tool_names_used, final_response, user_input)

        cost_detail = {}
        if self._cost_tracker is not None:
            with contextlib.suppress(Exception):
                cost_detail = self._cost_tracker.get_summary()

        self._emit_event(
            session_id=sid,
            task_id=task_id,
            event_type="agent.run.complete",
            result="allow",
            detail={
                "provider": provider.name,
                "tools_used": all_tool_names_used,
                **cost_detail,
            },
        )

        return final_response

    def run_stream(self, user_input: str, session_id: str | None = None):
        """Stream the response token-by-token for real-time CLI output.

        For tool-calling loops, this falls back to ``run()`` and yields the
        full response as a single chunk (streaming during multi-step tool
        calls is not practical since tool results must be processed first).

        For single-turn completions, yields text deltas from the provider's
        ``stream()`` method.

        Args:
            user_input: The user's message text.
            session_id: Optional existing session ID.

        Yields:
            String chunks of the model's response.
        """
        session = self._resolve_session(session_id)
        sid = str(session.id)

        try:
            provider = self._get_provider()
        except ProviderError:
            raise

        # If tools are available, use the full run loop (no streaming mid-tool-call)
        tools = self._get_tools()
        if tools and self.config.max_iterations > 1:
            result = self.run(user_input, session_id=session_id)
            yield result
            return

        # Single-turn streaming
        history = self._load_history(sid)
        system_prompt, messages = self._build_context_messages(user_input, history)
        msg_objects = self._dicts_to_messages(system_prompt, messages)

        self._acquire_rate_limit()

        full_text = ""
        try:
            for chunk in provider.stream(msg_objects, system=system_prompt):
                full_text += chunk
                yield chunk
        except Exception:
            # Fall back to non-streaming
            response = self._single_turn(
                provider=provider,
                system_prompt=system_prompt,
                messages=messages,
                session_id=sid,
                task_id=str(self._session_mgr.generate_task_id()),
            )
            yield response.content
            full_text = response.content

        # Persist turns
        self._save_turn(sid, "user", user_input)
        self._save_turn(sid, "assistant", full_text, provider=provider.name)

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        provider,
        system_prompt: str,
        messages: list[dict],
        session_id: str,
        task_id: str,
        user_input: str = "",
    ) -> tuple[str, list[str]]:
        """Execute the multi-step provider loop.

        Wraps the provider in the circuit breaker.  If tools are available
        and ``max_iterations > 1``, runs the tool-call loop; otherwise
        falls back to a single-turn completion.

        Args:
            provider: The resolved :class:`~missy.providers.base.BaseProvider`.
            system_prompt: Enriched system prompt string.
            messages: Initial message list (user turn last).
            session_id: Session ID for kwargs forwarding.
            task_id: Task ID for kwargs forwarding.
            user_input: Original user prompt, forwarded to the tool loop for
                checkpointing.

        Returns:
            A 2-tuple of ``(final_response_text, list_of_tool_names_used)``.
        """
        tools = self._get_tools()
        use_tool_loop = bool(tools) and self.config.max_iterations > 1

        if use_tool_loop:
            return self._tool_loop(
                provider=provider,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                session_id=session_id,
                task_id=task_id,
                user_input=user_input,
            )
        else:
            response = self._single_turn(
                provider=provider,
                system_prompt=system_prompt,
                messages=messages,
                session_id=session_id,
                task_id=task_id,
            )
            return response.content, []

    def _tool_loop(
        self,
        provider,
        system_prompt: str,
        messages: list[dict],
        tools: list,
        session_id: str,
        task_id: str,
        user_input: str = "",
    ) -> tuple[str, list[str]]:
        """Inner agentic tool-call loop.

        Iterates up to ``max_iterations`` times.  Each iteration:

        1. Calls ``provider.complete_with_tools()`` via the circuit breaker.
        2. If the model requests tool calls, executes them and appends
           results as messages.
        3. Injects a verification prompt so the model can assess outcomes.
        4. After each round of tool results, saves a checkpoint and checks
           for repeated tool failures (strategy rotation).
        5. If the model returns a final text response, exits.

        Falls back to single-turn :meth:`complete` if
        ``complete_with_tools`` is not implemented or fails at the protocol
        level.

        Args:
            provider: The resolved provider.
            system_prompt: Enriched system prompt.
            messages: Starting message list.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            session_id: For audit events.
            task_id: For audit events.
            user_input: Original user prompt, used for checkpointing.

        Returns:
            A 2-tuple of ``(final_response_text, list_of_tool_names_used)``.
        """
        from missy.agent.done_criteria import make_verification_prompt

        # --- Feature #7: failure tracker (graceful degradation) ---
        try:
            from missy.agent.failure_tracker import FailureTracker as _FailureTracker

            failure_tracker: _FailureTracker | None = _FailureTracker(threshold=3)
        except ImportError:
            failure_tracker = None

        # --- Feature #8: checkpoint manager (graceful degradation) ---
        _cm = None
        _checkpoint_id = None
        try:
            from missy.agent.checkpoint import CheckpointManager as _CheckpointManager

            _cm = _CheckpointManager()
            _checkpoint_id = _cm.create(session_id, task_id, user_input)
        except Exception:
            _cm = None
            _checkpoint_id = None

        tool_names_used: list[str] = []
        # Mutable message list for the loop; starts from what context manager gave us
        loop_messages: list[dict] = list(messages)

        try:
            for iteration in range(self.config.max_iterations):
                provider_messages = self._dicts_to_messages(system_prompt, loop_messages)

                # Rate limit before calling provider
                self._acquire_rate_limit()

                try:
                    response: CompletionResponse = self._circuit_breaker.call(
                        provider.complete_with_tools,
                        provider_messages,
                        tools,
                        system_prompt,
                    )
                except AttributeError:
                    # Provider doesn't implement complete_with_tools; fall back
                    logger.debug(
                        "Provider %r does not implement complete_with_tools; using complete()",
                        provider.name,
                    )
                    fallback = self._single_turn(
                        provider=provider,
                        system_prompt=system_prompt,
                        messages=loop_messages,
                        session_id=session_id,
                        task_id=task_id,
                    )
                    return fallback.content, tool_names_used

                # Record cost and enforce budget
                self._record_cost(response, session_id=session_id)
                self._check_budget(session_id=session_id, task_id=task_id)

                if response.finish_reason == "tool_calls" and response.tool_calls:
                    # Execute each tool call
                    tool_results: list[ToolResult] = []
                    for tc in response.tool_calls:
                        tool_names_used.append(tc.name)
                        tr = self._execute_tool(tc, session_id=session_id, task_id=task_id)
                        tool_results.append(tr)

                        # Feature #7: track failures / successes per tool
                        if failure_tracker is not None:
                            if tr.is_error:
                                should_inject = failure_tracker.record_failure(tc.name, tr.content)
                            else:
                                failure_tracker.record_success(tc.name)
                                should_inject = False
                        else:
                            should_inject = False

                    # Append assistant message with tool_calls to loop history
                    loop_messages.append(
                        {
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                                for tc in response.tool_calls
                            ],
                        }
                    )

                    # Append tool result messages
                    for tr in tool_results:
                        loop_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr.tool_call_id,
                                "name": tr.name,
                                "content": tr.content,
                                "is_error": tr.is_error,
                            }
                        )

                    # Feature #7: inject strategy-rotation prompt when threshold hit
                    if should_inject and failure_tracker is not None:
                        # Use the last tc/tr from the loop (they stay in scope)
                        strategy_prompt = failure_tracker.get_strategy_prompt(tc.name, tr.content)
                        loop_messages.append({"role": "user", "content": strategy_prompt})
                        with contextlib.suppress(Exception):
                            self._emit_event(
                                session_id=session_id,
                                task_id=task_id,
                                event_type="agent.tool.strategy_rotation",
                                result="allow",
                                detail={
                                    "tool_name": tc.name,
                                    "failure_count": failure_tracker.threshold,
                                },
                            )

                        # Feature #9: error-driven code evolution analysis
                        self._analyze_for_evolution(tc.name, tr.content, failure_tracker)

                    # Feature #8: checkpoint after each round of tool results
                    if _cm is not None and _checkpoint_id is not None:
                        with contextlib.suppress(Exception):
                            _cm.update(_checkpoint_id, loop_messages, tool_names_used, iteration)

                    # Inject verification prompt
                    verification = make_verification_prompt()
                    loop_messages.append({"role": "user", "content": verification})

                    # Continue loop to get model's next response
                    continue

                # finish_reason == "stop" or "length": we have a final response
                final_text = response.content or ""
                logger.debug(
                    "Tool loop completed after %d iteration(s); finish_reason=%r",
                    iteration + 1,
                    response.finish_reason,
                )
                # Feature #8: mark checkpoint complete on success
                if _cm is not None and _checkpoint_id is not None:
                    with contextlib.suppress(Exception):
                        _cm.complete(_checkpoint_id)
                return final_text, tool_names_used

        except Exception:
            # Feature #8: mark checkpoint failed on unhandled exception
            if _cm is not None and _checkpoint_id is not None:
                with contextlib.suppress(Exception):
                    _cm.fail(_checkpoint_id)
            raise

        # Iteration limit reached: return whatever content we have
        logger.warning(
            "Agent hit max_iterations=%d without a final stop response.",
            self.config.max_iterations,
        )
        # Make one final plain completion attempt
        try:
            fallback = self._single_turn(
                provider=provider,
                system_prompt=system_prompt,
                messages=loop_messages,
                session_id=session_id,
                task_id=task_id,
            )
            return fallback.content, tool_names_used
        except Exception:
            return "[Agent reached iteration limit without a final response.]", tool_names_used

    def _single_turn(
        self,
        provider,
        system_prompt: str,
        messages: list[dict],
        session_id: str,
        task_id: str,
    ) -> CompletionResponse:
        """Execute a single provider.complete() call via the circuit breaker.

        Args:
            provider: The resolved provider.
            system_prompt: Enriched system prompt string.
            messages: Message list (user turn last).
            session_id: For provider kwargs.
            task_id: For provider kwargs.

        Returns:
            A :class:`~missy.providers.base.CompletionResponse`.
        """
        msg_objects = self._dicts_to_messages(system_prompt, messages)
        complete_kwargs: dict = {
            "session_id": session_id,
            "task_id": task_id,
            "temperature": self.config.temperature,
        }
        if self.config.model:
            complete_kwargs["model"] = self.config.model

        # Rate limit before calling provider
        self._acquire_rate_limit()

        result = self._circuit_breaker.call(
            provider.complete,
            msg_objects,
            **complete_kwargs,
        )
        self._record_cost(result, session_id=session_id)
        return result

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    # Safe tools allowed in safe-chat mode (read-only, no side effects)
    _SAFE_CHAT_TOOLS = frozenset(
        {
            "calculator",
            "file_read",
            "list_files",
            "web_fetch",
            "browser_get_content",
            "browser_get_url",
            "browser_screenshot",
            "x11_screenshot",
            "x11_window_list",
            "atspi_get_tree",
            "atspi_get_text",
        }
    )

    def _get_tools(self) -> list:
        """Return registered tools, or an empty list when unavailable.

        Respects :attr:`AgentConfig.capability_mode`:

        - ``"full"``: all registered tools
        - ``"safe-chat"``: only read-only/safe tools
        - ``"no-tools"``: empty list (pure chat)

        Returns:
            A list of :class:`~missy.tools.base.BaseTool` instances, or
            ``[]`` when the registry is not initialised.
        """
        if self.config.capability_mode == "no-tools":
            return []

        try:
            registry = get_tool_registry()
            tool_names = registry.list_tools()
            tools = [registry.get(name) for name in tool_names if registry.get(name) is not None]
        except RuntimeError:
            return []

        if self.config.capability_mode == "safe-chat":
            tools = [t for t in tools if getattr(t, "name", "") in self._SAFE_CHAT_TOOLS]

        return tools

    def _execute_tool(
        self, tool_call: ToolCall, session_id: str = "", task_id: str = ""
    ) -> ToolResult:
        """Execute a single tool call via the tool registry.

        Args:
            tool_call: The :class:`~missy.providers.base.ToolCall` to execute.
            session_id: For audit events.
            task_id: For audit events.

        Returns:
            A :class:`~missy.providers.base.ToolResult` with the outcome.
        """
        try:
            registry = get_tool_registry()
            logger.info(
                "Executing tool %r with args: %s",
                tool_call.name,
                tool_call.arguments,
            )
            # Strip session_id/task_id from tool args to avoid colliding
            # with the explicit kwargs we pass to registry.execute().
            tool_args = {
                k: v for k, v in tool_call.arguments.items() if k not in ("session_id", "task_id")
            }
            result = registry.execute(
                tool_call.name,
                session_id=session_id,
                task_id=task_id,
                **tool_args,
            )
            content = str(result.output) if result.output is not None else ""
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content if result.success else (result.error or "Tool failed"),
                is_error=not result.success,
            )
        except KeyError as exc:
            logger.warning("Tool %r not found in registry: %s", tool_call.name, exc)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Tool not found: {tool_call.name}",
                is_error=True,
            )
        except RuntimeError as exc:
            logger.warning("Tool registry not available: %s", exc)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="Tool registry not initialised.",
                is_error=True,
            )
        except Exception as exc:
            logger.exception("Unexpected error executing tool %r", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Unexpected error: {exc}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Context / memory helpers
    # ------------------------------------------------------------------

    def _build_context_messages(
        self, user_input: str, history: list[dict]
    ) -> tuple[str, list[dict]]:
        """Assemble the system prompt and message list via ContextManager.

        Falls back to a minimal system + user message when the context
        manager is unavailable.

        Args:
            user_input: The new user input text.
            history: Loaded history dicts from the memory store.

        Returns:
            A 2-tuple of ``(system_prompt_str, messages_list)``.
        """
        if self._context_manager is not None:
            try:
                return self._context_manager.build_messages(
                    system=self.config.system_prompt,
                    new_message=user_input,
                    history=history,
                )
            except Exception as exc:
                logger.debug("ContextManager.build_messages failed: %s", exc)

        # Minimal fallback
        return self.config.system_prompt, [{"role": "user", "content": user_input}]

    def _load_history(self, session_id: str) -> list[dict]:
        """Load conversation history from the memory store.

        Args:
            session_id: Session ID to load history for.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts, or ``[]``
            when the memory store is unavailable.
        """
        if self._memory_store is None:
            return []
        try:
            turns = self._memory_store.get_session_turns(session_id, limit=50)
            return [{"role": t.role, "content": t.content} for t in turns]
        except Exception as exc:
            logger.debug("Failed to load history from memory store: %s", exc)
            return []

    def _save_turn(self, session_id: str, role: str, content: str, provider: str = "") -> None:
        """Persist a conversation turn to the memory store.

        Args:
            session_id: Session to write to.
            role: Speaker role (``"user"`` or ``"assistant"``).
            content: Message content.
            provider: Provider name (for assistant turns).
        """
        if self._memory_store is None:
            return
        try:
            self._memory_store.add_turn(
                session_id=session_id,
                role=role,
                content=content,
                provider=provider,
            )
        except Exception as exc:
            logger.debug("Failed to save turn to memory store: %s", exc)

    def _record_learnings(
        self,
        tool_names_used: list[str],
        final_response: str,
        prompt: str,
    ) -> None:
        """Extract and log learnings from a completed tool-augmented run.

        Args:
            tool_names_used: All tool names invoked during the run.
            final_response: The final assistant response text.
            prompt: The original user prompt.
        """
        try:
            from missy.agent.learnings import extract_learnings

            learning = extract_learnings(
                tool_names_used=tool_names_used,
                final_response=final_response,
                prompt=prompt,
            )
            logger.debug(
                "Task learning: type=%r outcome=%r lesson=%r",
                learning.task_type,
                learning.outcome,
                learning.lesson,
            )
        except Exception as exc:
            logger.debug("Failed to extract learnings: %s", exc)

    def _analyze_for_evolution(
        self,
        tool_name: str,
        error_content: str,
        failure_tracker,
    ) -> None:
        """Analyze repeated tool failures for possible code self-evolution.

        When a tool fails repeatedly and the error traceback points to Missy's
        own source code, this method logs a skeleton evolution proposal that
        the agent (or operator) can later flesh out and apply.

        This is a best-effort operation; failures are silently swallowed.

        Args:
            tool_name: Name of the tool that failed.
            error_content: The error message / traceback text.
            failure_tracker: The active
                :class:`~missy.agent.failure_tracker.FailureTracker`.
        """
        try:
            from missy.agent.code_evolution import CodeEvolutionManager

            mgr = CodeEvolutionManager()
            stats = failure_tracker.get_stats()
            total = stats.get(tool_name, {}).get("total_failures", 0)

            skeleton = mgr.analyze_error_for_evolution(
                error_message=error_content[:200],
                traceback_text=error_content,
                tool_name=tool_name,
                failure_count=total,
            )
            if skeleton is not None:
                logger.info(
                    "Code evolution skeleton proposed: %s — %s",
                    skeleton.id,
                    skeleton.title,
                )
                self._emit_event(
                    session_id="system",
                    task_id="code_evolution",
                    event_type="code_evolution.skeleton_proposed",
                    result="allow",
                    detail={
                        "proposal_id": skeleton.id,
                        "title": skeleton.title,
                        "tool_name": tool_name,
                        "failure_count": total,
                    },
                )
        except Exception as exc:
            logger.debug("Code evolution analysis failed: %s", exc)

    # ------------------------------------------------------------------
    # Message format conversion
    # ------------------------------------------------------------------

    def _dicts_to_messages(self, system_prompt: str, message_dicts: list[dict]) -> list[Message]:
        """Convert context-manager message dicts to provider Message objects.

        Prepends the system prompt as a Message with ``role="system"``.
        Skips dict entries with roles not in ``("user", "assistant")`` that
        providers may not understand (e.g. tool result messages).

        Args:
            system_prompt: System prompt string.
            message_dicts: List of message dicts.

        Returns:
            A list of :class:`~missy.providers.base.Message` objects with
            the system prompt first.
        """
        result: list[Message] = [Message(role="system", content=system_prompt)]
        for d in message_dicts:
            role = d.get("role", "")
            content = d.get("content", "")
            if role == "tool":
                # Represent tool results as user messages for providers that
                # don't support native tool_result role
                content_str = f"[Tool result for {d.get('name', 'unknown')}]: {content}"
                if d.get("is_error"):
                    content_str = f"[Tool error for {d.get('name', 'unknown')}]: {content}"
                result.append(Message(role="user", content=content_str))
            elif role in ("user", "assistant"):
                result.append(Message(role=role, content=str(content)))
        return result

    # ------------------------------------------------------------------
    # Lazy subsystem factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_circuit_breaker(provider_name: str):
        """Create a :class:`~missy.agent.circuit_breaker.CircuitBreaker`.

        Args:
            provider_name: Used as the breaker name for logging.

        Returns:
            A :class:`~missy.agent.circuit_breaker.CircuitBreaker` instance,
            or a no-op stub when the module is unavailable.
        """
        try:
            from missy.agent.circuit_breaker import CircuitBreaker

            return CircuitBreaker(name=provider_name)
        except Exception:
            return _NoOpCircuitBreaker()

    @staticmethod
    def _make_context_manager():
        """Create a :class:`~missy.agent.context.ContextManager`.

        Returns:
            A :class:`~missy.agent.context.ContextManager` instance, or
            ``None`` when the module is unavailable.
        """
        try:
            from missy.agent.context import ContextManager

            return ContextManager()
        except Exception:
            return None

    @staticmethod
    def _make_memory_store():
        """Create a :class:`~missy.memory.store.MemoryStore`.

        Returns:
            A :class:`~missy.memory.store.MemoryStore` instance, or ``None``
            when unavailable.
        """
        try:
            from missy.memory.store import MemoryStore

            return MemoryStore()
        except Exception:
            return None

    def _make_cost_tracker(self):
        """Create a :class:`~missy.agent.cost_tracker.CostTracker`.

        Uses ``max_spend_usd`` from the agent config when set.

        Returns:
            A :class:`~missy.agent.cost_tracker.CostTracker` instance, or
            ``None`` when the module is unavailable.
        """
        try:
            from missy.agent.cost_tracker import CostTracker

            return CostTracker(max_spend_usd=self.config.max_spend_usd)
        except Exception:
            return None

    @staticmethod
    def _make_rate_limiter():
        """Create a :class:`~missy.providers.rate_limiter.RateLimiter`.

        Returns:
            A :class:`~missy.providers.rate_limiter.RateLimiter` instance,
            or ``None`` when the module is unavailable.
        """
        try:
            from missy.providers.rate_limiter import RateLimiter

            return RateLimiter(requests_per_minute=60, tokens_per_minute=100_000)
        except Exception:
            return None

    @staticmethod
    def _scan_checkpoints() -> list:
        """Scan for incomplete checkpoints from previous runs.

        Returns a list of :class:`~missy.agent.checkpoint.RecoveryResult`
        objects that the caller (e.g. CLI) can present to the user.
        Fails gracefully — returns an empty list if checkpoint module
        is unavailable or the DB doesn't exist.
        """
        try:
            from missy.agent.checkpoint import scan_for_recovery

            results = scan_for_recovery()
            if results:
                logger.info(
                    "Found %d incomplete checkpoint(s) from previous runs.",
                    len(results),
                )
            return results
        except Exception:
            return []

    @property
    def pending_recovery(self) -> list:
        """Incomplete checkpoints from previous runs.

        Returns a list of :class:`~missy.agent.checkpoint.RecoveryResult`
        objects.  Each has an ``action`` field (``"resume"``, ``"restart"``,
        or ``"abandon"``) indicating the recommended recovery strategy.
        """
        return list(self._pending_recovery)

    def _record_cost(self, response, session_id: str = "") -> None:
        """Record token usage in the cost tracker and persist to SQLite."""
        if self._cost_tracker is None:
            return
        try:
            rec = self._cost_tracker.record_from_response(response)
            # Persist to SQLite for historical cost queries
            if rec is not None and session_id and self._memory_store is not None:
                try:
                    store = self._memory_store
                    # Unwrap resilient store to get at SQLite store
                    if hasattr(store, "_primary"):
                        store = store._primary
                    if hasattr(store, "record_cost"):
                        store.record_cost(
                            session_id=session_id,
                            model=rec.model,
                            prompt_tokens=rec.prompt_tokens,
                            completion_tokens=rec.completion_tokens,
                            cost_usd=rec.cost_usd,
                        )
                except Exception as exc:
                    logger.debug("Failed to persist cost to store: %s", exc)
        except Exception as exc:
            logger.debug("Failed to record cost: %s", exc)

    def _acquire_rate_limit(self) -> None:
        """Block until the rate limiter allows the next API call."""
        if self._rate_limiter is None:
            return
        try:
            self._rate_limiter.acquire()
        except Exception as exc:
            logger.warning("Rate limiter error: %s", exc)

    def _check_budget(self, session_id: str = "", task_id: str = "") -> None:
        """Enforce budget limits after recording cost.

        Raises :class:`~missy.agent.cost_tracker.BudgetExceededError` if the
        accumulated session cost exceeds ``max_spend_usd``.  An audit event
        is emitted before the exception propagates.
        """
        if self._cost_tracker is None:
            return
        try:
            self._cost_tracker.check_budget()
        except Exception:
            # Emit audit event for budget breach
            with contextlib.suppress(Exception):
                self._emit_event(
                    session_id=session_id,
                    task_id=task_id,
                    event_type="agent.budget.exceeded",
                    result="deny",
                    detail=self._cost_tracker.get_summary(),
                )
            raise

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_session(self, session_id: str | None) -> Session:
        """Return the current thread session, creating one if needed.

        Args:
            session_id: Optional caller-supplied identifier.  When provided
                it is stored in the new session's metadata so that audit
                events can be correlated by the caller's own ID scheme.

        Returns:
            The active :class:`~missy.core.session.Session` for this thread.
        """
        existing = self._session_mgr.get_current_session()
        if existing is not None:
            return existing

        metadata: dict = {}
        if session_id is not None:
            metadata["caller_session_id"] = session_id

        return self._session_mgr.create_session(metadata=metadata)

    def _get_provider(self):
        """Resolve the configured provider with automatic fallback.

        Returns:
            A :class:`~missy.providers.base.BaseProvider` instance.

        Raises:
            ProviderError: When no provider is available.
        """
        registry = get_registry()
        provider = registry.get(self.config.provider)

        if provider is not None and provider.is_available():
            return provider

        if provider is not None:
            logger.warning(
                "Configured provider %r is not available; falling back.",
                self.config.provider,
            )

        available = registry.get_available()
        if available:
            fallback = available[0]
            logger.info("Using fallback provider %r.", fallback.name)
            return fallback

        raise ProviderError(
            f"No providers available. Configured provider was {self.config.provider!r}. "
            "Ensure at least one provider is initialised and its API key is set."
        )

    def _build_messages(self, user_input: str) -> list[Message]:
        """Construct the message list to send to the provider.

        This is the legacy single-turn helper retained for backward
        compatibility with code that calls it directly.

        Args:
            user_input: The user's text input for this turn.

        Returns:
            A list of :class:`~missy.providers.base.Message` objects with
            the system prompt first, followed by the user turn.
        """
        return [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=user_input),
        ]

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        event_type: str,
        result: str,
        detail: dict,
    ) -> None:
        """Publish an agent lifecycle audit event.

        Args:
            session_id: Session identifier.
            task_id: Task identifier.
            event_type: Dotted event type string.
            result: One of ``"allow"`` or ``"error"``.
            detail: Structured event data.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type=event_type,
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event %r", event_type)


# ---------------------------------------------------------------------------
# No-op circuit breaker stub for graceful degradation
# ---------------------------------------------------------------------------


class _NoOpCircuitBreaker:
    """Passthrough stub used when the circuit_breaker module is unavailable."""

    def call(self, func, *args, **kwargs):
        return func(*args, **kwargs)
