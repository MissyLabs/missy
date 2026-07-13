"""Shell policy enforcement for the Missy framework.

Shell command execution is evaluated against a :class:`ShellPolicy` instance.
When ``policy.enabled`` is ``False`` every command is denied regardless of the
allow-list.  When enabled, the first token of the command (the program name)
must begin with an entry in ``policy.allowed_commands`` -- unless
``policy.unrestricted`` is ``True``, in which case any non-empty command is
allowed immediately: no allow-list matching (an empty ``allowed_commands`` no
longer denies everything), and no subshell/brace-group/malformed-quoting
rejection either (those checks exist only to protect the allow-list match
from being bypassed via a hidden subcommand -- with no allow-list to protect
in this mode, ``$(...)``, backticks, and brace groups are all permitted too).
``unrestricted`` still requires ``enabled: True``, and does not affect any
other, independent policy layer (e.g. redirect targets still route through
the filesystem policy engine).

Every check emits an :class:`~missy.core.events.AuditEvent`.

Example::

    from missy.config.settings import ShellPolicy
    from missy.policy.shell import ShellPolicyEngine

    policy = ShellPolicy(enabled=True, allowed_commands=["git", "ls"])
    engine = ShellPolicyEngine(policy)
    engine.check_command("git status")   # -> True
    engine.check_command("rm -rf /")     # -> raises PolicyViolationError

    unrestricted_policy = ShellPolicy(enabled=True, unrestricted=True)
    engine = ShellPolicyEngine(unrestricted_policy)
    engine.check_command("rm -rf /tmp/x")          # -> True
    engine.check_command("echo $(whoami)")         # -> True (subshell allowed too)
"""

from __future__ import annotations

import logging
import shlex

from missy.config.settings import ShellPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError

logger = logging.getLogger(__name__)


class ShellPolicyEngine:
    """Evaluates shell commands against a :class:`ShellPolicy`.

    Args:
        policy: The shell policy to enforce.
    """

    def __init__(self, policy: ShellPolicy) -> None:
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_command(
        self,
        command: str,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Return ``True`` if executing *command* is permitted by policy.

        Evaluation order:

        1. If ``policy.enabled`` is ``False``, deny unconditionally.
        2. If ``policy.unrestricted`` is ``True``, allow any non-empty
           command immediately -- no program-name extraction, no
           subshell/brace-group rejection, no allow-list check.
        3. Parse the leading token (program name) from *command* using
           POSIX shell splitting rules.  An empty or whitespace-only command
           string is always denied.
        4. Check whether the program name starts with any entry in
           ``policy.allowed_commands``.  This permits both bare names
           (``"git"``) and path-qualified names (``"/usr/bin/git"``).

        Args:
            command: The shell command string to evaluate.
            session_id: Optional identifier of the calling session.
            task_id: Optional identifier of the calling task.

        Returns:
            ``True`` when the command is explicitly allowed.

        Raises:
            PolicyViolationError: When the command is denied.
        """
        # Step 1 – shell execution globally disabled.
        if not self._policy.enabled:
            self._emit_event(command, "deny", "shell_disabled", session_id, task_id)
            raise PolicyViolationError(
                "Shell execution is disabled by policy.",
                category="shell",
                detail="ShellPolicy.enabled is False; no shell commands are permitted.",
            )

        # Step 2 – fully unrestricted mode: allow any non-empty command
        # immediately, before Step 3's parsing-safety check ever runs.
        #
        # Step 3's subshell/brace-group/malformed-quoting rejection exists
        # ONLY to protect Step 4's allow-list match from being bypassed via
        # a hidden subcommand simple tokenisation can't see into (e.g.
        # `echo $(rm -rf /)` with only "echo" allowlisted). With no
        # allow-list to protect in unrestricted mode (Step 4 never runs
        # either), that protection has nothing left to guard -- keeping it
        # active here would leave `unrestricted: true` still rejecting
        # perfectly ordinary commands that happen to use `$(...)`,
        # backticks, or a brace group, defeating the point of the setting.
        # Still gated on enabled=True (Step 1 above), and still doesn't
        # touch any other, independent policy layer --
        # PolicyEngine.check_shell()'s SR-1.7 redirect-target-to-
        # filesystem-policy routing runs regardless of this flag.
        if self._policy.unrestricted:
            if not command or not command.strip():
                self._emit_event(command, "deny", None, session_id, task_id)
                raise PolicyViolationError(
                    "Shell command denied: empty command.",
                    category="shell",
                    detail="An empty or whitespace-only command has nothing to execute.",
                )
            # Best-effort program-name extraction for launcher-command
            # observability only. Extraction failure (subshell markers,
            # unparseable quoting, etc.) is expected here and must not
            # block execution -- unlike the allow-list-matching path
            # below, nothing downstream depends on these names.
            for program in self._extract_all_programs(command) or []:
                self._warn_if_launcher(program)
            self._emit_event(command, "allow", "unrestricted", session_id, task_id)
            return True

        # Step 3 – extract ALL program tokens (handles compound commands).
        programs = self._extract_all_programs(command)
        if programs is None:
            self._emit_event(command, "deny", None, session_id, task_id)
            raise PolicyViolationError(
                "Shell command denied: empty, unparseable, or contains subshell.",
                category="shell",
                detail=f"Could not determine the program name(s) from command {command!r}.",
            )

        # Step 4 – allow-list check.
        # SR-1.8: enabled=True with an empty allowed_commands list must deny
        # ALL commands, matching ShellPolicy.allowed_commands's own
        # documented contract ("An empty list means no commands are allowed
        # even when enabled is True"). Configuration ambiguity must never
        # become allow-all -- a previous version of this engine inverted
        # that contract and treated an empty list as unrestricted shell
        # access whenever enabled=True, which is exactly backwards. (An
        # operator who actually wants that behavior now has an explicit,
        # auditable way to ask for it: ShellPolicy.unrestricted=True above.)
        if not self._policy.allowed_commands:
            self._emit_event(command, "deny", "empty_allowlist", session_id, task_id)
            raise PolicyViolationError(
                "Shell command denied: allowed_commands is empty.",
                category="shell",
                detail=(
                    "ShellPolicy.enabled is True but allowed_commands is empty -- "
                    "per policy, an empty allowlist denies all commands rather than "
                    "permitting them. Configure allowed_commands explicitly to permit "
                    "specific programs, or set shell.unrestricted: true to skip "
                    "allow-list matching entirely."
                ),
            )

        # Every program in a compound command must be allowed.
        for program in programs:
            rule = self._match_allowed(program)
            if not rule:
                self._emit_event(command, "deny", None, session_id, task_id)
                raise PolicyViolationError(
                    f"Shell command denied: {program!r} is not in the allowed commands list.",
                    category="shell",
                    detail=(
                        f"Program {program!r} (from command {command!r}) did not match "
                        "any allowed_commands entry."
                    ),
                )
            self._warn_if_launcher(program)

        # All programs matched — allow.
        self._emit_event(command, "allow", f"cmd:compound({len(programs)})", session_id, task_id)
        return True

    @classmethod
    def _warn_if_launcher(cls, program: str) -> None:
        """Log a warning if *program* is a command launcher (see ``_LAUNCHER_COMMANDS``)."""
        import os.path

        basename = os.path.basename(program)
        if basename in cls._LAUNCHER_COMMANDS:
            logger.warning(
                "ShellPolicyEngine: %r is a command launcher that can execute "
                "arbitrary subcommands — consider removing it from allowed_commands",
                basename,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Shell metacharacters that can chain additional commands.
    _CHAIN_OPERATORS = ("&&", "||", ";", "|", "&", "\n")
    _SUBSHELL_MARKERS = ("$(", "`", "<(", ">(", "<<(", "<<<", "<<")

    # Brace groups can execute compound statements; reject at the start of
    # a (sub-)command or anywhere a bare ``{`` could introduce a group.
    _BRACE_GROUP_MARKERS = ("{ ", "{;")

    # Commands that can execute arbitrary subcommands — warn when whitelisted.
    _LAUNCHER_COMMANDS = frozenset(
        {
            "env",
            "xargs",
            "find",
            "nice",
            "nohup",
            "strace",
            "ltrace",
            "time",
            "watch",
            "sudo",
            "su",
            "doas",
            "bash",
            "sh",
            "zsh",
            "dash",
            "python",
            "python3",
            "perl",
            "ruby",
            "node",
            "eval",
            "exec",  # bash builtins that execute arbitrary commands
        }
    )

    # SR-1.7: redirection operators that write to / read from a filesystem
    # path. ">&"/"<&" (fd duplication, e.g. "2>&1") are deliberately
    # excluded -- their target is a file descriptor number, not a path.
    _REDIRECT_WRITE_OPS = frozenset({">", ">>", ">|", "&>", "&>>"})
    _REDIRECT_READ_OPS = frozenset({"<", "<>"})

    def extract_redirect_targets(self, command: str) -> tuple[list[str], list[str]]:
        """Return ``(write_targets, read_targets)`` for every redirection
        operator in *command*, across every sub-command of a compound chain.

        SR-1.7: an allowed program name says nothing about what files a
        shell redirection lets it touch — ``echo x > /etc/cron.d/pwn``
        with only ``echo`` allowlisted writes an arbitrary host file with
        no filesystem policy check at all. This extracts every redirect
        target so the caller (:meth:`~missy.policy.engine.PolicyEngine.check_shell`)
        can route them through the filesystem policy engine too.

        Uses POSIX-punctuation-aware tokenisation (``shlex`` with
        ``punctuation_chars=True``) so operators are recognised whether or
        not they're surrounded by whitespace — ``echo x>file`` and
        ``echo x > file`` both correctly yield ``file`` as a write target.
        Content inside quotes is treated as a literal argument and not
        re-scanned for nested operators, matching real shell semantics —
        this deliberately does NOT see redirects hidden inside a quoted
        string handed to a launcher (e.g. ``sh -c 'echo x > file'``); that
        is a distinct, known limitation of static command-string analysis,
        not a gap in this parser specifically (see ``_LAUNCHER_COMMANDS`
        and the module-level residual-risk note).

        Args:
            command: The raw shell command string (as passed to
                :meth:`check_command`).

        Returns:
            A ``(write_targets, read_targets)`` tuple of path strings, in
            the order encountered. Empty lists when the command has no
            redirections or cannot be tokenised (malformed quoting is
            already denied earlier in :meth:`check_command`'s own
            tokenisation, so this treats that case as "nothing to add"
            rather than raising here).
        """
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return ([], [])

        write_targets: list[str] = []
        read_targets: list[str] = []
        for i in range(len(tokens) - 1):
            tok = tokens[i]
            nxt = tokens[i + 1]
            if nxt.startswith("&"):
                # Duplicates a file descriptor (e.g. "2>&1", ">&2") rather
                # than naming a file — not a filesystem target.
                continue
            if tok in self._REDIRECT_WRITE_OPS:
                write_targets.append(nxt)
            elif tok in self._REDIRECT_READ_OPS:
                read_targets.append(nxt)

        return (write_targets, read_targets)

    @staticmethod
    def _extract_program(command: str) -> str | None:
        """Return the program token from a shell command string.

        Uses :func:`shlex.split` for POSIX-correct tokenisation.  Returns
        ``None`` if the command is empty, contains only whitespace, or cannot
        be parsed (e.g. unmatched quotes).

        Args:
            command: Raw shell command string.

        Returns:
            The first token (program name), or ``None``.
        """
        if not command or not command.strip():
            return None
        try:
            tokens = shlex.split(command)
        except ValueError:
            # Malformed quoting – deny.
            logger.debug("ShellPolicyEngine: could not parse command %r", command)
            return None
        return tokens[0] if tokens else None

    @classmethod
    def _extract_all_programs(cls, command: str) -> list[str] | None:
        """Extract ALL program names from a potentially compound command.

        Splits on shell chain operators (&&, ||, ;, |, newline) and returns
        the first token of each sub-command.  Returns ``None`` if the command
        is empty, contains subshell markers ($(...) or backticks), or any
        sub-command cannot be parsed.

        Subshell markers are rejected outright because the inner commands
        cannot be reliably extracted via simple tokenisation.
        """
        if not command or not command.strip():
            return None

        # Reject subshell / command substitution — these can hide arbitrary
        # commands inside $(...) or backticks that simple splitting cannot
        # reliably extract.
        for marker in cls._SUBSHELL_MARKERS:
            if marker in command:
                logger.debug(
                    "ShellPolicyEngine: rejecting command with subshell marker %r: %s",
                    marker,
                    command[:200],
                )
                return None

        # Reject brace groups — ``{ cmd1; cmd2; }`` can execute compound
        # statements that bypass individual command whitelisting.
        # Scan the entire command, not just the start, because brace groups
        # can appear after semicolons: ``echo hi; { rm -rf /; }``
        for marker in cls._BRACE_GROUP_MARKERS:
            if marker in command:
                logger.debug(
                    "ShellPolicyEngine: rejecting command with brace group: %s",
                    command[:200],
                )
                return None

        # Split on chain operators to get individual sub-commands.
        import re

        # Replace chain operators with a unique delimiter, then split.
        # Order matters: && and || must be checked before single & or |.
        # Also split on bare & (background execution) via negative lookahead.
        # Bug fix (found while implementing SR-1.7): a lone "&" preceded by
        # "<" or ">" is a file-descriptor-duplication redirect (2>&1, >&2,
        # <&0), not the background-execution operator -- the previous
        # pattern split "cmd 2>&1" into "cmd 2>" and "1", denying the
        # extremely common ">&1"/"2>&1" idiom outright by misparsing "1" as
        # a fake sub-command's program name. The negative lookbehind
        # excludes that case while leaving genuine background "&" (not
        # preceded by a redirect operator) splitting unchanged.
        pattern = r"\s*(?:&&|\|\||(?<![<>])&(?!&)|[;|\n])\s*"
        parts = re.split(pattern, command)

        programs: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            prog = cls._extract_program(part)
            if prog is None:
                return None
            programs.append(prog)

        return programs if programs else None

    def _match_allowed(self, program: str) -> str | None:
        """Return the first ``allowed_commands`` entry matched by *program*.

        Matching is performed as a prefix check on the *basename* of the
        program so that ``/usr/bin/git`` matches the entry ``"git"``, and the
        entry ``"git"`` does not accidentally match ``"gitk"`` (because the
        comparison is against the full basename or path, not a substring).

        Specifically, the program matches an entry when:

        * ``program`` equals ``entry`` exactly, or
        * ``program`` ends with ``"/" + entry`` (path-qualified match), or
        * ``entry`` ends with ``"/" + basename(program)`` (entry is qualified).

        Args:
            program: The program name or path extracted from the command.

        Returns:
            The matching entry string prefixed with ``"cmd:"``, or ``None``.
        """
        import os.path

        program_basename = os.path.basename(program)

        for entry in self._policy.allowed_commands:
            entry_basename = os.path.basename(entry)
            # Match if basenames agree.
            if program_basename == entry_basename:
                return f"cmd:{entry}"
            # Also allow: configured entry is a prefix of the program token
            # for commands like "git" matching "git-commit" style sub-commands
            # is intentionally NOT done here – we match exact basename only
            # to avoid unintended over-permissioning.

        return None

    def _emit_event(
        self,
        command: str,
        result: str,
        rule: str | None,
        session_id: str,
        task_id: str,
    ) -> None:
        """Publish a shell audit event.

        Args:
            command: The original command string.
            result: ``"allow"`` or ``"deny"``.
            rule: The matching policy rule, or ``None``.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
        """
        event = AuditEvent.now(
            session_id=session_id,
            task_id=task_id,
            event_type="shell_check",
            category="shell",
            result=result,  # type: ignore[arg-type]
            policy_rule=rule,
            detail={"command": command},
        )
        event_bus.publish(event)
