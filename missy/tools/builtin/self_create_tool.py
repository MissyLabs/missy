"""Built-in tool: create, list, and delete agent-authored custom tool proposals.

SR-4.5: this tool only writes scripts to ~/.missy/custom-tools/ plus a
sidecar metadata JSON. Nothing in Missy scans that directory or loads
its contents back into the live ToolRegistry -- a script written here
can never actually be called as a tool. This is intentional (dynamic
loading and auto-execution of agent-authored code is a significant,
currently-unimplemented security surface -- see AUDIT_SECURITY.md's
"SR-4.5" section), not an oversight, so every user-facing string this
tool returns says "proposal"/"written for review", never "created" or
"registered".
"""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

CUSTOM_TOOLS_DIR = Path("~/.missy/custom-tools")
ALLOWED_LANGUAGES = {"bash": ".sh", "python": ".py", "node": ".js"}


class SelfCreateTool(BaseTool):
    """Write, list, or delete agent-authored custom tool *proposals*.

    These scripts are NOT automatically loaded or made callable -- see the
    module docstring. This tool only manages files on disk for later human
    review; it does not expand what the agent itself can do.
    """

    name = "self_create_tool"
    description = (
        "Write, list, or delete custom tool PROPOSAL scripts for human review. "
        "IMPORTANT: proposals are NOT automatically registered or callable -- "
        "writing one here does not give you, or any future turn, the ability "
        "to invoke it. A human operator must review the script and wire it "
        "into the tool registry manually before it can run. "
        "Use action='create' to write a proposal; action='list' to see existing "
        "proposals; action='delete' to remove one."
    )
    permissions = ToolPermissions(filesystem_write=True)
    parameters = {
        "action": {
            "type": "string",
            "enum": ["create", "list", "delete"],
            "description": "Operation: 'create' writes a new script tool, 'list' shows existing, 'delete' removes one.",
            "required": True,
        },
        "tool_name": {
            "type": "string",
            "description": "Name of the tool (alphanumeric/underscore/hyphen). Required for create and delete.",
        },
        "language": {
            "type": "string",
            "enum": ["bash", "python", "node"],
            "description": "Script language for create (default: python).",
        },
        "script": {
            "type": "string",
            "description": "Full script source code for create. Python scripts should print output to stdout.",
        },
        "tool_description": {
            "type": "string",
            "description": "Human-readable description for the tool being created.",
        },
    }

    def execute(
        self,
        *,
        action: str,
        tool_name: str = "",
        language: str = "bash",
        script: str = "",
        tool_description: str = "",
        **_kwargs,
    ) -> ToolResult:
        tools_dir = CUSTOM_TOOLS_DIR.expanduser()

        if action == "list":
            if not tools_dir.exists():
                return ToolResult(success=True, output="No custom tool proposals on file.", error=None)
            entries = []
            for meta_file in sorted(tools_dir.glob("*.json")):
                try:
                    meta = json.loads(meta_file.read_text())
                    entries.append(f"- {meta.get('name', '?')}: {meta.get('description', '')}")
                except Exception as _meta_exc:
                    logger.debug("self_create_tool: failed to load %s: %s", meta_file, _meta_exc)
            header = (
                "Custom tool PROPOSALS on file (not registered/callable -- "
                "pending human review):"
            )
            body = "\n".join(entries) if entries else "No custom tool proposals on file."
            return ToolResult(
                success=True,
                output=f"{header}\n{body}" if entries else body,
                error=None,
            )

        if action == "delete":
            import re

            if not tool_name or not re.match(r"^[a-zA-Z0-9_-]+$", tool_name):
                return ToolResult(
                    success=False,
                    output="",
                    error="tool_name must be alphanumeric/underscore/hyphen only.",
                )
            removed = False
            for ext in ALLOWED_LANGUAGES.values():
                p = tools_dir / f"{tool_name}{ext}"
                # Verify the resolved path is still under tools_dir
                if p.resolve().parent != tools_dir.resolve():
                    continue
                if p.exists():
                    p.unlink()
                    removed = True
            meta = tools_dir / f"{tool_name}.json"
            if meta.resolve().parent == tools_dir.resolve() and meta.exists():
                meta.unlink()
                removed = True
            if removed:
                return ToolResult(
                    success=True, output=f"Deleted custom tool proposal: {tool_name}", error=None
                )
            return ToolResult(
                success=False, output="", error=f"Tool proposal not found: {tool_name}"
            )

        if action == "create":
            import re

            if not tool_name or not re.match(r"^[a-zA-Z0-9_-]+$", tool_name):
                return ToolResult(
                    success=False,
                    output="",
                    error="tool_name must be alphanumeric/underscore/hyphen only.",
                )
            if language not in ALLOWED_LANGUAGES:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"language must be one of: {list(ALLOWED_LANGUAGES)}",
                )
            if not script:
                return ToolResult(success=False, output="", error="script is required for create.")

            # Scan script content for dangerous patterns
            _DANGEROUS_PATTERNS = [
                "curl ",
                "wget ",
                "nc ",
                "ncat ",
                "socat ",
                "/dev/tcp/",
                "/dev/udp/",
                "eval(",
                "exec(",
                "os.system(",
                "subprocess.call(",
                "subprocess.Popen(",
                "subprocess.run(",
                "import socket",
                "import http",
                "reverse_shell",
                "bind_shell",
                "chmod +s",
                "chmod u+s",
                "setuid",
                # Indirect execution patterns
                "__import__(",
                "getattr(",
                "importlib.",
                "compile(",
                "code.interact(",
                # Builtins access for sandbox escape
                "__builtins__",
                # File I/O for arbitrary reads/writes
                "open(",
                # Process execution functions
                "os.exec",
                "os.fork",
                "os.spawn",
                "os.popen(",
                "os.startfile(",
                # Destructive file operations
                "shutil.rmtree",
                "shutil.move",
                "os.remove(",
                "os.unlink(",
                "os.rmdir(",
                # Node.js patterns
                "child_process",
                "require('fs')",
                'require("fs")',
                # Shell expansion patterns
                "$(",
                "`",
            ]
            script_lower = script.lower()
            for pattern in _DANGEROUS_PATTERNS:
                if pattern.lower() in script_lower:
                    logger.warning(
                        "self_create_tool: dangerous pattern %r found in script for %r",
                        pattern,
                        tool_name,
                    )
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Script contains potentially dangerous pattern: {pattern!r}. "
                        "Custom tool scripts must not include network access, code execution, "
                        "or privilege escalation patterns.",
                    )

            tools_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            ext = ALLOWED_LANGUAGES[language]
            script_path = tools_dir / f"{tool_name}{ext}"
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(stat.S_IRWXU)

            meta = {"name": tool_name, "description": tool_description, "language": language}
            (tools_dir / f"{tool_name}.json").write_text(json.dumps(meta, indent=2))

            return ToolResult(
                success=True,
                output=(
                    f"Proposal script '{tool_name}' written to {script_path} for human "
                    "review. This is NOT a registered or callable tool -- no mechanism "
                    "loads scripts from this directory into the active tool registry. "
                    "A human operator must review and manually wire it in before it can "
                    "ever run. Tell the user/operator the proposal exists; do not treat "
                    "it as available to call."
                ),
                error=None,
            )

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")
