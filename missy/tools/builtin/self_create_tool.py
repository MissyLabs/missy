"""Built-in tool: create, list, and delete agent-authored custom tools.

Custom tools are scripts stored in ~/.missy/custom-tools/ and registered
at startup. The agent can create bash, python, or node scripts as tools.
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
    """Create, list, or delete agent-authored persistent custom tools."""

    name = "self_create_tool"
    description = (
        "Create, list, or delete persistent custom tools. "
        "Use action='create' to write a script; action='list' to see existing; action='delete' to remove."
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
                return ToolResult(success=True, output="No custom tools defined.", error=None)
            entries = []
            for meta_file in sorted(tools_dir.glob("*.json")):
                try:
                    meta = json.loads(meta_file.read_text())
                    entries.append(f"- {meta.get('name', '?')}: {meta.get('description', '')}")
                except Exception as _meta_exc:
                    logger.debug("self_create_tool: failed to load %s: %s", meta_file, _meta_exc)
            return ToolResult(
                success=True, output="\n".join(entries) or "No custom tools defined.", error=None
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
                    success=True, output=f"Deleted custom tool: {tool_name}", error=None
                )
            return ToolResult(success=False, output="", error=f"Tool not found: {tool_name}")

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
                output=f"Custom tool '{tool_name}' created at {script_path}",
                error=None,
            )

        return ToolResult(success=False, output="", error=f"Unknown action: {action}")
