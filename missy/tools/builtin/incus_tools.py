"""Built-in tools: Incus container and VM management.

Provides full control over Incus instances, images, networks, storage,
profiles, and snapshots via the ``incus`` CLI.  All commands use
``--format json`` (where supported) for structured output.

Requires ``shell=True`` permission — the policy engine must whitelist
the ``incus`` command.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_OUTPUT_BYTES = 65_536
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600
_SHELL_PERMS = ToolPermissions(shell=True)


def _run_incus(
    args: list[str],
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    stdin_data: str | None = None,
) -> ToolResult:
    """Execute ``incus <args>`` and return a :class:`ToolResult`.

    Centralises subprocess invocation, output truncation, and error
    handling for every Incus tool.
    """
    timeout = min(int(timeout), _MAX_TIMEOUT)
    cmd = ["incus"] + args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            input=stdin_data.encode("utf-8") if stdin_data else None,
        )
        combined: bytes = proc.stdout + proc.stderr
        if len(combined) > _MAX_OUTPUT_BYTES:
            combined = combined[:_MAX_OUTPUT_BYTES] + b"\n[Output truncated]"
        output = combined.decode("utf-8", errors="replace")

        # Try to parse JSON output for structured results
        if output.strip().startswith(("{", "[")):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                output = json.loads(output)

        success = proc.returncode == 0
        error = f"Exit code {proc.returncode}" if not success else None
        return ToolResult(success=success, output=output, error=error)
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False, output=None,
            error=f"Command timed out after {timeout}s",
        )
    except FileNotFoundError:
        return ToolResult(
            success=False, output=None,
            error="incus binary not found — is Incus installed?",
        )
    except Exception as exc:
        return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# 1. List instances
# ---------------------------------------------------------------------------
class IncusListTool(BaseTool):
    """List Incus instances (containers and VMs)."""

    name = "incus_list"
    description = (
        "List Incus instances (containers and virtual machines). "
        "Returns structured JSON with name, status, type, IPv4/IPv6, and more. "
        "Supports filtering by type or project."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        project: str | None = None,
        instance_type: str | None = None,
        all_projects: bool = False,
        **_kwargs: Any,
    ) -> ToolResult:
        args = ["list", "--format", "json"]
        if all_projects:
            args.append("--all-projects")
        elif project:
            args.extend(["--project", project])
        if instance_type:
            args.extend(["--type", instance_type])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Incus project to list from (default: current project).",
                    },
                    "instance_type": {
                        "type": "string",
                        "enum": ["container", "virtual-machine"],
                        "description": "Filter by instance type.",
                    },
                    "all_projects": {
                        "type": "boolean",
                        "description": "List instances across all projects.",
                    },
                },
                "required": [],
            },
        }


# ---------------------------------------------------------------------------
# 2. Launch instance
# ---------------------------------------------------------------------------
class IncusLaunchTool(BaseTool):
    """Launch a new Incus container or VM."""

    name = "incus_launch"
    description = (
        "Launch a new Incus container or virtual machine from an image. "
        "Example images: 'images:ubuntu/24.04', 'images:debian/12', 'images:alpine/3.19'. "
        "Use --vm flag for virtual machines. Supports profiles, config, and resource limits."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        image: str,
        name: str | None = None,
        vm: bool = False,
        profiles: list[str] | None = None,
        config: dict[str, str] | None = None,
        project: str | None = None,
        ephemeral: bool = False,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        args = ["launch", image]
        if name:
            args.append(name)
        if vm:
            args.append("--vm")
        if ephemeral:
            args.append("--ephemeral")
        if project:
            args.extend(["--project", project])
        for p in (profiles or []):
            args.extend(["--profile", p])
        for k, v in (config or {}).items():
            args.extend(["--config", f"{k}={v}"])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": (
                            "Image to launch, e.g. 'images:ubuntu/24.04', "
                            "'images:debian/12', 'images:alpine/3.19'."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": "Instance name (auto-generated if omitted).",
                    },
                    "vm": {
                        "type": "boolean",
                        "description": "Launch as a virtual machine instead of container.",
                    },
                    "profiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Profiles to apply (e.g. ['default', 'gpu']).",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Instance config overrides, e.g. "
                            "{'limits.cpu': '2', 'limits.memory': '4GiB'}."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project to launch into.",
                    },
                    "ephemeral": {
                        "type": "boolean",
                        "description": "Delete instance on shutdown.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60, max: 600).",
                    },
                },
                "required": ["image"],
            },
        }


# ---------------------------------------------------------------------------
# 3. Instance lifecycle actions
# ---------------------------------------------------------------------------
class IncusInstanceActionTool(BaseTool):
    """Start, stop, restart, pause, delete, or rename an instance."""

    name = "incus_instance_action"
    description = (
        "Perform a lifecycle action on an Incus instance: "
        "start, stop, restart, pause, delete, or rename. "
        "Use force=true for forced stop/restart/delete."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        action: str,
        force: bool = False,
        new_name: str | None = None,
        project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"start", "stop", "restart", "pause", "delete", "rename"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "rename":
            if not new_name:
                return ToolResult(
                    success=False, output=None,
                    error="new_name is required for rename action",
                )
            args = ["rename", instance, new_name]
        else:
            args = [action, instance]
            if force and action in {"stop", "restart", "delete"}:
                args.append("--force")
        if project:
            args.extend(["--project", project])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Name of the instance.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "restart", "pause", "delete", "rename"],
                        "description": "Lifecycle action to perform.",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force the action (for stop/restart/delete).",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New name (required for rename action).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["instance", "action"],
            },
        }


# ---------------------------------------------------------------------------
# 4. Instance info
# ---------------------------------------------------------------------------
class IncusInfoTool(BaseTool):
    """Get detailed information about an Incus instance."""

    name = "incus_info"
    description = (
        "Get detailed information about an Incus instance including status, "
        "resources, IPs, snapshots, disk usage, and processes."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        show_resources: bool = False,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        args = ["info", instance]
        if show_resources:
            args.append("--resources")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "show_resources": {
                        "type": "boolean",
                        "description": "Include resource usage details.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["instance"],
            },
        }


# ---------------------------------------------------------------------------
# 5. Execute command in instance
# ---------------------------------------------------------------------------
class IncusExecTool(BaseTool):
    """Execute a command inside an Incus instance."""

    name = "incus_exec"
    description = (
        "Execute a command inside a running Incus instance. "
        "Equivalent to 'incus exec <instance> -- <command>'. "
        "Supports setting environment variables and working directory."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: int | None = None,
        group: int | None = None,
        project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        args = ["exec", instance]
        if cwd:
            args.extend(["--cwd", cwd])
        if user is not None:
            args.extend(["--user", str(user)])
        if group is not None:
            args.extend(["--group", str(group)])
        for k, v in (env or {}).items():
            args.extend(["--env", f"{k}={v}"])
        if project:
            args.extend(["--project", project])
        args.append("--")
        if isinstance(command, str):
            args.extend(["bash", "-c", command])
        else:
            args.extend(command)
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "command": {
                        "description": (
                            "Command to execute. String is passed to 'bash -c'. "
                            "Array is executed directly."
                        ),
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory inside the instance.",
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Environment variables to set.",
                    },
                    "user": {
                        "type": "integer",
                        "description": "UID to run as inside the instance.",
                    },
                    "group": {
                        "type": "integer",
                        "description": "GID to run as inside the instance.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60, max: 600).",
                    },
                },
                "required": ["instance", "command"],
            },
        }


# ---------------------------------------------------------------------------
# 6. File transfer
# ---------------------------------------------------------------------------
class IncusFileTool(BaseTool):
    """Push or pull files between the host and an Incus instance."""

    name = "incus_file"
    description = (
        "Transfer files to/from an Incus instance. "
        "push: host → instance, pull: instance → host. "
        "Paths use 'instance/path' format for the instance side."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        instance: str,
        instance_path: str,
        host_path: str,
        create_dirs: bool = False,
        recursive: bool = False,
        project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        if action not in {"push", "pull"}:
            return ToolResult(
                success=False, output=None,
                error="action must be 'push' or 'pull'",
            )
        instance_ref = f"{instance}{instance_path}"
        if action == "push":
            args = ["file", "push", host_path, instance_ref]
        else:
            args = ["file", "pull", instance_ref, host_path]
        if create_dirs:
            args.append("--create-dirs")
        if recursive:
            args.append("--recursive")
        if project:
            args.extend(["--project", project])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["push", "pull"],
                        "description": "Direction: push (host→instance) or pull (instance→host).",
                    },
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "instance_path": {
                        "type": "string",
                        "description": "Path inside the instance (must start with '/').",
                    },
                    "host_path": {
                        "type": "string",
                        "description": "Path on the host.",
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories as needed.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Transfer directories recursively.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["action", "instance", "instance_path", "host_path"],
            },
        }


# ---------------------------------------------------------------------------
# 7. Snapshots
# ---------------------------------------------------------------------------
class IncusSnapshotTool(BaseTool):
    """Manage Incus instance snapshots."""

    name = "incus_snapshot"
    description = (
        "Create, restore, delete, or list snapshots for an Incus instance. "
        "Snapshots are instant and can include stateful memory dumps."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        action: str,
        snapshot_name: str | None = None,
        stateful: bool = False,
        project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"create", "restore", "delete", "list"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            args = ["info", instance, "--format", "json"]
            if project:
                args.extend(["--project", project])
            return _run_incus(args, timeout=timeout)
        if action in {"create", "delete"} and not snapshot_name:
            return ToolResult(
                success=False, output=None,
                error=f"snapshot_name is required for {action}",
            )
        if action == "create":
            args = ["snapshot", "create", instance, snapshot_name]
            if stateful:
                args.append("--stateful")
        elif action == "restore":
            if snapshot_name:
                args = ["snapshot", "restore", instance, snapshot_name]
            else:
                args = ["snapshot", "restore", instance]
        elif action == "delete":
            args = ["snapshot", "delete", instance, snapshot_name]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["create", "restore", "delete", "list"],
                        "description": "Snapshot action.",
                    },
                    "snapshot_name": {
                        "type": "string",
                        "description": "Snapshot name (required for create/delete).",
                    },
                    "stateful": {
                        "type": "boolean",
                        "description": "Include memory state (create only).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["instance", "action"],
            },
        }


# ---------------------------------------------------------------------------
# 8. Instance config
# ---------------------------------------------------------------------------
class IncusConfigTool(BaseTool):
    """Get or set Incus instance configuration."""

    name = "incus_config"
    description = (
        "Get, set, or show full configuration for an Incus instance. "
        "Can set resource limits (limits.cpu, limits.memory), devices, "
        "security options, and more."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        action: str = "show",
        key: str | None = None,
        value: str | None = None,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"show", "get", "set", "unset"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "show":
            args = ["config", "show", instance]
        elif action == "get":
            if not key:
                return ToolResult(
                    success=False, output=None, error="key is required for get",
                )
            args = ["config", "get", instance, key]
        elif action == "set":
            if not key or value is None:
                return ToolResult(
                    success=False, output=None,
                    error="key and value are required for set",
                )
            args = ["config", "set", instance, key, value]
        elif action == "unset":
            if not key:
                return ToolResult(
                    success=False, output=None, error="key is required for unset",
                )
            args = ["config", "unset", instance, key]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["show", "get", "set", "unset"],
                        "description": "Config action (default: show).",
                    },
                    "key": {
                        "type": "string",
                        "description": (
                            "Config key, e.g. 'limits.cpu', 'limits.memory', "
                            "'security.nesting', 'security.privileged'."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to set.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["instance"],
            },
        }


# ---------------------------------------------------------------------------
# 9. Image management
# ---------------------------------------------------------------------------
class IncusImageTool(BaseTool):
    """Manage Incus images."""

    name = "incus_image"
    description = (
        "List, info, copy, delete, or alias Incus images. "
        "Use list to browse available images from remotes like 'images:'. "
        "Use copy to cache a remote image locally."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        image: str | None = None,
        alias: str | None = None,
        remote: str | None = None,
        project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"list", "info", "delete", "copy", "alias"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            target = f"{remote}:" if remote else ""
            args = ["image", "list", target, "--format", "json"]
        elif action == "info":
            if not image:
                return ToolResult(
                    success=False, output=None, error="image is required for info",
                )
            args = ["image", "info", image]
        elif action == "delete":
            if not image:
                return ToolResult(
                    success=False, output=None, error="image is required for delete",
                )
            args = ["image", "delete", image]
        elif action == "copy":
            if not image:
                return ToolResult(
                    success=False, output=None, error="image is required for copy",
                )
            dest = "local:"
            args = ["image", "copy", image, dest]
            if alias:
                args.extend(["--alias", alias])
        elif action == "alias":
            if not image or not alias:
                return ToolResult(
                    success=False, output=None,
                    error="image and alias are required for alias action",
                )
            args = ["image", "alias", "create", alias, image]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "info", "delete", "copy", "alias"],
                        "description": "Image action.",
                    },
                    "image": {
                        "type": "string",
                        "description": "Image identifier (fingerprint, alias, or remote:alias).",
                    },
                    "alias": {
                        "type": "string",
                        "description": "Alias name (for copy --alias or alias create).",
                    },
                    "remote": {
                        "type": "string",
                        "description": "Remote server name for listing (e.g. 'images').",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 10. Network management
# ---------------------------------------------------------------------------
class IncusNetworkTool(BaseTool):
    """Manage Incus networks."""

    name = "incus_network"
    description = (
        "List, create, delete, show, or configure Incus networks. "
        "Supports managed bridges, OVN, macvlan, and more."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        name: str | None = None,
        network_type: str | None = None,
        config: dict[str, str] | None = None,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"list", "create", "delete", "show", "set", "attach", "detach"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            args = ["network", "list", "--format", "json"]
        elif action == "create":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for create",
                )
            args = ["network", "create", name]
            if network_type:
                args.extend(["--type", network_type])
            for k, v in (config or {}).items():
                args.append(f"{k}={v}")
        elif action == "delete":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for delete",
                )
            args = ["network", "delete", name]
        elif action == "show":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for show",
                )
            args = ["network", "show", name]
        elif action == "set":
            if not name or not config:
                return ToolResult(
                    success=False, output=None,
                    error="name and config are required for set",
                )
            for k, v in config.items():
                result = _run_incus(["network", "set", name, k, v])
                if not result.success:
                    return result
            return ToolResult(success=True, output="Configuration updated")
        elif action == "attach":
            if not name:
                return ToolResult(
                    success=False, output=None,
                    error="name is required for attach (format: 'network_name instance_name')",
                )
            parts = name.split()
            if len(parts) < 2:
                return ToolResult(
                    success=False, output=None,
                    error="For attach, name should be 'network_name instance_name'.",
                )
            args = ["network", "attach", parts[0], parts[1]]
        elif action == "detach":
            if not name:
                return ToolResult(
                    success=False, output=None,
                    error="name is required for detach (format: 'network_name instance_name')",
                )
            parts = name.split()
            if len(parts) < 2:
                return ToolResult(
                    success=False, output=None,
                    error="For detach, name should be 'network_name instance_name'.",
                )
            args = ["network", "detach", parts[0], parts[1]]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "delete", "show", "set", "attach", "detach"],
                        "description": "Network action.",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "Network name. For attach/detach: 'network_name instance_name'."
                        ),
                    },
                    "network_type": {
                        "type": "string",
                        "enum": ["bridge", "ovn", "macvlan", "sriov", "physical"],
                        "description": "Network type (create only).",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Network config key-value pairs.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 11. Storage management
# ---------------------------------------------------------------------------
class IncusStorageTool(BaseTool):
    """Manage Incus storage pools and volumes."""

    name = "incus_storage"
    description = (
        "Manage Incus storage pools and volumes. "
        "Supports pool operations (list/create/delete/show) and "
        "volume operations (volume-list/volume-create/volume-delete/volume-attach/volume-detach)."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        pool: str | None = None,
        volume: str | None = None,
        instance: str | None = None,
        driver: str | None = None,
        size: str | None = None,
        config: dict[str, str] | None = None,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {
            "list", "create", "delete", "show",
            "volume-list", "volume-create", "volume-delete",
            "volume-attach", "volume-detach",
        }
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )

        if action == "list":
            args = ["storage", "list", "--format", "json"]
        elif action == "create":
            if not pool or not driver:
                return ToolResult(
                    success=False, output=None,
                    error="pool and driver are required for create",
                )
            args = ["storage", "create", pool, driver]
            if size:
                args.append(f"size={size}")
            for k, v in (config or {}).items():
                args.append(f"{k}={v}")
        elif action == "delete":
            if not pool:
                return ToolResult(
                    success=False, output=None, error="pool is required for delete",
                )
            args = ["storage", "delete", pool]
        elif action == "show":
            if not pool:
                return ToolResult(
                    success=False, output=None, error="pool is required for show",
                )
            args = ["storage", "show", pool]
        elif action == "volume-list":
            if not pool:
                return ToolResult(
                    success=False, output=None,
                    error="pool is required for volume-list",
                )
            args = ["storage", "volume", "list", pool, "--format", "json"]
        elif action == "volume-create":
            if not pool or not volume:
                return ToolResult(
                    success=False, output=None,
                    error="pool and volume are required for volume-create",
                )
            args = ["storage", "volume", "create", pool, volume]
            if size:
                args.append(f"size={size}")
            for k, v in (config or {}).items():
                args.append(f"{k}={v}")
        elif action == "volume-delete":
            if not pool or not volume:
                return ToolResult(
                    success=False, output=None,
                    error="pool and volume are required for volume-delete",
                )
            args = ["storage", "volume", "delete", pool, volume]
        elif action == "volume-attach":
            if not pool or not volume or not instance:
                return ToolResult(
                    success=False, output=None,
                    error="pool, volume, and instance are required for volume-attach",
                )
            args = ["storage", "volume", "attach", pool, volume, instance]
        elif action == "volume-detach":
            if not pool or not volume or not instance:
                return ToolResult(
                    success=False, output=None,
                    error="pool, volume, and instance are required for volume-detach",
                )
            args = ["storage", "volume", "detach", pool, volume, instance]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list", "create", "delete", "show",
                            "volume-list", "volume-create", "volume-delete",
                            "volume-attach", "volume-detach",
                        ],
                        "description": "Storage action.",
                    },
                    "pool": {
                        "type": "string",
                        "description": "Storage pool name.",
                    },
                    "volume": {
                        "type": "string",
                        "description": "Volume name (for volume-* actions).",
                    },
                    "instance": {
                        "type": "string",
                        "description": "Instance name (for volume-attach/detach).",
                    },
                    "driver": {
                        "type": "string",
                        "enum": ["dir", "zfs", "btrfs", "lvm", "ceph", "cephfs", "cephobject"],
                        "description": "Storage driver (for pool create).",
                    },
                    "size": {
                        "type": "string",
                        "description": "Pool or volume size, e.g. '50GiB'.",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Additional config key-value pairs.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 12. Profile management
# ---------------------------------------------------------------------------
class IncusProfileTool(BaseTool):
    """Manage Incus profiles."""

    name = "incus_profile"
    description = (
        "List, show, create, delete, or edit Incus profiles. "
        "Profiles are reusable configuration templates applied to instances."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        name: str | None = None,
        config: dict[str, str] | None = None,
        yaml_content: str | None = None,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"list", "show", "create", "delete", "set", "edit"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            args = ["profile", "list", "--format", "json"]
        elif action == "show":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for show",
                )
            args = ["profile", "show", name]
        elif action == "create":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for create",
                )
            args = ["profile", "create", name]
        elif action == "delete":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for delete",
                )
            args = ["profile", "delete", name]
        elif action == "set":
            if not name or not config:
                return ToolResult(
                    success=False, output=None,
                    error="name and config are required for set",
                )
            for k, v in config.items():
                result = _run_incus(["profile", "set", name, k, v])
                if not result.success:
                    return result
            return ToolResult(success=True, output="Profile configuration updated")
        elif action == "edit":
            if not name or not yaml_content:
                return ToolResult(
                    success=False, output=None,
                    error="name and yaml_content are required for edit",
                )
            args = ["profile", "edit", name]
            if project:
                args.extend(["--project", project])
            return _run_incus(args, stdin_data=yaml_content)
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "show", "create", "delete", "set", "edit"],
                        "description": "Profile action.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Profile name.",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Config key-value pairs (for set action).",
                    },
                    "yaml_content": {
                        "type": "string",
                        "description": "Full YAML profile definition (for edit action).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 13. Project management
# ---------------------------------------------------------------------------
class IncusProjectTool(BaseTool):
    """Manage Incus projects."""

    name = "incus_project"
    description = (
        "List, create, delete, show, or switch Incus projects. "
        "Projects provide multi-tenancy isolation for instances, images, "
        "profiles, networks, and storage."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        action: str,
        name: str | None = None,
        config: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"list", "create", "delete", "show", "switch"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            args = ["project", "list", "--format", "json"]
        elif action == "create":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for create",
                )
            args = ["project", "create", name]
            for k, v in (config or {}).items():
                args.extend(["--config", f"{k}={v}"])
        elif action == "delete":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for delete",
                )
            args = ["project", "delete", name]
        elif action == "show":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for show",
                )
            args = ["project", "show", name]
        elif action == "switch":
            if not name:
                return ToolResult(
                    success=False, output=None, error="name is required for switch",
                )
            args = ["project", "switch", name]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "delete", "show", "switch"],
                        "description": "Project action.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Project name.",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Project config (for create).",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 14. Device management
# ---------------------------------------------------------------------------
class IncusDeviceTool(BaseTool):
    """Manage devices attached to Incus instances."""

    name = "incus_device"
    description = (
        "Add, remove, list, or show devices on an Incus instance. "
        "Devices include disks, GPUs, NICs, USB, unix-char, proxy ports, etc."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        instance: str,
        action: str,
        device_name: str | None = None,
        device_type: str | None = None,
        config: dict[str, str] | None = None,
        project: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        valid = {"list", "add", "remove", "show"}
        if action not in valid:
            return ToolResult(
                success=False, output=None,
                error=f"Invalid action '{action}'. Must be one of: {', '.join(sorted(valid))}",
            )
        if action == "list":
            args = ["config", "device", "list", instance, "--format", "json"]
        elif action == "show":
            if not device_name:
                return ToolResult(
                    success=False, output=None,
                    error="device_name is required for show",
                )
            args = ["config", "device", "show", instance]
        elif action == "add":
            if not device_name or not device_type:
                return ToolResult(
                    success=False, output=None,
                    error="device_name and device_type are required for add",
                )
            args = ["config", "device", "add", instance, device_name, device_type]
            for k, v in (config or {}).items():
                args.append(f"{k}={v}")
        elif action == "remove":
            if not device_name:
                return ToolResult(
                    success=False, output=None,
                    error="device_name is required for remove",
                )
            args = ["config", "device", "remove", instance, device_name]
        else:
            return ToolResult(success=False, output=None, error="Unreachable")
        if project:
            args.extend(["--project", project])
        return _run_incus(args)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "instance": {
                        "type": "string",
                        "description": "Instance name.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "remove", "show"],
                        "description": "Device action.",
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Device name.",
                    },
                    "device_type": {
                        "type": "string",
                        "enum": [
                            "disk", "gpu", "infiniband", "nic", "pci",
                            "proxy", "tpm", "unix-block", "unix-char",
                            "unix-hotplug", "usb",
                        ],
                        "description": "Device type (for add).",
                    },
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Device config, e.g. {'source': '/dev/gpu0'} or "
                            "{'listen': 'tcp:0.0.0.0:80', 'connect': 'tcp:127.0.0.1:80'}."
                        ),
                    },
                    "project": {
                        "type": "string",
                        "description": "Incus project.",
                    },
                },
                "required": ["instance", "action"],
            },
        }


# ---------------------------------------------------------------------------
# 15. Copy/Move instances
# ---------------------------------------------------------------------------
class IncusCopyMoveTool(BaseTool):
    """Copy or move Incus instances."""

    name = "incus_copy_move"
    description = (
        "Copy or move an Incus instance. Supports copying across "
        "remotes, projects, and storage pools. Can create instances "
        "from snapshots."
    )
    permissions = _SHELL_PERMS

    def execute(
        self,
        *,
        source: str,
        destination: str,
        action: str = "copy",
        stateless: bool = False,
        instance_only: bool = False,
        storage: str | None = None,
        project: str | None = None,
        target_project: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        action = action.lower()
        if action not in {"copy", "move"}:
            return ToolResult(
                success=False, output=None,
                error="action must be 'copy' or 'move'",
            )
        args = [action, source, destination]
        if stateless and action == "copy":
            args.append("--stateless")
        if instance_only:
            args.append("--instance-only")
        if storage:
            args.extend(["--storage", storage])
        if project:
            args.extend(["--project", project])
        if target_project:
            args.extend(["--target-project", target_project])
        return _run_incus(args, timeout=timeout)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source instance (or instance/snapshot).",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination instance name.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["copy", "move"],
                        "description": "Copy or move (default: copy).",
                    },
                    "stateless": {
                        "type": "boolean",
                        "description": "Copy without state (copy only).",
                    },
                    "instance_only": {
                        "type": "boolean",
                        "description": "Skip snapshots when copying.",
                    },
                    "storage": {
                        "type": "string",
                        "description": "Target storage pool.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Source project.",
                    },
                    "target_project": {
                        "type": "string",
                        "description": "Destination project.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["source", "destination"],
            },
        }
