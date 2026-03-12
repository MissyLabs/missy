"""Built-in tools bundled with Missy.

All tools in this package are ready to use; import them directly or call
:func:`register_builtin_tools` to add every tool to the active registry in
one step.

Example::

    from missy.tools.registry import init_tool_registry
    from missy.tools.builtin import register_builtin_tools

    registry = init_tool_registry()
    register_builtin_tools(registry)
"""
from missy.tools.builtin.calculator import CalculatorTool
from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import FileReadTool
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.list_files import ListFilesTool
from missy.tools.builtin.self_create_tool import SelfCreateTool
from missy.tools.builtin.shell_exec import ShellExecTool
from missy.tools.builtin.web_fetch import WebFetchTool

__all__ = [
    "CalculatorTool",
    "FileDeleteTool",
    "FileReadTool",
    "FileWriteTool",
    "ListFilesTool",
    "SelfCreateTool",
    "ShellExecTool",
    "WebFetchTool",
    "register_builtin_tools",
]

_ALL_TOOL_CLASSES = [
    CalculatorTool,
    FileDeleteTool,
    FileReadTool,
    FileWriteTool,
    ListFilesTool,
    SelfCreateTool,
    ShellExecTool,
    WebFetchTool,
]


def register_builtin_tools(registry=None) -> None:
    """Register all built-in tools with a :class:`~missy.tools.registry.ToolRegistry`.

    When *registry* is ``None`` the process-level registry returned by
    :func:`~missy.tools.registry.get_tool_registry` is used.  If that
    registry has not yet been initialised a :class:`RuntimeError` is raised
    by the registry module.

    Args:
        registry: An explicit :class:`~missy.tools.registry.ToolRegistry`
            instance to register tools into, or ``None`` to use the
            process-level singleton.

    Raises:
        RuntimeError: When *registry* is ``None`` and
            :func:`~missy.tools.registry.init_tool_registry` has not been
            called yet.
    """
    if registry is None:
        from missy.tools.registry import get_tool_registry

        registry = get_tool_registry()

    for tool_cls in _ALL_TOOL_CLASSES:
        registry.register(tool_cls())
