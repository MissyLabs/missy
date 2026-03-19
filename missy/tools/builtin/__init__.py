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

from missy.tools.builtin.atspi_tools import (
    AtSpiClickTool,
    AtSpiGetTextTool,
    AtSpiGetTreeTool,
    AtSpiSetValueTool,
)
from missy.tools.builtin.browser_tools import (
    BrowserClickTool,
    BrowserCloseTool,
    BrowserEvaluateTool,
    BrowserFillTool,
    BrowserGetContentTool,
    BrowserGetUrlTool,
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserWaitTool,
)
from missy.tools.builtin.calculator import CalculatorTool
from missy.tools.builtin.code_evolve import CodeEvolveTool
from missy.tools.builtin.discord_upload import DiscordUploadTool
from missy.tools.builtin.file_delete import FileDeleteTool
from missy.tools.builtin.file_read import FileReadTool
from missy.tools.builtin.file_write import FileWriteTool
from missy.tools.builtin.incus_tools import (
    IncusConfigTool,
    IncusCopyMoveTool,
    IncusDeviceTool,
    IncusExecTool,
    IncusFileTool,
    IncusImageTool,
    IncusInfoTool,
    IncusInstanceActionTool,
    IncusLaunchTool,
    IncusListTool,
    IncusNetworkTool,
    IncusProfileTool,
    IncusProjectTool,
    IncusSnapshotTool,
    IncusStorageTool,
)
from missy.tools.builtin.list_files import ListFilesTool
from missy.tools.builtin.memory_tools import (
    MemoryDescribeTool,
    MemoryExpandTool,
    MemorySearchTool,
)
from missy.tools.builtin.self_create_tool import SelfCreateTool
from missy.tools.builtin.shell_exec import ShellExecTool
from missy.tools.builtin.tts_speak import AudioListDevicesTool, AudioSetVolumeTool, TTSSpeakTool
from missy.tools.builtin.web_fetch import WebFetchTool
from missy.tools.builtin.x11_launch import X11LaunchTool
from missy.tools.builtin.vision_tools import (
    VisionAnalyzeTool,
    VisionCaptureTool,
    VisionDevicesTool,
    VisionSceneMemoryTool,
)
from missy.tools.builtin.x11_tools import (
    X11ClickTool,
    X11KeyTool,
    X11ReadScreenTool,
    X11ScreenshotTool,
    X11TypeTool,
    X11WindowListTool,
)

__all__ = [
    "AtSpiClickTool",
    "AtSpiGetTextTool",
    "AtSpiGetTreeTool",
    "AtSpiSetValueTool",
    "BrowserClickTool",
    "BrowserCloseTool",
    "BrowserEvaluateTool",
    "BrowserFillTool",
    "BrowserGetContentTool",
    "BrowserGetUrlTool",
    "BrowserNavigateTool",
    "BrowserScreenshotTool",
    "BrowserWaitTool",
    "CalculatorTool",
    "CodeEvolveTool",
    "DiscordUploadTool",
    "FileDeleteTool",
    "IncusCopyMoveTool",
    "IncusConfigTool",
    "IncusDeviceTool",
    "IncusExecTool",
    "IncusFileTool",
    "IncusImageTool",
    "IncusInfoTool",
    "IncusInstanceActionTool",
    "IncusLaunchTool",
    "IncusListTool",
    "IncusNetworkTool",
    "IncusProfileTool",
    "IncusProjectTool",
    "IncusSnapshotTool",
    "IncusStorageTool",
    "FileReadTool",
    "FileWriteTool",
    "ListFilesTool",
    "MemoryDescribeTool",
    "MemoryExpandTool",
    "MemorySearchTool",
    "SelfCreateTool",
    "ShellExecTool",
    "WebFetchTool",
    "X11ClickTool",
    "X11LaunchTool",
    "X11KeyTool",
    "X11ReadScreenTool",
    "X11ScreenshotTool",
    "X11TypeTool",
    "X11WindowListTool",
    "AudioListDevicesTool",
    "AudioSetVolumeTool",
    "TTSSpeakTool",
    "VisionAnalyzeTool",
    "VisionCaptureTool",
    "VisionDevicesTool",
    "VisionSceneMemoryTool",
    "register_builtin_tools",
]

_ALL_TOOL_CLASSES = [
    AtSpiClickTool,
    AtSpiGetTextTool,
    AtSpiGetTreeTool,
    AtSpiSetValueTool,
    BrowserClickTool,
    BrowserCloseTool,
    BrowserEvaluateTool,
    BrowserFillTool,
    BrowserGetContentTool,
    BrowserGetUrlTool,
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserWaitTool,
    CalculatorTool,
    CodeEvolveTool,
    DiscordUploadTool,
    FileDeleteTool,
    FileReadTool,
    IncusCopyMoveTool,
    IncusConfigTool,
    IncusDeviceTool,
    IncusExecTool,
    IncusFileTool,
    IncusImageTool,
    IncusInfoTool,
    IncusInstanceActionTool,
    IncusLaunchTool,
    IncusListTool,
    IncusNetworkTool,
    IncusProfileTool,
    IncusProjectTool,
    IncusSnapshotTool,
    IncusStorageTool,
    FileWriteTool,
    ListFilesTool,
    MemoryDescribeTool,
    MemoryExpandTool,
    MemorySearchTool,
    SelfCreateTool,
    ShellExecTool,
    TTSSpeakTool,
    AudioListDevicesTool,
    AudioSetVolumeTool,
    WebFetchTool,
    X11ClickTool,
    X11KeyTool,
    X11LaunchTool,
    X11ReadScreenTool,
    X11ScreenshotTool,
    X11TypeTool,
    X11WindowListTool,
    VisionCaptureTool,
    VisionAnalyzeTool,
    VisionDevicesTool,
    VisionSceneMemoryTool,
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
