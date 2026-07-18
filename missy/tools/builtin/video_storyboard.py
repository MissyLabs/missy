"""Built-in tool: multi-scene video storyboard orchestration (F16).

A single-call composition of the existing ``video_generate`` and ``video_edit``
tools: given an ordered list of *scenes* (each a prompt + optional duration,
caption, and transition), it generates a clip per scene, trims each to its
requested duration, splices them (with an optional crossfade), overlays an
optional title, and returns the final MP4. This is the documented "iteration is
just calling the tool again" contract packaged into one deterministic
orchestration so the model doesn't have to hand-chain a dozen calls.

The heavy lifting is delegated to the two underlying tools, which enforce their
own GPU/model preflights and produce real files; this tool only sequences them
and reports honest per-scene progress. The underlying tools are injectable for
testing.
"""

from __future__ import annotations

from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_SCENES = 12


class VideoStoryboardTool(BaseTool):
    """Generate, trim, splice, and title a multi-scene video in one call."""

    name = "video_storyboard"
    description = (
        "Produce a multi-scene video from an ordered list of scenes. Each scene "
        "has a 'prompt' and optional 'duration' (seconds), 'caption', and "
        "'transition' ('crossfade'/'none'). Generates a clip per scene, trims, "
        "splices, and adds an optional overall 'title'. Returns the final MP4 path."
    )
    # Union of what the delegated tools need: video_generate (network → ComfyUI)
    # and video_edit (shell → ffmpeg, filesystem). So the storyboard tool is
    # gated at the registry with the same posture as its parts.
    permissions = ToolPermissions(
        network=True, shell=True, filesystem_read=True, filesystem_write=True
    )

    def __init__(self, generate_tool: Any = None, edit_tool: Any = None) -> None:
        self._generate_tool = generate_tool
        self._edit_tool = edit_tool

    def _gen(self) -> Any:
        if self._generate_tool is None:
            from missy.tools.builtin.video_generate import VideoGenerateTool

            self._generate_tool = VideoGenerateTool()
        return self._generate_tool

    def _edit(self) -> Any:
        if self._edit_tool is None:
            from missy.tools.builtin.video_edit import VideoEditTool

            self._edit_tool = VideoEditTool()
        return self._edit_tool

    # -- policy declarations delegate to the underlying tools ---------------

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        try:
            return self._gen().resolve_network_hosts(kwargs)
        except Exception:
            return []

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "ffmpeg && ffprobe"

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> dict[str, list[str]]:
        try:
            return self._edit().resolve_filesystem_targets(kwargs)
        except Exception:
            return {}

    # -- orchestration ------------------------------------------------------

    def execute(
        self,
        *,
        scenes: list[dict] | None = None,
        title: str = "",
        title_seconds: float = 2.0,
        transition: str = "crossfade",
        backend: str = "wan",
        **shared: Any,
    ) -> ToolResult:
        """Generate and assemble a storyboard from *scenes*.

        Args:
            scenes: Ordered list of ``{prompt, duration?, caption?, transition?}``.
            title: Optional overall title overlaid on the opening.
            title_seconds: How long the title stays visible.
            transition: Default transition between scenes when a scene doesn't
                specify its own ("crossfade" or "none").
            backend: video_generate backend for every scene.
            **shared: Extra kwargs forwarded to every video_generate call
                (e.g. width/height/fps/seed).

        Returns:
            A ToolResult whose output describes the final MP4 and per-scene clips.
        """
        if not scenes or not isinstance(scenes, list):
            return ToolResult(success=False, output=None, error="scenes must be a non-empty list")
        if len(scenes) > _MAX_SCENES:
            return ToolResult(
                success=False,
                output=None,
                error=f"too many scenes ({len(scenes)}); max is {_MAX_SCENES}",
            )
        for i, scene in enumerate(scenes):
            if not isinstance(scene, dict) or not str(scene.get("prompt", "")).strip():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"scene {i} is missing a non-empty 'prompt'",
                )

        gen = self._gen()
        edit = self._edit()
        clips: list[str] = []
        progress: list[dict] = []

        # 1. Generate (and optionally trim) each scene.
        for i, scene in enumerate(scenes):
            g = gen.execute(backend=backend, prompt=str(scene["prompt"]), **shared)
            if not g.success or not isinstance(g.output, dict) or not g.output.get("path"):
                return ToolResult(
                    success=False,
                    output={"completed_scenes": progress},
                    error=f"scene {i} generation failed: {g.error or 'no output path'}",
                )
            clip_path = g.output["path"]
            duration = scene.get("duration")
            if duration:
                t = edit.execute(
                    operation="trim", input=clip_path, start=0.0, duration=float(duration)
                )
                if t.success and isinstance(t.output, dict) and t.output.get("path"):
                    clip_path = t.output["path"]
            clips.append(clip_path)
            progress.append({"scene": i, "path": clip_path})

        # 2. Splice scenes together (single scene needs no concat).
        if len(clips) == 1:
            final_path = clips[0]
        else:
            per_scene_transition = str(scenes[-1].get("transition", transition) or transition)
            c = edit.execute(
                operation="concat", inputs=list(clips), transition=per_scene_transition
            )
            if not c.success or not isinstance(c.output, dict) or not c.output.get("path"):
                return ToolResult(
                    success=False,
                    output={"clips": clips},
                    error=f"concat failed: {c.error or 'no output path'}",
                )
            final_path = c.output["path"]

        # 3. Optional title overlay on the opening.
        if title:
            tx = edit.execute(
                operation="text",
                input=final_path,
                text=title,
                start=0.0,
                end=float(title_seconds),
                position="center",
            )
            if tx.success and isinstance(tx.output, dict) and tx.output.get("path"):
                final_path = tx.output["path"]

        return ToolResult(
            success=True,
            output={
                "path": final_path,
                "scene_count": len(clips),
                "clips": clips,
                "title": title,
                "scenes": progress,
            },
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "scenes": {
                        "type": "array",
                        "description": (
                            "Ordered scenes. Each item: {prompt (required), "
                            "duration (seconds, optional), caption (optional), "
                            "transition ('crossfade'/'none', optional)}."
                        ),
                        "items": {"type": "object"},
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title overlaid on the opening.",
                    },
                    "title_seconds": {
                        "type": "number",
                        "description": "Seconds the title stays visible (default 2).",
                    },
                    "transition": {
                        "type": "string",
                        "description": "Default transition between scenes.",
                        "example": "crossfade",
                    },
                    "backend": {
                        "type": "string",
                        "description": "video_generate backend for all scenes.",
                        "example": "wan",
                    },
                },
                "required": ["scenes"],
            },
        }
