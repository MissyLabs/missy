"""Built-in tool: multi-scene video storyboard orchestration (F16, Part III).

A single-call composition of the existing ``video_generate`` and ``video_edit``
tools: given an ordered list of *scenes* (each a prompt + optional duration,
caption, transition, and per-scene generation overrides), it generates a clip
per scene, trims each to its requested duration, overlays per-scene captions,
splices them (honouring per-scene transitions), overlays an optional title,
optionally lays a continuous soundtrack over the assembly, and returns the
final MP4. This is the documented "iteration is just calling the tool again"
contract packaged into one deterministic orchestration so the model doesn't
have to hand-chain a dozen calls.

Coherence: ``continuity=True`` chains scenes visually — the last frame of each
scene's finished clip is extracted (``video_edit`` ``extract_frame``) and fed
as the next scene's ``image_path``, so an i2v-capable backend (``wan``/``svd``)
starts every scene exactly where the previous one ended. This is the standard
multi-scene technique for Wan 2.2 (see ``video.md`` Part III).

Reproducibility: the output's ``scenes`` list records each scene's echoed
``seed``/``prompt``/``path``, and scenes accept per-scene overrides (``seed``,
``negative_prompt``, ``image_path``, ``audio_prompt``, ``steps``, ``cfg``,
``video_frames``), so a single bad scene can be re-rolled or reproduced
without regenerating the rest.

The heavy lifting is delegated to the two underlying tools, which enforce their
own GPU/model preflights and produce real files; this tool only sequences them
and reports honest per-scene progress. The underlying tools are injectable for
testing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_SCENES = 12
_VALID_TRANSITIONS = frozenset({"none", "crossfade"})
# Backends whose generation can be seeded from an image (continuity chaining).
_I2V_BACKENDS = frozenset({"wan", "svd"})
# Scene keys forwarded to video_generate as per-scene overrides of the shared
# kwargs. A whitelist: everything else in a scene dict is storyboard-level
# vocabulary (prompt/duration/caption/transition), not generation knobs.
_SCENE_OVERRIDE_KEYS = (
    "seed",
    "negative_prompt",
    "image_path",
    "audio_prompt",
    "steps",
    "cfg",
    "video_frames",
)


class VideoStoryboardTool(BaseTool):
    """Generate, trim, caption, splice, title, and score a multi-scene video."""

    name = "video_storyboard"
    description = (
        "Produce a multi-scene video from an ordered list of scenes. Each scene "
        "has a 'prompt' and optional 'duration' (seconds), 'caption' (overlaid "
        "on that scene), 'transition' ('crossfade'/'none' -- the join INTO that "
        "scene), and per-scene overrides (seed, negative_prompt, image_path, "
        "audio_prompt, steps, cfg, video_frames). Generates a clip per scene, "
        "trims, captions, splices, adds an optional overall 'title', and can "
        "lay one continuous soundtrack over the result (audio_path). "
        "continuity=true visually chains scenes by starting each one from the "
        "previous scene's last frame (wan/svd backends). The output lists each "
        "scene's seed so any single scene can be re-rolled. Returns the final "
        "MP4 path."
    )
    # Union of what the delegated tools need: video_generate (network → ComfyUI)
    # and video_edit (shell → ffmpeg, filesystem). So the storyboard tool is
    # gated at the registry with the same posture as its parts.
    permissions = ToolPermissions(
        network=True, shell=True, filesystem_read=True, filesystem_write=True
    )

    def __init__(
        self,
        generate_tool: Any = None,
        edit_tool: Any = None,
        child_executor: Callable[..., ToolResult] | None = None,
    ) -> None:
        self._generate_tool = generate_tool
        self._edit_tool = edit_tool
        if child_executor is not None:
            self._child_executor = child_executor
        elif generate_tool is not None or edit_tool is not None:
            # Explicit dependency injection is a library/test seam. Production
            # construction supplies neither object and always uses the live
            # registry reference monitor below.
            injected = {"video_generate": generate_tool, "video_edit": edit_tool}

            def execute_injected(name: str, **kwargs: Any) -> ToolResult:
                tool = injected.get(name)
                if tool is None:
                    raise RuntimeError(f"No injected child tool for {name!r}")
                method = getattr(tool, "execute")  # noqa: B009 - deliberate injection seam
                return method(**kwargs)

            self._child_executor = execute_injected
        else:
            self._child_executor = self._execute_registered_child

    @staticmethod
    def _execute_registered_child(name: str, **kwargs: Any) -> ToolResult:
        from missy.tools.registry import get_tool_registry

        return get_tool_registry().execute(name, **kwargs)

    def _gen(self) -> Any:
        if self._generate_tool is None:
            from missy.tools.registry import get_tool_registry

            self._generate_tool = get_tool_registry().get("video_generate")
            if self._generate_tool is None:
                raise RuntimeError("video_generate is not registered")
        return self._generate_tool

    def _edit(self) -> Any:
        if self._edit_tool is None:
            from missy.tools.registry import get_tool_registry

            self._edit_tool = get_tool_registry().get("video_edit")
            if self._edit_tool is None:
                raise RuntimeError("video_edit is not registered")
        return self._edit_tool

    # -- policy declarations delegate to the underlying tools ---------------

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        try:
            return self._gen().resolve_network_hosts(kwargs)
        except Exception:
            return []

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "ffmpeg && ffprobe"

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        # Resolver failures must propagate to ToolRegistry's fail-closed
        # policy path.  Returning an empty mapping previously both violated
        # the resolver contract and could make an unresolved composite look
        # like it touched no files.
        return self._edit().resolve_filesystem_targets(kwargs)

    # -- orchestration helpers ----------------------------------------------

    @staticmethod
    def _scene_transitions(scenes: list[dict], default: str) -> list[str]:
        """The join *into* each scene beyond the first: ``joins[i]`` is the
        transition between scene ``i`` and scene ``i+1``."""
        return [str(s.get("transition") or default).strip().lower() for s in scenes[1:]]

    def _concat_all(self, clips: list[str], joins: list[str]) -> ToolResult:
        """Splice ``clips`` honouring the per-join transitions.

        Uniform joins collapse into a single concat call (one encode);
        mixed joins left-fold pairwise so each join gets its own
        transition.
        """
        if len(set(joins)) == 1:
            return self._child_executor(
                "video_edit", operation="concat", inputs=list(clips), transition=joins[0]
            )
        result = ToolResult(success=True, output={"path": clips[0]})
        for idx, join in enumerate(joins):
            result = self._child_executor(
                "video_edit",
                operation="concat",
                inputs=[result.output["path"], clips[idx + 1]],
                transition=join,
            )
            if not result.success or not isinstance(result.output, dict):
                return result
        return result

    # -- orchestration ------------------------------------------------------

    def execute(
        self,
        *,
        scenes: list[dict] | None = None,
        title: str = "",
        title_seconds: float = 2.0,
        transition: str = "crossfade",
        backend: str = "wan",
        continuity: bool = False,
        audio_path: str = "",
        audio_mode: str = "replace",
        audio_loop: bool = False,
        **shared: Any,
    ) -> ToolResult:
        """Generate and assemble a storyboard from *scenes*.

        Args:
            scenes: Ordered list of ``{prompt, duration?, caption?,
                transition?, seed?, negative_prompt?, image_path?,
                audio_prompt?, steps?, cfg?, video_frames?}``. A scene's
                ``transition`` is the join *into* that scene (ignored on
                the first scene).
            title: Optional overall title overlaid on the opening.
            title_seconds: How long the title stays visible (0.5-30 s).
            transition: Default transition between scenes when a scene
                doesn't specify its own ("crossfade" or "none").
            backend: video_generate backend for every scene.
            continuity: Start each scene from the previous scene's last
                frame (extracted via video_edit) so the story flows
                visually. Requires an image-capable backend (wan/svd);
                a scene's own ``image_path`` takes precedence.
            audio_path: Optional audio file laid over the *final*
                assembled video as one continuous soundtrack.
            audio_mode: "replace" (default) or "mix" for ``audio_path``.
            audio_loop: Loop a short ``audio_path`` to the video length.
            **shared: Extra kwargs forwarded to every video_generate call
                (e.g. width/height/fps/seed); per-scene overrides win.

        Returns:
            A ToolResult whose output describes the final MP4 and, per
            scene, the clip path and echoed seed (for re-rolling any
            single scene).
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
        transition = (transition or "crossfade").strip().lower()
        joins = self._scene_transitions(scenes, transition)
        bad = sorted({t for t in [transition, *joins] if t not in _VALID_TRANSITIONS})
        if bad:
            return ToolResult(
                success=False,
                output=None,
                error=f"invalid transition(s) {bad}; valid: {sorted(_VALID_TRANSITIONS)}",
            )
        title_seconds = max(0.5, min(30.0, float(title_seconds)))
        backend = (backend or "wan").strip().lower()
        if continuity and backend not in _I2V_BACKENDS:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"continuity=True needs an image-to-video backend "
                    f"({sorted(_I2V_BACKENDS)}); backend={backend!r} is text-to-video only."
                ),
            )
        if continuity and backend == "svd" and not scenes[0].get("image_path"):
            return ToolResult(
                success=False,
                output=None,
                error="backend='svd' with continuity=True needs scene 0 to supply 'image_path' "
                "(svd always animates from an image).",
            )

        clips: list[str] = []
        progress: list[dict] = []
        chained_image = ""  # continuity: last frame of the previous scene

        # 1. Generate, trim, and caption each scene.
        for i, scene in enumerate(scenes):
            scene_kwargs = dict(shared)
            for key in _SCENE_OVERRIDE_KEYS:
                if scene.get(key) is not None and key in scene:
                    scene_kwargs[key] = scene[key]
            if continuity and chained_image and not scene.get("image_path"):
                scene_kwargs["image_path"] = chained_image
            g = self._child_executor(
                "video_generate", backend=backend, prompt=str(scene["prompt"]), **scene_kwargs
            )
            if not g.success or not isinstance(g.output, dict) or not g.output.get("path"):
                return ToolResult(
                    success=False,
                    output={"completed_scenes": progress},
                    error=f"scene {i} generation failed: {g.error or 'no output path'}",
                )
            clip_path = g.output["path"]
            duration = scene.get("duration")
            if duration:
                t = self._child_executor(
                    "video_edit",
                    operation="trim",
                    input=clip_path,
                    start=0.0,
                    duration=float(duration),
                )
                if t.success and isinstance(t.output, dict) and t.output.get("path"):
                    clip_path = t.output["path"]
            caption = str(scene.get("caption") or "").strip()
            if caption:
                c = self._child_executor(
                    "video_edit",
                    operation="text",
                    input=clip_path,
                    text=caption,
                    position="bottom",
                )
                if c.success and isinstance(c.output, dict) and c.output.get("path"):
                    clip_path = c.output["path"]
            if continuity and i < len(scenes) - 1:
                # Chain from the exact frame the viewer sees before the cut
                # (post-trim/caption). A failure here is a hard error --
                # silently dropping continuity mid-story defeats its purpose.
                f = self._child_executor(
                    "video_edit", operation="extract_frame", input=clip_path, at=-1.0
                )
                if not f.success or not isinstance(f.output, dict) or not f.output.get("path"):
                    return ToolResult(
                        success=False,
                        output={"completed_scenes": progress},
                        error=(
                            f"continuity frame extraction after scene {i} failed: "
                            f"{f.error or 'no output path'}"
                        ),
                    )
                chained_image = f.output["path"]
            clips.append(clip_path)
            progress.append(
                {
                    "scene": i,
                    "path": clip_path,
                    "prompt": str(scene["prompt"]),
                    "seed": g.output.get("seed"),
                }
            )

        # 2. Splice scenes together (single scene needs no concat).
        if len(clips) == 1:
            final_path = clips[0]
        else:
            c = self._concat_all(clips, joins)
            if not c.success or not isinstance(c.output, dict) or not c.output.get("path"):
                return ToolResult(
                    success=False,
                    output={"clips": clips},
                    error=f"concat failed: {c.error or 'no output path'}",
                )
            final_path = c.output["path"]

        # 3. Optional title overlay on the opening.
        if title:
            tx = self._child_executor(
                "video_edit",
                operation="text",
                input=final_path,
                text=title,
                start=0.0,
                end=float(title_seconds),
                position="center",
            )
            if tx.success and isinstance(tx.output, dict) and tx.output.get("path"):
                final_path = tx.output["path"]

        # 4. Optional continuous soundtrack over the whole assembly.
        if audio_path:
            a = self._child_executor(
                "video_edit",
                operation="audio",
                input=final_path,
                audio_file=audio_path,
                audio_mode=audio_mode,
                loop=bool(audio_loop),
            )
            if not a.success or not isinstance(a.output, dict) or not a.output.get("path"):
                return ToolResult(
                    success=False,
                    output={"clips": clips, "assembled": final_path, "scenes": progress},
                    error=f"soundtrack mux failed: {a.error or 'no output path'}",
                )
            final_path = a.output["path"]

        return ToolResult(
            success=True,
            output={
                "path": final_path,
                "scene_count": len(clips),
                "clips": clips,
                "title": title,
                "continuity": bool(continuity),
                "soundtrack": audio_path or None,
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
                            "duration (seconds, optional), caption (optional, "
                            "overlaid on that scene), transition ('crossfade'/"
                            "'none', the join INTO that scene, optional), plus "
                            "optional per-scene generation overrides: seed, "
                            "negative_prompt, image_path, audio_prompt, steps, "
                            "cfg, video_frames}."
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
                        "enum": sorted(_VALID_TRANSITIONS),
                        "description": (
                            "Default transition between scenes (a scene's own "
                            "'transition' overrides its join)."
                        ),
                    },
                    "backend": {
                        "type": "string",
                        "description": "video_generate backend for all scenes.",
                        "example": "wan",
                    },
                    "continuity": {
                        "type": "boolean",
                        "description": (
                            "Start each scene from the previous scene's last frame "
                            "so the story flows visually (wan/svd backends only; "
                            "default false)."
                        ),
                    },
                    "audio_path": {
                        "type": "string",
                        "description": (
                            "Optional audio file laid over the final assembled "
                            "video as one continuous soundtrack."
                        ),
                    },
                    "audio_mode": {
                        "type": "string",
                        "enum": ["mix", "replace"],
                        "description": (
                            "How audio_path is applied: 'replace' (default) or "
                            "'mix' with existing per-scene audio."
                        ),
                    },
                    "audio_loop": {
                        "type": "boolean",
                        "description": "Loop a short audio_path to the video length.",
                    },
                },
                "required": ["scenes"],
            },
        }
