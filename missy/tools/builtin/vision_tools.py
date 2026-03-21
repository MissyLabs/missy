"""Vision tools for the Missy agent framework.

Provides tools that the agent can invoke to capture images, inspect visual
content, and perform domain-specific analysis.  All tools integrate with
the vision subsystem in ``missy/vision/``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VisionCaptureTool
# ---------------------------------------------------------------------------


class VisionCaptureTool(BaseTool):
    """Capture an image from a webcam or other source.

    Saves the image to ~/.missy/captures/ and returns the file path
    along with metadata.  Use discord_upload_file to share the image.
    """

    name = "vision_capture"
    description = (
        "Capture an image from a USB webcam, file, or screenshot. "
        "Saves to ~/.missy/captures/ and returns the file path and metadata. "
        "Use discord_upload_file with the saved_to path to share the image."
    )
    permissions = ToolPermissions(filesystem_read=True, filesystem_write=True)
    parameters = {
        "source": {
            "type": "string",
            "description": (
                "Image source: 'webcam' (default), 'screenshot', or a file path."
            ),
            "required": False,
        },
        "device": {
            "type": "string",
            "description": "Camera device path (e.g. /dev/video0). Auto-detected if omitted.",
            "required": False,
        },
        "save_path": {
            "type": "string",
            "description": "Optional path to save the captured image.",
            "required": False,
        },
    }

    def execute(
        self,
        *,
        source: str = "webcam",
        device: str = "",
        save_path: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            from missy.vision.pipeline import ImagePipeline
            from missy.vision.sources import (
                FileSource,
                ScreenshotSource,
                WebcamSource,
            )

            # Determine source
            if source in ("webcam", "camera"):
                if not device:
                    from missy.vision.discovery import find_preferred_camera
                    cam = find_preferred_camera()
                    if cam is None:
                        return ToolResult(
                            success=False,
                            output=None,
                            error="No camera found. Connect a USB webcam.",
                        )
                    device = cam.device_path
                img_source = WebcamSource(device)
            elif source == "screenshot":
                img_source = ScreenshotSource()
            else:
                # Treat as file path
                img_source = FileSource(source)

            if not img_source.is_available():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Image source not available: {source}",
                )

            frame = img_source.acquire()

            # Preprocess
            pipeline = ImagePipeline()
            processed = pipeline.process(frame.image)
            quality = pipeline.assess_quality(frame.image)

            # Always save to disk — base64 is too large for tool results
            # and will cause Discord 400 errors.
            import cv2

            if not save_path:
                captures_dir = Path.home() / ".missy" / "captures"
                captures_dir.mkdir(parents=True, exist_ok=True)
                ts = frame.timestamp.strftime("%Y%m%d_%H%M%S")
                save_path = str(captures_dir / f"capture_{ts}.jpg")

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                save_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            result = {
                "source_type": frame.source_type.value,
                "source_path": frame.source_path,
                "width": frame.width,
                "height": frame.height,
                "quality": quality,
                "saved_to": save_path,
                "timestamp": frame.timestamp.isoformat(),
            }

            return ToolResult(success=True, output=json.dumps(result))

        except ImportError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Vision dependencies not installed: {exc}. Install with: pip install -e '.[vision]'",
            )
        except Exception as exc:
            logger.error("Vision capture failed: %s", exc, exc_info=True)
            return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# VisionBurstCaptureTool
# ---------------------------------------------------------------------------


class VisionBurstCaptureTool(BaseTool):
    """Capture a rapid burst of frames and optionally pick the sharpest.

    Useful for motion tasks, selecting the clearest frame, or recording
    a sequence of changes.
    """

    name = "vision_burst"
    description = (
        "Capture a burst of frames from a webcam. Returns metadata for each "
        "frame or just the sharpest frame if best_only is true."
    )
    permissions = ToolPermissions(filesystem_read=True, filesystem_write=True)
    parameters = {
        "count": {
            "type": "integer",
            "description": "Number of frames to capture (1-20, default 3).",
            "required": False,
        },
        "interval": {
            "type": "number",
            "description": "Seconds between captures (default 0.3).",
            "required": False,
        },
        "best_only": {
            "type": "boolean",
            "description": "If true, return only the sharpest frame from the burst.",
            "required": False,
        },
        "device": {
            "type": "string",
            "description": "Camera device path. Auto-detected if omitted.",
            "required": False,
        },
    }

    def execute(
        self,
        *,
        count: int = 3,
        interval: float = 0.3,
        best_only: bool = False,
        device: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            from missy.vision.capture import CameraHandle
            from missy.vision.discovery import find_preferred_camera
            from missy.vision.pipeline import ImagePipeline

            if not device:
                cam = find_preferred_camera()
                if cam is None:
                    return ToolResult(
                        success=False, output=None,
                        error="No camera found. Connect a USB webcam.",
                    )
                device = cam.device_path

            handle = CameraHandle(device)
            pipeline = ImagePipeline()

            try:
                handle.open()

                if best_only:
                    result = handle.capture_best(burst_count=count)
                    if not result.success:
                        return ToolResult(
                            success=False, output=None,
                            error=result.error or "No frames captured",
                        )

                    processed = pipeline.process(result.image)
                    quality = pipeline.assess_quality(result.image)

                    import cv2

                    captures_dir = Path.home() / ".missy" / "captures"
                    captures_dir.mkdir(parents=True, exist_ok=True)
                    save_path = str(
                        captures_dir / f"burst_best_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(
                        save_path, processed,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )

                    return ToolResult(
                        success=True,
                        output=json.dumps({
                            "mode": "best",
                            "burst_count": count,
                            "width": result.width,
                            "height": result.height,
                            "quality": quality,
                            "saved_to": save_path,
                        }),
                    )

                # Full burst
                results = handle.capture_burst(count=count, interval=interval)
                frames = []
                for i, r in enumerate(results):
                    if r.success:
                        quality = pipeline.assess_quality(r.image)
                        frames.append({
                            "index": i,
                            "width": r.width,
                            "height": r.height,
                            "quality": quality,
                            "success": True,
                        })
                    else:
                        frames.append({"index": i, "success": False, "error": r.error})

                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "mode": "burst",
                        "count": count,
                        "interval": interval,
                        "frames": frames,
                        "successful": sum(1 for f in frames if f.get("success")),
                    }),
                )

            finally:
                handle.close()

        except ImportError as exc:
            return ToolResult(
                success=False, output=None,
                error=f"Vision dependencies not installed: {exc}",
            )
        except Exception as exc:
            logger.error("Vision burst capture failed: %s", exc, exc_info=True)
            return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# VisionAnalyzeTool
# ---------------------------------------------------------------------------


class VisionAnalyzeTool(BaseTool):
    """Analyze a captured image with domain-specific prompts.

    Builds analysis prompts for puzzle assistance, painting feedback,
    or general inspection.  Returns the prompt for the agent to use
    with the image when calling the LLM.
    """

    name = "vision_analyze"
    description = (
        "Build a domain-specific analysis prompt for a captured image. "
        "Supports modes: general, puzzle, painting, inspection."
    )
    permissions = ToolPermissions()
    parameters = {
        "mode": {
            "type": "string",
            "description": "Analysis mode: 'general', 'puzzle', 'painting', 'inspection'.",
            "required": True,
        },
        "context": {
            "type": "string",
            "description": "Additional context or user question about the image.",
            "required": False,
        },
        "is_followup": {
            "type": "boolean",
            "description": "Whether this is a follow-up analysis (uses scene memory).",
            "required": False,
        },
    }

    def execute(
        self,
        *,
        mode: str = "general",
        context: str = "",
        is_followup: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            import numpy as np

            from missy.vision.analysis import (
                AnalysisMode,
                AnalysisPromptBuilder,
                AnalysisRequest,
            )
            from missy.vision.scene_memory import get_scene_manager

            analysis_mode = AnalysisMode(mode)

            # Get scene memory context for follow-ups
            previous_observations: list[str] = []
            previous_state: dict[str, Any] = {}

            if is_followup:
                mgr = get_scene_manager()
                session = mgr.get_active_session()
                if session:
                    previous_observations = session.observations
                    previous_state = session.state

            request = AnalysisRequest(
                image=np.zeros((1, 1, 3), dtype=np.uint8),  # placeholder
                mode=analysis_mode,
                context=context,
                previous_observations=previous_observations,
                previous_state=previous_state,
                is_followup=is_followup,
            )

            builder = AnalysisPromptBuilder()
            prompt = builder.build_prompt(request)

            return ToolResult(
                success=True,
                output=json.dumps({
                    "mode": mode,
                    "prompt": prompt,
                    "is_followup": is_followup,
                    "has_scene_memory": bool(previous_observations),
                }),
            )

        except Exception as exc:
            logger.error("Vision analyze failed: %s", exc, exc_info=True)
            return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# VisionDevicesTool
# ---------------------------------------------------------------------------


class VisionDevicesTool(BaseTool):
    """List available camera devices."""

    name = "vision_devices"
    description = "Enumerate available USB cameras and their details."
    permissions = ToolPermissions(filesystem_read=True)
    parameters = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            from missy.vision.discovery import KNOWN_CAMERAS, CameraDiscovery

            disc = CameraDiscovery()
            cameras = disc.discover(force=True)

            devices = []
            for cam in cameras:
                devices.append({
                    "device_path": cam.device_path,
                    "name": cam.name,
                    "usb_id": cam.usb_id,
                    "bus_info": cam.bus_info,
                    "known_model": KNOWN_CAMERAS.get(cam.usb_id, ""),
                })

            preferred = disc.find_preferred()
            result = {
                "camera_count": len(cameras),
                "cameras": devices,
                "preferred": preferred.device_path if preferred else None,
            }

            return ToolResult(success=True, output=json.dumps(result))

        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# VisionSceneMemoryTool
# ---------------------------------------------------------------------------


class VisionSceneMemoryTool(BaseTool):
    """Manage vision scene memory for multi-step visual tasks."""

    name = "vision_scene"
    description = (
        "Manage scene memory for multi-step visual tasks. "
        "Actions: create, add_observation, update_state, summarize, close."
    )
    permissions = ToolPermissions()
    parameters = {
        "action": {
            "type": "string",
            "description": "Action: 'create', 'add_observation', 'update_state', 'summarize', 'close'.",
            "required": True,
        },
        "task_id": {
            "type": "string",
            "description": "Task identifier for the scene session.",
            "required": False,
        },
        "task_type": {
            "type": "string",
            "description": "Task type: 'puzzle', 'painting', 'general', 'inspection'.",
            "required": False,
        },
        "observation": {
            "type": "string",
            "description": "Observation text to record (for add_observation action).",
            "required": False,
        },
        "state_updates": {
            "type": "string",
            "description": "JSON string of state updates (for update_state action).",
            "required": False,
        },
    }

    def execute(
        self,
        *,
        action: str,
        task_id: str = "",
        task_type: str = "general",
        observation: str = "",
        state_updates: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            from missy.vision.scene_memory import TaskType, get_scene_manager

            mgr = get_scene_manager()

            if action == "create":
                if not task_id:
                    task_id = f"task_{int(time.time())}"
                tt = TaskType(task_type)
                session = mgr.create_session(task_id, tt)
                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "action": "created",
                        "task_id": task_id,
                        "task_type": task_type,
                    }),
                )

            elif action == "add_observation":
                session = mgr.get_session(task_id) or mgr.get_active_session()
                if not session:
                    return ToolResult(
                        success=False, output=None, error="No active scene session"
                    )
                session.add_observation(observation)
                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "action": "observation_added",
                        "task_id": session.task_id,
                        "total_observations": len(session.observations),
                    }),
                )

            elif action == "update_state":
                session = mgr.get_session(task_id) or mgr.get_active_session()
                if not session:
                    return ToolResult(
                        success=False, output=None, error="No active scene session"
                    )
                updates = json.loads(state_updates) if state_updates else {}
                session.update_state(**updates)
                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "action": "state_updated",
                        "task_id": session.task_id,
                        "state": session.state,
                    }),
                )

            elif action == "summarize":
                session = mgr.get_session(task_id) or mgr.get_active_session()
                if not session:
                    sessions = mgr.list_sessions()
                    return ToolResult(
                        success=True,
                        output=json.dumps({"sessions": sessions}),
                    )
                return ToolResult(
                    success=True,
                    output=json.dumps(session.summarize()),
                )

            elif action == "close":
                if task_id:
                    mgr.close_session(task_id)
                else:
                    session = mgr.get_active_session()
                    if session:
                        mgr.close_session(session.task_id)
                return ToolResult(
                    success=True,
                    output=json.dumps({"action": "closed"}),
                )

            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}. Use: create, add_observation, update_state, summarize, close",
                )

        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))
