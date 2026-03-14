"""Faster-Whisper STT engine implementation for the Missy voice channel.

Wraps the ``faster-whisper`` library (CTranslate2-based Whisper) to provide
low-latency, on-device speech recognition.

Install the optional dependency before use::

    pip install faster-whisper

Example::

    from missy.channels.voice.stt.whisper import FasterWhisperSTT

    engine = FasterWhisperSTT(model_size="base.en", device="auto")
    engine.load()
    result = await engine.transcribe(pcm_bytes, sample_rate=16000)
    print(result.text, result.confidence, result.processing_ms)
    engine.unload()
"""

from __future__ import annotations

import logging
import time
from typing import Any

from missy.channels.voice.stt.base import STTEngine, TranscriptionResult
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

# Session/task IDs used when emitting audit events from this module.
_SESSION_ID = "voice"
_TASK_ID = "stt"


class FasterWhisperSTT(STTEngine):
    """Speech-to-text engine backed by the ``faster-whisper`` library.

    Uses CTranslate2 as the inference runtime for significantly faster
    transcription than the reference OpenAI Whisper implementation.  The
    engine automatically selects compute device and precision unless both are
    specified explicitly.

    Args:
        model_size: Whisper model variant to load.  Accepted values include
            ``"tiny"``, ``"base"``, ``"base.en"``, ``"small"``, ``"medium"``,
            ``"large-v2"``, ``"large-v3"``, etc.  English-only ``.en`` variants
            are faster for English-only workloads.
        device: Target inference device.  ``"auto"`` (default) tries CUDA
            first and falls back to CPU when CUDA is unavailable.  Pass
            ``"cpu"`` or ``"cuda"`` to force a specific device.
        compute_type: CTranslate2 quantisation mode.  ``"auto"`` (default)
            selects ``"int8"`` for CPU and ``"float16"`` for GPU.  Explicit
            values such as ``"int8"``, ``"int8_float16"``, ``"float16"``, or
            ``"float32"`` are forwarded verbatim to ``WhisperModel``.

    Example::

        engine = FasterWhisperSTT(model_size="small", device="cpu")
        engine.load()
        result = await engine.transcribe(audio_bytes)
        engine.unload()
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None  # faster_whisper.WhisperModel
        self._resolved_device: str | None = None
        self._resolved_compute_type: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the Whisper model into memory.

        Attempts to import ``faster_whisper`` at call time so that the module
        can be imported without the optional dependency installed.  Device and
        compute type are resolved here if ``"auto"`` was requested.

        Raises:
            ImportError: If ``faster-whisper`` is not installed.
            RuntimeError: If the underlying model fails to initialise.
        """
        if self._model is not None:
            return  # Already loaded — idempotent.

        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install faster-whisper: pip install faster-whisper") from exc

        device, compute_type = self._resolve_device_and_compute()
        self._resolved_device = device
        self._resolved_compute_type = compute_type

        logger.info(
            "FasterWhisperSTT: loading model=%s device=%s compute_type=%s",
            self._model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("FasterWhisperSTT: model loaded")

    def unload(self) -> None:
        """Release the Whisper model from memory.

        Sets the internal model reference to ``None``.  The CTranslate2
        runtime will free GPU/CPU memory when the object is garbage-collected.
        """
        self._model = None
        self._resolved_device = None
        self._resolved_compute_type = None
        logger.info("FasterWhisperSTT: model unloaded")

    def is_loaded(self) -> bool:
        """Return ``True`` if the model is resident in memory.

        Returns:
            Boolean indicating readiness for inference.
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> TranscriptionResult:
        """Transcribe raw PCM-16 audio bytes into text.

        Converts the raw PCM bytes to a normalised float32 numpy array, runs
        Whisper beam-search decoding, and aggregates all output segments into
        a single transcript.  Confidence is derived from the per-segment
        ``no_speech_prob`` values when available.

        Args:
            audio: Raw 16-bit signed little-endian PCM audio data.
            sample_rate: Sampling rate in Hz.  Defaults to ``16000``.
            channels: Number of audio channels.  Defaults to ``1`` (mono).
                Multi-channel audio is mixed down to mono by averaging across
                channels before transcription.

        Returns:
            A :class:`~missy.channels.voice.stt.base.TranscriptionResult`
            containing the transcript, confidence, wall-clock processing time,
            and detected language code.

        Raises:
            RuntimeError: If :meth:`load` has not been called.
            ImportError: If ``numpy`` is not available.
        """
        if self._model is None:
            raise RuntimeError("FasterWhisperSTT.load() must be called before transcribe().")

        try:
            import numpy as np  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("numpy is required for audio conversion: pip install numpy") from exc

        # Convert raw PCM-16 bytes → float32 array normalised to [-1.0, 1.0].
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Mix down multi-channel audio to mono.
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)

        t_start = time.perf_counter()

        # faster_whisper returns a generator of segments plus metadata.
        segments_gen, info = self._model.transcribe(
            audio_array,
            beam_size=5,
            language=None,  # auto-detect language
        )

        # Materialise generator to collect all segments.
        segments = list(segments_gen)

        t_end = time.perf_counter()
        processing_ms = int((t_end - t_start) * 1000)

        # Concatenate text from every segment.
        text = "".join(seg.text for seg in segments).strip()

        # Derive confidence from no_speech_prob if available on segments.
        no_speech_probs: list[float] = []
        for seg in segments:
            prob = getattr(seg, "no_speech_prob", None)
            if prob is not None:
                no_speech_probs.append(float(prob))

        if no_speech_probs:
            confidence = float(1.0 - (sum(no_speech_probs) / len(no_speech_probs)))
            # Clamp to [0.0, 1.0] in case of floating-point drift.
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = -1.0

        # Detected language is available on the info object.
        language: str = getattr(info, "language", "") or ""

        result = TranscriptionResult(
            text=text,
            confidence=confidence,
            processing_ms=processing_ms,
            language=language,
        )

        self._emit_audit(result)
        return result

    # ------------------------------------------------------------------
    # Device/compute resolution
    # ------------------------------------------------------------------

    def _resolve_device_and_compute(self) -> tuple[str, str]:
        """Resolve ``"auto"`` device and compute_type to concrete values.

        Returns:
            A ``(device, compute_type)`` tuple of resolved string values.
        """
        device = self._device
        compute_type = self._compute_type

        if device == "auto":
            device = self._detect_device()

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        return device, compute_type

    @staticmethod
    def _detect_device() -> str:
        """Probe for CUDA availability; fall back to CPU.

        Returns:
            ``"cuda"`` if a CUDA-capable GPU is available, otherwise ``"cpu"``.
        """
        try:
            import torch  # type: ignore[import]

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        # CTranslate2 exposes its own CUDA check.
        try:
            import ctranslate2  # type: ignore[import]

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except (ImportError, Exception):
            pass

        return "cpu"

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _emit_audit(self, result: TranscriptionResult) -> None:
        """Publish a ``voice.stt.complete`` audit event.

        Args:
            result: The completed :class:`TranscriptionResult`.
        """
        try:
            event = AuditEvent.now(
                session_id=_SESSION_ID,
                task_id=_TASK_ID,
                event_type="voice.stt.complete",
                # Use "plugin" because "channel" is not yet in EventCategory.
                category="plugin",
                result="allow",
                detail={
                    "model_size": self._model_size,
                    "device": self._resolved_device or self._device,
                    "compute_type": self._resolved_compute_type or self._compute_type,
                    "processing_ms": result.processing_ms,
                    "text_length": len(result.text),
                    "language": result.language,
                    "confidence": result.confidence,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.debug("FasterWhisperSTT: audit emit failed", exc_info=True)
