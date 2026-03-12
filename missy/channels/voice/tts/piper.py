"""Piper TTS engine implementation for the Missy voice channel.

Wraps the `piper` command-line binary (https://github.com/rhasspy/piper) to
provide high-quality, on-device neural text-to-speech synthesis.  The binary
communicates over stdin/stdout: text is written to stdin and raw PCM-16 audio
is read from stdout.

Prerequisites::

    # Download piper binary from https://github.com/rhasspy/piper/releases
    # Place it on PATH or pass the full path to PiperTTS(piper_bin=...)
    # Download an ONNX voice model to ~/.local/share/piper-voices/

Example::

    from missy.channels.voice.tts.piper import PiperTTS

    engine = PiperTTS(voice="en_US-lessac-medium")
    engine.load()
    voices = engine.list_voices()
    buffer = await engine.synthesize("Hello, world!")
    # buffer.data contains WAV bytes ready to play
    engine.unload()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import wave
from io import BytesIO
from pathlib import Path

from missy.channels.voice.tts.base import AudioBuffer, TTSEngine
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

# Default directory where Piper voice ONNX models are stored.
_DEFAULT_VOICES_DIR = Path.home() / ".local" / "share" / "piper-voices"

# Fallback path for user-local Piper install.
_LOCAL_PIPER_BIN = Path.home() / ".local" / "bin" / "piper"

# Default sample rate output by Piper (model-dependent; most Piper voices use 22050 Hz).
_DEFAULT_SAMPLE_RATE = 22050

# Session/task IDs forwarded to audit events.
_SESSION_ID = "voice"
_TASK_ID = "tts"


def _piper_subprocess_env() -> dict[str, str]:
    """Return env dict with LD_LIBRARY_PATH set for Piper shared libs."""
    env = {**os.environ}
    piper_lib = str(_LOCAL_PIPER_BIN.parent)
    existing = env.get("LD_LIBRARY_PATH", "")
    if piper_lib not in existing:
        env["LD_LIBRARY_PATH"] = f"{piper_lib}:{existing}" if existing else piper_lib
    return env


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw PCM-16 bytes in a RIFF WAV container.

    Uses the stdlib :mod:`wave` module to write a valid WAV header so that
    the returned bytes can be written directly to a ``.wav`` file or passed
    to any standard audio player.

    Args:
        pcm_bytes: Raw 16-bit signed little-endian PCM audio data.
        sample_rate: Sampling rate in Hz (e.g. ``22050``).
        channels: Number of audio channels (e.g. ``1`` for mono).

    Returns:
        Bytes comprising a complete WAV file with header and audio data.
    """
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM → 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class PiperTTS(TTSEngine):
    """Text-to-speech engine that delegates synthesis to the Piper binary.

    Piper is invoked as a subprocess for each synthesis request.  Text is
    written to the process's stdin and raw PCM-16 audio is collected from
    stdout.  The raw PCM is then wrapped in a WAV container before being
    returned as an :class:`~missy.channels.voice.tts.base.AudioBuffer`.

    Args:
        model_path: Absolute path to a ``.onnx`` Piper voice model file.
            When ``None`` the engine looks for a model matching :attr:`voice`
            inside ``~/.local/share/piper-voices/``.
        piper_bin: Name or absolute path of the Piper executable.  Resolved
            via :func:`shutil.which` if a bare name is given.  Defaults to
            ``"piper"``.
        voice: Voice name used both as the default synthesis voice and as a
            filename stem when locating a model automatically (e.g.
            ``"en_US-lessac-medium"`` maps to
            ``~/.local/share/piper-voices/en_US-lessac-medium.onnx``).

    Raises:
        RuntimeError: On :meth:`load` if the Piper binary cannot be located
            or the requested model file does not exist.

    Example::

        engine = PiperTTS(voice="en_US-lessac-medium")
        engine.load()
        buffer = await engine.synthesize("Hello!")
        engine.unload()
    """

    name = "piper"

    def __init__(
        self,
        model_path: str | None = None,
        piper_bin: str = "piper",
        voice: str = "en_US-lessac-medium",
    ) -> None:
        self._model_path_arg: str | None = model_path
        self._piper_bin_arg: str = piper_bin
        self._voice: str = voice

        # Resolved values populated during load().
        self._piper_bin: str | None = None
        self._model_file: Path | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Verify the Piper binary and model file are accessible.

        Resolves the Piper binary path via :func:`shutil.which` and locates
        the ONNX model file.  Sets :attr:`_loaded` to ``True`` on success.

        Raises:
            RuntimeError: If the Piper binary is not found on PATH or at the
                specified path, or if the model file cannot be located.
        """
        if self._loaded:
            return  # Idempotent.

        # Resolve binary — check PATH first, then ~/.local/bin/piper.
        resolved_bin = shutil.which(self._piper_bin_arg)
        if resolved_bin is None and _LOCAL_PIPER_BIN.is_file():
            resolved_bin = str(_LOCAL_PIPER_BIN)
        if resolved_bin is None:
            raise RuntimeError(
                f"Piper binary '{self._piper_bin_arg}' not found. "
                "Download from https://github.com/rhasspy/piper/releases "
                "and place it on PATH or at ~/.local/bin/piper, "
                "or pass piper_bin='/absolute/path/to/piper'."
            )
        self._piper_bin = resolved_bin

        # Locate model file.
        self._model_file = self._resolve_model()
        self._loaded = True

        logger.info(
            "PiperTTS: loaded — binary=%s model=%s voice=%s",
            self._piper_bin,
            self._model_file,
            self._voice,
        )

    def unload(self) -> None:
        """No-op — Piper runs as short-lived subprocesses; nothing to release.

        Sets :attr:`_loaded` to ``False`` to prevent synthesis after unload.
        """
        self._loaded = False
        self._piper_bin = None
        self._model_file = None
        logger.info("PiperTTS: unloaded")

    def is_loaded(self) -> bool:
        """Return ``True`` if :meth:`load` completed successfully.

        Returns:
            Boolean indicating whether the engine is ready for synthesis.
        """
        return self._loaded

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    def list_voices(self) -> list[str]:
        """Scan the default voices directory for available ONNX models.

        Searches ``~/.local/share/piper-voices/`` for files with a ``.onnx``
        extension and returns their base names (without the ``.onnx`` suffix).

        Returns:
            Sorted list of voice name strings, or an empty list if the voices
            directory does not exist or contains no ONNX files.
        """
        if not _DEFAULT_VOICES_DIR.is_dir():
            return []
        voices = sorted(p.stem for p in _DEFAULT_VOICES_DIR.glob("*.onnx"))
        return voices

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> AudioBuffer:
        """Synthesise *text* to speech using the Piper subprocess.

        Spawns a Piper subprocess, writes *text* to its stdin, reads raw
        PCM-16 from stdout, wraps the PCM in a WAV container, and returns
        an :class:`~missy.channels.voice.tts.base.AudioBuffer`.

        Args:
            text: The plain-text string to synthesise.
            voice: Optional voice name.  When provided and different from the
                engine's configured :attr:`_voice`, a model file for the new
                voice is resolved at synthesis time.  When ``None`` the
                engine's default voice and model are used.

        Returns:
            An :class:`~missy.channels.voice.tts.base.AudioBuffer` with
            ``format="wav"`` containing the synthesised audio.

        Raises:
            RuntimeError: If :meth:`load` has not been called, the Piper
                subprocess returns a non-zero exit code, or no audio data
                is produced.
        """
        if not self._loaded or self._piper_bin is None or self._model_file is None:
            raise RuntimeError(
                "PiperTTS.load() must be called before synthesize()."
            )

        # Resolve model for override voice, or use the loaded default.
        if voice is not None and voice != self._voice:
            model_file = self._resolve_model_for_voice(voice)
            effective_voice = voice
        else:
            model_file = self._model_file
            effective_voice = self._voice

        cmd = [
            self._piper_bin,
            "--model", str(model_file),
            "--output_raw",
            "--quiet",
        ]

        t_start = time.perf_counter()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_piper_subprocess_env(),
        )

        # Write text followed by a newline; Piper processes one utterance per line.
        stdin_data = (text + "\n").encode("utf-8")
        pcm_bytes, stderr_bytes = await proc.communicate(input=stdin_data)

        t_end = time.perf_counter()
        processing_ms = int((t_end - t_start) * 1000)

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Piper exited with code {proc.returncode}: {stderr_text}"
            )

        if not pcm_bytes:
            raise RuntimeError(
                "Piper produced no audio output. "
                "Check that the model file is valid and the input text is non-empty."
            )

        wav_bytes = _pcm_to_wav(pcm_bytes, _DEFAULT_SAMPLE_RATE, channels=1)

        buffer = AudioBuffer(
            data=wav_bytes,
            sample_rate=_DEFAULT_SAMPLE_RATE,
            channels=1,
            format="wav",
        )

        self._emit_audit(
            voice=effective_voice,
            text_length=len(text),
            duration_ms=buffer.duration_ms,
            processing_ms=processing_ms,
        )

        return buffer

    # ------------------------------------------------------------------
    # Model resolution helpers
    # ------------------------------------------------------------------

    def _resolve_model(self) -> Path:
        """Locate and validate the ONNX model file.

        Checks :attr:`_model_path_arg` first; if ``None`` falls back to the
        default voices directory.

        Returns:
            Resolved :class:`~pathlib.Path` to the ONNX model file.

        Raises:
            RuntimeError: If the model file cannot be found.
        """
        if self._model_path_arg is not None:
            path = Path(self._model_path_arg).expanduser().resolve()
            if not path.is_file():
                raise RuntimeError(
                    f"Piper model file not found: {path}"
                )
            return path

        return self._resolve_model_for_voice(self._voice)

    def _resolve_model_for_voice(self, voice: str) -> Path:
        """Resolve an ONNX model file for *voice* from the default voices dir.

        Args:
            voice: Voice name stem, e.g. ``"en_US-lessac-medium"``.

        Returns:
            Resolved :class:`~pathlib.Path` to the ONNX model file.

        Raises:
            RuntimeError: If no matching ``.onnx`` file is found.
        """
        candidate = (_DEFAULT_VOICES_DIR / voice).with_suffix(".onnx")
        if candidate.is_file():
            return candidate

        raise RuntimeError(
            f"Piper voice model '{voice}.onnx' not found in {_DEFAULT_VOICES_DIR}. "
            "Download a voice from https://github.com/rhasspy/piper#voices and "
            f"place the .onnx file in {_DEFAULT_VOICES_DIR}, "
            "or pass model_path='/path/to/your-voice.onnx'."
        )

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def _emit_audit(
        self,
        *,
        voice: str,
        text_length: int,
        duration_ms: int,
        processing_ms: int,
    ) -> None:
        """Publish a ``voice.tts.complete`` audit event.

        Args:
            voice: Effective voice name used for synthesis.
            text_length: Character count of the input text.
            duration_ms: Computed playback duration of the synthesised audio.
            processing_ms: Wall-clock synthesis time.
        """
        try:
            event = AuditEvent.now(
                session_id=_SESSION_ID,
                task_id=_TASK_ID,
                event_type="voice.tts.complete",
                # Use "plugin" because "channel" is not yet in EventCategory.
                category="plugin",
                result="allow",
                detail={
                    "engine": self.name,
                    "voice": voice,
                    "text_length": text_length,
                    "duration_ms": duration_ms,
                    "processing_ms": processing_ms,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.debug("PiperTTS: audit emit failed", exc_info=True)
