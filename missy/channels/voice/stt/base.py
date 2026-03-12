"""Abstract speech-to-text engine interface for the Missy voice channel.

Defines the :class:`TranscriptionResult` data class and :class:`STTEngine`
abstract base class that all STT backends must implement.

Example::

    from missy.channels.voice.stt.base import STTEngine, TranscriptionResult
    from missy.channels.voice.stt.whisper import FasterWhisperSTT

    engine = FasterWhisperSTT(model_size="base.en")
    engine.load()
    result = await engine.transcribe(pcm_bytes, sample_rate=16000)
    print(result.text)
    engine.unload()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptionResult:
    """The output produced by an STT engine for a single audio chunk.

    Attributes:
        text: The recognised transcript text.
        confidence: A normalised confidence score in the range ``[0.0, 1.0]``.
            Set to ``-1.0`` when the backend does not report confidence.
        processing_ms: Wall-clock time taken to produce the transcription,
            in milliseconds.
        language: BCP-47 language code detected by the engine (e.g. ``"en"``).
            Empty string when not reported.
    """

    text: str
    confidence: float  # 0.0–1.0; -1.0 if not available
    processing_ms: int
    language: str = field(default="")  # detected language code e.g. "en"


class STTEngine(ABC):
    """Abstract speech-to-text engine.

    All concrete STT backends must subclass :class:`STTEngine` and implement
    every abstract method.  The :attr:`name` class attribute must be set to a
    unique, human-readable identifier for the engine (e.g. ``"faster-whisper"``).

    Lifecycle::

        engine.load()          # initialise model into memory
        result = await engine.transcribe(audio_bytes)
        engine.unload()        # release resources
    """

    #: Unique, human-readable identifier for this engine implementation.
    name: str

    @abstractmethod
    def load(self) -> None:
        """Initialise the engine and load the underlying model into memory.

        Implementations should be idempotent: calling :meth:`load` on an
        already-loaded engine must not raise an error.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources held by the engine.

        After calling this method :meth:`is_loaded` must return ``False``
        and any subsequent call to :meth:`transcribe` must raise
        :exc:`RuntimeError` or equivalent.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return ``True`` if the engine is ready to transcribe audio.

        Returns:
            Boolean indicating whether the underlying model is resident in
            memory and ready for inference.
        """
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> TranscriptionResult:
        """Transcribe raw PCM-16 audio bytes into text.

        Args:
            audio: Raw 16-bit signed little-endian PCM audio data.
            sample_rate: Number of samples per second (Hz).  Defaults to
                ``16000``, which is the native rate for Whisper-class models.
            channels: Number of audio channels.  Defaults to ``1`` (mono).

        Returns:
            A :class:`TranscriptionResult` containing the recognised text,
            confidence score, processing time, and detected language.

        Raises:
            RuntimeError: If the engine has not been loaded via :meth:`load`.
        """
        ...
