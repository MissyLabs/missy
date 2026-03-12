"""Abstract text-to-speech engine interface for the Missy voice channel.

Defines the :class:`AudioBuffer` data class and :class:`TTSEngine` abstract
base class that all TTS backends must implement.

Example::

    from missy.channels.voice.tts.base import TTSEngine, AudioBuffer
    from missy.channels.voice.tts.piper import PiperTTS

    engine = PiperTTS(voice="en_US-lessac-medium")
    engine.load()
    buffer = await engine.synthesize("Hello, world!")
    print(buffer.duration_ms, buffer.format)
    engine.unload()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AudioBuffer:
    """A block of synthesised audio data.

    Attributes:
        data: Raw audio bytes.  The meaning depends on :attr:`format`.
        sample_rate: Number of samples per second (Hz), e.g. ``22050``.
        channels: Number of audio channels.  Usually ``1`` (mono).
        format: Container/encoding identifier.  Currently ``"wav"`` (PCM
            wrapped in a RIFF WAV header) or ``"pcm_s16le"`` (raw 16-bit
            signed little-endian PCM with no header).
        duration_ms: Computed playback duration in milliseconds.  Derived
            automatically from :attr:`data`, :attr:`sample_rate`,
            :attr:`channels`, and the assumed 16-bit (2-byte) sample width.
    """

    data: bytes
    sample_rate: int
    channels: int
    format: str  # "wav" or "pcm_s16le"
    duration_ms: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute :attr:`duration_ms` from the audio parameters."""
        bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
        total_samples = len(self.data) / (self.channels * bytes_per_sample)
        self.duration_ms = int(total_samples / self.sample_rate * 1000)


class TTSEngine(ABC):
    """Abstract text-to-speech engine.

    All concrete TTS backends must subclass :class:`TTSEngine` and implement
    every abstract method.  The :attr:`name` class attribute must be set to a
    unique, human-readable identifier for the engine (e.g. ``"piper"``).

    Lifecycle::

        engine.load()                                   # verify binary / model
        voices = engine.list_voices()
        buffer = await engine.synthesize("Hello!")      # returns AudioBuffer
        engine.unload()                                 # release resources
    """

    #: Unique, human-readable identifier for this engine implementation.
    name: str

    @abstractmethod
    def load(self) -> None:
        """Initialise the engine, verify binaries, and validate model files.

        Implementations should be idempotent: calling :meth:`load` on an
        already-loaded engine must not raise an error.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources held by the engine.

        After calling this method :meth:`is_loaded` must return ``False``.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return ``True`` if the engine is ready to synthesise audio.

        Returns:
            Boolean indicating readiness for synthesis.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Return the names of all voices available to this engine.

        Returns:
            A list of voice name strings.  The exact format is
            implementation-defined (e.g. ``"en_US-lessac-medium"``).
        """
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> AudioBuffer:
        """Convert *text* to speech and return the audio data.

        Args:
            text: The plain-text string to synthesise.
            voice: Optional voice name to use.  When ``None`` the engine
                uses its configured default voice.

        Returns:
            An :class:`AudioBuffer` containing the synthesised audio.

        Raises:
            RuntimeError: If the engine has not been loaded via :meth:`load`.
        """
        ...
