"""Text-to-speech via Kokoro — warm, natural voice for supportive responses."""

import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    """Kokoro TTS engine for generating spoken responses.

    Runs on CPU to keep GPU available for LLMs.
    """

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        self.sample_rate = 24000  # Kokoro default

    async def initialize(self):
        """Load the Kokoro TTS pipeline. Call once at startup."""
        try:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="a")  # American English
            logger.info("Kokoro TTS loaded with voice '%s'", self.voice)
        except ImportError:
            logger.warning(
                "kokoro not installed. Install with: pip install -e '.[audio]'"
            )

    async def synthesize(self, text: str, *, voice: str | None = None) -> np.ndarray:
        """Convert text to speech audio.

        Args:
            text: Text to speak.
            voice: Override voice (defaults to self.voice).

        Returns:
            Float32 numpy array of audio at self.sample_rate.
        """
        if self._pipeline is None:
            logger.error("TTS not initialized — call initialize() first")
            return np.array([], dtype=np.float32)

        voice = voice or self.voice
        audio_chunks = []

        for _, _, audio in self._pipeline(text, voice=voice, speed=self.speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_chunks)

    async def synthesize_to_wav(self, text: str, *, voice: str | None = None) -> bytes:
        """Convert text to WAV bytes for streaming to speakers."""
        audio = await self.synthesize(text, voice=voice)
        if len(audio) == 0:
            return b""

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue()
