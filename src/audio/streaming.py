"""Audio streaming — receives audio from ReSpeaker Lite nodes over WebSocket."""

import asyncio
import io
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioBuffer:
    """Accumulates audio chunks and detects voice activity."""

    room: str
    sample_rate: int = 16000
    chunks: list[bytes] = field(default_factory=list)
    _silence_frames: int = 0
    _speech_frames: int = 0

    # VAD thresholds
    energy_threshold: float = 0.01
    speech_start_frames: int = 3  # consecutive frames with speech to trigger
    silence_end_frames: int = 15  # consecutive silent frames to end utterance

    @property
    def is_speaking(self) -> bool:
        return self._speech_frames >= self.speech_start_frames

    @property
    def is_done(self) -> bool:
        return self.is_speaking and self._silence_frames >= self.silence_end_frames

    def add_chunk(self, chunk: bytes) -> None:
        """Add an audio chunk and update VAD state."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(audio**2)))

        if energy > self.energy_threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            self.chunks.append(chunk)
        else:
            if self.is_speaking:
                self._silence_frames += 1
                self.chunks.append(chunk)  # keep trailing silence for natural cutoff

    def get_audio(self) -> np.ndarray:
        """Return accumulated audio as float32 numpy array."""
        if not self.chunks:
            return np.array([], dtype=np.float32)
        raw = b"".join(self.chunks)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def reset(self) -> None:
        self.chunks.clear()
        self._silence_frames = 0
        self._speech_frames = 0


@dataclass
class SpeakerProfile:
    """Stores a speaker embedding for identification."""

    label: str  # "primary_user", "speaker_2", etc.
    embedding: np.ndarray | None = None
    is_primary: bool = False


class AudioStreamManager:
    """Manages audio streams from multiple ReSpeaker nodes."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffers: dict[str, AudioBuffer] = {}
        self.speakers: dict[str, SpeakerProfile] = {}
        self._on_utterance_callbacks: list = []

    def get_buffer(self, room: str) -> AudioBuffer:
        if room not in self.buffers:
            self.buffers[room] = AudioBuffer(room=room, sample_rate=self.sample_rate)
        return self.buffers[room]

    def register_speaker(self, label: str, embedding: np.ndarray | None = None, *, is_primary: bool = False):
        """Register a known speaker profile."""
        self.speakers[label] = SpeakerProfile(
            label=label, embedding=embedding, is_primary=is_primary
        )

    def on_utterance(self, callback):
        """Register a callback for when a complete utterance is detected."""
        self._on_utterance_callbacks.append(callback)

    async def process_chunk(self, room: str, chunk: bytes) -> dict | None:
        """Process an audio chunk. Returns utterance info when speech segment ends."""
        buf = self.get_buffer(room)
        buf.add_chunk(chunk)

        if buf.is_done:
            audio = buf.get_audio()
            buf.reset()

            result = {
                "room": room,
                "audio": audio,
                "duration_s": len(audio) / self.sample_rate,
            }

            for cb in self._on_utterance_callbacks:
                await cb(result)

            return result

        return None
