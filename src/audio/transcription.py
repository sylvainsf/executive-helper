"""Whisper-based transcription with speaker diarization."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A transcribed segment with speaker label."""

    text: str
    speaker: str  # "primary_user", "speaker_2", "unknown_1", etc.
    start_time: float
    end_time: float
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Full transcription result with speaker-labeled segments."""

    segments: list[TranscriptionSegment]
    room: str
    has_wake_word: bool = False
    wake_word_speaker: str | None = None

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def speakers(self) -> set[str]:
        return {s.speaker for s in self.segments}

    @property
    def is_conversation(self) -> bool:
        return len(self.speakers) > 1

    def format_transcript(self) -> str:
        """Format as labeled transcript for the model."""
        lines = []
        for seg in self.segments:
            lines.append(f"[{seg.speaker}]: {seg.text}")
        return "\n".join(lines)


class TranscriptionEngine:
    """Whisper transcription + speaker diarization.

    Uses faster-whisper for ASR and a lightweight speaker embedding model
    for diarization (speaker identification against enrolled profiles).
    """

    def __init__(
        self,
        whisper_model: str = "tiny",
        wake_words: list[str] | None = None,
    ):
        self.whisper_model_name = whisper_model
        self.wake_words = [w.lower() for w in (wake_words or ["hey helper"])]
        self._whisper = None
        self._speaker_embedder = None
        self._enrolled_embeddings: dict[str, np.ndarray] = {}

    async def initialize(self):
        """Load models. Call once at startup."""
        try:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(
                self.whisper_model_name,
                device="cpu",  # keep GPU for LLMs
                compute_type="int8",
            )
            logger.info("Whisper model '%s' loaded", self.whisper_model_name)
        except ImportError:
            logger.warning(
                "faster-whisper not installed. Install with: pip install -e '.[audio]'"
            )

    def enroll_speaker(self, label: str, embedding: np.ndarray):
        """Enroll a speaker embedding for identification."""
        self._enrolled_embeddings[label] = embedding / np.linalg.norm(embedding)

    async def transcribe(
        self, audio: np.ndarray, room: str, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio with speaker labels.

        Args:
            audio: Float32 audio array, mono, normalized to [-1, 1].
            room: Room identifier from the source node.
            sample_rate: Audio sample rate.

        Returns:
            TranscriptionResult with speaker-labeled segments.
        """
        if self._whisper is None:
            # Stub mode — return raw text without real transcription
            return TranscriptionResult(
                segments=[
                    TranscriptionSegment(
                        text="[transcription unavailable — whisper not loaded]",
                        speaker="unknown",
                        start_time=0.0,
                        end_time=0.0,
                    )
                ],
                room=room,
            )

        # Transcribe with word-level timestamps for diarization alignment
        segments_iter, info = self._whisper.transcribe(
            audio,
            beam_size=3,
            word_timestamps=True,
            language="en",
        )

        segments = []
        full_text = ""

        for whisper_seg in segments_iter:
            text = whisper_seg.text.strip()
            if not text:
                continue

            full_text += " " + text

            # Speaker identification: extract embedding for this segment's audio
            speaker = await self._identify_speaker(
                audio, whisper_seg.start, whisper_seg.end, sample_rate
            )

            segments.append(
                TranscriptionSegment(
                    text=text,
                    speaker=speaker,
                    start_time=whisper_seg.start,
                    end_time=whisper_seg.end,
                    confidence=whisper_seg.avg_logprob,
                )
            )

        # Check for wake word
        has_wake_word = False
        wake_word_speaker = None
        full_text_lower = full_text.lower()
        for ww in self.wake_words:
            if ww in full_text_lower:
                has_wake_word = True
                # Attribute wake word to the speaker of the segment containing it
                for seg in segments:
                    if ww in seg.text.lower():
                        wake_word_speaker = seg.speaker
                        break
                break

        return TranscriptionResult(
            segments=segments,
            room=room,
            has_wake_word=has_wake_word,
            wake_word_speaker=wake_word_speaker,
        )

    async def _identify_speaker(
        self, audio: np.ndarray, start: float, end: float, sample_rate: int
    ) -> str:
        """Identify speaker for an audio segment using enrolled embeddings.

        Falls back to "unknown_N" labeling if no enrolled speakers match
        or if the embedding model is unavailable.
        """
        if not self._enrolled_embeddings:
            return "unknown"

        # Extract the segment audio
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < sample_rate * 0.5:
            # Too short for reliable speaker ID
            return "unknown"

        # TODO: Extract speaker embedding from segment_audio using a lightweight
        # speaker verification model (e.g., SpeechBrain ECAPA-TDNN, ~25MB).
        # Compare against enrolled embeddings via cosine similarity.
        # Threshold: >0.7 = match, otherwise "unknown_N"
        #
        # For now, return "unknown" — speaker ID will be implemented when
        # the embedding model is integrated.
        return "unknown"
