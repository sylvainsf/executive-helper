"""End-to-end audio pipeline: stream → transcribe → route → respond → speak."""

import asyncio
import json
import logging

import numpy as np

from configs.settings import settings
from src.audio.ha_monitor import HAActivityMonitor
from src.audio.intent_tracker import IntentTracker, TrackedIntent
from src.audio.streaming import AudioStreamManager
from src.audio.transcription import TranscriptionEngine, TranscriptionResult
from src.audio.tts import TTSEngine
from src.gateway.models import chat, request_ef_support

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Orchestrates the full audio flow.

    ReSpeaker (stream) → VAD → Whisper + Speaker ID → Route:
      - Wake word detected → automation model (command mode)
      - Conversation detected → automation model (analysis mode)
      - No speech / ambient → ignore

    Automation model may:
      - Execute device actions
      - Escalate to EF model for behavioral support
      - Respond to user via TTS
    """

    def __init__(self):
        self.stream_manager = AudioStreamManager(sample_rate=settings.audio_sample_rate)
        self.transcriber = TranscriptionEngine(
            whisper_model=settings.whisper_model,
            wake_words=settings.wake_words,
        )
        self.tts = TTSEngine(voice=settings.tts_voice, speed=settings.tts_speed)
        self.intent_tracker = IntentTracker()
        self.ha_monitor = HAActivityMonitor()

        # Callbacks for sending audio responses back to room nodes
        self._response_callbacks: dict[str, list] = {}

    async def initialize(self):
        """Load all models. Call once at startup."""
        await self.transcriber.initialize()
        await self.tts.initialize()
        self.stream_manager.on_utterance(self._handle_utterance)

        # Wire up intent tracker
        self.intent_tracker.set_activity_checker(self.ha_monitor.check_intent_activity)
        self.intent_tracker.on_escalate(self._handle_intent_escalation)
        await self.intent_tracker.start(check_interval_seconds=60)
        logger.info("Audio pipeline initialized")

    def on_response(self, room: str, callback):
        """Register callback to receive audio responses for a room."""
        self._response_callbacks.setdefault(room, []).append(callback)

    def enroll_primary_speaker(self, label: str, embedding: np.ndarray):
        """Enroll the primary user's voice for speaker identification."""
        self.transcriber.enroll_speaker(label, embedding)
        self.stream_manager.register_speaker(label, embedding, is_primary=True)

    async def process_audio_chunk(self, room: str, chunk: bytes) -> None:
        """Feed an audio chunk from a room node into the pipeline."""
        await self.stream_manager.process_chunk(room, chunk)

    async def _handle_utterance(self, utterance: dict) -> None:
        """Process a complete utterance detected by VAD."""
        room = utterance["room"]
        audio = utterance["audio"]

        logger.info(
            "Utterance detected in %s (%.1fs)", room, utterance["duration_s"]
        )

        # Transcribe with speaker identification
        result = await self.transcriber.transcribe(
            audio, room, settings.audio_sample_rate
        )

        if not result.segments or not result.full_text.strip():
            return

        logger.info("Transcript [%s]: %s", room, result.full_text[:100])

        # Route based on content
        if result.has_wake_word:
            await self._handle_command(result)
        elif result.is_conversation:
            await self._handle_conversation(result)
        else:
            # Single speaker, no wake word — context-dependent
            # Could be ambient speech or a passive observation opportunity
            await self._handle_ambient(result)

    async def _handle_command(self, result: TranscriptionResult) -> None:
        """Handle a wake-word-triggered command."""
        # Strip the wake word from the transcript for cleaner input
        command_text = result.full_text
        for ww in self.transcriber.wake_words:
            command_text = command_text.lower().replace(ww, "").strip()

        speaker = result.wake_word_speaker or "unknown"
        room = result.room

        messages = [
            {
                "role": "user",
                "content": (
                    f"[Room: {room}] [Speaker: {speaker}] "
                    f"Voice command: {command_text}"
                ),
            }
        ]

        response = await chat("auto", messages)
        logger.info("Auto model response: %s", response[:200])

        # Check for EF escalation in the response
        ef_response = await self._check_ef_escalation(response)

        # Parse and execute any EF actions
        if ef_response:
            ef_actions = self._extract_ef_actions(ef_response)
            if ef_actions:
                await self._execute_ef_actions(ef_actions, room, user_input=command_text)

        # Determine what to speak back
        spoken = (
            self._extract_spoken_text(ef_response) if ef_response
            else self._extract_spoken_text(response)
        )
        if spoken:
            await self._speak(room, spoken)

        # Log the decision for the journal
        await self._log_decision(
            room=room,
            speaker=speaker,
            input_text=command_text,
            model="auto",
            response=response,
            ef_intervention=ef_response,
            trigger="wake_word",
        )

    async def _handle_conversation(self, result: TranscriptionResult) -> None:
        """Handle a detected multi-speaker conversation."""
        transcript = result.format_transcript()
        room = result.room

        messages = [
            {
                "role": "user",
                "content": (
                    f"[Room: {room}] Conversation transcript:\n{transcript}"
                ),
            }
        ]

        response = await chat("auto", messages, temperature=0.5)

        ef_response = await self._check_ef_escalation(response)

        # Parse and execute any EF actions
        if ef_response:
            ef_actions = self._extract_ef_actions(ef_response)
            if ef_actions:
                await self._execute_ef_actions(ef_actions, room, user_input=transcript)

        # Conversations don't always need a spoken response —
        # only speak if the model explicitly generates user-facing text
        spoken = (
            self._extract_spoken_text(ef_response) if ef_response
            else self._extract_spoken_text(response)
        )
        if spoken:
            await self._speak(room, spoken)

        await self._log_decision(
            room=room,
            speaker="multi",
            input_text=transcript,
            model="auto",
            response=response,
            ef_intervention=ef_response,
            trigger="conversation",
        )

    async def _handle_ambient(self, result: TranscriptionResult) -> None:
        """Handle single-speaker, non-command speech (passive monitoring).

        Sends ambient speech to the automation model for intent detection.
        The model may respond with a track_intent action (silent timer)
        or determine no action is needed.
        """
        if not result.segments:
            return

        speaker = result.segments[0].speaker
        if speaker == "unknown":
            return  # Don't process unidentified ambient speech

        room = result.room
        text = result.full_text.strip()
        logger.debug("Ambient speech in %s from %s: %s", room, speaker, text[:80])

        # Ask the automation model to analyze for task intent signals
        messages = [
            {
                "role": "user",
                "content": (
                    f"[Room: {room}] [Speaker: {speaker}] "
                    f"[Mode: ambient_analysis] "
                    f"Ambient speech (no wake word): {text}"
                ),
            }
        ]

        response = await chat("auto", messages, temperature=0.3)

        # Check for track_intent action
        await self._check_track_intent(response, room, speaker, text)

        # Check for EF escalation (model might detect distress directly)
        ef_response = await self._check_ef_escalation(response)

        # Ambient analysis should almost never produce a spoken response —
        # only if the model detects something urgent (e.g., safety concern)
        if ef_response:
            ef_actions = self._extract_ef_actions(ef_response)
            if ef_actions:
                await self._execute_ef_actions(ef_actions, room, user_input=text)
            spoken = self._extract_spoken_text(ef_response)
            if spoken:
                await self._speak(room, spoken)

        await self._log_decision(
            room=room,
            speaker=speaker,
            input_text=text,
            model="auto",
            response=response,
            ef_intervention=ef_response,
            trigger="ambient",
        )

    async def _check_ef_escalation(self, auto_response: str) -> str | None:
        """Check if auto model wants to escalate to EF model."""
        if "request_ef_support" not in auto_response:
            return None

        try:
            start = auto_response.index("{")
            depth = 0
            end = start
            for i, c in enumerate(auto_response[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            parsed = json.loads(auto_response[start:end])
            if parsed.get("action") == "request_ef_support":
                context = parsed.get("context", parsed)
                return await request_ef_support(context)
        except (ValueError, json.JSONDecodeError, KeyError):
            pass
        return None

    def _extract_spoken_text(self, response: str) -> str | None:
        """Extract user-facing spoken text from a model response.

        The model may return JSON actions, spoken text, or both.
        Extract only the conversational part for TTS.
        """
        lines = response.strip().split("\n")
        spoken_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip JSON blocks
            if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
                continue
            if stripped:
                spoken_lines.append(stripped)

        text = " ".join(spoken_lines).strip()
        return text if text else None

    def _extract_ef_actions(self, response: str) -> list[dict]:
        """Extract system action JSON blocks from an EF model response.

        The EF model may append action JSON after spoken text, e.g.:
            Grab a trash bag and do one lap.
            {"action": "set_timer", "minutes": 10, "label": "cleanup"}

        Returns a list of parsed action dicts.
        """
        actions = []
        for line in response.strip().split("\n"):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if "action" in parsed:
                        actions.append(parsed)
                except json.JSONDecodeError:
                    pass
        return actions

    async def _execute_ef_actions(
        self,
        actions: list[dict],
        room: str,
        user_input: str = "",
    ) -> None:
        """Execute system actions returned by the EF model.

        Actions that need callbacks (reminders, body-double check-ins) are
        implemented via HA timer entities. The conversation context is stored
        in the journal DB. When the timer fires, HA calls the gateway's
        /ef-reminder-callback endpoint, which loads the context and asks the
        EF model for a contextual follow-up.

        Supported actions:
          - set_timer: Start a countdown timer via HA
          - set_reminder: Schedule a contextual follow-up via HA timer + DB
          - play_music: Start media in the room
          - brighten_lights / dim_lights: Adjust room lighting
          - body_double_checkin: Schedule a contextual check-in via HA timer + DB
          - dismiss_intent: Stop tracking the current task intent
        """
        from src.journal.store import create_reminder

        for action in actions:
            action_type = action.get("action")
            try:
                if action_type == "set_timer":
                    minutes = action.get("minutes", 5)
                    label = action.get("label", "timer")
                    timer_entity = f"timer.eh_{room}_timer"
                    await self.ha_monitor.call_service(
                        "timer", "start",
                        entity_id=timer_entity,
                        data={"duration": f"00:{minutes:02d}:00"},
                    )
                    logger.info("Set timer: %dm (%s) in %s", minutes, label, room)

                elif action_type == "set_reminder":
                    minutes = action.get("minutes", 15)
                    label = action.get("label", "reminder")
                    timer_entity = f"timer.eh_{room}_reminder"

                    # Store context so the callback can reconstruct the conversation
                    reminder_id = await create_reminder(
                        room=room,
                        label=label,
                        minutes=minutes,
                        action_type="reminder",
                        original_user_input=user_input,
                        timer_entity_id=timer_entity,
                    )

                    await self.ha_monitor.call_service(
                        "timer", "start",
                        entity_id=timer_entity,
                        data={"duration": f"00:{minutes:02d}:00"},
                    )
                    logger.info(
                        "Set reminder #%d: %dm (%s) in %s",
                        reminder_id, minutes, label, room,
                    )

                elif action_type == "play_music":
                    await self.ha_monitor.call_service(
                        "media_player", "play_media",
                        entity_id=f"media_player.{room}",
                        data={"media_content_type": "music", "media_content_id": ""},
                    )
                    logger.info("Play music in %s", room)

                elif action_type == "brighten_lights":
                    await self.ha_monitor.call_service(
                        "light", "turn_on",
                        entity_id=f"light.{room}",
                        data={"brightness": 255},
                    )
                    logger.info("Brighten lights in %s", room)

                elif action_type == "dim_lights":
                    await self.ha_monitor.call_service(
                        "light", "turn_on",
                        entity_id=f"light.{room}",
                        data={"brightness": 50},
                    )
                    logger.info("Dim lights in %s", room)

                elif action_type == "body_double_checkin":
                    minutes = action.get("minutes", 10)
                    timer_entity = f"timer.eh_{room}_checkin"

                    # Store context for the check-in callback
                    reminder_id = await create_reminder(
                        room=room,
                        label="body_double_checkin",
                        minutes=minutes,
                        action_type="body_double_checkin",
                        original_user_input=user_input,
                        timer_entity_id=timer_entity,
                    )

                    await self.ha_monitor.call_service(
                        "timer", "start",
                        entity_id=timer_entity,
                        data={"duration": f"00:{minutes:02d}:00"},
                    )
                    logger.info(
                        "Body-double check-in #%d: %dm in %s",
                        reminder_id, minutes, room,
                    )

                elif action_type == "dismiss_intent":
                    active = self.intent_tracker.get_active()
                    if active:
                        self.intent_tracker.dismiss(active[-1].id, reason="EF model dismissed")
                        logger.info("Dismissed latest active intent")

                else:
                    logger.warning("Unknown EF action type: %s", action_type)

            except Exception as e:
                logger.warning("Failed to execute EF action %s: %s", action_type, e)

    async def _speak(self, room: str, text: str) -> None:
        """Synthesize speech and send to room node."""
        logger.info("Speaking in %s: %s", room, text[:100])
        wav_bytes = await self.tts.synthesize_to_wav(text)
        if not wav_bytes:
            logger.warning("TTS produced no audio")
            return

        for cb in self._response_callbacks.get(room, []):
            await cb(wav_bytes)

    async def _check_track_intent(self, response: str, room: str, speaker: str, transcript: str) -> None:
        """Check if the automation model wants to silently track an intent."""
        if "track_intent" not in response:
            return

        try:
            start = response.index("{")
            depth = 0
            end = start
            for i, c in enumerate(response[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            parsed = json.loads(response[start:end])
            if parsed.get("action") != "track_intent":
                return

            self.intent_tracker.track(
                task=parsed.get("task", "unknown task"),
                room=room,
                speaker=speaker,
                transcript=transcript,
                urgency=parsed.get("urgency", "medium"),
                expected_rooms=parsed.get("expected_rooms", []),
                expected_entities=parsed.get("expected_entities", []),
                grace_minutes=parsed.get("grace_minutes", 30),
            )
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            logger.debug("Failed to parse track_intent: %s", e)

    async def _handle_intent_escalation(self, intent: TrackedIntent) -> None:
        """Called by IntentTracker when an intent needs EF support."""
        context = self.intent_tracker.build_escalation_context(intent)
        ef_response = await request_ef_support(context)

        if ef_response:
            # Speak in the room where the user currently is (or was last detected)
            target_room = intent.room_detected

            # Parse and execute any actions from the EF model
            ef_actions = self._extract_ef_actions(ef_response)
            if ef_actions:
                await self._execute_ef_actions(
                    ef_actions, target_room, user_input=intent.transcript,
                )

            spoken = self._extract_spoken_text(ef_response)
            if spoken:
                await self._speak(target_room, spoken)

        await self._log_decision(
            room=intent.room_detected,
            speaker=intent.speaker,
            input_text=f"[Intent escalation] {intent.task} ({int(intent.age_minutes)}m overdue)",
            model="ef",
            response=ef_response or "",
            trigger="intent_escalation",
        )

    async def _log_decision(self, **kwargs) -> None:
        """Log a system decision to the journal for review."""
        # Import here to avoid circular dependency
        from src.journal.store import log_decision

        await log_decision(**kwargs)
