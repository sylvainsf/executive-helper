"""Intent tracker — monitors expressed intentions and checks for follow-through.

Flow:
  1. User says something like "I should start dinner" (ambient, no wake word)
  2. Automation model identifies this as a task intent
  3. IntentTracker registers it with a silent grace timer
  4. Timer checks HA sensors for follow-through activity
  5. If no activity detected → escalate to EF model for gentle nudge
  6. If activity detected → silently resolve the intent (no interruption)

All tracking is silent — the user is never told they're being monitored.
Interventions are gentle, autonomy-supportive, and logged for review.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class IntentStatus(Enum):
    ACTIVE = "active"          # Timer running, watching for follow-through
    FULFILLED = "fulfilled"    # Activity detected, silently resolved
    ESCALATED = "escalated"    # No activity, EF support triggered
    DISMISSED = "dismissed"    # User explicitly dismissed or overridden
    EXPIRED = "expired"        # Max tracking time exceeded without action


class Urgency(Enum):
    LOW = "low"          # Casual mention ("I should do laundry sometime")
    MEDIUM = "medium"    # Time-relevant ("I need to start dinner")
    HIGH = "high"        # Health/safety ("I need to take my meds")


@dataclass
class TrackedIntent:
    """A detected user intention being monitored for follow-through."""

    id: str
    task: str                          # "prepare dinner", "take medication", etc.
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    room_detected: str = "unknown"     # Where the user was when they said it
    speaker: str = "unknown"
    transcript: str = ""               # The original speech that triggered detection
    urgency: Urgency = Urgency.MEDIUM

    # Follow-through detection
    expected_rooms: list[str] = field(default_factory=list)   # e.g., ["kitchen"]
    expected_entities: list[str] = field(default_factory=list) # e.g., ["sensor.kitchen_motion"]
    grace_minutes: int = 30            # How long to wait before first nudge
    max_tracking_minutes: int = 120    # Stop tracking after this long

    # Escalation
    escalation_count: int = 0          # How many times we've nudged
    max_escalations: int = 3           # Stop after this many nudges
    escalation_interval_minutes: int = 15  # Time between nudges

    # State
    status: IntentStatus = IntentStatus.ACTIVE
    resolved_at: datetime | None = None
    resolution_note: str = ""

    @property
    def age_minutes(self) -> float:
        delta = datetime.now(timezone.utc) - self.detected_at
        return delta.total_seconds() / 60

    @property
    def is_overdue(self) -> bool:
        return self.age_minutes > self.grace_minutes

    @property
    def should_escalate(self) -> bool:
        if self.status != IntentStatus.ACTIVE:
            return False
        if not self.is_overdue:
            return False
        if self.escalation_count >= self.max_escalations:
            return False
        if self.escalation_count == 0:
            return True
        # For subsequent escalations, check interval
        minutes_since_last = self.age_minutes - (
            self.grace_minutes + self.escalation_count * self.escalation_interval_minutes
        )
        return minutes_since_last >= 0

    @property
    def should_expire(self) -> bool:
        return self.age_minutes > self.max_tracking_minutes


class IntentTracker:
    """Manages tracked intents and their lifecycle.

    The tracker is purely a state machine — it doesn't directly call HA or the
    EF model. Instead, it exposes callbacks that the audio pipeline wires up.
    """

    def __init__(self):
        self._intents: dict[str, TrackedIntent] = {}
        self._next_id: int = 0
        self._check_task: asyncio.Task | None = None

        # Callbacks
        self._on_escalate: list = []  # called with (TrackedIntent,) when nudge needed
        self._on_fulfill: list = []   # called with (TrackedIntent,) when resolved
        self._check_activity: None | object = None  # async fn(intent) -> bool

    def on_escalate(self, callback):
        """Register callback for when an intent needs EF escalation."""
        self._on_escalate.append(callback)

    def on_fulfill(self, callback):
        """Register callback for when an intent is silently fulfilled."""
        self._on_fulfill.append(callback)

    def set_activity_checker(self, checker):
        """Set the async function that checks HA for activity.

        checker(intent: TrackedIntent) -> bool
        Returns True if follow-through activity was detected.
        """
        self._check_activity = checker

    def track(
        self,
        task: str,
        room: str = "unknown",
        speaker: str = "unknown",
        transcript: str = "",
        urgency: str = "medium",
        expected_rooms: list[str] | None = None,
        expected_entities: list[str] | None = None,
        grace_minutes: int = 30,
    ) -> TrackedIntent:
        """Register a new intent to track."""
        self._next_id += 1
        intent_id = f"intent_{self._next_id}"

        intent = TrackedIntent(
            id=intent_id,
            task=task,
            room_detected=room,
            speaker=speaker,
            transcript=transcript,
            urgency=Urgency(urgency),
            expected_rooms=expected_rooms or [],
            expected_entities=expected_entities or [],
            grace_minutes=grace_minutes,
        )

        self._intents[intent_id] = intent
        logger.info(
            "Tracking intent '%s' from %s in %s (grace: %dm)",
            task, speaker, room, grace_minutes,
        )
        return intent

    def dismiss(self, intent_id: str, reason: str = ""):
        """Manually dismiss an intent (e.g., user explicitly says they changed plans)."""
        if intent_id in self._intents:
            intent = self._intents[intent_id]
            intent.status = IntentStatus.DISMISSED
            intent.resolved_at = datetime.now(timezone.utc)
            intent.resolution_note = reason
            logger.info("Intent '%s' dismissed: %s", intent.task, reason)

    def get_active(self) -> list[TrackedIntent]:
        """Get all currently active intents."""
        return [i for i in self._intents.values() if i.status == IntentStatus.ACTIVE]

    def get_overdue(self) -> list[TrackedIntent]:
        """Get active intents that are past their grace period."""
        return [i for i in self.get_active() if i.is_overdue]

    async def start(self, check_interval_seconds: int = 60):
        """Start the background check loop."""
        self._check_task = asyncio.create_task(self._check_loop(check_interval_seconds))
        logger.info("Intent tracker started (check interval: %ds)", check_interval_seconds)

    async def stop(self):
        """Stop the background check loop."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Intent tracker stopped")

    async def _check_loop(self, interval: int):
        """Periodically check all active intents."""
        while True:
            await asyncio.sleep(interval)
            await self._check_all()

    async def _check_all(self):
        """Check all active intents for follow-through or escalation."""
        for intent in list(self._intents.values()):
            if intent.status != IntentStatus.ACTIVE:
                continue

            # Check for expiration
            if intent.should_expire:
                intent.status = IntentStatus.EXPIRED
                intent.resolved_at = datetime.now(timezone.utc)
                intent.resolution_note = "Tracking time exceeded"
                logger.info("Intent '%s' expired after %.0fm", intent.task, intent.age_minutes)
                continue

            # Check for activity (if we have a checker)
            if self._check_activity:
                try:
                    activity_detected = await self._check_activity(intent)
                    if activity_detected:
                        intent.status = IntentStatus.FULFILLED
                        intent.resolved_at = datetime.now(timezone.utc)
                        intent.resolution_note = "Activity detected"
                        logger.info("Intent '%s' fulfilled — activity detected", intent.task)
                        for cb in self._on_fulfill:
                            await cb(intent)
                        continue
                except Exception as e:
                    logger.warning("Activity check failed for '%s': %s", intent.task, e)

            # Check if escalation is needed
            if intent.should_escalate:
                intent.escalation_count += 1
                intent.status = IntentStatus.ESCALATED if intent.escalation_count >= intent.max_escalations else IntentStatus.ACTIVE
                logger.info(
                    "Escalating intent '%s' (attempt %d/%d, %.0fm overdue)",
                    intent.task, intent.escalation_count, intent.max_escalations,
                    intent.age_minutes - intent.grace_minutes,
                )
                for cb in self._on_escalate:
                    await cb(intent)

    def build_escalation_context(self, intent: TrackedIntent) -> dict:
        """Build context dict for EF model escalation."""
        return {
            "scheduled_task": intent.task,
            "scheduled_time": intent.detected_at.strftime("%H:%M"),
            "current_time": datetime.now(timezone.utc).strftime("%H:%M"),
            "user_state": (
                f"User mentioned '{intent.task}' {int(intent.age_minutes)} minutes ago "
                f"in {intent.room_detected} but no follow-through activity detected"
                + (f" in {', '.join(intent.expected_rooms)}" if intent.expected_rooms else "")
            ),
            "urgency": intent.urgency.value,
            "escalation_attempt": intent.escalation_count,
            "original_transcript": intent.transcript,
        }
