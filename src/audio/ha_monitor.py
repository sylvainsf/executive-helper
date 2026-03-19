"""Home Assistant activity monitor — checks device states for follow-through detection."""

import logging

import httpx

from configs.settings import settings

logger = logging.getLogger(__name__)


class HAActivityMonitor:
    """Checks Home Assistant entity states to detect user activity.

    Used by the IntentTracker to determine if a user has followed through
    on an expressed intention (e.g., "I should start dinner" → check for
    kitchen motion, stove usage, kitchen lights on).
    """

    def __init__(self, ha_url: str | None = None, ha_token: str | None = None):
        self.ha_url = (ha_url or settings.ha_url).rstrip("/")
        self.ha_token = ha_token or settings.ha_token
        self._headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json",
        }

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """Get the current state of a Home Assistant entity."""
        if not self.ha_token:
            logger.debug("No HA token configured — skipping state check")
            return None

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.ha_url}/api/states/{entity_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            logger.warning("Failed to get state for %s: %s", entity_id, e)
            return None

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str = "",
        data: dict | None = None,
    ) -> bool:
        """Call a Home Assistant service.

        Used by the EF model pipeline to execute actions like
        setting timers, playing music, or adjusting lights.
        """
        if not self.ha_token:
            logger.debug("No HA token configured — skipping service call")
            return False

        payload: dict = {}
        if entity_id:
            payload["entity_id"] = entity_id
        if data:
            payload.update(data)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.ha_url}/api/services/{domain}/{service}",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
                logger.info("HA service called: %s.%s on %s", domain, service, entity_id)
                return True
        except httpx.HTTPError as e:
            logger.warning(
                "Failed to call %s.%s on %s: %s", domain, service, entity_id, e
            )
            return False

    async def check_room_activity(self, room: str, since_minutes: int = 5) -> bool:
        """Check if there's been recent activity in a room.

        Looks for common activity indicators:
        - Motion sensors in the room
        - Lights turned on
        - Appliances in use

        This uses HA naming conventions (entity_id contains room name).
        """
        if not self.ha_token:
            return False

        # Check common entity patterns for the room
        indicators = [
            f"binary_sensor.{room}_motion",
            f"binary_sensor.{room}_occupancy",
            f"light.{room}",
            f"switch.{room}",
        ]

        for entity_id in indicators:
            state = await self.get_entity_state(entity_id)
            if state and state.get("state") == "on":
                logger.debug("Activity detected: %s is on", entity_id)
                return True

        return False

    async def check_entities_active(self, entity_ids: list[str]) -> bool:
        """Check if any of the specified entities are active/on."""
        for entity_id in entity_ids:
            state = await self.get_entity_state(entity_id)
            if not state:
                continue

            entity_state = state.get("state", "")
            # Consider various "active" states
            if entity_state in ("on", "playing", "heating", "cooling", "open", "detected"):
                logger.debug("Entity active: %s = %s", entity_id, entity_state)
                return True

        return False

    async def check_intent_activity(self, intent) -> bool:
        """Check if there's follow-through activity for a tracked intent.

        Checks both explicitly configured entities and room-based heuristics.
        """
        # Check explicit entities first
        if intent.expected_entities:
            if await self.check_entities_active(intent.expected_entities):
                return True

        # Fall back to room-based activity check
        for room in intent.expected_rooms:
            if await self.check_room_activity(room):
                return True

        return False
