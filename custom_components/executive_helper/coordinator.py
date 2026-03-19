"""Data update coordinator for Executive Helper."""

from datetime import timedelta
import logging

import httpx

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=30)


class EHCoordinator(DataUpdateCoordinator):
    """Fetch status from the Executive Helper backend."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=SCAN_INTERVAL)
        self._host = entry.data["host"]
        self._port = entry.data["port"]
        self._base_url = f"http://{self._host}:{self._port}"

    async def _async_update_data(self) -> dict:
        """Poll the backend for current status."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Health check
                health_resp = await client.get(f"{self._base_url}/health")
                health_resp.raise_for_status()
                health = health_resp.json()

                # Journal stats
                journal_resp = await client.get(f"{self._base_url}/api/journal/stats")
                journal_resp.raise_for_status()
                journal = journal_resp.json()

                # Active intents
                intents_resp = await client.get(f"{self._base_url}/api/intents/active")
                if intents_resp.status_code == 200:
                    intents = intents_resp.json()
                else:
                    intents = []

                # Connected room nodes
                devices_resp = await client.get(f"{self._base_url}/api/devices")
                if devices_resp.status_code == 200:
                    devices = devices_resp.json()
                else:
                    devices = []

            return {
                "connected": True,
                "ollama": health.get("ollama", False),
                "active_intents": len(intents) if isinstance(intents, list) else 0,
                "intent_details": intents if isinstance(intents, list) else [],
                "unrated_decisions": journal.get("unrated", 0),
                "total_decisions": journal.get("total", 0),
                "helpful_count": journal.get("helpful", 0),
                "unhelpful_count": journal.get("unhelpful", 0),
                "connected_rooms": len(devices) if isinstance(devices, list) else 0,
                "rooms": devices if isinstance(devices, list) else [],
            }
        except httpx.HTTPError as err:
            raise UpdateFailed(f"Error communicating with backend: {err}") from err

    async def request_ef_support(self, scheduled_task: str, urgency: str, user_state: str):
        """Call the backend to request EF support."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{self._base_url}/ef-support",
                json={
                    "scheduled_task": scheduled_task,
                    "urgency": urgency,
                    "user_state": user_state,
                },
            )

    async def dismiss_intent(self, intent_id: str, reason: str):
        """Dismiss a tracked intent."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{self._base_url}/api/intents/{intent_id}/dismiss",
                json={"reason": reason},
            )

    async def enroll_speaker(self, room_id: str, label: str, role: str):
        """Start speaker enrollment."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{self._base_url}/api/speakers/enroll",
                json={"room_id": room_id, "label": label, "role": role},
            )

    async def fire_reminder_callback(self, timer_entity_id: str):
        """Notify the backend that an EF timer has fired.

        The backend looks up the stored context for this timer and asks
        the EF model for a contextual follow-up message.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._base_url}/ef-reminder-callback",
                    json={"timer_entity_id": timer_entity_id},
                )
                resp.raise_for_status()
        except httpx.HTTPError as err:
            _LOGGER.warning("Reminder callback failed for %s: %s", timer_entity_id, err)
