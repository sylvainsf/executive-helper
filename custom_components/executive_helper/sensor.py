"""Sensor entities for Executive Helper."""

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import EHCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Executive Helper sensors."""
    coordinator: EHCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        EHActiveIntentsSensor(coordinator, entry),
        EHLastInterventionSensor(coordinator, entry),
        EHUnratedDecisionsSensor(coordinator, entry),
        EHHelpfulRateSensor(coordinator, entry),
    ]

    # Create a sensor for each configured room
    rooms = entry.data.get("rooms", {})
    for room_id, room_info in rooms.items():
        entities.append(EHRoomNodeSensor(coordinator, entry, room_id, room_info))

    async_add_entities(entities)


class EHSensorBase(CoordinatorEntity, SensorEntity):
    """Base class for Executive Helper sensors."""

    def __init__(self, coordinator: EHCoordinator, entry: ConfigEntry):
        super().__init__(coordinator)
        self._entry = entry

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Executive Helper",
            "manufacturer": "Executive Helper",
            "model": "EH Gateway",
        }


class EHActiveIntentsSensor(EHSensorBase):
    """Number of currently active tracked intents."""

    _attr_name = "Active Intents"
    _attr_icon = "mdi:timer-sand"

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_active_intents"

    @property
    def native_value(self):
        return self.coordinator.data.get("active_intents", 0)

    @property
    def extra_state_attributes(self):
        intents = self.coordinator.data.get("intent_details", [])
        if not intents:
            return {}
        return {
            "intents": [
                {"task": i.get("task", "?"), "age_minutes": i.get("age_minutes", 0)}
                for i in intents[:5]
            ]
        }


class EHLastInterventionSensor(EHSensorBase):
    """When the last EF intervention occurred."""

    _attr_name = "Last EF Intervention"
    _attr_icon = "mdi:heart-pulse"

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_last_intervention"

    @property
    def native_value(self):
        return self.coordinator.data.get("total_decisions", 0)


class EHUnratedDecisionsSensor(EHSensorBase):
    """Number of decisions awaiting review."""

    _attr_name = "Unrated Decisions"
    _attr_icon = "mdi:star-half-full"

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_unrated"

    @property
    def native_value(self):
        return self.coordinator.data.get("unrated_decisions", 0)


class EHHelpfulRateSensor(EHSensorBase):
    """Helpful vs unhelpful ratio."""

    _attr_name = "Helpful Rate"
    _attr_icon = "mdi:thumb-up"
    _attr_native_unit_of_measurement = "%"

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_helpful_rate"

    @property
    def native_value(self):
        helpful = self.coordinator.data.get("helpful_count", 0)
        unhelpful = self.coordinator.data.get("unhelpful_count", 0)
        total = helpful + unhelpful
        if total == 0:
            return None
        return round(helpful / total * 100, 1)


class EHRoomNodeSensor(CoordinatorEntity, SensorEntity):
    """Status sensor for a room audio node."""

    _attr_icon = "mdi:microphone"

    def __init__(
        self,
        coordinator: EHCoordinator,
        entry: ConfigEntry,
        room_id: str,
        room_info: dict,
    ):
        super().__init__(coordinator)
        self._entry = entry
        self._room_id = room_id
        self._room_info = room_info
        self._attr_name = f"{room_info.get('label', room_id)} Node"

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_room_{self._room_id}"

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, f"{self._entry.entry_id}_{self._room_id}")},
            "name": f"EH {self._room_info.get('label', self._room_id)}",
            "manufacturer": "Seeedstudio",
            "model": "ReSpeaker Lite",
            "via_device": (DOMAIN, self._entry.entry_id),
        }

    @property
    def native_value(self):
        rooms = self.coordinator.data.get("rooms", [])
        for r in rooms:
            if r.get("room_id") == self._room_id:
                return "connected" if r.get("connected") else "disconnected"
        return "unknown"

    @property
    def extra_state_attributes(self):
        return {
            "room_id": self._room_id,
            "respeaker_ip": self._room_info.get("respeaker_ip", ""),
            "motion_sensor": self._room_info.get("ha_motion_sensor", ""),
            "grace_minutes": self._room_info.get("grace_minutes", 30),
        }
