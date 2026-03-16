"""Binary sensor entities for Executive Helper."""

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import EHCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Executive Helper binary sensors."""
    coordinator: EHCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([EHBackendConnectedSensor(coordinator, entry)])


class EHBackendConnectedSensor(CoordinatorEntity, BinarySensorEntity):
    """Whether the Executive Helper backend is reachable."""

    _attr_name = "EH Backend Connected"
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    def __init__(self, coordinator: EHCoordinator, entry: ConfigEntry):
        super().__init__(coordinator)
        self._entry = entry

    @property
    def unique_id(self):
        return f"{self._entry.entry_id}_backend_connected"

    @property
    def is_on(self):
        return self.coordinator.data.get("connected", False)

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Executive Helper",
            "manufacturer": "Executive Helper",
            "model": "EH Gateway",
        }
