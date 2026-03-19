"""Executive Helper — HA custom integration for executive dysfunction support."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, Event

from .const import DOMAIN
from .coordinator import EHCoordinator

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Executive Helper from a config entry."""
    coordinator = EHCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    await _register_services(hass, coordinator)

    # Listen for EF timer completions → callback to gateway
    _register_timer_listener(hass, coordinator)

    # Register the custom panel for the decision journal
    hass.http.register_static_path(
        f"/executive-helper/journal",
        hass.config.path("custom_components/executive_helper/panel/index.html"),
        cache_headers=False,
    )
    hass.components.frontend.async_register_built_in_panel(
        component_name="iframe",
        sidebar_title="EF Journal",
        sidebar_icon="mdi:notebook-heart",
        frontend_url_path="executive-helper-journal",
        config={
            "url": f"http://{entry.data['host']}:{entry.data['port']}/journal"
        },
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok


async def _register_services(hass: HomeAssistant, coordinator: EHCoordinator):
    """Register Executive Helper services."""

    async def handle_ef_support(call):
        """Handle the request_ef_support service call."""
        await coordinator.request_ef_support(
            scheduled_task=call.data.get("scheduled_task", ""),
            urgency=call.data.get("urgency", "medium"),
            user_state=call.data.get("user_state", ""),
        )

    async def handle_dismiss_intent(call):
        """Handle the dismiss_intent service call."""
        await coordinator.dismiss_intent(
            intent_id=call.data["intent_id"],
            reason=call.data.get("reason", ""),
        )

    async def handle_enroll_speaker(call):
        """Handle the enroll_speaker service call."""
        await coordinator.enroll_speaker(
            room_id=call.data["room_id"],
            label=call.data["label"],
            role=call.data.get("role", "household_member"),
        )

    hass.services.async_register(DOMAIN, "request_ef_support", handle_ef_support)
    hass.services.async_register(DOMAIN, "dismiss_intent", handle_dismiss_intent)
    hass.services.async_register(DOMAIN, "enroll_speaker", handle_enroll_speaker)


def _register_timer_listener(hass: HomeAssistant, coordinator: EHCoordinator):
    """Listen for timer.finished events on EF-managed timers.

    When the EF model sets a reminder or body-double check-in, it creates
    an HA timer entity prefixed with 'eh_'. When that timer fires, we call
    back to the gateway with the timer entity ID so the EF model can
    deliver a contextual follow-up.
    """
    import logging
    _LOGGER = logging.getLogger(__name__)

    async def _handle_timer_finished(event: Event):
        entity_id = event.data.get("entity_id", "")
        # Only handle our timers (prefixed with eh_)
        if not entity_id.startswith("timer.eh_"):
            return

        _LOGGER.info("EF timer finished: %s", entity_id)
        await coordinator.fire_reminder_callback(entity_id)

    hass.bus.async_listen("timer.finished", _handle_timer_finished)
