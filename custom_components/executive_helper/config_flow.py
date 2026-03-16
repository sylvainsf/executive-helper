"""Config flow for Executive Helper integration."""

from __future__ import annotations

import httpx
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import DEFAULT_GRACE_MINUTES, DEFAULT_PORT, DEFAULT_WAKE_WORD, DOMAIN


class EHConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Executive Helper."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step — connect to the backend server."""
        errors = {}

        if user_input is not None:
            host = user_input["host"]
            port = user_input["port"]

            # Test connection to the backend
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"http://{host}:{port}/health")
                    resp.raise_for_status()
                    health = resp.json()
                    if health.get("status") not in ("ok", "degraded"):
                        errors["base"] = "cannot_connect"
            except httpx.HTTPError:
                errors["base"] = "cannot_connect"

            if not errors:
                await self.async_set_unique_id(f"eh_{host}_{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Executive Helper",
                    data={"host": host, "port": port, "rooms": {}},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("host", default="localhost"): str,
                    vol.Required("port", default=DEFAULT_PORT): int,
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return EHOptionsFlow(config_entry)


class EHOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Executive Helper — add/edit room nodes."""

    def __init__(self, config_entry: config_entries.ConfigEntry):
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Show the main options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["global_settings", "add_room", "manage_rooms", "manage_speakers"],
        )

    async def async_step_global_settings(self, user_input=None):
        """Configure global settings."""
        if user_input is not None:
            opts = dict(self._config_entry.options)
            opts.update(user_input)
            return self.async_create_entry(title="", data=opts)

        current = self._config_entry.options
        return self.async_show_form(
            step_id="global_settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        "wake_word",
                        default=current.get("wake_word", DEFAULT_WAKE_WORD),
                    ): str,
                    vol.Optional(
                        "tts_voice",
                        default=current.get("tts_voice", "af_heart"),
                    ): vol.In(["af_heart", "af_sarah", "af_bella", "am_adam", "am_michael"]),
                    vol.Optional(
                        "tts_speed",
                        default=current.get("tts_speed", 1.0),
                    ): vol.Coerce(float),
                    vol.Optional(
                        "whisper_model",
                        default=current.get("whisper_model", "tiny"),
                    ): vol.In(["tiny", "base", "small"]),
                }
            ),
        )

    async def async_step_add_room(self, user_input=None):
        """Add a new room node."""
        errors = {}

        if user_input is not None:
            room_id = user_input["room_id"].strip().lower().replace(" ", "_")

            # Register the room with the backend
            host = self._config_entry.data["host"]
            port = self._config_entry.data["port"]
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"http://{host}:{port}/api/devices",
                        json={
                            "room_id": room_id,
                            "label": user_input["label"],
                            "respeaker_ip": user_input.get("respeaker_ip", ""),
                            "ha_motion_sensor": user_input.get("ha_motion_sensor", ""),
                            "ha_entities": [],
                            "wake_word": user_input.get("wake_word", ""),
                            "grace_minutes": user_input.get("grace_minutes", DEFAULT_GRACE_MINUTES),
                        },
                    )
                    resp.raise_for_status()
            except httpx.HTTPError:
                errors["base"] = "backend_error"

            if not errors:
                # Store room in config entry data
                data = dict(self._config_entry.data)
                rooms = dict(data.get("rooms", {}))
                rooms[room_id] = {
                    "label": user_input["label"],
                    "respeaker_ip": user_input.get("respeaker_ip", ""),
                    "ha_motion_sensor": user_input.get("ha_motion_sensor", ""),
                    "grace_minutes": user_input.get("grace_minutes", DEFAULT_GRACE_MINUTES),
                }
                data["rooms"] = rooms
                self.hass.config_entries.async_update_entry(self._config_entry, data=data)
                return self.async_create_entry(title="", data=self._config_entry.options)

        return self.async_show_form(
            step_id="add_room",
            data_schema=vol.Schema(
                {
                    vol.Required("room_id"): str,
                    vol.Required("label"): str,
                    vol.Optional("respeaker_ip"): str,
                    vol.Optional("wake_word"): str,
                    vol.Optional("ha_motion_sensor"): str,
                    vol.Optional("grace_minutes", default=DEFAULT_GRACE_MINUTES): int,
                }
            ),
            errors=errors,
        )

    async def async_step_manage_rooms(self, user_input=None):
        """Show existing rooms for editing/removal."""
        rooms = self._config_entry.data.get("rooms", {})

        if not rooms:
            return self.async_abort(reason="no_rooms")

        if user_input is not None:
            room_id = user_input["room_id"]
            if user_input.get("action") == "remove":
                data = dict(self._config_entry.data)
                rooms = dict(data.get("rooms", {}))
                rooms.pop(room_id, None)
                data["rooms"] = rooms
                self.hass.config_entries.async_update_entry(self._config_entry, data=data)
            return self.async_create_entry(title="", data=self._config_entry.options)

        room_options = {rid: f"{info['label']} ({rid})" for rid, info in rooms.items()}

        return self.async_show_form(
            step_id="manage_rooms",
            data_schema=vol.Schema(
                {
                    vol.Required("room_id"): vol.In(room_options),
                    vol.Required("action"): vol.In({"edit": "Edit", "remove": "Remove"}),
                }
            ),
        )

    async def async_step_manage_speakers(self, user_input=None):
        """Speaker enrollment step — triggers enrollment on the backend."""
        errors = {}

        if user_input is not None:
            host = self._config_entry.data["host"]
            port = self._config_entry.data["port"]
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"http://{host}:{port}/api/speakers/enroll",
                        json={
                            "room_id": user_input["room_id"],
                            "label": user_input["label"],
                            "role": user_input["role"],
                        },
                    )
                    resp.raise_for_status()
            except httpx.HTTPError:
                errors["base"] = "enrollment_failed"

            if not errors:
                return self.async_create_entry(title="", data=self._config_entry.options)

        rooms = self._config_entry.data.get("rooms", {})
        room_options = {rid: info["label"] for rid, info in rooms.items()}

        if not room_options:
            return self.async_abort(reason="no_rooms")

        return self.async_show_form(
            step_id="manage_speakers",
            data_schema=vol.Schema(
                {
                    vol.Required("room_id"): vol.In(room_options),
                    vol.Required("label"): str,
                    vol.Required("role", default="primary_user"): vol.In(
                        {
                            "primary_user": "Primary User (receives EF support)",
                            "household_member": "Household Member",
                            "guest": "Guest (limited access)",
                        }
                    ),
                }
            ),
            errors=errors,
        )
