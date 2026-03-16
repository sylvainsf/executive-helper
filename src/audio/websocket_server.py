"""WebSocket server for receiving audio streams from ReSpeaker Lite nodes."""

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from src.audio.pipeline import AudioPipeline

logger = logging.getLogger(__name__)


class AudioWebSocketServer:
    """Manages WebSocket connections from room audio nodes.

    Each ReSpeaker Lite node connects via WebSocket and streams raw
    16-bit PCM audio chunks. The server feeds them into the audio pipeline.

    Protocol:
        1. Node connects to /ws/audio/{room_id}
        2. Node sends binary frames (raw PCM int16, mono, 16kHz)
        3. Server sends back WAV audio frames when there's a TTS response
    """

    def __init__(self, pipeline: AudioPipeline):
        self.pipeline = pipeline
        self.connections: dict[str, WebSocket] = {}

    async def handle_connection(self, websocket: WebSocket, room_id: str):
        """Handle a WebSocket connection from a room audio node."""
        await websocket.accept()
        self.connections[room_id] = websocket

        # Register response callback to send TTS audio back to this room
        async def send_response(wav_bytes: bytes):
            try:
                await websocket.send_bytes(wav_bytes)
            except Exception:
                logger.warning("Failed to send audio response to %s", room_id)

        self.pipeline.on_response(room_id, send_response)
        logger.info("Room node connected: %s", room_id)

        try:
            while True:
                data = await websocket.receive_bytes()
                await self.pipeline.process_audio_chunk(room_id, data)
        except WebSocketDisconnect:
            logger.info("Room node disconnected: %s", room_id)
        finally:
            self.connections.pop(room_id, None)

    @property
    def connected_rooms(self) -> list[str]:
        return list(self.connections.keys())
