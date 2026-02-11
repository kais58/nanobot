"""Web channel for dashboard chat via WebSocket."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

    from nanobot.bus.progress import ProgressEvent


class WebChannel(BaseChannel):
    """Web-based chat channel with WebSocket delivery."""

    name = "web"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self._connections: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the web channel (managed by FastAPI lifecycle)."""
        self._running = True
        logger.info("Web channel started")

    async def stop(self) -> None:
        """Stop the web channel."""
        self._running = False
        self._connections.clear()
        logger.info("Web channel stopped")

    def register(self, session_id: str, ws: "WebSocket") -> None:
        """Register a WebSocket connection for a session."""
        self._connections[session_id] = ws

    def unregister(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        self._connections.pop(session_id, None)

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to a web client via WebSocket."""
        ws = self._connections.get(msg.chat_id)
        if ws:
            try:
                await ws.send_json({"type": "message", "content": msg.content})
            except Exception as e:
                logger.debug(f"WebSocket send failed for {msg.chat_id}: {e}")
                self.unregister(msg.chat_id)
        else:
            logger.debug(f"Web message to {msg.chat_id}: no active connection")

    async def on_progress(self, event: "ProgressEvent") -> None:
        """Send progress events to web clients."""
        ws = self._connections.get(event.chat_id)
        if ws:
            try:
                await ws.send_json(
                    {
                        "type": "progress",
                        "kind": event.kind,
                        "detail": event.detail,
                        "tool_name": event.tool_name,
                    }
                )
            except Exception:
                pass
