"""Web channel for dashboard chat (placeholder for WebSocket integration)."""

from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class WebChannel(BaseChannel):
    """Web-based chat channel for the dashboard."""

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
        logger.info("Web channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to a web client."""
        # WebSocket delivery will be implemented in a future update
        logger.debug(f"Web message to {msg.chat_id}: {msg.content[:100]}")
