"""Tool to configure Discord notification channel via natural language."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.channels.discord import DiscordChannel


class DiscordSetNotificationChannelTool(Tool):
    """Tool to set the Discord channel for proactive notifications."""

    def __init__(self, discord_channel: "DiscordChannel"):
        self._discord = discord_channel

    @property
    def name(self) -> str:
        return "discord_set_notification_channel"

    @property
    def description(self) -> str:
        return (
            "Set the Discord channel for proactive notifications (cron jobs, alerts, etc.). "
            "Use this when the user asks to change where notifications are sent. "
            "Example: 'send notifications to #alerts' or 'use #general for updates'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel_name": {
                    "type": "string",
                    "description": "The channel name without # (e.g., 'general', 'alerts', 'bot-notifications')",
                }
            },
            "required": ["channel_name"],
        }

    async def execute(self, channel_name: str, **kwargs: Any) -> str:
        """Execute the tool to update the Discord notification channel."""
        # Get the Discord client and guild
        guild = self._discord.get_guild()
        if not guild:
            return "Error: Discord bot is not connected to a guild"

        # Find channel by name (case-insensitive)
        channel_name_clean = channel_name.strip().lstrip("#").lower()
        target_channel = None

        for channel in guild.text_channels:
            if channel.name.lower() == channel_name_clean:
                target_channel = channel
                break

        if not target_channel:
            # List available channels for helpful error
            available = [f"#{c.name}" for c in guild.text_channels[:10]]
            available_str = ", ".join(available)
            if len(guild.text_channels) > 10:
                available_str += f" (and {len(guild.text_channels) - 10} more)"
            return (
                f"Error: Channel '#{channel_name_clean}' not found in this server. "
                f"Available channels: {available_str}"
            )

        # Update runtime config
        await self._discord.update_default_channel(str(target_channel.id))

        # Persist to config file
        try:
            config_path = Path.home() / ".nanobot" / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config_data = json.load(f)

                # Update the discord default_channel_id
                if "channels" not in config_data:
                    config_data["channels"] = {}
                if "discord" not in config_data["channels"]:
                    config_data["channels"]["discord"] = {}

                config_data["channels"]["discord"]["default_channel_id"] = str(target_channel.id)

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)

                logger.info(
                    f"Updated Discord default_channel_id to {target_channel.id} (#{target_channel.name})"
                )
        except Exception as e:
            logger.error(f"Failed to persist Discord channel config: {e}")
            return (
                f"Notifications will now be sent to #{target_channel.name} for this session, "
                f"but I couldn't save this to config: {e}"
            )

        return f"Notifications will now be sent to #{target_channel.name}"
