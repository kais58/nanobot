"""MCP (Model Context Protocol) support for nanobot."""

from nanobot.mcp.client import MCPClient
from nanobot.mcp.manager import MCPManager
from nanobot.mcp.protocol import (
    MCPServerCapabilities,
    MCPServerInfo,
    MCPToolCallResult,
    MCPToolInfo,
)
from nanobot.mcp.tools import MCPToolProxy, create_mcp_tool_proxies

__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPServerCapabilities",
    "MCPServerInfo",
    "MCPToolCallResult",
    "MCPToolInfo",
    "MCPToolProxy",
    "create_mcp_tool_proxies",
]
