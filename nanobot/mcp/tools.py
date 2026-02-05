"""MCP tool proxy - wraps MCP tools as native Tool instances."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool
from nanobot.mcp.protocol import MCPToolInfo

if TYPE_CHECKING:
    from nanobot.mcp.manager import MCPManager


class MCPToolProxy(Tool):
    """
    Proxy that wraps an MCP tool as a native nanobot Tool.

    This allows MCP tools to be registered in the ToolRegistry
    and used seamlessly alongside native tools.
    """

    def __init__(self, prefixed_name: str, tool_info: MCPToolInfo, manager: "MCPManager"):
        """
        Initialize an MCP tool proxy.

        Args:
            prefixed_name: Tool name with server prefix (e.g., "github_create_issue").
            tool_info: Tool information from the MCP server.
            manager: MCP manager for executing tool calls.
        """
        self._prefixed_name = prefixed_name
        self._tool_info = tool_info
        self._manager = manager

    @property
    def name(self) -> str:
        """Tool name used in function calls."""
        return self._prefixed_name

    @property
    def description(self) -> str:
        """Description of what the tool does."""
        return self._tool_info.description

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        # MCP uses inputSchema, which is already in JSON Schema format
        schema = self._tool_info.input_schema.copy()

        # Ensure it has a type
        if "type" not in schema:
            schema["type"] = "object"

        return schema

    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the MCP tool.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            String result of the tool execution.
        """
        return await self._manager.call_tool(self._prefixed_name, kwargs)


def create_mcp_tool_proxies(manager: "MCPManager") -> list[MCPToolProxy]:
    """
    Create tool proxies for all available MCP tools.

    Args:
        manager: MCP manager with initialized servers.

    Returns:
        List of MCPToolProxy instances.
    """
    proxies = []

    for prefixed_name, tool_info in manager.get_all_tools():
        proxy = MCPToolProxy(prefixed_name, tool_info, manager)
        proxies.append(proxy)

    return proxies
