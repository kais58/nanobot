"""MCP (Model Context Protocol) types and constants."""

from dataclasses import dataclass, field
from typing import Any

# JSON-RPC 2.0 Constants
JSONRPC_VERSION = "2.0"

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request message."""

    method: str
    id: int | str
    params: dict[str, Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        return result


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response message."""

    id: int | str | None
    result: Any = None
    error: dict[str, Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JsonRpcResponse":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
        )


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    method: str
    params: dict[str, Any] | None = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        return result


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )


@dataclass
class MCPServerCapabilities:
    """MCP server capabilities from initialization."""

    tools: bool = False
    resources: bool = False
    prompts: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerCapabilities":
        """Create from capabilities dict."""
        caps = data.get("capabilities", {})
        return cls(
            tools="tools" in caps,
            resources="resources" in caps,
            prompts="prompts" in caps,
        )


@dataclass
class MCPServerInfo:
    """Information about an MCP server after initialization."""

    name: str
    version: str
    capabilities: MCPServerCapabilities = field(default_factory=MCPServerCapabilities)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerInfo":
        """Create from initialization result."""
        server_info = data.get("serverInfo", {})
        return cls(
            name=server_info.get("name", "unknown"),
            version=server_info.get("version", "unknown"),
            capabilities=MCPServerCapabilities.from_dict(data),
        )


@dataclass
class MCPToolCallResult:
    """Result from an MCP tool call."""

    content: list[dict[str, Any]]
    is_error: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPToolCallResult":
        """Create from tool call result."""
        return cls(
            content=data.get("content", []),
            is_error=data.get("isError", False),
        )

    def to_string(self) -> str:
        """Convert content to string for tool result."""
        parts = []
        for item in self.content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[Image: {item.get('mimeType', 'image')}]")
            elif item.get("type") == "resource":
                uri = item.get("resource", {}).get("uri", "unknown")
                parts.append(f"[Resource: {uri}]")
            else:
                parts.append(str(item))
        return "\n".join(parts)


# Standard JSON-RPC error codes
class JsonRpcError:
    """Standard JSON-RPC error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
