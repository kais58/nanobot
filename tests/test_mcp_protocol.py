"""Tests for MCP protocol types."""

from nanobot.mcp.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    MCPServerCapabilities,
    MCPServerInfo,
    MCPToolCallResult,
    MCPToolInfo,
)


def test_jsonrpc_request_to_dict() -> None:
    """Test JSON-RPC request serialization."""
    request = JsonRpcRequest(method="tools/list", id=1, params={"cursor": None})
    result = request.to_dict()

    assert result["jsonrpc"] == "2.0"
    assert result["method"] == "tools/list"
    assert result["id"] == 1
    assert result["params"] == {"cursor": None}


def test_jsonrpc_request_no_params() -> None:
    """Test JSON-RPC request without params."""
    request = JsonRpcRequest(method="notifications/initialized", id=2)
    result = request.to_dict()

    assert "params" not in result
    assert result["method"] == "notifications/initialized"


def test_jsonrpc_response_from_dict() -> None:
    """Test JSON-RPC response deserialization."""
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": [{"name": "test", "description": "Test tool"}]},
    }
    response = JsonRpcResponse.from_dict(data)

    assert response.id == 1
    assert response.result == {"tools": [{"name": "test", "description": "Test tool"}]}
    assert response.error is None


def test_jsonrpc_response_error() -> None:
    """Test JSON-RPC error response."""
    data = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32601, "message": "Method not found"},
    }
    response = JsonRpcResponse.from_dict(data)

    assert response.id == 1
    assert response.result is None
    assert response.error == {"code": -32601, "message": "Method not found"}


def test_mcp_tool_info_from_dict() -> None:
    """Test MCPToolInfo creation from dictionary."""
    data = {
        "name": "create_issue",
        "description": "Create a GitHub issue",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title"],
        },
    }
    tool = MCPToolInfo.from_dict(data)

    assert tool.name == "create_issue"
    assert tool.description == "Create a GitHub issue"
    assert tool.input_schema["type"] == "object"
    assert "title" in tool.input_schema["properties"]


def test_mcp_tool_info_missing_description() -> None:
    """Test MCPToolInfo with missing description."""
    data = {"name": "simple_tool", "inputSchema": {"type": "object"}}
    tool = MCPToolInfo.from_dict(data)

    assert tool.name == "simple_tool"
    assert tool.description == ""


def test_mcp_server_capabilities_from_dict() -> None:
    """Test MCPServerCapabilities creation."""
    data = {
        "capabilities": {
            "tools": {},
            "resources": {},
        }
    }
    caps = MCPServerCapabilities.from_dict(data)

    assert caps.tools is True
    assert caps.resources is True
    assert caps.prompts is False


def test_mcp_server_info_from_dict() -> None:
    """Test MCPServerInfo creation."""
    data = {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "test-server",
            "version": "1.0.0",
        },
        "capabilities": {
            "tools": {},
        },
    }
    info = MCPServerInfo.from_dict(data)

    assert info.name == "test-server"
    assert info.version == "1.0.0"
    assert info.capabilities.tools is True
    assert info.capabilities.resources is False


def test_mcp_tool_call_result_text() -> None:
    """Test MCPToolCallResult with text content."""
    data = {
        "content": [
            {"type": "text", "text": "Issue created: #123"},
        ],
        "isError": False,
    }
    result = MCPToolCallResult.from_dict(data)

    assert result.is_error is False
    assert result.to_string() == "Issue created: #123"


def test_mcp_tool_call_result_error() -> None:
    """Test MCPToolCallResult with error."""
    data = {
        "content": [
            {"type": "text", "text": "Permission denied"},
        ],
        "isError": True,
    }
    result = MCPToolCallResult.from_dict(data)

    assert result.is_error is True
    assert result.to_string() == "Permission denied"


def test_mcp_tool_call_result_multiple_content() -> None:
    """Test MCPToolCallResult with multiple content items."""
    data = {
        "content": [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
            {"type": "image", "mimeType": "image/png"},
        ],
    }
    result = MCPToolCallResult.from_dict(data)

    output = result.to_string()
    assert "Line 1" in output
    assert "Line 2" in output
    assert "[Image: image/png]" in output
