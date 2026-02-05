"""Tests for MCP manager."""

import pytest

from nanobot.config.schema import MCPConfig, MCPServerConfig, MCPToolConfig
from nanobot.mcp.manager import MCPManager
from nanobot.mcp.protocol import MCPToolInfo


def test_mcp_manager_disabled() -> None:
    """Test MCP manager when disabled."""
    config = MCPConfig(enabled=False)
    manager = MCPManager(config)

    assert manager.is_enabled is False
    assert manager.get_all_tools() == []


def test_mcp_manager_no_servers() -> None:
    """Test MCP manager with no servers configured."""
    config = MCPConfig(enabled=True, servers={})
    manager = MCPManager(config)

    assert manager.is_enabled is True
    assert manager.get_all_tools() == []


def test_mcp_manager_tool_map_lookup() -> None:
    """Test tool server lookup returns None for unknown tools."""
    config = MCPConfig(enabled=True)
    manager = MCPManager(config)

    assert manager.get_tool_server("unknown_tool") is None


@pytest.mark.asyncio
async def test_mcp_manager_call_unknown_tool() -> None:
    """Test calling unknown tool returns error."""
    config = MCPConfig(enabled=True)
    manager = MCPManager(config)

    result = await manager.call_tool("unknown_tool", {})
    assert "Error:" in result
    assert "not found" in result


def test_mcp_config_server_disabled() -> None:
    """Test server config with enabled=False."""
    server_config = MCPServerConfig(
        command="npx",
        args=["-y", "some-server"],
        enabled=False,
    )
    config = MCPConfig(
        enabled=True,
        servers={"test": server_config},
    )
    manager = MCPManager(config)

    # Server is disabled, so no clients should be created during start
    assert manager.is_enabled is True


def test_mcp_config_tool_disabled() -> None:
    """Test disabling specific tools in config."""
    config = MCPConfig(
        enabled=True,
        tools={
            "github_create_issue": MCPToolConfig(enabled=False),
            "github_list_repos": MCPToolConfig(enabled=True),
        },
    )

    assert config.tools["github_create_issue"].enabled is False
    assert config.tools["github_list_repos"].enabled is True


def test_mcp_manager_server_status_empty() -> None:
    """Test server status with no servers."""
    config = MCPConfig(enabled=True)
    manager = MCPManager(config)

    status = manager.get_server_status()
    assert status == {}


def test_mcp_config_env_variables() -> None:
    """Test server config with environment variables."""
    server_config = MCPServerConfig(
        command="uvx",
        args=["mcp-server-github"],
        env={"GITHUB_TOKEN": "test-token"},
    )

    assert server_config.env == {"GITHUB_TOKEN": "test-token"}
    assert server_config.timeout == 30  # default


def test_mcp_config_custom_timeout() -> None:
    """Test server config with custom timeout."""
    server_config = MCPServerConfig(
        command="python",
        args=["-m", "slow_server"],
        timeout=120,
    )

    assert server_config.timeout == 120
