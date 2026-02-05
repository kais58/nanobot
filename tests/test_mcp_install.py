"""Tests for MCP install tool."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.tools.mcp_install import InstallMCPServerTool
from nanobot.config.loader import get_config_path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def tool(workspace: Path) -> InstallMCPServerTool:
    """Create an InstallMCPServerTool instance."""
    return InstallMCPServerTool(workspace=workspace)


def test_tool_properties(tool: InstallMCPServerTool) -> None:
    """Test tool properties."""
    assert tool.name == "install_mcp_server"
    assert "MCP server" in tool.description
    assert "npm" in tool.description.lower() or "pip" in tool.description.lower()


def test_tool_parameters(tool: InstallMCPServerTool) -> None:
    """Test tool parameters schema."""
    params = tool.parameters
    assert params["type"] == "object"
    assert "server_name" in params["properties"]
    assert "package" in params["properties"]
    assert "command" in params["properties"]
    assert "args" in params["properties"]
    assert "env" in params["properties"]
    assert "install_command" in params["properties"]
    assert "server_name" in params["required"]
    assert "package" in params["required"]


def test_tool_set_context(tool: InstallMCPServerTool) -> None:
    """Test setting context."""
    tool.set_context("telegram", "123456")
    assert tool._context_channel == "telegram"
    assert tool._context_chat_id == "123456"


@pytest.mark.asyncio
async def test_invalid_server_name(tool: InstallMCPServerTool) -> None:
    """Test validation of server name."""
    result = await tool.execute(
        server_name="invalid name!",
        package="test-package",
        command="npx",
        args=["-y", "test"],
    )
    assert "Error:" in result
    assert "alphanumeric" in result


@pytest.mark.asyncio
async def test_valid_server_name_formats(tool: InstallMCPServerTool) -> None:
    """Test valid server name formats."""
    # Test _is_npm_package detection
    assert tool._is_npm_package("@anthropic/mcp-server-filesystem") is True
    assert tool._is_npm_package("mcp-server-github") is True
    assert tool._is_npm_package("some/package") is True
    assert tool._is_npm_package("simple-package") is False


@pytest.mark.asyncio
async def test_server_already_exists(workspace: Path, tmp_path: Path) -> None:
    """Test error when server already exists in config."""
    # Create a config file with existing server
    config_path = tmp_path / "config.json"
    config_data = {
        "tools": {
            "mcp": {
                "enabled": True,
                "servers": {
                    "existing": {
                        "command": "npx",
                        "args": ["-y", "existing-server"],
                    }
                },
            }
        }
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    tool = InstallMCPServerTool(workspace=workspace)

    with patch("nanobot.agent.tools.mcp_install.get_config_path", return_value=config_path):
        with patch("nanobot.agent.tools.mcp_install.load_config") as mock_load:
            from nanobot.config.schema import Config, MCPConfig, MCPServerConfig, ToolsConfig

            mock_config = Config(
                tools=ToolsConfig(
                    mcp=MCPConfig(
                        enabled=True,
                        servers={"existing": MCPServerConfig(command="npx", args=[])},
                    )
                )
            )
            mock_load.return_value = mock_config

            result = await tool.execute(
                server_name="existing",
                package="test-package",
                command="npx",
                args=["-y", "test"],
            )

            assert "Error:" in result
            assert "already exists" in result


@pytest.mark.asyncio
async def test_successful_install(workspace: Path, tmp_path: Path) -> None:
    """Test successful MCP server installation."""
    config_path = tmp_path / "config.json"
    config_data = {"tools": {"mcp": {"enabled": False, "servers": {}}}}
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    tool = InstallMCPServerTool(workspace=workspace)
    tool.set_context("telegram", "123456")

    with patch("nanobot.agent.tools.mcp_install.get_config_path", return_value=config_path):
        with patch("nanobot.agent.tools.mcp_install.load_config") as mock_load:
            with patch("nanobot.agent.tools.mcp_install.save_config") as mock_save:
                with patch.object(tool, "_run_install", new_callable=AsyncMock) as mock_install:
                    from nanobot.config.schema import Config, MCPConfig, ToolsConfig

                    mock_config = Config(tools=ToolsConfig(mcp=MCPConfig(enabled=False)))
                    mock_load.return_value = mock_config
                    mock_install.return_value = "OK"

                    result = await tool.execute(
                        server_name="github",
                        package="mcp-server-github",
                        command="uvx",
                        args=["mcp-server-github"],
                        env={"GITHUB_TOKEN": "test"},
                    )

                    assert "Installed" in result
                    assert "github" in result
                    assert "restart" in result.lower()

                    # Verify config was saved
                    mock_save.assert_called_once()

                    # Verify restart signal was written
                    signal_file = workspace / ".restart_signal"
                    assert signal_file.exists()

                    signal_data = json.loads(signal_file.read_text(encoding="utf-8"))
                    assert "github" in signal_data["reason"]
                    assert signal_data["verify_job"]["name"] == "verify_mcp_github"
                    assert signal_data["verify_job"]["channel"] == "telegram"
                    assert signal_data["verify_job"]["to"] == "123456"


@pytest.mark.asyncio
async def test_install_failure(workspace: Path, tmp_path: Path) -> None:
    """Test handling of installation failure."""
    config_path = tmp_path / "config.json"

    tool = InstallMCPServerTool(workspace=workspace)

    with patch("nanobot.agent.tools.mcp_install.get_config_path", return_value=config_path):
        with patch("nanobot.agent.tools.mcp_install.load_config") as mock_load:
            with patch.object(tool, "_run_install", new_callable=AsyncMock) as mock_install:
                from nanobot.config.schema import Config, MCPConfig, ToolsConfig

                mock_config = Config(tools=ToolsConfig(mcp=MCPConfig(enabled=False)))
                mock_load.return_value = mock_config
                mock_install.return_value = "Error: Package not found"

                result = await tool.execute(
                    server_name="test",
                    package="nonexistent-package",
                    command="npx",
                    args=["-y", "nonexistent"],
                )

                assert "Error:" in result
                assert "Package not found" in result
