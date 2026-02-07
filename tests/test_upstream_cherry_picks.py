"""Tests for cherry-picked upstream improvements."""

import asyncio
from pathlib import Path

import pytest

from nanobot.config.loader import _migrate_config
from nanobot.config.schema import (
    Config,
    ExecToolConfig,
    ProviderConfig,
    ProvidersConfig,
    ToolsConfig,
)
from nanobot.providers.resolver import ProviderResolver


# ---------------------------------------------------------------------------
# Config migration: tools.exec.restrictToWorkspace -> tools.restrictToWorkspace
# ---------------------------------------------------------------------------


class TestConfigMigration:
    def test_migrate_moves_restrict_to_workspace(self) -> None:
        data = {
            "tools": {
                "exec": {
                    "timeout": 30,
                    "restrictToWorkspace": True,
                },
            },
        }
        result = _migrate_config(data)
        assert result["tools"]["restrictToWorkspace"] is True
        assert "restrictToWorkspace" not in result["tools"]["exec"]

    def test_migrate_preserves_tools_level_value(self) -> None:
        """If tools.restrictToWorkspace already set, exec-level should not overwrite."""
        data = {
            "tools": {
                "restrictToWorkspace": False,
                "exec": {
                    "restrictToWorkspace": True,
                },
            },
        }
        result = _migrate_config(data)
        # tools-level already existed, exec-level stays untouched
        assert result["tools"]["restrictToWorkspace"] is False

    def test_migrate_no_tools_section(self) -> None:
        data = {"agents": {}}
        result = _migrate_config(data)
        assert result == data

    def test_migrate_no_exec_section(self) -> None:
        data = {"tools": {"web": {}}}
        result = _migrate_config(data)
        assert result == data

    def test_migrate_no_restrict_key(self) -> None:
        data = {"tools": {"exec": {"timeout": 60}}}
        result = _migrate_config(data)
        assert "restrictToWorkspace" not in result["tools"]


# ---------------------------------------------------------------------------
# New providers in schema
# ---------------------------------------------------------------------------


class TestNewProviders:
    def test_deepseek_provider_config(self) -> None:
        providers = ProvidersConfig(
            deepseek=ProviderConfig(api_key="ds-key-123"),
        )
        assert providers.deepseek.api_key == "ds-key-123"

    def test_dashscope_provider_config(self) -> None:
        providers = ProvidersConfig(
            dashscope=ProviderConfig(api_key="dash-key"),
        )
        assert providers.dashscope.api_key == "dash-key"

    def test_moonshot_provider_config(self) -> None:
        providers = ProvidersConfig(
            moonshot=ProviderConfig(api_key="ms-key"),
        )
        assert providers.moonshot.api_key == "ms-key"

    def test_aihubmix_provider_config(self) -> None:
        providers = ProvidersConfig(
            aihubmix=ProviderConfig(
                api_key="ahm-key",
                extra_headers={"APP-Code": "abc123"},
            ),
        )
        assert providers.aihubmix.api_key == "ahm-key"
        assert providers.aihubmix.extra_headers == {"APP-Code": "abc123"}

    def test_extra_headers_default_none(self) -> None:
        provider = ProviderConfig(api_key="test")
        assert provider.extra_headers is None


# ---------------------------------------------------------------------------
# New providers in resolver
# ---------------------------------------------------------------------------


class TestNewProviderResolution:
    def test_resolve_deepseek(self) -> None:
        providers = ProvidersConfig(
            deepseek=ProviderConfig(api_key="ds-key"),
        )
        resolver = ProviderResolver(providers)
        api_key, api_base = resolver.resolve("deepseek")
        assert api_key == "ds-key"
        assert api_base is None

    def test_resolve_dashscope(self) -> None:
        providers = ProvidersConfig(
            dashscope=ProviderConfig(api_key="dash-key"),
        )
        resolver = ProviderResolver(providers)
        api_key, _ = resolver.resolve("dashscope")
        assert api_key == "dash-key"

    def test_resolve_moonshot(self) -> None:
        providers = ProvidersConfig(
            moonshot=ProviderConfig(api_key="ms-key"),
        )
        resolver = ProviderResolver(providers)
        api_key, _ = resolver.resolve("moonshot")
        assert api_key == "ms-key"

    def test_resolve_aihubmix_default_base(self) -> None:
        providers = ProvidersConfig(
            aihubmix=ProviderConfig(api_key="ahm-key"),
        )
        resolver = ProviderResolver(providers)
        api_key, api_base = resolver.resolve("aihubmix")
        assert api_key == "ahm-key"
        assert api_base == "https://aihubmix.com/v1"

    def test_resolve_with_headers(self) -> None:
        providers = ProvidersConfig(
            aihubmix=ProviderConfig(
                api_key="ahm-key",
                extra_headers={"APP-Code": "xyz"},
            ),
        )
        resolver = ProviderResolver(providers)
        api_key, api_base, headers = resolver.resolve_with_headers("aihubmix")
        assert api_key == "ahm-key"
        assert headers == {"APP-Code": "xyz"}

    def test_resolve_with_headers_no_headers(self) -> None:
        providers = ProvidersConfig(
            openrouter=ProviderConfig(api_key="sk-or-key"),
        )
        resolver = ProviderResolver(providers)
        _, _, headers = resolver.resolve_with_headers("openrouter")
        assert headers is None

    def test_fallback_includes_new_providers(self) -> None:
        """New providers should appear in priority-based fallback."""
        providers = ProvidersConfig(
            deepseek=ProviderConfig(api_key="ds-only"),
        )
        resolver = ProviderResolver(providers)
        api_key, _ = resolver.resolve()
        assert api_key == "ds-only"

    def test_new_providers_lower_priority_than_existing(self) -> None:
        """Existing providers should still win in fallback order."""
        providers = ProvidersConfig(
            anthropic=ProviderConfig(api_key="ant-key"),
            deepseek=ProviderConfig(api_key="ds-key"),
        )
        resolver = ProviderResolver(providers)
        api_key, _ = resolver.resolve()
        assert api_key == "ant-key"


# ---------------------------------------------------------------------------
# ToolsConfig.restrict_to_workspace
# ---------------------------------------------------------------------------


class TestToolsConfigRestrictToWorkspace:
    def test_default_false(self) -> None:
        tools = ToolsConfig()
        assert tools.restrict_to_workspace is False

    def test_set_true(self) -> None:
        tools = ToolsConfig(**{"restrictToWorkspace": True})
        assert tools.restrict_to_workspace is True

    def test_full_config_with_restriction(self) -> None:
        config = Config(
            tools=ToolsConfig(**{"restrictToWorkspace": True}),
        )
        assert config.tools.restrict_to_workspace is True


# ---------------------------------------------------------------------------
# Filesystem tool workspace restriction
# ---------------------------------------------------------------------------


class TestFilesystemRestriction:
    def test_read_file_outside_workspace_blocked(self, tmp_path: Path) -> None:
        from nanobot.agent.tools.filesystem import ReadFileTool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        tool = ReadFileTool(allowed_dir=workspace)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(path=str(outside))
        )
        assert "outside the allowed directory" in result

    def test_read_file_inside_workspace_allowed(self, tmp_path: Path) -> None:
        from nanobot.agent.tools.filesystem import ReadFileTool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        inside = workspace / "test.txt"
        inside.write_text("hello", encoding="utf-8")

        tool = ReadFileTool(allowed_dir=workspace)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(path=str(inside))
        )
        assert result == "hello"

    def test_write_file_outside_workspace_blocked(self, tmp_path: Path) -> None:
        from nanobot.agent.tools.filesystem import WriteFileTool

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        tool = WriteFileTool(allowed_dir=workspace)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(path=str(tmp_path / "hack.txt"), content="pwned")
        )
        assert "outside the allowed directory" in result

    def test_edit_file_outside_workspace_blocked(self, tmp_path: Path) -> None:
        from nanobot.agent.tools.filesystem import EditFileTool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("old", encoding="utf-8")

        tool = EditFileTool(allowed_dir=workspace)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(path=str(outside), old_text="old", new_text="new")
        )
        assert "outside the allowed directory" in result

    def test_no_restriction_when_allowed_dir_none(self, tmp_path: Path) -> None:
        from nanobot.agent.tools.filesystem import ReadFileTool

        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        tool = ReadFileTool(allowed_dir=None)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(path=str(test_file))
        )
        assert result == "content"


# ---------------------------------------------------------------------------
# Prefix rules in LiteLLMProvider
# ---------------------------------------------------------------------------


class TestPrefixRules:
    def test_deepseek_model_gets_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="deepseek-chat")
        assert provider._apply_model_prefix("deepseek-chat") == "deepseek/deepseek-chat"

    def test_deepseek_already_prefixed(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="test-model")
        result = provider._apply_model_prefix("deepseek/deepseek-chat")
        assert result == "deepseek/deepseek-chat"

    def test_qwen_gets_dashscope_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="test-model")
        result = provider._apply_model_prefix("qwen-turbo")
        assert result == "dashscope/qwen-turbo"

    def test_moonshot_gets_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="test-model")
        result = provider._apply_model_prefix("moonshot-v1-8k")
        assert result == "moonshot/moonshot-v1-8k"

    def test_kimi_gets_moonshot_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="test-model")
        result = provider._apply_model_prefix("kimi-latest")
        assert result == "moonshot/kimi-latest"

    def test_gemini_gets_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="test-model")
        result = provider._apply_model_prefix("gemini-2.0-flash")
        assert result == "gemini/gemini-2.0-flash"

    def test_openrouter_model_gets_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key="sk-or-test",
            default_model="anthropic/claude-3.5-sonnet",
        )
        result = provider._apply_model_prefix("meta-llama/llama-3.3-70b")
        assert result == "openrouter/meta-llama/llama-3.3-70b"

    def test_aihubmix_gets_openai_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key="test",
            api_base="https://aihubmix.com/v1",
            default_model="gpt-4o",
        )
        result = provider._apply_model_prefix("gpt-4o")
        assert result == "openai/gpt-4o"

    def test_unknown_model_no_prefix(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test", default_model="anthropic/claude-3.5")
        result = provider._apply_model_prefix("anthropic/claude-3.5")
        assert result == "anthropic/claude-3.5"

    def test_extra_headers_stored(self) -> None:
        from nanobot.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key="test",
            default_model="test",
            extra_headers={"X-Custom": "value"},
        )
        assert provider.extra_headers == {"X-Custom": "value"}


# ---------------------------------------------------------------------------
# Runtime info in context builder
# ---------------------------------------------------------------------------


class TestRuntimeInfo:
    def test_runtime_info_in_system_prompt(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        builder = ContextBuilder(workspace=tmp_path)
        prompt = builder.build_system_prompt()
        assert "Python" in prompt
        # Should contain OS info (macOS or Linux or similar)
        assert any(
            os_name in prompt
            for os_name in ("macOS", "Linux", "Windows", "Darwin")
        )

    def test_get_runtime_info_format(self) -> None:
        from nanobot.agent.context import ContextBuilder

        info = ContextBuilder._get_runtime_info()
        assert "Python" in info
        # Should have format like "macOS arm64, Python 3.13.1"
        assert ", " in info
