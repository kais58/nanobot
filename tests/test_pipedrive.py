"""Tests for Pipedrive API wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.config.schema import PipedriveConfig, MarketingConfig


class TestPipedriveConfig:
    def test_defaults(self):
        config = PipedriveConfig()
        assert config.enabled is False
        assert config.api_token == ""
        assert config.api_url == "https://api.pipedrive.com/v1"

    def test_from_alias(self):
        config = PipedriveConfig(
            **{
                "apiToken": "test-token",
                "syncIntervalMinutes": 15,
            }
        )
        assert config.api_token == "test-token"
        assert config.sync_interval_minutes == 15


class TestPipedriveClient:
    @pytest.fixture
    def client(self):
        from nanobot.marketing.pipedrive import PipedriveClient

        return PipedriveClient(api_token="test-token")

    @pytest.mark.asyncio
    async def test_search_persons(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "items": [
                    {
                        "item": {
                            "id": 1,
                            "name": "Test Person",
                        }
                    }
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client, "_get_client"
        ) as mock_get:
            mock_http = AsyncMock()
            mock_http.request = AsyncMock(
                return_value=mock_response
            )
            mock_get.return_value = mock_http

            results = await client.search_persons("Test")
            assert len(results) == 1
            assert results[0]["name"] == "Test Person"

    @pytest.mark.asyncio
    async def test_create_person(self, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {"id": 1, "name": "New Person"},
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client, "_get_client"
        ) as mock_get:
            mock_http = AsyncMock()
            mock_http.request = AsyncMock(
                return_value=mock_response
            )
            mock_get.return_value = mock_http

            result = await client.create_person(
                "New Person", email="test@example.com"
            )
            assert result["name"] == "New Person"
