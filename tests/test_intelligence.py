"""Tests for market intelligence tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.tools.intelligence import MarketIntelligenceTool


class TestMarketIntelligenceTool:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.get_signal_stats.return_value = {
            "total": 5,
            "by_status": {"new": 3, "reviewed": 2},
            "by_type": {"restructuring": 2, "ma": 3},
        }
        store.get_signals.return_value = [
            {
                "id": 1,
                "company_name": "Test GmbH",
                "signal_type": "restructuring",
                "title": "Major restructuring",
                "relevance_score": 0.8,
                "source_url": "https://example.com",
                "source_name": "Test News",
            },
        ]
        store.get_recommendations.return_value = []
        store.add_signal.return_value = {"id": 1}
        return store

    @pytest.fixture
    def tool(self, mock_store):
        return MarketIntelligenceTool(
            intel_store=mock_store,
            brave_api_key=None,
            provider=None,
        )

    @pytest.mark.asyncio
    async def test_get_signal_stats(self, tool):
        result = await tool.execute(action="get_signal_stats")
        assert "total: 5" in result
        assert "restructuring" in result

    @pytest.mark.asyncio
    async def test_generate_report(self, tool):
        result = await tool.execute(action="generate_report")
        assert "Market Intelligence Summary" in result
        assert "Test GmbH" in result

    @pytest.mark.asyncio
    async def test_scan_news_no_api_key(self, tool):
        result = await tool.execute(action="scan_news")
        assert "Error" in result
        assert "Brave API key" in result

    @pytest.mark.asyncio
    async def test_scan_company_missing_name(self, tool):
        result = await tool.execute(action="scan_company")
        assert "Error" in result
        assert "company_name" in result

    @pytest.mark.asyncio
    async def test_analyze_signals_no_provider(self, tool):
        result = await tool.execute(
            action="analyze_signals",
            text="Test text about restructuring",
        )
        assert "Error" in result
        assert "LLM provider" in result

    @pytest.mark.asyncio
    async def test_analyze_signals_with_provider(self, mock_store):
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(
            return_value=MagicMock(
                content='[{"company_name": "Test GmbH", '
                '"signal_type": "restructuring", '
                '"title": "Major restructuring announced", '
                '"relevance_score": 0.8, '
                '"kp_service_match": "turnaround"}]'
            )
        )
        tool = MarketIntelligenceTool(
            intel_store=mock_store,
            provider=mock_provider,
            model="test-model",
        )
        result = await tool.execute(
            action="analyze_signals",
            text="Test GmbH announces major restructuring.",
            source_url="https://example.com",
            source_name="Test News",
        )
        assert "1 signal(s)" in result
        assert "Test GmbH" in result
        mock_store.add_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        result = await tool.execute(action="unknown")
        assert "Error" in result
