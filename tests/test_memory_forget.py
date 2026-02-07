"""Tests for MemoryForgetTool rate limiting."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.tools.memory_forget import (
    MAX_DELETIONS_PER_WINDOW,
    WINDOW_SECONDS,
    MemoryForgetTool,
)


@pytest.fixture
def vector_store() -> AsyncMock:
    """Create a mock vector store."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.delete = AsyncMock(return_value=True)
    return store


@pytest.fixture
def tool(vector_store: AsyncMock) -> MemoryForgetTool:
    return MemoryForgetTool(vector_store=vector_store)


def _make_results(n: int) -> list[dict]:
    """Create n fake vector store search results."""
    return [
        {"id": f"id-{i}", "text": f"memory text {i}", "similarity": 0.8}
        for i in range(n)
    ]


class TestRateLimit:
    """Tests for time-based sliding window rate limiting."""

    @pytest.mark.asyncio
    async def test_fresh_tool_has_full_quota(self, tool: MemoryForgetTool) -> None:
        """A new tool instance has the full deletion budget."""
        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW

    @pytest.mark.asyncio
    async def test_deletions_reduce_quota(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """Each successful deletion reduces the remaining quota."""
        vector_store.search = AsyncMock(return_value=_make_results(3))

        result = await tool.execute(query="test", confirm=True)

        assert "Deleted 3" in result
        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW - 3

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_when_exhausted(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """Tool returns rate limit message when window is exhausted."""
        # Fill the quota by injecting timestamps
        now = time.monotonic()
        tool._deletion_timestamps = [now] * MAX_DELETIONS_PER_WINDOW

        result = await tool.execute(query="anything", confirm=True)

        assert "rate limit reached" in result.lower()
        assert f"{WINDOW_SECONDS // 60} minutes" in result

    @pytest.mark.asyncio
    async def test_old_timestamps_expire(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """Timestamps older than the window are pruned, freeing quota."""
        # Place all timestamps just outside the window
        old = time.monotonic() - WINDOW_SECONDS - 1
        tool._deletion_timestamps = [old] * MAX_DELETIONS_PER_WINDOW

        # Should have full quota again
        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW

    @pytest.mark.asyncio
    async def test_partial_window_expiry(
        self, tool: MemoryForgetTool
    ) -> None:
        """Only expired timestamps are pruned; recent ones remain."""
        now = time.monotonic()
        old = now - WINDOW_SECONDS - 1
        # 5 old + 3 recent
        tool._deletion_timestamps = [old] * 5 + [now] * 3

        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW - 3

    @pytest.mark.asyncio
    async def test_mid_batch_rate_limit(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """If quota runs out mid-batch, stops deleting gracefully."""
        # Leave room for only 2 more deletions
        now = time.monotonic()
        tool._deletion_timestamps = [now] * (MAX_DELETIONS_PER_WINDOW - 2)
        vector_store.search = AsyncMock(return_value=_make_results(5))

        result = await tool.execute(query="test", confirm=True)

        assert "Deleted 2" in result
        assert tool._remaining_deletions() == 0

    @pytest.mark.asyncio
    async def test_preview_does_not_consume_quota(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """Preview mode (confirm=false) does not affect the rate limit."""
        vector_store.search = AsyncMock(return_value=_make_results(5))

        result = await tool.execute(query="test", confirm=False)

        assert "confirm=true" in result
        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW

    @pytest.mark.asyncio
    async def test_rate_limit_message_includes_retry_time(
        self, tool: MemoryForgetTool
    ) -> None:
        """Rate limit message tells the user when they can retry."""
        now = time.monotonic()
        tool._deletion_timestamps = [now] * MAX_DELETIONS_PER_WINDOW

        result = await tool.execute(query="anything", confirm=True)

        assert "Try again in" in result
        assert "minute(s)" in result

    @pytest.mark.asyncio
    async def test_failed_deletes_dont_consume_quota(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        """Failed deletions don't count against the rate limit."""
        vector_store.search = AsyncMock(return_value=_make_results(3))
        vector_store.delete = AsyncMock(return_value=False)

        result = await tool.execute(query="test", confirm=True)

        assert "Deleted 0" in result
        assert tool._remaining_deletions() == MAX_DELETIONS_PER_WINDOW


class TestBasicFunctionality:
    """Ensure core behavior still works after refactoring."""

    @pytest.mark.asyncio
    async def test_no_query_returns_error(self, tool: MemoryForgetTool) -> None:
        result = await tool.execute(query="", confirm=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_no_vector_store(self) -> None:
        tool = MemoryForgetTool(vector_store=None)
        result = await tool.execute(query="test", confirm=True)
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_no_results_found(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        vector_store.search = AsyncMock(return_value=[])
        result = await tool.execute(query="nonexistent", confirm=True)
        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_preview_shows_results(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        vector_store.search = AsyncMock(return_value=_make_results(2))
        result = await tool.execute(query="test", confirm=False)
        assert "Found 2" in result
        assert "memory text 0" in result

    @pytest.mark.asyncio
    async def test_successful_deletion(
        self, tool: MemoryForgetTool, vector_store: AsyncMock
    ) -> None:
        vector_store.search = AsyncMock(return_value=_make_results(3))
        result = await tool.execute(query="test", confirm=True)
        assert "Deleted 3" in result
        assert vector_store.delete.call_count == 3
