"""Memory forget tool for removing specific memories from the vector store."""

import time
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

# Rate limit: max deletions within a sliding time window
MAX_DELETIONS_PER_WINDOW = 20
WINDOW_SECONDS = 1800  # 30 minutes


class MemoryForgetTool(Tool):
    """
    Remove specific memories from the vector store.

    Use when the user asks to forget something or when incorrect
    information was stored. Rate-limited to 20 deletions per 30 minutes.
    Supports a preview mode (confirm=False) to show what would be
    deleted before actually deleting.
    """

    def __init__(self, vector_store: Any):
        """
        Initialize the memory forget tool.

        Args:
            vector_store: VectorStore instance with search and delete methods.
        """
        self._vector_store = vector_store
        self._deletion_timestamps: list[float] = []

    @property
    def name(self) -> str:
        return "memory_forget"

    @property
    def description(self) -> str:
        return (
            "Remove specific memories from the vector store. Use when "
            "the user asks you to forget something or when incorrect "
            "information was stored. Set confirm=false first to preview "
            "what would be deleted, then confirm=true to execute. "
            "Rate-limited to 20 deletions per 30 minutes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Description of what to forget. Be specific "
                        "about the topic or information to remove."
                    ),
                },
                "confirm": {
                    "type": "boolean",
                    "description": (
                        "Set to false to preview what would be deleted. "
                        "Set to true to actually delete the memories."
                    ),
                },
            },
            "required": ["query", "confirm"],
        }

    def _prune_old_timestamps(self) -> None:
        """Remove deletion timestamps older than the rate-limit window."""
        cutoff = time.monotonic() - WINDOW_SECONDS
        self._deletion_timestamps = [ts for ts in self._deletion_timestamps if ts > cutoff]

    def _remaining_deletions(self) -> int:
        """Return how many deletions are allowed in the current window."""
        self._prune_old_timestamps()
        return MAX_DELETIONS_PER_WINDOW - len(self._deletion_timestamps)

    async def execute(self, **kwargs: Any) -> str:
        """Execute the memory forget operation."""
        query = kwargs.get("query", "")
        confirm = kwargs.get("confirm", False)

        if not query:
            return "Error: query is required."

        if self._vector_store is None:
            return "Memory forget is not available (vector store not initialized)."

        remaining = self._remaining_deletions()
        if remaining <= 0:
            # Find when the oldest deletion in the window expires
            oldest = min(self._deletion_timestamps)
            reset_at = oldest + WINDOW_SECONDS
            seconds_left = int(reset_at - time.monotonic())
            minutes_left = max(1, (seconds_left + 59) // 60)
            return (
                f"Deletion rate limit reached ({MAX_DELETIONS_PER_WINDOW} "
                f"per {WINDOW_SECONDS // 60} minutes). "
                f"Try again in ~{minutes_left} minute(s)."
            )

        try:
            # Search for matching memories
            results = await self._vector_store.search(
                query=query,
                top_k=8,
                min_similarity=0.6,
            )

            if not results:
                return f"No memories found matching: {query}"

            # Preview mode: show what would be deleted
            if not confirm:
                lines = [f"Found {len(results)} memories that would be deleted:\n"]
                for i, result in enumerate(results, 1):
                    text = result.get("text", "")
                    similarity = result.get("similarity", 0)
                    created_at = result.get("created_at", "")
                    preview = text[:200] + "..." if len(text) > 200 else text
                    lines.append(f"{i}. (similarity: {similarity:.2f}) {preview}")
                    if created_at:
                        lines.append(f"   Date: {created_at[:10]}")
                    lines.append("")

                lines.append("Call again with confirm=true to delete these memories.")
                return "\n".join(lines)

            # Deletion mode — cap by remaining quota
            deleted = 0
            for result in results:
                if self._remaining_deletions() <= 0:
                    break
                entry_id = result.get("id")
                if entry_id is None:
                    continue
                try:
                    if await self._vector_store.delete(entry_id):
                        deleted += 1
                        self._deletion_timestamps.append(time.monotonic())
                except Exception as e:
                    logger.warning(f"Failed to delete memory entry: {e}")

            remaining = self._remaining_deletions()

            logger.debug(
                f"Deleted {deleted} memories for query '{query}', "
                f"{remaining} deletions remaining in window"
            )
            return (
                f"Deleted {deleted} memories matching: {query}. Remaining deletions: {remaining}."
            )

        except Exception as e:
            return f"Error during memory forget: {e}"
