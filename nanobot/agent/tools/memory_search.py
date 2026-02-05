"""Memory search tool for querying past conversations."""

from typing import Any

from nanobot.agent.tools.base import Tool


class MemorySearchTool(Tool):
    """
    Search semantic memory from past conversations.

    Allows the agent to recall information from previous sessions,
    including facts, decisions, and conversation context.
    """

    def __init__(self, vector_store: Any):
        """
        Initialize the memory search tool.

        Args:
            vector_store: VectorStore instance for semantic search.
        """
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search past conversations and extracted facts from memory. "
            "Use this to recall information discussed in previous sessions, "
            "user preferences, decisions made, or any other context from the past."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing what you're looking for. "
                        "Be specific about the topic or type of information."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Execute the memory search."""
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)

        if not query:
            return "Error: query is required"

        if self._vector_store is None:
            return "Memory search is not available (vector store not initialized)"

        try:
            results = await self._vector_store.search(
                query=query,
                top_k=limit,
            )

            if not results:
                return f"No memories found matching: {query}"

            # Format results
            output = [f"Found {len(results)} relevant memories:\n"]

            for i, result in enumerate(results, 1):
                similarity = result.get("similarity", 0)
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                created_at = result.get("created_at", "")

                # Format metadata
                session_key = metadata.get("session_key", "unknown")
                entry_type = metadata.get("type", "conversation")

                output.append(f"--- Memory {i} (similarity: {similarity:.2f}) ---")
                output.append(f"Type: {entry_type}")
                output.append(f"Session: {session_key}")
                if created_at:
                    output.append(f"Date: {created_at[:10]}")
                output.append(f"Content: {text[:500]}{'...' if len(text) > 500 else ''}")
                output.append("")

            return "\n".join(output)

        except Exception as e:
            return f"Error searching memory: {e}"
