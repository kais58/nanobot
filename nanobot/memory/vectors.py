"""SQLite-based vector store for semantic memory search."""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.llm.embeddings import EmbeddingService
from nanobot.utils.helpers import ensure_dir


class VectorStore:
    """
    SQLite-based vector store for semantic memory.

    Stores embeddings with metadata and supports cosine similarity search.
    Handles 100k+ entries efficiently.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize the vector store.

        Args:
            db_path: Path to SQLite database file.
            embedding_service: Service for generating embeddings.
        """
        self.db_path = Path(db_path).expanduser()
        self.embedding_service = embedding_service

        # Ensure directory exists
        ensure_dir(self.db_path.parent)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON vectors(content_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON vectors(created_at)
            """)
            conn.commit()

    def _compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        return json.dumps(embedding).encode("utf-8")

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize embedding from bytes."""
        return json.loads(data.decode("utf-8"))

    async def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a text entry to the vector store.

        Args:
            text: Text to index.
            metadata: Optional metadata (session_key, timestamp, etc.).

        Returns:
            True if added, False if already exists.
        """
        content_hash = self._compute_hash(text)

        # Check for duplicate
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM vectors WHERE content_hash = ?", (content_hash,))
            if cursor.fetchone():
                logger.debug(f"Skipping duplicate entry: {content_hash[:8]}...")
                return False

        # Generate embedding
        try:
            embedding = await self.embedding_service.embed_single(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return False

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO vectors
                (content_hash, text, embedding, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    content_hash,
                    text,
                    self._serialize_embedding(embedding),
                    json.dumps(metadata) if metadata else None,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

        logger.debug(f"Added vector entry: {content_hash[:8]}...")
        return True

    async def add_batch(
        self,
        entries: list[tuple[str, dict[str, Any] | None]],
    ) -> int:
        """
        Add multiple entries efficiently.

        Args:
            entries: List of (text, metadata) tuples.

        Returns:
            Number of entries added.
        """
        if not entries:
            return 0

        # Filter out duplicates
        to_add = []
        with sqlite3.connect(self.db_path) as conn:
            for text, metadata in entries:
                content_hash = self._compute_hash(text)
                cursor = conn.execute(
                    "SELECT id FROM vectors WHERE content_hash = ?", (content_hash,)
                )
                if not cursor.fetchone():
                    to_add.append((text, metadata, content_hash))

        if not to_add:
            return 0

        # Generate embeddings in batch
        texts = [t[0] for t in to_add]
        try:
            embeddings = await self.embedding_service.embed(texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return 0

        # Store in database
        added = 0
        with sqlite3.connect(self.db_path) as conn:
            for i, (text, metadata, content_hash) in enumerate(to_add):
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO vectors
                        (content_hash, text, embedding, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            content_hash,
                            text,
                            self._serialize_embedding(embeddings[i]),
                            json.dumps(metadata) if metadata else None,
                            datetime.now().isoformat(),
                        ),
                    )
                    added += 1
                except Exception as e:
                    logger.warning(f"Failed to add entry: {e}")
            conn.commit()

        logger.debug(f"Added {added} vector entries in batch")
        return added

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Search for similar entries.

        Args:
            query: Search query text.
            top_k: Maximum results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of matching entries with similarity scores.
        """
        # Generate query embedding
        try:
            query_embedding = await self.embedding_service.embed_single(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Load all vectors and compute similarities
        # Note: For very large stores (100k+), consider using approximate methods
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT text, embedding, metadata, created_at FROM vectors")
            for row in cursor:
                text, embedding_blob, metadata_json, created_at = row
                embedding = self._deserialize_embedding(embedding_blob)
                similarity = self._cosine_similarity(query_embedding, embedding)

                if similarity >= min_similarity:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    results.append(
                        {
                            "text": text,
                            "similarity": similarity,
                            "metadata": metadata,
                            "created_at": created_at,
                        }
                    )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def count(self) -> int:
        """Get the number of entries in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all entries from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM vectors")
            conn.commit()
        logger.info("Cleared vector store")
