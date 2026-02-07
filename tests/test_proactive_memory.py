"""Tests for ProactiveMemory dismiss capability."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.memory.proactive import ProactiveMemory


@pytest.fixture
def vector_store() -> AsyncMock:
    """Create a mock vector store."""
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def proactive_memory(tmp_path: Path, vector_store: AsyncMock) -> ProactiveMemory:
    """Create a ProactiveMemory instance with a temp data dir."""
    data_dir = tmp_path / "memory"
    data_dir.mkdir()
    return ProactiveMemory(
        vector_store=vector_store,
        data_dir=data_dir,
    )


class TestDismissReminder:
    """Tests for the dismiss_reminder capability."""

    def test_dismiss_creates_file(self, proactive_memory: ProactiveMemory) -> None:
        """Dismissing a reminder creates the dismissed.json file."""
        proactive_memory.dismiss_reminder("remind me to do laundry")
        assert proactive_memory._dismissed_path.exists()

    def test_dismiss_stores_hash(self, proactive_memory: ProactiveMemory) -> None:
        """Dismissed text hash is persisted to disk."""
        text = "remind me to check email at 9am"
        proactive_memory.dismiss_reminder(text)

        dismissed = proactive_memory._load_dismissed()
        expected_hash = ProactiveMemory._text_hash(text)
        assert expected_hash in dismissed

    def test_dismiss_multiple(self, proactive_memory: ProactiveMemory) -> None:
        """Multiple dismissals accumulate."""
        proactive_memory.dismiss_reminder("first reminder")
        proactive_memory.dismiss_reminder("second reminder")

        dismissed = proactive_memory._load_dismissed()
        assert len(dismissed) == 2

    def test_dismiss_idempotent(self, proactive_memory: ProactiveMemory) -> None:
        """Dismissing the same text twice doesn't duplicate."""
        proactive_memory.dismiss_reminder("same reminder")
        proactive_memory.dismiss_reminder("same reminder")

        dismissed = proactive_memory._load_dismissed()
        assert len(dismissed) == 1

    @pytest.mark.asyncio
    async def test_dismissed_filtered_from_reminders(
        self,
        proactive_memory: ProactiveMemory,
        vector_store: AsyncMock,
    ) -> None:
        """Dismissed reminders are filtered out of get_reminders results."""
        reminder_text = "remind me to call the dentist"

        # Mock vector store to return a commitment-matching entry
        vector_store.search = AsyncMock(
            return_value=[{"text": reminder_text, "similarity": 0.9}]
        )

        # Before dismiss: reminder should appear
        reminders = await proactive_memory.get_reminders()
        assert reminder_text in reminders

        # Dismiss it
        proactive_memory.dismiss_reminder(reminder_text)

        # After dismiss: reminder should be filtered out
        reminders = await proactive_memory.get_reminders()
        assert reminder_text not in reminders

    @pytest.mark.asyncio
    async def test_non_dismissed_still_surface(
        self,
        proactive_memory: ProactiveMemory,
        vector_store: AsyncMock,
    ) -> None:
        """Non-dismissed reminders still appear normally."""
        kept = "remind me to buy groceries"
        dismissed_text = "remind me to do laundry"

        vector_store.search = AsyncMock(
            return_value=[
                {"text": kept, "similarity": 0.9},
                {"text": dismissed_text, "similarity": 0.8},
            ]
        )

        proactive_memory.dismiss_reminder(dismissed_text)

        reminders = await proactive_memory.get_reminders()
        assert kept in reminders
        assert dismissed_text not in reminders


class TestDismissedPersistence:
    """Tests for dismissed state persistence."""

    def test_load_empty(self, proactive_memory: ProactiveMemory) -> None:
        """Loading when no file exists returns empty set."""
        dismissed = proactive_memory._load_dismissed()
        assert dismissed == set()

    def test_roundtrip(self, proactive_memory: ProactiveMemory) -> None:
        """Save and reload dismissed set."""
        original = {"abc123", "def456"}
        proactive_memory._save_dismissed(original)
        loaded = proactive_memory._load_dismissed()
        assert loaded == original

    def test_corrupt_file_returns_empty(self, proactive_memory: ProactiveMemory) -> None:
        """Corrupt dismissed file returns empty set gracefully."""
        proactive_memory._dismissed_path.write_text("not json!", encoding="utf-8")
        dismissed = proactive_memory._load_dismissed()
        assert dismissed == set()
