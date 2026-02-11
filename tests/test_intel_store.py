"""Tests for the intelligence store."""

import pytest
from pathlib import Path

from nanobot.marketing.intel_store import IntelStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_intel.db"
    s = IntelStore(db_path=db_path)
    yield s
    s.close()


class TestSignals:
    def test_add_and_get(self, store):
        signal = store.add_signal(
            company_name="Test GmbH",
            signal_type="restructuring",
            title="Test restructuring signal",
            source_url="https://example.com/news",
            source_name="Example News",
            relevance_score=0.8,
        )
        assert signal["company_name"] == "Test GmbH"
        assert signal["signal_type"] == "restructuring"
        assert signal["status"] == "new"

    def test_filter_by_status(self, store):
        store.add_signal(
            "A", "restructuring", "T1", "http://a", "src"
        )
        store.add_signal("B", "ma", "T2", "http://b", "src")
        store.update_signal_status(1, "reviewed")
        new = store.get_signals(status="new")
        assert len(new) == 1
        assert new[0]["company_name"] == "B"

    def test_filter_by_relevance(self, store):
        store.add_signal(
            "A",
            "restructuring",
            "T1",
            "http://a",
            "src",
            relevance_score=0.3,
        )
        store.add_signal(
            "B",
            "ma",
            "T2",
            "http://b",
            "src",
            relevance_score=0.9,
        )
        high = store.get_signals(min_relevance=0.7)
        assert len(high) == 1
        assert high[0]["company_name"] == "B"


class TestRecommendations:
    def test_add_and_get(self, store):
        rec = store.add_recommendation(
            company_name="Test Corp",
            consultant_name="Max Mustermann",
            service_area="change_management",
        )
        assert rec["status"] == "pending"
        recs = store.get_recommendations(status="pending")
        assert len(recs) == 1

    def test_approve(self, store):
        store.add_recommendation(company_name="Test Corp")
        assert store.update_recommendation_status(
            1, "approved", "admin"
        )
        recs = store.get_recommendations(status="approved")
        assert len(recs) == 1
        assert recs[0]["approved_by"] == "admin"


class TestConsultants:
    def test_add_and_filter(self, store):
        store.add_consultant(
            name="Dr. Schmidt",
            specializations=[
                "change_management",
                "strategy",
            ],
            industries=["automotive"],
        )
        store.add_consultant(
            name="Frau Mueller",
            specializations=["leadership"],
            industries=["healthcare"],
        )
        auto = store.get_consultants(industry="automotive")
        assert len(auto) == 1
        assert auto[0]["name"] == "Dr. Schmidt"


class TestStats:
    def test_signal_stats(self, store):
        store.add_signal(
            "A", "restructuring", "T1", "http://a", "src"
        )
        store.add_signal("B", "ma", "T2", "http://b", "src")
        stats = store.get_signal_stats()
        assert stats["total"] == 2
        assert stats["by_type"]["restructuring"] == 1
