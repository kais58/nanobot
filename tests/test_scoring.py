"""Tests for lead scoring and recommendation engine."""

from datetime import datetime, timezone

from nanobot.marketing.scoring import (
    LeadScorer,
    RecommendationEngine,
    ScoredLead,
)


class TestLeadScorer:
    def setup_method(self):
        self.scorer = LeadScorer()

    def test_no_signals_low_score(self):
        result = self.scorer.score_lead("Empty Corp", [])
        assert result.total_score < 0.5
        assert result.tier == "cold"
        assert result.signal_count == 0

    def test_strong_signals_high_score(self):
        now = datetime.now(timezone.utc).isoformat()
        signals = [
            {
                "signal_type": "restructuring",
                "relevance_score": 0.9,
                "detected_at": now,
            },
            {
                "signal_type": "leadership_change",
                "relevance_score": 0.8,
                "detected_at": now,
            },
        ]
        result = self.scorer.score_lead(
            "Hot Corp",
            signals,
            {"industry": "automotive", "region": "germany", "employees": 1000},
        )
        assert result.total_score > 0.5
        assert result.tier in ("hot", "warm")
        assert result.signal_count == 2
        assert "turnaround" in result.recommended_services

    def test_tier_assignment(self):
        assert LeadScorer._assign_tier(0.8) == "hot"
        assert LeadScorer._assign_tier(0.5) == "warm"
        assert LeadScorer._assign_tier(0.3) == "cold"

    def test_industry_fit_scoring(self):
        result_auto = self.scorer.score_lead(
            "Auto GmbH",
            [{"signal_type": "restructuring", "relevance_score": 0.5,
              "detected_at": datetime.now(timezone.utc).isoformat()}],
            {"industry": "automotive", "region": "germany"},
        )
        result_unknown = self.scorer.score_lead(
            "Unknown GmbH",
            [{"signal_type": "restructuring", "relevance_score": 0.5,
              "detected_at": datetime.now(timezone.utc).isoformat()}],
            {"industry": "unknown_industry", "region": "unknown_region"},
        )
        assert result_auto.fit_score > result_unknown.fit_score

    def test_engagement_scoring(self):
        now = datetime.now(timezone.utc).isoformat()
        signals = [
            {"signal_type": "expansion", "relevance_score": 0.6,
             "detected_at": now},
        ]
        result_positive = self.scorer.score_lead(
            "Good Corp", signals,
            engagement_history=[
                {"outcome": "replied"},
                {"outcome": "meeting"},
            ],
        )
        result_negative = self.scorer.score_lead(
            "Bad Corp", signals,
            engagement_history=[
                {"outcome": "bounced"},
                {"outcome": "unsubscribed"},
            ],
        )
        assert result_positive.engagement_score > result_negative.engagement_score


class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine()

    def test_match_by_specialization(self):
        lead = ScoredLead(
            company_name="Test Corp",
            total_score=0.75,
            tier="hot",
            recommended_services=["change_management", "strategy"],
            details={"industry": "automotive", "region": "germany"},
        )
        consultants = [
            {
                "name": "Dr. Schmidt",
                "email": "schmidt@kp.de",
                "specializations": ["change_management"],
                "industries": ["automotive"],
                "regions": ["germany"],
                "active": True,
            },
            {
                "name": "Frau Mueller",
                "email": "mueller@kp.de",
                "specializations": ["leadership"],
                "industries": ["healthcare"],
                "regions": ["germany"],
                "active": True,
            },
        ]
        rec = self.engine.generate_recommendation(lead, consultants)
        assert rec is not None
        assert rec.consultant_name == "Dr. Schmidt"
        assert rec.service_area == "change_management"

    def test_no_consultants_returns_none(self):
        lead = ScoredLead(
            company_name="Test Corp",
            recommended_services=["strategy"],
            details={},
        )
        rec = self.engine.generate_recommendation(lead, [])
        assert rec is None

    def test_inactive_consultants_skipped(self):
        lead = ScoredLead(
            company_name="Test Corp",
            recommended_services=["strategy"],
            details={},
        )
        consultants = [
            {
                "name": "Inactive",
                "email": "x@kp.de",
                "specializations": ["strategy"],
                "industries": [],
                "regions": [],
                "active": False,
            },
        ]
        rec = self.engine.generate_recommendation(lead, consultants)
        assert rec is None
