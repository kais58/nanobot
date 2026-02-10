"""Lead scoring tool for the agent."""

from typing import Any

from nanobot.agent.tools.base import Tool


class LeadScoringTool(Tool):
    """Tool to score leads and generate consultant-lead recommendations."""

    def __init__(
        self,
        intel_store: Any,
        scorer: Any,
        recommendation_engine: Any,
    ):
        self._store = intel_store
        self._scorer = scorer
        self._engine = recommendation_engine

    @property
    def name(self) -> str:
        return "lead_scoring"

    @property
    def description(self) -> str:
        return (
            "Score leads and match consultants. Actions: score_lead "
            "(score a specific company), score_all (score all companies "
            "with signals), get_recommendations (list pending), "
            "match_consultant (find best consultant for a lead)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "score_lead",
                        "score_all",
                        "get_recommendations",
                        "match_consultant",
                    ],
                    "description": "Action to perform",
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name (for score_lead, match_consultant)",
                },
                "industry": {
                    "type": "string",
                    "description": "Company industry (for score_lead)",
                },
                "region": {
                    "type": "string",
                    "description": "Company region (for score_lead)",
                },
                "employees": {
                    "type": "integer",
                    "description": "Employee count (for score_lead)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        company_name: str | None = None,
        industry: str | None = None,
        region: str | None = None,
        employees: int | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "score_lead":
                return self._score_lead(company_name, industry, region, employees)
            elif action == "score_all":
                return self._score_all(limit)
            elif action == "get_recommendations":
                return self._get_recommendations(limit)
            elif action == "match_consultant":
                return self._match_consultant(company_name, industry, region, employees)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    def _score_lead(
        self,
        company_name: str | None,
        industry: str | None,
        region: str | None,
        employees: int | None,
    ) -> str:
        if not company_name:
            return "Error: company_name is required"

        signals = self._store.get_signals(limit=100)
        company_signals = [
            s for s in signals if s.get("company_name", "").lower() == company_name.lower()
        ]

        company_info: dict[str, Any] = {}
        if industry:
            company_info["industry"] = industry
        if region:
            company_info["region"] = region
        if employees:
            company_info["employees"] = employees

        scored = self._scorer.score_lead(company_name, company_signals, company_info)

        return (
            f"Lead Score: {company_name}\n"
            f"  Total: {scored.total_score:.0%} ({scored.tier})\n"
            f"  Signal: {scored.signal_score:.0%} "
            f"({scored.signal_count} signals)\n"
            f"  Fit: {scored.fit_score:.0%}\n"
            f"  Engagement: {scored.engagement_score:.0%}\n"
            f"  Top signal: {scored.top_signal_type}\n"
            f"  Services: {', '.join(scored.recommended_services)}"
        )

    def _score_all(self, limit: int) -> str:
        signals = self._store.get_signals(limit=500)

        # Group by company
        by_company: dict[str, list[dict[str, Any]]] = {}
        for s in signals:
            name = s.get("company_name", "Unknown")
            by_company.setdefault(name, []).append(s)

        scored_leads = []
        for name, sigs in by_company.items():
            scored = self._scorer.score_lead(name, sigs)
            scored_leads.append(scored)

        # Sort by score descending
        scored_leads.sort(key=lambda x: x.total_score, reverse=True)
        scored_leads = scored_leads[:limit]

        if not scored_leads:
            return "No leads with signals found."

        lines = ["Scored Leads:\n"]
        for i, lead in enumerate(scored_leads, 1):
            lines.append(
                f"{i}. {lead.company_name}: "
                f"{lead.total_score:.0%} ({lead.tier}) - "
                f"{lead.signal_count} signal(s)"
            )
        return "\n".join(lines)

    def _get_recommendations(self, limit: int) -> str:
        recs = self._store.get_recommendations(status="pending", limit=limit)
        if not recs:
            return "No pending recommendations."

        lines = ["Pending Recommendations:\n"]
        for r in recs:
            lines.append(
                f"- [{r['id']}] {r['company_name']}: "
                f"{r.get('service_area', 'N/A')} "
                f"({r.get('consultant_name', 'unassigned')})"
            )
        return "\n".join(lines)

    def _match_consultant(
        self,
        company_name: str | None,
        industry: str | None,
        region: str | None,
        employees: int | None,
    ) -> str:
        if not company_name:
            return "Error: company_name is required"

        signals = self._store.get_signals(limit=100)
        company_signals = [
            s for s in signals if s.get("company_name", "").lower() == company_name.lower()
        ]

        company_info: dict[str, Any] = {}
        if industry:
            company_info["industry"] = industry
        if region:
            company_info["region"] = region
        if employees:
            company_info["employees"] = employees

        scored = self._scorer.score_lead(company_name, company_signals, company_info)

        consultants = self._store.get_consultants()
        rec = self._engine.generate_recommendation(scored, consultants)

        if not rec:
            return "No suitable consultant match found."

        # Store the recommendation
        self._store.add_recommendation(
            company_name=rec.company_name,
            signal_id=(company_signals[0].get("id") if company_signals else None),
            consultant_name=rec.consultant_name,
            service_area=rec.service_area,
            outreach_channel=rec.outreach_channel,
            pitch_summary=(f"Score: {rec.lead_score:.0%} ({rec.lead_tier}). {rec.signal_summary}"),
        )

        return (
            f"Recommendation for {company_name}:\n"
            f"  Consultant: {rec.consultant_name}\n"
            f"  Service: {rec.service_area}\n"
            f"  Channel: {rec.outreach_channel}\n"
            f"  Match score: {rec.match_score:.0%}\n"
            f"  Lead score: {rec.lead_score:.0%} ({rec.lead_tier})"
        )
