"""Report generation tool for the agent."""

from typing import Any

from nanobot.agent.tools.base import Tool


class MarketReportTool(Tool):
    """Tool to generate and manage intelligence reports."""

    def __init__(self, report_generator: Any, intel_store: Any):
        self._generator = report_generator
        self._store = intel_store

    @property
    def name(self) -> str:
        return "market_report"

    @property
    def description(self) -> str:
        return (
            "Generate and manage market intelligence reports. "
            "Actions: daily_brief, weekly_report, monthly_summary, "
            "list_reports, get_report, render_outreach, list_templates."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "daily_brief",
                        "weekly_report",
                        "monthly_summary",
                        "list_reports",
                        "get_report",
                        "render_outreach",
                        "list_templates",
                    ],
                    "description": "Action to perform",
                },
                "report_id": {
                    "type": "integer",
                    "description": "Report ID (for get_report)",
                },
                "template_name": {
                    "type": "string",
                    "description": "Template filename (for render_outreach)",
                },
                "template_vars": {
                    "type": "object",
                    "description": "Variables for template rendering",
                },
                "report_type": {
                    "type": "string",
                    "description": "Filter: daily, weekly, or monthly",
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
        report_id: int | None = None,
        template_name: str | None = None,
        template_vars: dict[str, Any] | None = None,
        report_type: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "daily_brief":
                return self._daily_brief()
            elif action == "weekly_report":
                return self._weekly_report()
            elif action == "monthly_summary":
                return self._monthly_summary()
            elif action == "list_reports":
                return self._list_reports(report_type, limit)
            elif action == "get_report":
                return self._get_report(report_id)
            elif action == "render_outreach":
                return self._render_outreach(template_name, template_vars)
            elif action == "list_templates":
                return self._list_templates()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    def _daily_brief(self) -> str:
        signals = self._store.get_signals(status="new", limit=50)
        recommendations = self._store.get_recommendations(status="pending", limit=20)
        content = self._generator.generate_daily_brief(signals, recommendations)
        self._store.add_report("daily", "Daily Brief", content)
        return content

    def _weekly_report(self) -> str:
        from collections import defaultdict
        from datetime import datetime, timedelta

        from nanobot.marketing.scoring import LeadScorer

        now = datetime.utcnow()
        week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = now.strftime("%Y-%m-%d")
        signals = self._store.get_signals(limit=200)
        rec_stats = {
            "pending": 0,
            "approved": 0,
            "sent": 0,
            "rejected": 0,
        }
        for r in self._store.get_recommendations(limit=200):
            s = r.get("status", "pending")
            if s in rec_stats:
                rec_stats[s] += 1

        # Compute top leads using LeadScorer
        scorer = LeadScorer()
        by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sig in signals:
            by_company[sig.get("company_name", "")].append(sig)
        scored_leads = [scorer.score_lead(name, sigs) for name, sigs in by_company.items()]
        top_leads = [
            {
                "company_name": lead.company_name,
                "total_score": lead.total_score,
                "tier": lead.tier,
                "signal_count": lead.signal_count,
                "recommended_services": lead.recommended_services,
            }
            for lead in sorted(scored_leads, key=lambda x: x.total_score, reverse=True)[:10]
        ]

        content = self._generator.generate_weekly_report(
            signals=signals,
            top_leads=top_leads,
            recommendation_stats=rec_stats,
            week_start=week_start,
            week_end=week_end,
        )
        self._store.add_report("weekly", "Weekly Report", content)
        return content

    def _monthly_summary(self) -> str:
        stats = self._store.get_signal_stats()
        content = self._generator.generate_monthly_summary(
            total_signals=stats.get("total", 0),
            signal_trends=stats.get("by_type", {}),
        )
        self._store.add_report("monthly", "Monthly Summary", content)
        return content

    def _list_reports(self, report_type: str | None, limit: int) -> str:
        reports = self._store.get_reports(report_type=report_type, limit=limit)
        if not reports:
            return "No reports found."
        lines = ["Reports:\n"]
        for r in reports:
            lines.append(
                f"- [{r['id']}] {r['report_type']}: {r['title']} ({r['generated_at'][:10]})"
            )
        return "\n".join(lines)

    def _get_report(self, report_id: int | None) -> str:
        if not report_id:
            return "Error: report_id is required"
        report = self._store.get_report(report_id)
        if report:
            return report.get("content", "Empty report")
        return f"Error: Report {report_id} not found"

    def _render_outreach(
        self,
        template_name: str | None,
        template_vars: dict[str, Any] | None,
    ) -> str:
        if not template_name:
            return "Error: template_name is required"
        return self._generator.render_outreach(template_name, template_vars or {})

    def _list_templates(self) -> str:
        templates = self._generator.list_templates()
        if not templates:
            return "No templates found."
        return "Available templates:\n" + "\n".join(f"- {t}" for t in sorted(templates))
