"""Market intelligence tool for proactive signal discovery."""

import json
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class MarketIntelligenceTool(Tool):
    """Tool for market intelligence gathering and analysis.

    Provides proactive scanning of German-language business news,
    company-specific deep dives, and structured signal extraction.
    """

    def __init__(
        self,
        intel_store: Any,
        brave_api_key: str | None = None,
        provider: Any = None,
        model: str | None = None,
        search_queries: list[str] | None = None,
    ):
        self._store = intel_store
        self._brave_api_key = brave_api_key
        self._provider = provider
        self._model = model
        self._search_queries = search_queries or [
            "Umstrukturierung Deutschland Unternehmen",
            "CEO Wechsel DAX MDAX",
            "digitale Transformation Mittelstand",
            "Restrukturierung Insolvenz Deutschland",
            "Fuehrungswechsel Vorstand Deutschland",
            "Unternehmensberatung Auftrag DACH",
        ]

    @property
    def name(self) -> str:
        return "market_intelligence"

    @property
    def description(self) -> str:
        return (
            "Gather and analyze market intelligence for consulting "
            "opportunities. Actions: scan_news (batch search for "
            "signals), scan_company (deep-dive on a company), "
            "analyze_signals (extract structured signals from text), "
            "generate_report (create intelligence report), "
            "get_signal_stats (view signal statistics)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "scan_news",
                        "scan_company",
                        "analyze_signals",
                        "generate_report",
                        "get_signal_stats",
                    ],
                    "description": "Action to perform",
                },
                "company_name": {
                    "type": "string",
                    "description": ("Company name (for scan_company action)"),
                },
                "text": {
                    "type": "string",
                    "description": ("Raw text to analyze for signals (for analyze_signals action)"),
                },
                "source_url": {
                    "type": "string",
                    "description": "Source URL for the text being analyzed",
                },
                "source_name": {
                    "type": "string",
                    "description": "Source name for the text being analyzed",
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Custom search queries (overrides defaults)"),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per query",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        company_name: str | None = None,
        text: str | None = None,
        source_url: str | None = None,
        source_name: str | None = None,
        queries: list[str] | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "scan_news":
                return await self._scan_news(queries, limit)
            elif action == "scan_company":
                return await self._scan_company(company_name, limit)
            elif action == "analyze_signals":
                return await self._analyze_signals(text, source_url, source_name)
            elif action == "generate_report":
                return self._generate_report()
            elif action == "get_signal_stats":
                return self._get_signal_stats()
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    async def _scan_news(self, queries: list[str] | None, limit: int) -> str:
        """Batch search for market signals via Brave API."""
        if not self._brave_api_key:
            return "Error: No Brave API key configured. Cannot perform web searches."

        search_queries = queries or self._search_queries
        all_results: list[dict[str, Any]] = []

        from nanobot.agent.tools.web import WebSearchTool

        searcher = WebSearchTool(api_key=self._brave_api_key)

        for query in search_queries:
            try:
                result_text = await searcher.execute(query=query, max_results=limit)
                all_results.append(
                    {
                        "query": query,
                        "results": result_text,
                    }
                )
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        if not all_results:
            return "No search results found."

        lines = [
            f"News scan complete: {len(search_queries)} queries, "
            f"{len(all_results)} returned results.\n"
        ]
        for r in all_results:
            lines.append(f"## Query: {r['query']}")
            lines.append(r["results"][:500])
            lines.append("")

        return "\n".join(lines) + (
            "\n\nUse analyze_signals action to extract structured signals from these results."
        )

    async def _scan_company(self, company_name: str | None, limit: int) -> str:
        """Deep-dive search on a specific company."""
        if not company_name:
            return "Error: company_name is required for scan_company"

        if not self._brave_api_key:
            return "Error: No Brave API key configured."

        queries = [
            f"{company_name} Restrukturierung",
            f"{company_name} Vorstandswechsel CEO",
            f"{company_name} Strategie Transformation",
            f"{company_name} Stellenangebote Consulting",
        ]

        from nanobot.agent.tools.web import WebSearchTool

        searcher = WebSearchTool(api_key=self._brave_api_key)
        lines = [f"Company scan: {company_name}\n"]

        for query in queries:
            try:
                result = await searcher.execute(query=query, max_results=limit)
                lines.append(f"### {query}")
                lines.append(result[:500])
                lines.append("")
            except Exception as e:
                logger.warning(f"Company scan failed for '{query}': {e}")

        return "\n".join(lines) + ("\n\nUse analyze_signals to extract structured signals.")

    async def _analyze_signals(
        self,
        text: str | None,
        source_url: str | None,
        source_name: str | None,
    ) -> str:
        """Extract structured signals from text using LLM."""
        if not text:
            return "Error: text is required for analyze_signals"

        if not self._provider:
            return (
                "Error: No LLM provider configured for signal analysis. "
                "The agent should extract signals manually."
            )

        prompt = (
            "Analyze the following text for consulting opportunity signals. "
            "Extract company names and classify signals.\n\n"
            "Signal types:\n"
            "- restructuring: company reorganization, cost-cutting\n"
            "- leadership_change: new CEO, board changes\n"
            "- ma: mergers, acquisitions, divestitures\n"
            "- digital_transformation: digitalization initiatives\n"
            "- expansion: market entry, new products, growth\n"
            "- cost_cutting: layoffs, efficiency programs\n\n"
            "K&P service areas: change_management, strategy, leadership, "
            "turnaround, process_optimization, sales_management\n\n"
            "Output JSON array of objects:\n"
            '{"company_name": "...", "signal_type": "...", '
            '"title": "short description", "relevance_score": 0.0-1.0, '
            '"kp_service_match": "best matching service area"}\n\n'
            "Only include clearly identified signals. "
            "If none found, output [].\n\n"
            f"Text to analyze:\n{text[:3000]}\n\nJSON:"
        )

        response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            max_tokens=1024,
            temperature=0.2,
        )

        content = (response.content or "").strip()
        if not content:
            return "No signals extracted from text."

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            signals = json.loads(content)
        except json.JSONDecodeError:
            return f"Failed to parse signal extraction output: {content[:200]}"

        if not isinstance(signals, list) or not signals:
            return "No signals found in the text."

        # Store extracted signals
        stored = 0
        for signal in signals:
            cn = signal.get("company_name", "").strip()
            st = signal.get("signal_type", "").strip()
            title = signal.get("title", "").strip()
            if not cn or not st or not title:
                continue

            self._store.add_signal(
                company_name=cn,
                signal_type=st,
                title=title,
                source_url=source_url or "unknown",
                source_name=source_name or "LLM extraction",
                relevance_score=signal.get("relevance_score", 0.5),
                kp_service_match=signal.get("kp_service_match", ""),
            )
            stored += 1

        lines = [f"Extracted {stored} signal(s):\n"]
        for s in signals[:10]:
            lines.append(
                f"- {s.get('company_name')}: {s.get('signal_type')} "
                f"({s.get('relevance_score', 0):.0%}) - {s.get('title')}"
            )

        return "\n".join(lines)

    def _generate_report(self) -> str:
        """Generate a quick intelligence summary from stored signals."""
        stats = self._store.get_signal_stats()
        new_signals = self._store.get_signals(status="new", limit=10)
        pending_recs = self._store.get_recommendations(status="pending", limit=10)

        lines = ["Market Intelligence Summary\n"]
        lines.append(f"Total signals: {stats.get('total', 0)}")

        by_status = stats.get("by_status", {})
        for status, count in by_status.items():
            lines.append(f"  {status}: {count}")

        by_type = stats.get("by_type", {})
        if by_type:
            lines.append("\nBy type:")
            for stype, count in by_type.items():
                lines.append(f"  {stype}: {count}")

        if new_signals:
            lines.append(f"\nLatest {len(new_signals)} new signals:")
            for s in new_signals:
                lines.append(f"  - {s['company_name']}: {s['title']} ({s['relevance_score']:.0%})")

        if pending_recs:
            lines.append(f"\n{len(pending_recs)} pending recommendation(s)")

        return "\n".join(lines)

    def _get_signal_stats(self) -> str:
        """Get signal statistics."""
        stats = self._store.get_signal_stats()
        lines = [f"Signal Statistics (total: {stats.get('total', 0)})\n"]

        by_status = stats.get("by_status", {})
        if by_status:
            lines.append("By status:")
            for status, count in by_status.items():
                lines.append(f"  {status}: {count}")

        by_type = stats.get("by_type", {})
        if by_type:
            lines.append("\nBy type:")
            for stype, count in by_type.items():
                lines.append(f"  {stype}: {count}")

        return "\n".join(lines)
