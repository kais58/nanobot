"""Unified CRM tool routing to Pipedrive and local intelligence store."""

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class CRMTool(Tool):
    """Unified CRM tool that routes to Pipedrive and local intelligence store.

    Provides a single interface for the agent to manage leads, contacts,
    signals, recommendations, and activities across both systems.
    """

    def __init__(
        self,
        pipedrive_client: Any = None,
        intel_store: Any = None,
        consent_store: Any = None,
    ):
        self._pipedrive = pipedrive_client
        self._store = intel_store
        self._consent = consent_store

    @property
    def name(self) -> str:
        return "crm"

    @property
    def description(self) -> str:
        return (
            "Manage CRM data across Pipedrive and local intelligence store. "
            "Actions: search_leads, create_lead, add_signal, get_signals, "
            "get_recommendations, approve_recommendation, log_activity, "
            "get_consultants, add_consultant, sync_pipedrive, "
            "gdpr_delete."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "search_leads",
                        "create_lead",
                        "add_signal",
                        "get_signals",
                        "get_recommendations",
                        "approve_recommendation",
                        "log_activity",
                        "get_consultants",
                        "add_consultant",
                        "sync_pipedrive",
                        "gdpr_delete",
                    ],
                    "description": "Action to perform",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search_leads)",
                },
                "name": {
                    "type": "string",
                    "description": "Person/org/consultant name",
                },
                "email": {
                    "type": "string",
                    "description": "Email address",
                },
                "phone": {
                    "type": "string",
                    "description": "Phone number",
                },
                "company_name": {
                    "type": "string",
                    "description": "Company/organization name",
                },
                "title": {
                    "type": "string",
                    "description": "Lead title or signal title",
                },
                "signal_type": {
                    "type": "string",
                    "enum": [
                        "restructuring",
                        "leadership_change",
                        "ma",
                        "digital_transformation",
                        "expansion",
                        "cost_cutting",
                    ],
                    "description": "Type of market signal",
                },
                "source_url": {
                    "type": "string",
                    "description": "Source URL for the signal",
                },
                "source_name": {
                    "type": "string",
                    "description": "Name of the source",
                },
                "description": {
                    "type": "string",
                    "description": "Signal description or activity note",
                },
                "relevance_score": {
                    "type": "number",
                    "description": "Signal relevance (0.0 - 1.0)",
                },
                "recommendation_id": {
                    "type": "integer",
                    "description": "Recommendation ID (for approve)",
                },
                "signal_id": {
                    "type": "integer",
                    "description": "Signal ID (for update/reference)",
                },
                "status": {
                    "type": "string",
                    "description": "Status filter or new status",
                },
                "activity_type": {
                    "type": "string",
                    "description": ("Activity type (call, email, meeting, task)"),
                },
                "subject": {
                    "type": "string",
                    "description": "Activity subject",
                },
                "specializations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Consultant specializations",
                },
                "industries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Consultant industries",
                },
                "regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Consultant regions",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        try:
            if action == "search_leads":
                return await self._search_leads(**kwargs)
            elif action == "create_lead":
                return await self._create_lead(**kwargs)
            elif action == "add_signal":
                return self._add_signal(**kwargs)
            elif action == "get_signals":
                return self._get_signals(**kwargs)
            elif action == "get_recommendations":
                return self._get_recommendations(**kwargs)
            elif action == "approve_recommendation":
                return self._approve_recommendation(**kwargs)
            elif action == "log_activity":
                return await self._log_activity(**kwargs)
            elif action == "get_consultants":
                return self._get_consultants(**kwargs)
            elif action == "add_consultant":
                return self._add_consultant(**kwargs)
            elif action == "sync_pipedrive":
                return await self._sync_pipedrive()
            elif action == "gdpr_delete":
                return self._gdpr_delete(**kwargs)
            else:
                return f"Error: Unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    async def _search_leads(self, query: str = "", **kw: Any) -> str:
        if not query:
            return "Error: query is required for search_leads"
        if not self._pipedrive:
            return "Error: Pipedrive not configured"

        persons = await self._pipedrive.search_persons(query)
        if not persons:
            return f"No leads found matching '{query}'"

        lines = [f"Found {len(persons)} result(s):\n"]
        for p in persons[:10]:
            name = p.get("name", "Unknown")
            pid = p.get("id", "?")
            org = p.get("organization", {})
            org_name = org.get("name", "") if org else ""
            lines.append(f"- [{pid}] {name}" + (f" ({org_name})" if org_name else ""))
        return "\n".join(lines)

    async def _create_lead(
        self,
        name: str = "",
        email: str = "",
        phone: str = "",
        company_name: str = "",
        title: str = "",
        **kw: Any,
    ) -> str:
        if not name:
            return "Error: name is required for create_lead"
        if not self._pipedrive:
            return "Error: Pipedrive not configured"

        # Create org if company name provided
        org_id = None
        if company_name:
            orgs = await self._pipedrive.search_organizations(company_name)
            if orgs:
                org_id = orgs[0].get("id")
            else:
                org = await self._pipedrive.create_organization(company_name)
                org_id = org.get("id")

        # Create person
        person = await self._pipedrive.create_person(
            name=name,
            email=email or None,
            phone=phone or None,
            org_id=org_id,
        )
        person_id = person.get("id")

        # Create lead
        lead_title = title or f"Lead: {name}"
        lead = await self._pipedrive.create_lead(
            title=lead_title,
            person_id=person_id,
            org_id=org_id,
        )

        return (
            f"Created lead '{lead_title}' in Pipedrive "
            f"(person ID: {person_id}, lead ID: {lead.get('id', '?')})"
        )

    def _add_signal(
        self,
        company_name: str = "",
        signal_type: str = "",
        title: str = "",
        source_url: str = "",
        source_name: str = "",
        description: str = "",
        relevance_score: float = 0.5,
        **kw: Any,
    ) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"
        if not company_name:
            return "Error: company_name is required"
        if not signal_type:
            return "Error: signal_type is required"
        if not title:
            return "Error: title is required"
        if not source_url:
            return "Error: source_url is required (GDPR traceability)"

        signal = self._store.add_signal(
            company_name=company_name,
            signal_type=signal_type,
            title=title,
            source_url=source_url,
            source_name=source_name or "manual",
            description=description,
            relevance_score=relevance_score,
        )
        return (
            f"Signal added: [{signal.get('id')}] {company_name} - "
            f"{signal_type}: {title} (relevance: {relevance_score:.0%})"
        )

    def _get_signals(
        self,
        status: str = "",
        signal_type: str = "",
        limit: int = 20,
        **kw: Any,
    ) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"

        signals = self._store.get_signals(
            status=status or None,
            signal_type=signal_type or None,
            limit=limit,
        )
        if not signals:
            return "No signals found."

        lines = [f"Signals ({len(signals)}):\n"]
        for s in signals:
            lines.append(
                f"- [{s['id']}] {s['company_name']}: "
                f"{s['signal_type']} - {s['title']} "
                f"({s['relevance_score']:.0%}, {s['status']})"
            )
        return "\n".join(lines)

    def _get_recommendations(self, status: str = "", limit: int = 20, **kw: Any) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"

        recs = self._store.get_recommendations(status=status or None, limit=limit)
        if not recs:
            return "No recommendations found."

        lines = [f"Recommendations ({len(recs)}):\n"]
        for r in recs:
            lines.append(
                f"- [{r['id']}] {r['company_name']}: "
                f"{r.get('service_area', 'N/A')} "
                f"({r.get('consultant_name', 'unassigned')}) "
                f"[{r['status']}]"
            )
        return "\n".join(lines)

    def _approve_recommendation(self, recommendation_id: int = 0, **kw: Any) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"
        if not recommendation_id:
            return "Error: recommendation_id is required"

        ok = self._store.update_recommendation_status(recommendation_id, "approved", "agent")
        if ok:
            return f"Recommendation {recommendation_id} approved"
        return f"Error: Recommendation {recommendation_id} not found"

    async def _log_activity(
        self,
        activity_type: str = "email",
        subject: str = "",
        name: str = "",
        company_name: str = "",
        description: str = "",
        **kw: Any,
    ) -> str:
        if not self._pipedrive:
            return "Error: Pipedrive not configured"
        if not subject:
            return "Error: subject is required for log_activity"

        person_id = None
        org_id = None

        if name:
            persons = await self._pipedrive.search_persons(name)
            if persons:
                person_id = persons[0].get("id")

        if company_name:
            orgs = await self._pipedrive.search_organizations(company_name)
            if orgs:
                org_id = orgs[0].get("id")

        activity = await self._pipedrive.create_activity(
            activity_type=activity_type,
            subject=subject,
            person_id=person_id,
            org_id=org_id,
            note=description,
        )
        return (
            f"Activity logged in Pipedrive: {activity_type} - {subject} "
            f"(ID: {activity.get('id', '?')})"
        )

    def _get_consultants(
        self,
        specializations: list[str] | None = None,
        industries: list[str] | None = None,
        **kw: Any,
    ) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"

        spec = specializations[0] if specializations else None
        ind = industries[0] if industries else None
        consultants = self._store.get_consultants(specialization=spec, industry=ind)
        if not consultants:
            return "No consultants found."

        lines = [f"Consultants ({len(consultants)}):\n"]
        for c in consultants:
            specs = ", ".join(c.get("specializations", []))
            lines.append(f"- [{c['id']}] {c['name']}: {specs}")
        return "\n".join(lines)

    def _add_consultant(
        self,
        name: str = "",
        email: str = "",
        specializations: list[str] | None = None,
        industries: list[str] | None = None,
        regions: list[str] | None = None,
        **kw: Any,
    ) -> str:
        if not self._store:
            return "Error: Intelligence store not configured"
        if not name:
            return "Error: name is required"

        consultant = self._store.add_consultant(
            name=name,
            email=email,
            specializations=specializations,
            industries=industries,
            regions=regions,
        )
        return f"Consultant added: [{consultant.get('id')}] {name}"

    async def _sync_pipedrive(self) -> str:
        if not self._pipedrive:
            return "Error: Pipedrive not configured"

        leads = await self._pipedrive.get_leads(limit=50)
        deals = await self._pipedrive.get_deals(limit=50)

        return f"Pipedrive sync complete: {len(leads)} leads, {len(deals)} deals"

    def _gdpr_delete(self, email: str = "", **kw: Any) -> str:
        if not email:
            return "Error: email is required for gdpr_delete"
        if not self._consent:
            return "Error: Consent store not configured"

        stats = self._consent.gdpr_delete(email)
        return f"GDPR deletion complete for {email}: {json.dumps(stats)}"
