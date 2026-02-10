"""Lead pipeline route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/leads", tags=["leads"])


@router.get("", response_class=HTMLResponse)
async def leads_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    signals = store.get_signals(limit=500) if store else []

    # Group signals by company and compute basic scores
    by_company: dict[str, list] = {}
    for s in signals:
        name = s.get("company_name", "Unknown")
        by_company.setdefault(name, []).append(s)

    leads = []
    for name, sigs in by_company.items():
        avg_relevance = sum(s.get("relevance_score", 0) for s in sigs) / len(sigs)
        tier = "hot" if avg_relevance > 0.7 else ("warm" if avg_relevance > 0.4 else "cold")
        leads.append(
            {
                "company_name": name,
                "signal_count": len(sigs),
                "avg_relevance": avg_relevance,
                "tier": tier,
                "top_signal": (sigs[0].get("signal_type", "") if sigs else ""),
            }
        )

    leads.sort(key=lambda x: x["avg_relevance"], reverse=True)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "leads.html",
        {
            "request": request,
            "leads": leads,
        },
    )
