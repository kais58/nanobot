"""Recommendations approval route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_class=HTMLResponse)
async def recommendations_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    status = request.query_params.get("status")
    recs = store.get_recommendations(status=status or None, limit=50) if store else []

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "recommendations.html",
        {
            "request": request,
            "recommendations": recs,
            "current_status": status or "all",
        },
    )


@router.post("/{rec_id}/approve")
async def approve_rec(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        store.update_recommendation_status(rec_id, "approved", user)

    return RedirectResponse("/recommendations", status_code=303)


@router.post("/{rec_id}/reject")
async def reject_rec(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        store.update_recommendation_status(rec_id, "rejected", user)

    return RedirectResponse("/recommendations", status_code=303)
