"""Recommendations approval route."""

from datetime import datetime
from urllib.parse import quote

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

    # Enrich with tracking data
    for rec in recs:
        tracking = store.get_outreach_tracking(rec["id"])
        rec["tracking"] = tracking

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


@router.get("/{rec_id}/compose")
async def compose_email(request: Request, rec_id: int):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    rec = store.get_recommendation(rec_id)
    if not rec:
        return RedirectResponse("/recommendations", status_code=303)

    subject = f"K&P: {rec.get('service_area', '')} - {rec['company_name']}"
    body = rec.get("pitch_summary", "")
    mailto = f"mailto:?subject={quote(subject)}&body={quote(body)}"
    return RedirectResponse(mailto, status_code=303)


@router.post("/{rec_id}/back-to-pending")
async def back_to_pending(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        store.update_recommendation_status(rec_id, "pending", user)

    return RedirectResponse("/recommendations", status_code=303)


@router.post("/{rec_id}/mark-sent")
async def mark_sent(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        store.update_recommendation_status(rec_id, "sent", user)
        rec = store.get_recommendation(rec_id)
        if rec:
            store.create_outreach_tracking(rec_id, rec["company_name"])

    return RedirectResponse("/recommendations", status_code=303)


@router.post("/{rec_id}/response-received")
async def response_received(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        store.update_outreach_response(rec_id, responded=True)

    return RedirectResponse("/recommendations", status_code=303)


@router.post("/{rec_id}/no-response")
async def no_response(request: Request, rec_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    if store:
        tracking = store.get_outreach_tracking(rec_id)
        if tracking:
            now = datetime.utcnow().isoformat()
            store._conn.execute(
                """UPDATE outreach_tracking
                SET response_status = 'no_response', heat_status = 'cold',
                    updated_at = ?
                WHERE recommendation_id = ?""",
                (now, rec_id),
            )
            store._conn.commit()

    return RedirectResponse("/recommendations", status_code=303)
