"""Intelligence reports route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_class=HTMLResponse)
async def reports_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    report_type = request.query_params.get("type")
    reports = store.get_reports(report_type=report_type or None, limit=20) if store else []

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "reports": reports,
            "current_type": report_type or "all",
        },
    )


@router.get("/{report_id}", response_class=HTMLResponse)
async def report_detail(request: Request, report_id: int):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    reports = store.get_reports(limit=100) if store else []
    report = next((r for r in reports if r["id"] == report_id), None)

    if not report:
        return RedirectResponse("/reports")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "report_detail.html",
        {
            "request": request,
            "report": report,
        },
    )
