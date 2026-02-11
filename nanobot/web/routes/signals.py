"""Signal feed route."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_class=HTMLResponse)
async def signals_page(request: Request):
    auth = request.app.state.auth
    if not auth.require_auth(request):
        return RedirectResponse("/login")

    store = request.app.state.intel_store
    status = request.query_params.get("status")
    signal_type = request.query_params.get("type")

    signals = []
    if store:
        signals = store.get_signals(
            status=status or None,
            signal_type=signal_type or None,
            limit=50,
        )

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "signals.html",
        {
            "request": request,
            "signals": signals,
            "current_status": status or "all",
            "current_type": signal_type or "all",
        },
    )


@router.post("/{signal_id}/review")
async def review_signal(request: Request, signal_id: int):
    auth = request.app.state.auth
    user = auth.require_auth(request)
    if not user:
        return RedirectResponse("/login")

    form = await request.form()
    new_status = form.get("status", "reviewed")

    store = request.app.state.intel_store
    if store:
        store.update_signal_status(signal_id, new_status, user)

    return RedirectResponse("/signals", status_code=303)
