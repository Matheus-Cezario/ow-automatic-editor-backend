"""Render request state, written by the render domain (editor, previewer).

It used to live in `jobs.py` beside the job's own state; it moved here so that
each module writes only the tables of its own domain (see `TABLE_OWNER`).
"""

from __future__ import annotations

from .db import session
from .models import Render, RenderStage, RenderStatus, utcnow


def set_render_status(
    render_id: str,
    status: "RenderStatus | None" = None,
    *,
    stage: RenderStage | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    with session() as s:
        render = s.get(Render, render_id)
        if render is None:
            return
        # a cancelled request stays cancelled: the worker may still report
        # on its way out
        if render.status == RenderStatus.CANCELLED:
            return
        if status is RenderStatus.RENDERING and render.started_at is None:
            render.started_at = utcnow()
        if status is not None:
            render.status = status
        if stage is not None:
            render.stage = stage
        if progress is not None:
            render.progress = max(0.0, min(1.0, progress))
        if error is not None:
            render.error = error


def render_cancelled(render_id: str) -> bool:
    """Has the user stopped this request (or deleted it)?"""
    with session() as s:
        render = s.get(Render, render_id)
        return render is None or render.status == RenderStatus.CANCELLED


def fail_render(render_id: str, message: str) -> None:
    set_render_status(
        render_id, RenderStatus.FAILED, stage=RenderStage.ERROR, error=message[:4000]
    )
