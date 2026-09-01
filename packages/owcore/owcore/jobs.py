"""Job state operations shared across the services."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import select, update

from .db import session
from .models import (
    DETECTORS,
    DetectionEvent,
    DetectorReport,
    Event,
    Job,
    JobParams,
    JobStatus,
    Render,
    RenderStatus,
)


def set_status(
    job_id: str,
    status: JobStatus | None = None,
    *,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        if error is not None:
            job.error = error


def fail(job_id: str, message: str) -> None:
    set_status(job_id, JobStatus.FAILED, stage="erro", error=message[:4000])


def get_params(job_id: str) -> JobParams:
    with session() as s:
        job = s.get(Job, job_id)
        return JobParams(**(job.params or {})) if job else JobParams()


def save_events(job_id: str, detector: str, events: Sequence[DetectionEvent]) -> None:
    with session() as s:
        for e in events:
            s.add(
                Event(
                    job_id=job_id,
                    kind=str(e.kind),
                    t=e.t,
                    confidence=e.confidence,
                    meta={**e.meta, "detector": detector},
                )
            )


def record_report(
    job_id: str, detector: str, n_events: int, error: str | None = None
) -> None:
    """Idempotent: reprocessing the same message does not duplicate the report."""
    with session() as s:
        existing = s.scalar(
            select(DetectorReport).where(
                DetectorReport.job_id == job_id, DetectorReport.detector == detector
            )
        )
        if existing is not None:
            existing.n_events = n_events
            existing.ok = 0 if error else 1
            existing.error = error
            return
        s.add(
            DetectorReport(
                job_id=job_id,
                detector=detector,
                n_events=n_events,
                ok=0 if error else 1,
                error=error,
            )
        )


def reported_detectors(job_id: str) -> set[str]:
    with session() as s:
        rows = s.scalars(
            select(DetectorReport.detector).where(DetectorReport.job_id == job_id)
        ).all()
        return set(rows)


def all_detectors_done(job_id: str, expected: Sequence[str] = DETECTORS) -> bool:
    return set(expected).issubset(reported_detectors(job_id))


def expected_detectors(job_id: str) -> tuple[str, ...]:
    """Who has to report before the analysis is considered finished.

    The music's rhythm is not part of this: it is not part of analysing the
    match. Music comes in through the library, when the user brings it to the
    editor.
    """
    return DETECTORS


def claim_for_planning(job_id: str) -> bool:
    """Atomic detecting -> ready transition.

    The detectors finish almost together and all of them notify; without this
    claim two processes would close the analysis twice over -- and write the
    crossed events twice. The conditional UPDATE settles it in the database,
    the same way in SQLite and in Postgres.
    """
    with session() as s:
        result = s.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JobStatus.DETECTING)
            .values(status=JobStatus.READY, stage="analise concluida", progress=1.0)
        )
        return bool(result.rowcount)


def set_render_status(
    render_id: str,
    status: "RenderStatus | None" = None,
    *,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    with session() as s:
        render = s.get(Render, render_id)
        if render is None:
            return
        if status is not None:
            render.status = status
        if stage is not None:
            render.stage = stage
        if progress is not None:
            render.progress = max(0.0, min(1.0, progress))
        if error is not None:
            render.error = error


def fail_render(render_id: str, message: str) -> None:
    set_render_status(
        render_id, RenderStatus.FAILED, stage="erro", error=message[:4000]
    )


def stale_detecting_jobs(older_than_s: float) -> list[str]:
    """Jobs stuck in detection -- some detector died halfway through."""
    from datetime import timedelta

    from .models import utcnow

    cutoff = utcnow().replace(tzinfo=None) - timedelta(seconds=older_than_s)
    with session() as s:
        return list(
            s.scalars(
                select(Job.id).where(
                    Job.status == JobStatus.DETECTING, Job.updated_at < cutoff
                )
            ).all()
        )


def load_events(job_id: str) -> list[DetectionEvent]:
    with session() as s:
        rows = s.scalars(
            select(Event).where(Event.job_id == job_id).order_by(Event.t)
        ).all()
        return [
            DetectionEvent(kind=r.kind, t=r.t, confidence=r.confidence, meta=r.meta or {})
            for r in rows
        ]
