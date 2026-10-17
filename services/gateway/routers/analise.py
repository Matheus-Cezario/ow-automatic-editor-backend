"""Analysis: uploading a match, following it, its recording, proxy, frames,
the cuts package and deleting it."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from sqlalchemy import func, select

from owcore.bus import get_bus
from owcore.db import session
from owcore.ffmpeg import probe
from owcore.models import (
    STREAM_JOBS,
    STREAM_THUMBS,
    Event,
    Job,
    JobCreated,
    JobParams,
    JobStage,
    JobStatus,
    RenderStatus,
    ThumbsRequested,
    frame_key,
    new_id,
)
from owcore.models import Montage as MontageModel
from owcore.storage import get_storage

from common import (
    CHUNK,
    VIDEO_EXTS,
    VIDEO_MIME,
    _adopt_old_draft,
    _as_aware,
    _clip_dict,
    _iso,
    _media_dict,
    _montage_dict,
    _queue_position,
    _render_dict,
    _safe_suffix,
    _serve_blob,
    _store_upload,
)

LOG = logging.getLogger("gateway")
router = APIRouter()


def _job_dict(job: Job, *, full: bool = False) -> dict[str, Any]:
    has_cuts = any(
        (c.meta or {}).get("segments_zip_key") for c in job.clips
    )
    without_video = sum(1 for c in job.clips if not c.key)
    data = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "n_moments": job.n_moments,
        "progress": round(job.progress, 3),
        "error": job.error,
        "video_name": job.video_name,
        "duration_s": round(job.duration_s, 2),
        # `or 0` because a match analysed before this column existed reads it
        # as NULL until the backfill passes over it
        "fps": round(job.fps or 0.0, 3),
        "width": job.width or 0,
        "height": job.height or 0,
        "params": job.params,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
        "n_renders": len(job.renders),
        # the listing does not carry the whole requests, but the app needs to
        # know whether it is worth going on polling
        "has_active_render": any(
            r.status in (RenderStatus.PENDING, RenderStatus.RENDERING)
            for r in job.renders
        ),
        "n_clips": len(job.clips),
        # the recording itself: the montage preview seeks inside it
        "video_url": f"/api/jobs/{job.id}/video",
        # the reduced copy, when it exists. Jobs analysed before it came into
        # the world answer `null`, and the app falls back to the recording
        "proxy_url": f"/api/jobs/{job.id}/proxy" if job.proxy_key else None,
        # the whole match's package: it requires opening no video at all
        "zip_url": f"/api/jobs/{job.id}/cuts.zip" if job.clips else None,
        "has_cuts": has_cuts,
        #: clips whose assembly failed but whose cuts survived
        "clips_only_cuts": without_video,
    }
    if full:
        from sqlalchemy.orm import object_session

        db = object_session(job)
        data["renders"] = [
            _render_dict(
                r,
                job.clips,
                queue_position=_queue_position(db, r) if db is not None else None,
            )
            for r in sorted(job.renders, key=lambda r: r.created_at, reverse=True)
        ]
        data["events"] = [
            {
                "kind": e.kind,
                "t": round(e.t, 3),
                "confidence": round(e.confidence, 3),
                "meta": e.meta,
            }
            for e in sorted(job.events, key=lambda e: e.t)
        ]
        data["detectors"] = [
            {
                "detector": r.detector,
                "ok": bool(r.ok),
                "error": r.error,
                "n_events": r.n_events,
            }
            for r in job.reports
        ]
        data["clips"] = [_clip_dict(c) for c in sorted(job.clips, key=lambda c: -c.score)]
        library = sorted(job.media, key=lambda m: m.created_at)
        data["media"] = [_media_dict(m) for m in library]
        # `tracks` is still only the music: it is what the track picker uses,
        # and a two-second whoosh is not a song to build a montage on
        data["tracks"] = [
            _media_dict(m)
            for m in library
            if m.is_audio and not m.sfx_id and not m.voice
        ]
        # the montages come back with the job: that is how the screen rebuilds
        # itself after an F5, and it is the list the picker shows
        montages = sorted(job.montages, key=lambda m: _as_aware(m.updated_at), reverse=True)
        data["montages"] = [_montage_dict(m, full=True) for m in montages]
        # `draft` is still the most recent one, for an app older than Phase 8
        data["draft"] = (montages[0].data if montages else job.draft) or {}
        # the match audio's waveform, for the editor's ruler. Detail view only:
        # it is a few thousand numbers, and the listing has no use for them
        data["waveform"] = job.waveform or []
    return data


@router.post("/api/jobs", status_code=201)
def create_job(
    video: UploadFile = File(..., description="match recording"),
    params: str = Form("{}", description="JobParams as JSON"),
    size: int = Form(0, description="file size, to verify the upload"),
) -> dict[str, Any]:
    """First phase: the recording alone.

    No music comes in here. The analysis finds the moments; what becomes a
    video -- and with which music -- is decided in the editor afterwards, and
    sent to `POST /api/jobs/{id}/renders` as many times as the user wants.
    """
    try:
        parsed = JobParams(**json.loads(params or "{}"))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(422, f"invalid parameters: {exc}") from exc

    job_id = new_id()

    video_ext = _safe_suffix(video.filename, VIDEO_EXTS, ".mp4")
    video_key = _store_upload(f"{job_id}/source{video_ext}", video, size)

    with session() as s:
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            stage=JobStage.QUEUED,
            video_key=video_key,
            video_name=video.filename or "recording",
            params=parsed.model_dump(),
        )
        s.add(job)

    get_bus().publish(STREAM_JOBS, JobCreated(job_id=job_id).model_dump())
    return {"id": job_id, "status": JobStatus.PENDING}


@router.get("/api/jobs")
def list_jobs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    with session() as s:
        jobs = s.scalars(
            select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
        ).all()
        for j in jobs:
            _backfill_moments(s, j)
        return {"jobs": [_job_dict(j) for j in jobs]}


def _backfill_moments(s, job: Job) -> None:
    """Counts the moments of a match analysed before `n_moments` existed.

    Back then the count only lived inside a sentence in `stage` -- which is
    also rewritten here as the plain code, so the app never has to parse old
    prose. Once per job: the count is stored.
    """
    if job.status != JobStatus.READY or job.n_moments is not None:
        return
    job.n_moments = s.scalar(
        select(func.count()).select_from(Event).where(Event.job_id == job.id)
    ) or 0
    job.stage = JobStage.READY


#: while the analysis is not finished, the preprocessor is still going to
#: write the recording's fields -- there is nothing to patch up
_ANALYSING = frozenset(
    {JobStatus.PENDING, JobStatus.PREPROCESSING, JobStatus.DETECTING}
)


def _backfill_size(job: Job) -> None:
    """Finds the size of a recording analysed before this column existed.

    The schema reconciler adds the column, but has no way of knowing what it
    should hold -- only the upload knows. An old match would open the editor
    unable to say whether a 9:16 crops its frame. It costs one `ffprobe`, once
    in each job's life. ffmpeg reads the upload where it is -- via `Range`, if
    it is on S3 -- so measuring a two-gigabyte recording costs its header, not
    the two gigabytes.

    It is not worth it while the analysis is running, for two reasons. One: a
    match *being analysed now* is not an old match -- the preprocessor will
    write the real size within seconds. Two: the header is cheap, but reading
    it over the network while the preprocessor downloads the same upload
    competes for the same bandwidth, and the screen polls every two seconds.
    Measured: a query that answers in 0.5s went past 30s in that window, which
    the screen shows as "could not reach the server".
    """
    if job.width or not job.video_key or job.status in _ANALYSING:
        return
    try:
        info = probe(get_storage().url(job.video_key))
    except Exception:  # noqa: BLE001 - a backfill must not bring the editor down
        LOG.warning("could not measure the recording of %s", job.id, exc_info=True)
        return
    job.width, job.height = info.width, info.height
    if not job.fps:
        job.fps = info.fps


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _backfill_size(job)
        _backfill_moments(s, job)
        _adopt_old_draft(s, job)
        return _job_dict(job, full=True)


@router.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, request: Request) -> Response:
    """The original recording, with `Range`.

    It is what the montage screen's preview plays: instead of rendering the
    video on every adjustment -- which would cost a full trip through ffmpeg
    per drag -- the app opens the recording itself and seeks to the instant of
    the block under the playhead. The real cut still happens on the server;
    this is only for seeing before asking.
    """
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        key = job.video_key
    mime = VIDEO_MIME.get(Path(key).suffix.lower(), "video/mp4")
    return _serve_blob(key, request, mime)


@router.get("/api/jobs/{job_id}/proxy")
def job_proxy(job_id: str, request: Request) -> Response:
    """The reduced copy of the recording, with `Range`.

    It is what the editor's monitor opens. The original recording is hundreds
    of megabytes, and seeking inside it dozens of times a second while dragging
    used to bring the browser's video element down. This copy comes out of the
    same decode as the crops -- it costs almost nothing -- and the final cut
    still comes from the original upload.
    """
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        key = job.proxy_key
    if not key:
        raise HTTPException(
            404, "this match was analysed before proxies existed"
        )
    return _serve_blob(key, request, "video/mp4")


@router.get("/api/jobs/{job_id}/frame")
def job_frame(job_id: str, t: float, request: Request) -> Response:
    """A frame of the match at instant `t`, for the editor's sidebar.

    It only delivers what already exists: the `thumbs` service extracts. A 404
    here means "not extracted yet", and the app shows its placeholder instead
    of going without the item.
    """
    key = frame_key(job_id, t)
    if not get_storage().exists(key):
        raise HTTPException(404, "thumbnail not extracted yet")
    response = _serve_blob(key, request, "image/jpeg")
    # an instant's frame never changes: worth letting the browser cache it
    response.headers["cache-control"] = "public, max-age=86400"
    return response


@router.post("/api/jobs/{job_id}/frames", status_code=202)
def request_frames(job_id: str) -> dict[str, Any]:
    """Requests extraction of the missing thumbnails.

    New jobs already come with them -- the planner asks as soon as the analysis
    ends. This is for old ones, and for when one has failed: the app calls it
    when opening the editor, and the service skips what is already there.
    """
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job not found")
    get_bus().publish(STREAM_THUMBS, ThumbsRequested(job_id=job_id).model_dump())
    return {"job_id": job_id, "status": "requested"}


def _cuts_from(data: dict | None, job_id: str) -> bool:
    """Whether a saved montage brings a moment from `job_id`'s recording."""
    for layer in (data or {}).get("layers") or []:
        for clip in layer.get("clips") or []:
            if clip.get("job_id") == job_id:
                return True
    return False


@router.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    """Deletes a match -- unless another match's montage cuts moments from
    it: its recording would vanish from under that montage, which could then
    neither play nor render. The answer names who uses it."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        users: set[str] = set()
        others = MontageModel.job_id != job_id
        for m in s.scalars(select(MontageModel).where(others)):
            if _cuts_from(m.data, job_id):
                users.add(m.job_id)
        for other in s.scalars(select(Job).where(Job.id != job_id)):
            if _cuts_from(other.draft, job_id):
                users.add(other.id)
        if users:
            names = sorted(s.get(Job, u).video_name or u for u in users)
            raise HTTPException(
                409,
                "moments of this match are used in the montages of "
                + ", ".join(names)
                + " -- take them out there first",
            )
        s.delete(job)
    return Response(status_code=204)


@router.get("/api/jobs/{job_id}/cuts.zip")
def job_zip(job_id: str, request: Request) -> Response:
    """Everything the match generated, in a single file.

    Built on the spot from what is already in storage -- the final videos and
    each montage's loose cuts -- instead of keeping a third package with the
    same bytes. Since the zip only packs (the mp4s are already compressed), the
    cost is basically that of copying.
    """
    storage = get_storage()
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if not job.clips:
            raise HTTPException(404, "this match has no videos yet")
        base_name = Path(job.video_name or "match").stem
        # a job yields several requests over time; the package brings them all,
        # each in its own folder, so the same kind of video generated twice
        # with different music does not overwrite itself
        order = {r.id: n for n, r in enumerate(
            sorted(job.renders, key=lambda r: r.created_at), start=1
        )}
        items = [
            (i, order.get(c.render_id, 0), c.kind, c.key,
             (c.meta or {}).get("segments_zip_key"))
            for i, c in enumerate(
                sorted(job.clips, key=lambda c: (order.get(c.render_id, 0), -c.score)),
                start=1,
            )
        ]

    tmp = Path(tempfile.mkdtemp(prefix="owzip-"))
    package = tmp / "package.zip"
    try:
        with zipfile.ZipFile(package, "w", zipfile.ZIP_STORED) as zf:
            for i, n_request, kind, video_key, cuts_key in items:
                folder = f"request_{n_request:02d}" if n_request else "videos"
                if video_key and storage.exists(video_key):
                    local = storage.get_file(video_key, tmp / f"v{i}.mp4")
                    zf.write(local, f"{folder}/videos/{i:02d}_{kind}.mp4")
                    local.unlink(missing_ok=True)
                if not cuts_key or not storage.exists(cuts_key):
                    continue
                local = storage.get_file(cuts_key, tmp / f"c{i}.zip")
                with zipfile.ZipFile(local) as source:
                    for info in source.infolist():
                        dest = f"{folder}/cuts/{i:02d}_{kind}/{info.filename}"
                        # copy chunk by chunk: `source.read(name)` put a whole
                        # cut in memory at a time
                        with source.open(info) as src_fh, zf.open(dest, "w") as dst_fh:
                            shutil.copyfileobj(src_fh, dst_fh, CHUNK)
                local.unlink(missing_ok=True)
        total = package.stat().st_size
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    def send_chunks() -> Iterator[bytes]:
        """Delivers the package in chunks and only then deletes the temp dir.

        The `read_bytes()` that used to be here put the **whole** package in the
        gateway's memory -- a match with a few requests is hundreds of MB, per
        concurrent request, and the process serving the API is the same one
        serving the app. Streaming keeps the cost at one chunk at a time, and
        the temporary file was on disk anyway.
        """
        try:
            with open(package, "rb") as fh:
                while True:
                    chunk = fh.read(CHUNK)
                    if not chunk:
                        break
                    yield chunk
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return StreamingResponse(
        send_chunks(),
        media_type="application/zip",
        headers={
            "content-disposition": f'attachment; filename="{base_name}_cuts.zip"',
            "content-length": str(total),
        },
    )
