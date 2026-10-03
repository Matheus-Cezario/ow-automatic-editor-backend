"""The system's API: upload, progress tracking and video delivery.

It is the only service exposed to the world. It processes nothing -- it stores
the file, creates the job and publishes on the bus; the rest happens in the
workers.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from sqlalchemy import func, select

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import init_db, session
from owcore.models import (
    STREAM_JOBS,
    STREAM_MEDIA,
    STREAM_PREVIEW,
    STREAM_RENDER_READY,
    STREAM_THUMBS,
    Clip,
    Event,
    Job,
    JobCreated,
    JobParams,
    JobStage,
    JobStatus,
    Media,
    MediaKind,
    MediaUploaded,
    Montage as MontageModel,
    MontageDraft,
    MontageVersion,
    Preset,
    Preview,
    PreviewRequested,
    Render,
    RenderRequested,
    Recipe,
    RenderStage,
    RenderStatus,
    ThumbsRequested,
    Timeline,
    TrackStatus,
    frame_key,
    new_id,
    utcnow,
)
from owcore.ffmpeg import probe
from owcore.preview import preview_window
from owcore.storage import get_storage

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
CHUNK = 1024 * 256

#: The original recording is served to the app too: it is what the editing
#: screen's preview shows, seeking to each block's instant.
VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".ts": "video/mp2t",
}

#: The app's player asks for the music over HTTP; without the right type some
#: browsers refuse to play it (and without playing there is no way to place a
#: cut).
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".opus": "audio/ogg",
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


LOG = logging.getLogger("gateway")

app = FastAPI(
    title="OW Editor",
    description="Overwatch 2 match editor with automatic event detection.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the Flutter web app runs on another port during dev
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────── helpers ────────────────────────────────────


def _safe_suffix(filename: str | None, allowed: set[str], default: str) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in allowed else default


def _store_upload(key: str, upload: UploadFile, expected_bytes: int) -> str:
    """Stores the upload, checking that it arrived whole.

    A truncated upload looks like no error at all: the multipart closes
    properly, the `Content-Length` matches what actually arrived, and what is
    left is half a recording stored as if it were whole. The damage only
    showed up stages later, in the preprocessor, as an `ffprobe exited with 1`
    -- far from the upload screen and without saying what to do.

    Checking here costs one `stat` and returns the problem where it was born,
    with the only action that fixes it: upload again.

    `expected_bytes` of zero turns the check off -- it is an old client that
    does not send the size.
    """
    storage = get_storage()
    stored = storage.put_stream(key, upload.file)
    got = storage.size(stored)
    if expected_bytes and got != expected_bytes:
        storage.delete(stored)
        raise HTTPException(
            400,
            f"the file arrived incomplete: {got} of {expected_bytes} bytes. "
            "Upload it again.",
        )
    return stored


def _json_list(raw: Any, field: str) -> list:
    """Reads a multipart field carrying a JSON list as text."""
    if raw is None or raw == "":
        return []
    if not isinstance(raw, str):
        raise HTTPException(422, f"'{field}' must be JSON as text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"'{field}' is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise HTTPException(422, f"'{field}' must be a list")
    return value


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
        data["renders"] = [
            _render_dict(r, job.clips)
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
        # `tracks` is still only the music: it is what the track picker uses
        data["tracks"] = [_media_dict(m) for m in library if m.is_audio]
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


def _render_dict(r: Render, all_clips: list[Clip]) -> dict[str, Any]:
    clips_of_render = [c for c in all_clips if c.render_id == r.id]
    return {
        "id": r.id,
        "job_id": r.job_id,
        "status": r.status,
        "stage": r.stage,
        "progress": round(r.progress, 3),
        "error": r.error,
        "created_at": _iso(r.created_at),
        "updated_at": _iso(r.updated_at),
        "timelines": [
            {
                "title": tl.get("title") or "",
                "track_id": tl.get("track_id"),
                # an old request stores `cuts`; a new one, layers
                "n_cuts": len(tl.get("cuts") or []) or sum(
                    len(c.get("clips") or []) for c in (tl.get("layers") or [])
                ),
                "n_layers": len(tl.get("layers") or []) or 1,
            }
            for tl in (r.timelines or [])
        ],
        "clips": [_clip_dict(c) for c in sorted(clips_of_render, key=lambda c: -c.score)],
    }


def _clip_dict(c: Clip) -> dict[str, Any]:
    return {
        "id": c.id,
        "job_id": c.job_id,
        "render_id": c.render_id,
        "kind": c.kind,
        "title": c.title,
        "start_s": round(c.start_s, 2),
        "end_s": round(c.end_s, 2),
        "score": round(c.score, 2),
        "meta": c.meta,
        # no key: the assembly failed and only the cuts exist
        "video_url": f"/api/clips/{c.id}/video" if c.key else None,
        "thumb_url": f"/api/clips/{c.id}/thumb" if c.meta.get("thumb_key") else None,
        "segments_zip_url": (
            f"/api/clips/{c.id}/cuts.zip"
            if c.meta.get("segments_zip_key")
            else None
        ),
    }


_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def _serve_blob(key: str, request: Request, media_type: str) -> Response:
    """Serves a blob with Range support, so the player can seek."""
    storage = get_storage()
    if not storage.exists(key):
        raise HTTPException(404, "upload not found")
    total = storage.size(key)

    range_header = request.headers.get("range")
    match = _RANGE.match(range_header or "")
    if not match:
        def whole() -> Iterator[bytes]:
            pos = 0
            while pos < total:
                chunk = storage.open_range(key, pos, CHUNK)
                if not chunk:
                    break
                pos += len(chunk)
                yield chunk

        return StreamingResponse(
            whole(),
            media_type=media_type,
            headers={"content-length": str(total), "accept-ranges": "bytes"},
        )

    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else total - 1
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    length = end - start + 1

    def ranged() -> Iterator[bytes]:
        pos, left = start, length
        while left > 0:
            chunk = storage.open_range(key, pos, min(CHUNK, left))
            if not chunk:
                break
            pos += len(chunk)
            left -= len(chunk)
            yield chunk

    return StreamingResponse(
        ranged(),
        status_code=206,
        media_type=media_type,
        headers={
            "content-range": f"bytes {start}-{end}/{total}",
            "content-length": str(length),
            "accept-ranges": "bytes",
        },
    )


# ──────────────────────────────── routes ────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"ok": True, "mode": s.mode, "profile": s.profile}


@app.post("/api/jobs", status_code=201)
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


@app.post("/api/jobs/{job_id}/renders", status_code=201)
async def create_render(job_id: str, request: Request) -> dict[str, Any]:
    """Second phase: render the videos the user assembled.

    Takes a multipart with the `timelines` field (JSON): each montage with its
    blocks already placed and, if it has music, pointing at a track already
    uploaded to this job's library.

    There used to be a second field, `selections`, with the proposals the
    system offered ready-made. There are no proposals any more: what becomes a
    video comes out of the editor.

    It can be called as many times as wanted on the same job -- using a moment
    in one video does not use it up for the others.
    """
    form = await request.form()
    raw_timelines = _json_list(form.get("timelines"), "timelines")
    if not raw_timelines:
        raise HTTPException(422, "build at least one timeline")

    render_id = new_id()

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != JobStatus.READY:
            raise HTTPException(
                409, f"this job's analysis has not finished yet (status: {job.status})"
            )
        # a montage points at music; the library holds more than that
        music_ids = {m.id: m.status for m in job.media if m.is_audio}
        library = {m.id for m in job.media}

    montages = [
        _validated_timeline(item, music_ids, library) for item in raw_timelines
    ]

    with session() as s:
        s.add(
            Render(
                id=render_id,
                job_id=job_id,
                status=RenderStatus.PENDING,
                stage=RenderStage.QUEUED,
                timelines=[m.model_dump() for m in montages],
            )
        )

    # straight to the editor: there is no rhythm stage in between any more.
    # This montage's music came up through the library, already analysed.
    get_bus().publish(
        STREAM_RENDER_READY, RenderRequested(render_id=render_id).model_dump()
    )
    return {"id": render_id, "job_id": job_id, "status": RenderStatus.PENDING}


def _validated_timeline(
    item: Any, music_ids: dict[str, str], library: set[str]
) -> Timeline:
    """One montage from the app, checked against this job's library."""
    if not isinstance(item, dict):
        raise HTTPException(422, "each timeline must be an object")
    try:
        spec = Timeline(**item)
    except ValidationError as exc:
        raise HTTPException(422, f"invalid timeline: {exc}") from exc
    # a clip pointing at another job's media does not go in: the montage
    # would come out without it, and with no warning
    for clip in spec.clips:
        if clip.media_id and clip.media_id not in library:
            raise HTTPException(
                422,
                f"unknown media in this job: {clip.media_id!r}",
            )
    _check_layers(spec, music_ids)
    # the watermark comes from the same library, and refusing it here is
    # better than letting the whole render fail later because of it
    if spec.export.watermark_id and spec.export.watermark_id not in library:
        raise HTTPException(
            422,
            f"unknown watermark in this job: {spec.export.watermark_id!r}",
        )
    return spec


# ── the exact preview: a stretch rendered small by the server's own graph ───


def _preview_dict(p: Preview) -> dict[str, Any]:
    return {
        "id": p.id,
        "job_id": p.job_id,
        "status": p.status,
        "progress": round(p.progress, 3),
        "error": p.error,
        "from_s": p.from_s,
        "to_s": p.to_s,
        "video_url": f"/api/previews/{p.id}/video" if p.video_key else None,
    }


@app.post("/api/jobs/{job_id}/previews", status_code=201)
async def create_preview(job_id: str, request: Request) -> dict[str, Any]:
    """Renders a stretch of one montage, small and fast, through the same
    graph as the final video.

    JSON body: `{"timeline": {...}, "from_s": 0, "to_s": 10}` -- the window is
    in seconds of the montage, and is capped. Only the latest preview of a
    match is kept: asking for a new one discards the previous, finished or not.
    """
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(422, "the body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "the body must be an object")

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status != JobStatus.READY:
            raise HTTPException(
                409, f"this job's analysis has not finished yet (status: {job.status})"
            )
        music_ids = {m.id: m.status for m in job.media if m.is_audio}
        library = {m.id for m in job.media}

    spec = _validated_timeline(body.get("timeline"), music_ids, library)
    try:
        from_s, to_s = preview_window(
            spec.duration_s, body.get("from_s"), body.get("to_s")
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"invalid preview window: {exc}") from exc

    storage = get_storage()
    with session() as s:
        for old in s.query(Preview).filter(Preview.job_id == job_id).all():
            if old.video_key:
                storage.delete(old.video_key)
            s.delete(old)
        preview = Preview(
            job_id=job_id,
            status=RenderStatus.PENDING,
            timeline=spec.model_dump(mode="json"),
            from_s=from_s,
            to_s=to_s,
        )
        s.add(preview)
        s.flush()
        result = _preview_dict(preview)

    get_bus().publish(
        STREAM_PREVIEW, PreviewRequested(preview_id=result["id"]).model_dump()
    )
    return result


@app.get("/api/previews/{preview_id}")
def get_preview(preview_id: str) -> dict[str, Any]:
    with session() as s:
        p = s.get(Preview, preview_id)
        if p is None:
            raise HTTPException(404, "preview not found")
        return _preview_dict(p)


@app.get("/api/previews/{preview_id}/video")
def preview_video(preview_id: str, request: Request) -> Response:
    with session() as s:
        p = s.get(Preview, preview_id)
        if p is None:
            raise HTTPException(404, "preview not found")
        key = p.video_key
    if not key:
        raise HTTPException(404, "this preview is not ready")
    return _serve_blob(key, request, "video/mp4")


@app.get("/api/renders/{render_id}")
def get_render(render_id: str) -> dict[str, Any]:
    with session() as s:
        request_ = s.get(Render, render_id)
        if request_ is None:
            raise HTTPException(404, "request not found")
        return _render_dict(request_, list(request_.clips))


@app.delete("/api/renders/{render_id}", status_code=204)
def delete_render(render_id: str) -> Response:
    """Deletes a request and its videos. The montage stays saved: it can be
    requested again, with different music."""
    with session() as s:
        request_ = s.get(Render, render_id)
        if request_ is None:
            raise HTTPException(404, "request not found")
        s.delete(request_)
    return Response(status_code=204)


# ── the job's music: uploaded before any video exists ───────────────────────
#
# In the montage the music comes first. You cannot place a cut "on the turn of
# the chorus" without hearing the chorus, and you cannot snap a cut to the beat
# without knowing where the beats are. So it is uploaded, the system listens,
# and the app receives duration, BPM, beats and waveform to draw the ruler.
#
# The music belongs to the **job**, not to a request: the same track serves as
# many montages of that match as the user wants, with no re-upload.


def _media_dict(m: Media) -> dict[str, Any]:
    """One item of the match's media library.

    Audio goes in full -- beats and waveform included -- because that is what
    the montage screen uses to draw the music and snap cuts to the beat. Video
    and image go with their dimensions and the thumbnail and proxy addresses.
    """
    data = {
        "id": m.id,
        "job_id": m.job_id,
        "kind": m.kind,
        "status": m.status,
        "error": m.error,
        "name": m.name,
        "duration_s": round(m.duration_s, 3),
        "file_url": f"/api/media/{m.id}/file",
        "thumb_url": f"/api/media/{m.id}/thumb" if m.thumb_key else None,
        "proxy_url": f"/api/media/{m.id}/proxy" if m.proxy_key else None,
        "created_at": _iso(m.created_at),
    }
    if m.is_audio:
        data |= {
            "bpm": round(m.bpm, 2),
            "beats": m.beats or [],
            "peaks": m.peaks or [],
            # the app still asks for the music through here
            "audio_url": f"/api/media/{m.id}/file",
        }
    else:
        data |= {"width": m.width, "height": m.height, "fps": round(m.fps, 3)}
    return data


def _store_media(
    job_id: str, upload: UploadFile, kind: MediaKind, expected_bytes: int = 0
) -> str:
    """Stores the upload and asks for it to be analysed. Returns the id."""
    media_id = new_id()
    defaults = {
        MediaKind.AUDIO: (AUDIO_EXTS, ".mp3"),
        MediaKind.VIDEO: (VIDEO_EXTS, ".mp4"),
        MediaKind.IMAGE: (IMAGE_EXTS, ".png"),
    }[kind]
    ext = _safe_suffix(upload.filename, *defaults)
    key = _store_upload(f"{job_id}/media/{media_id}{ext}", upload, expected_bytes)

    with session() as s:
        s.add(
            Media(
                id=media_id,
                job_id=job_id,
                kind=kind,
                status=TrackStatus.PENDING,
                name=upload.filename or "upload",
                key=key,
            )
        )
    get_bus().publish(STREAM_MEDIA, MediaUploaded(media_id=media_id).model_dump())
    return media_id


def _kind_of(filename: str | None) -> MediaKind | None:
    """What kind the file is, from its extension.

    By extension and not by `content-type` because the browser often lies --
    it sends `application/octet-stream` for everything when the upload came
    from a place it does not know.
    """
    ext = Path(filename or "").suffix.lower()
    if ext in AUDIO_EXTS:
        return MediaKind.AUDIO
    if ext in VIDEO_EXTS:
        return MediaKind.VIDEO
    if ext in IMAGE_EXTS:
        return MediaKind.IMAGE
    return None


@app.post("/api/jobs/{job_id}/media", status_code=201)
def add_media(
    job_id: str,
    file: UploadFile = File(..., description="video, image or audio"),
    size: int = Form(0, description="file size, to verify the upload"),
) -> dict[str, Any]:
    """Brings a file into the match's library.

    Answers right away, with the item still `pending`: the analysis
    (dimensions, thumbnail, proxy; beats for audio) runs in the worker, and the
    app follows it through `GET /api/media/{id}`.
    """
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job not found")

    kind = _kind_of(file.filename)
    if kind is None:
        raise HTTPException(
            422,
            f"don't know what to do with {file.filename!r}: video, image and "
            "audio are accepted",
        )
    media_id = _store_media(job_id, file, kind, size)
    return {"id": media_id, "job_id": job_id, "kind": kind,
            "status": TrackStatus.PENDING}


@app.get("/api/media/{media_id}")
def get_media(media_id: str) -> dict[str, Any]:
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        return _media_dict(item)


@app.delete("/api/media/{media_id}", status_code=204)
def delete_media(media_id: str) -> Response:
    """Removes the item from the library. Videos already generated with it
    stay: the final mp4 already has what it needed inside."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        s.delete(item)
    return Response(status_code=204)


@app.get("/api/media/{media_id}/file")
def media_file(media_id: str, request: Request) -> Response:
    """The file itself, with `Range`."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        key, kind = item.key, item.kind
    ext = Path(key).suffix.lower()
    mime = (
        AUDIO_MIME.get(ext, "audio/mpeg")
        if kind == MediaKind.AUDIO
        else IMAGE_MIME.get(ext, "image/png")
        if kind == MediaKind.IMAGE
        else VIDEO_MIME.get(ext, "video/mp4")
    )
    return _serve_blob(key, request, mime)


@app.get("/api/media/{media_id}/thumb")
def media_thumb(media_id: str, request: Request) -> Response:
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        key = item.thumb_key
    if not key:
        raise HTTPException(404, "no thumbnail")
    response = _serve_blob(key, request, "image/jpeg")
    response.headers["cache-control"] = "public, max-age=86400"
    return response


@app.get("/api/media/{media_id}/proxy")
def media_proxy(media_id: str, request: Request) -> Response:
    """The reduced copy of the imported video -- what the monitor opens."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        key = item.proxy_key
    if not key:
        raise HTTPException(404, "this item has no proxy")
    return _serve_blob(key, request, "video/mp4")


# ── the music routes, now thin shells over the library ──────────────────────
#
# They still exist because the app uses them and because "the job's music" is
# a useful name. Underneath it is all `Media` of kind audio.


@app.post("/api/jobs/{job_id}/tracks", status_code=201)
def add_track(
    job_id: str,
    audio: UploadFile = File(..., description="music to build the montage on"),
    size: int = Form(0, description="file size, to verify the upload"),
) -> dict[str, Any]:
    """Uploads a track and has the system listen to it."""
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job not found")
    media_id = _store_media(job_id, audio, MediaKind.AUDIO, size)
    return {"id": media_id, "job_id": job_id, "status": TrackStatus.PENDING}


@app.get("/api/tracks/{track_id}")
def get_track(track_id: str) -> dict[str, Any]:
    return get_media(track_id)


@app.get("/api/tracks/{track_id}/audio")
def track_audio(track_id: str, request: Request) -> Response:
    return media_file(track_id, request)


@app.delete("/api/tracks/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    return delete_media(track_id)


@app.get("/api/jobs")
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


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _backfill_size(job)
        _backfill_moments(s, job)
        _adopt_old_draft(s, job)
        return _job_dict(job, full=True)


@app.get("/api/jobs/{job_id}/video")
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


@app.put("/api/jobs/{job_id}/draft")
def save_draft(job_id: str, draft: dict = Body(...)) -> dict[str, Any]:
    """Stores the montage in progress. **V1 legacy.**

    Since Phase 8 a match has several named montages, and the app saves by the
    id of one of them. This route writes to the most recent one -- and creates
    the first one if there is none -- so an older app keeps working instead of
    silently losing work.
    """
    draft_tl = _validate_montage(draft)

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        current = max(job.montages, key=lambda m: _as_aware(m.updated_at), default=None)
        if current is None:
            current = MontageModel(job_id=job_id, name="Montage 1")
            s.add(current)
        current.data = draft_tl.model_dump()
    return {"job_id": job_id, "n_cuts": len(draft_tl.clips)}


@app.delete("/api/jobs/{job_id}/draft", status_code=204)
def delete_draft(job_id: str) -> Response:
    """Throws this match's montages away and starts from scratch. **V1 legacy.**"""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        job.draft = {}
        for m in list(job.montages):
            s.delete(m)
    return Response(status_code=204)


@app.get("/api/jobs/{job_id}/proxy")
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


@app.get("/api/jobs/{job_id}/frame")
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


@app.post("/api/jobs/{job_id}/frames", status_code=202)
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


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        s.delete(job)
    return Response(status_code=204)


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: str, request: Request) -> Response:
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clip not found")
        key = clip.key
    if not key:
        raise HTTPException(
            404, "this clip's montage failed; download the cuts from cuts.zip"
        )
    return _serve_blob(key, request, "video/mp4")


@app.get("/api/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str, request: Request) -> Response:
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clip not found")
        key = (clip.meta or {}).get("thumb_key")
    if not key:
        raise HTTPException(404, "no thumbnail")
    return _serve_blob(key, request, "image/jpeg")


@app.get("/api/jobs/{job_id}/cuts.zip")
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


@app.get("/api/clips/{clip_id}/cuts.zip")
def clip_segments_zip(clip_id: str, request: Request) -> Response:
    """The montage's individual cuts, in a zip.

    For re-editing elsewhere: each file is a stretch, named after the instant
    it came from in the original recording.
    """
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clip not found")
        key = (clip.meta or {}).get("segments_zip_key")
        name = f"cuts_{clip.kind}_{clip.id}.zip"
    if not key:
        raise HTTPException(404, "this clip has no separate cuts")
    response = _serve_blob(key, request, "application/zip")
    response.headers["content-disposition"] = f'attachment; filename="{name}"'
    return response


@app.get("/api/profile")
def profile() -> dict[str, Any]:
    """Exposes the HUD profile for the app's calibration screen."""
    from owcore.profiles import load_profile

    return load_profile(get_settings().profile).data


# ── named montages ─────────────────────────────────────────────────────────


def _as_aware(t: datetime | None) -> datetime:
    """A record's timestamp, always with a timezone.

    SQLite stores `datetime` without a timezone, so a row read from the
    database comes back naive while one created in this same request still has
    the timezone `utcnow()` gave it. Sorting both together blows up -- which is
    exactly what happens when listing the montages right after creating one.
    """
    if t is None:
        return utcnow()
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _iso(t: datetime | None) -> str:
    """The timestamp as text, **with the timezone written out**.

    Without the timezone suffix, the reader on the other side treats the date
    as local time -- and Dart does exactly that. Since what leaves here is UTC,
    the app showed every time shifted by the user's timezone, and any "time
    left" computed from `created_at` came out negative.
    """
    return _as_aware(t).isoformat()


def _montage_dict(m: MontageModel, *, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": m.id,
        "job_id": m.job_id,
        "name": m.name,
        "created_at": _iso(m.created_at),
        "updated_at": _iso(m.updated_at),
        "n_versions": len(m.versions),
        **m.summary,
    }
    if full:
        d["data"] = m.data or {}
    return d


def _adopt_old_draft(s: Any, job: Job) -> None:
    """Brings the job's single montage into the list of named montages.

    Until Phase 8 there was a single one, in a column of the job itself. Doing
    this on read, and not in a database migration, is the same choice as the
    rest of the system: the code that reads is what knows how to convert the
    old format, and so a match untouched for months still opens.
    """
    if not job.draft or job.montages:
        return
    # through the relationship, not through `s.add`: that way `job.montages`
    # already sees it within this same request, which is where it must appear
    job.montages.append(
        MontageModel(name=job.draft.get("title") or "Montage 1", data=job.draft)
    )
    # the column is cleared so there are not two truths about the same montage
    job.draft = {}
    s.flush()


def _free_name(existing: Sequence[Any], base: str) -> str:
    """A name that is not on the list yet.

    Repeated names in a list to pick from are as good as no names at all.
    """
    names = {m.name for m in existing}
    if base not in names:
        return base
    for i in range(2, 100):
        attempt = f"{base} {i}"
        if attempt not in names:
            return attempt
    return f"{base} {new_id()[:4]}"


def _get_montage(s: Any, job_id: str, montage_id: str) -> MontageModel:
    m = s.get(MontageModel, montage_id)
    if m is None or m.job_id != job_id:
        raise HTTPException(404, "montage not found in this match")
    return m


def _check_layers(spec: Any, sounds: dict[str, str]) -> None:
    """A layer either draws or plays -- and its content must match its kind.

    Music on a video layer would make ffmpeg try to resize an audio stream, and
    the whole render would die with a message that explains nothing. An image
    on an audio layer would be worse: it has no sound, so it would come out
    silent, with no error at all, and the user would look for the problem in
    the mix.
    """
    for layer in spec.layers:
        for clip in layer.clips:
            is_sound = bool(clip.media_id) and clip.media_id in sounds
            if layer.is_audio and not is_sound:
                raise HTTPException(
                    422,
                    "an audio layer only accepts music from the library",
                )
            if not layer.is_audio and is_sound:
                raise HTTPException(
                    422,
                    f"media {clip.media_id!r} is sound: it goes on an audio "
                    "layer",
                )
            # without the analysis finished there is neither beat nor
            # duration; it is also a sign the app sent it too early
            if is_sound and sounds[clip.media_id] != TrackStatus.READY:
                raise HTTPException(
                    409,
                    f"music {clip.media_id} has not been analysed yet "
                    f"(status: {sounds[clip.media_id]})",
                )


def _validate_montage(data: dict) -> MontageDraft:
    try:
        return MontageDraft(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"invalid montage: {exc}") from exc


@app.get("/api/jobs/{job_id}/montages")
def list_montages(job_id: str) -> dict[str, Any]:
    """This match's montages, most recent first."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        montages = sorted(job.montages, key=lambda m: _as_aware(m.updated_at), reverse=True)
        return {
            "job_id": job_id,
            "items": [_montage_dict(m, full=True) for m in montages],
        }


@app.post("/api/jobs/{job_id}/montages", status_code=201)
def create_montage(job_id: str, body: dict = Body(default={})) -> dict[str, Any]:
    """Starts a new montage, empty or from given content.

    They are different jobs over the same material -- the 30 s cut for Shorts
    and the long montage -- and until now one had to be chosen.
    """
    data = body.get("data") or {}
    _validate_montage(data)
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        _adopt_old_draft(s, job)
        name = str(body.get("name") or "").strip()
        m = MontageModel(
            job_id=job_id,
            name=_free_name(job.montages, name or f"Montage {len(job.montages) + 1}"),
            data=data,
        )
        s.add(m)
        s.flush()
        return _montage_dict(m, full=True)


@app.put("/api/jobs/{job_id}/montages/{montage_id}")
def save_montage(
    job_id: str, montage_id: str, body: dict = Body(...)
) -> dict[str, Any]:
    """Stores the montage. It is what the app calls by itself while editing.

    An invalid cut is refused here: storing garbage now would mean handing
    garbage back on the next opening.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        if "data" in body:
            draft_tl = _validate_montage(body["data"])
            m.data = draft_tl.model_dump()
        if "name" in body:
            name = str(body["name"] or "").strip()
            if not name:
                raise HTTPException(422, "a montage without a name cannot be found")
            others = [o for o in m.job.montages if o.id != m.id]
            m.name = _free_name(others, name)
        s.flush()
        return _montage_dict(m)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/duplicate", status_code=201)
def duplicate_montage(job_id: str, montage_id: str) -> dict[str, Any]:
    """A copy, to experiment without risking the one that is already good.

    The copy does not take the original's history: the snapshots say where
    *that* montage has been, and the copy has not been anywhere yet.
    """
    with session() as s:
        original = _get_montage(s, job_id, montage_id)
        copy = MontageModel(
            job_id=job_id,
            name=_free_name(original.job.montages, f"{original.name} (copy)"),
            data=dict(original.data or {}),
        )
        s.add(copy)
        s.flush()
        return _montage_dict(copy, full=True)


@app.delete("/api/jobs/{job_id}/montages/{montage_id}", status_code=204)
def delete_montage(job_id: str, montage_id: str) -> Response:
    with session() as s:
        s.delete(_get_montage(s, job_id, montage_id))
    return Response(status_code=204)


# ── version history ─────────────────────────────────────────────────────────

#: how many snapshots each montage keeps. Past that, the oldest goes.
#:
#: This is not undo -- that lives in the app. These are markers, and twenty
#: markers is already more history than anyone scrolls through in a list.
MAX_VERSIONS = 20


def _version_dict(v: MontageVersion, *, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": v.id,
        "montage_id": v.montage_id,
        "label": v.label,
        "created_at": _iso(v.created_at),
        **MontageModel(data=v.data).summary,
    }
    if full:
        d["data"] = v.data or {}
    return d


def _take_snapshot(m: MontageModel, label: str) -> MontageVersion | None:
    """Takes a snapshot of the montage as it stands now.

    Refuses a snapshot identical to the last one: rendering the same video twice
    in a row produced no new version, and a list of identical states helps
    nobody find the "it was good yesterday".
    """
    if not m.data:
        return None
    latest = max(m.versions, key=lambda v: _as_aware(v.created_at), default=None)
    if latest is not None and latest.data == m.data:
        return None

    # the timestamp comes from here, and not from the column's `default`:
    # without it the fresh snapshot joins the list with a null `created_at` and
    # cannot even be sorted
    snapshot = MontageVersion(label=label, data=dict(m.data), created_at=utcnow())
    m.versions.append(snapshot)
    newest_first = sorted(m.versions, key=lambda v: _as_aware(v.created_at), reverse=True)
    for v in newest_first[MAX_VERSIONS:]:
        m.versions.remove(v)
    return snapshot


@app.get("/api/jobs/{job_id}/montages/{montage_id}/versions")
def list_versions(job_id: str, montage_id: str) -> dict[str, Any]:
    """This montage's snapshots, most recent first."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshots = sorted(m.versions, key=lambda v: _as_aware(v.created_at), reverse=True)
        return {"montage_id": montage_id, "items": [_version_dict(v) for v in snapshots]}


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions", status_code=201)
def create_version(
    job_id: str, montage_id: str, body: dict = Body(default={})
) -> dict[str, Any]:
    """Marks the montage as it stands: the "it was good like this"."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = _take_snapshot(m, str(body.get("label") or "marked by hand"))
        if snapshot is None:
            raise HTTPException(409, "there is nothing new to mark")
        s.flush()
        return _version_dict(snapshot, full=True)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}/restore")
def restore_version(job_id: str, montage_id: str, version_id: str) -> dict[str, Any]:
    """Rolls the montage back to a snapshot.

    The current state becomes a snapshot first -- restoring never deletes work,
    it only swaps what is in front.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = s.get(MontageVersion, version_id)
        if snapshot is None or snapshot.montage_id != montage_id:
            raise HTTPException(404, "version not found in this montage")
        _take_snapshot(m, "before restoring")
        m.data = dict(snapshot.data or {})
        s.flush()
        return _montage_dict(m, full=True)


@app.delete(
    "/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}", status_code=204
)
def delete_version(job_id: str, montage_id: str, version_id: str) -> Response:
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = s.get(MontageVersion, version_id)
        if snapshot is None or snapshot.montage_id != montage_id:
            raise HTTPException(404, "version not found in this montage")
        m.versions.remove(snapshot)
    return Response(status_code=204)


# ── presets ─────────────────────────────────────────────────────────────────


def _preset_dict(p: Preset) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "data": p.data or {},
        "created_at": _iso(p.created_at),
        "updated_at": _iso(p.updated_at),
    }


def _validate_recipe(data: dict) -> Recipe:
    try:
        return Recipe(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"invalid recipe: {exc}") from exc


@app.get("/api/presets")
def list_presets() -> dict[str, Any]:
    """The presets. They belong to no match -- crossing from one to another is
    why they exist."""
    with session() as s:
        items = s.execute(select(Preset).order_by(Preset.created_at)).scalars().all()
        return {"items": [_preset_dict(p) for p in items]}


@app.post("/api/presets", status_code=201)
def create_preset(body: dict = Body(...)) -> dict[str, Any]:
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "a preset without a name cannot be found")
    recipe = _validate_recipe(body.get("data") or {})
    with session() as s:
        existing = s.execute(select(Preset)).scalars().all()
        p = Preset(
            name=_free_name(existing, name), data=recipe.model_dump(mode="json")
        )
        s.add(p)
        s.flush()
        return _preset_dict(p)


@app.put("/api/presets/{preset_id}")
def update_preset(preset_id: str, body: dict = Body(...)) -> dict[str, Any]:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "preset not found")
        if "data" in body:
            p.data = _validate_recipe(body["data"]).model_dump(mode="json")
        if "name" in body:
            name = str(body["name"] or "").strip()
            if not name:
                raise HTTPException(422, "a preset without a name cannot be found")
            others = [
                o for o in s.execute(select(Preset)).scalars().all() if o.id != p.id
            ]
            p.name = _free_name(others, name)
        s.flush()
        return _preset_dict(p)


@app.delete("/api/presets/{preset_id}", status_code=204)
def delete_preset(preset_id: str) -> Response:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "preset not found")
        s.delete(p)
    return Response(status_code=204)


# -- the compiled Flutter app, when present (single-origin deploy) ----------
#
# With the app served by the gateway itself, Flutter calls the API on a
# relative path and the same build works on any host, with no recompiling and
# no CORS setup. If nobody ran `flutter build web`, the mount simply does not
# happen and the API stays available on its own. The check is on `index.html`,
# and not on the directory: in Docker the bind mount creates the folder empty
# even when nobody compiled the app, and mounting StaticFiles on it would turn
# the site root into a 404.

_web = Path(get_settings().web_dir)
if (_web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    class _Revalidated(StaticFiles):
        """The app's files, with the browser told to ask before reusing them.

        Flutter names its bundles the same on every build (`main.dart.js`), and
        with no `Cache-Control` the browser caches them by heuristic: a fresh
        build kept loading the previous one. `no-cache` still lets it keep the
        file -- it only revalidates, and an unchanged file costs a 304.
        """

        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/", _Revalidated(directory=str(_web), html=True), name="web")


# The mount comes **last** on purpose: it matches any path, and every route
# registered after it becomes unreachable -- a `POST` to one of them returns
# 405, because the file server is what answers.
