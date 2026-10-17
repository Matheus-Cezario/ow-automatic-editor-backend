"""Render: final renders, exact previews and the clips they produce."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import ValidationError

from owcore import fonts
from owcore.bus import get_bus
from owcore.db import session
from owcore.models import (
    STREAM_PREVIEW,
    STREAM_RENDER_READY,
    Clip,
    Job,
    JobStatus,
    Preview,
    PreviewRequested,
    Render,
    RenderRequested,
    RenderStage,
    RenderStatus,
    Timeline,
    TrackStatus,
    new_id,
    utcnow,
)
from owcore.preview import preview_window
from owcore.storage import get_storage

from common import _queue_position, _render_dict, _serve_blob

router = APIRouter()


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


@router.post("/api/jobs/{job_id}/renders", status_code=201)
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
    # a moment from another match needs that match's recording to be there
    others = spec.recording_jobs()
    if others:
        with session() as s:
            for other_id in sorted(others):
                other = s.get(Job, other_id)
                if other is None or not other.video_key:
                    raise HTTPException(
                        422, f"unknown match in this montage: {other_id!r}"
                    )
    _check_layers(spec, music_ids)
    # a font the server does not have would only fail at render time
    for clip in spec.clips:
        if clip.text_style.font:
            try:
                fonts.resolve(clip.text_style.font)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
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


@router.post("/api/jobs/{job_id}/previews", status_code=201)
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


@router.get("/api/previews/{preview_id}")
def get_preview(preview_id: str) -> dict[str, Any]:
    with session() as s:
        p = s.get(Preview, preview_id)
        if p is None:
            raise HTTPException(404, "preview not found")
        return _preview_dict(p)


@router.get("/api/previews/{preview_id}/video")
def preview_video(preview_id: str, request: Request) -> Response:
    with session() as s:
        p = s.get(Preview, preview_id)
        if p is None:
            raise HTTPException(404, "preview not found")
        key = p.video_key
    if not key:
        raise HTTPException(404, "this preview is not ready")
    return _serve_blob(key, request, "video/mp4")


@router.get("/api/renders/{render_id}")
def get_render(render_id: str) -> dict[str, Any]:
    with session() as s:
        request_ = s.get(Render, render_id)
        if request_ is None:
            raise HTTPException(404, "request not found")
        return _render_dict(
            request_,
            list(request_.clips),
            queue_position=_queue_position(s, request_),
        )


@router.get("/api/renders")
def list_renders(recent_s: float = 600) -> dict[str, Any]:
    """The render queue, across every match: what is waiting or rendering,
    and what finished in the last `recent_s` seconds -- oldest first, the
    order the editor takes them in."""
    from datetime import timedelta

    cutoff = (utcnow() - timedelta(seconds=recent_s)).replace(tzinfo=None)
    with session() as s:
        rows = (
            s.query(Render)
            .filter(
                (Render.status.in_([RenderStatus.PENDING, RenderStatus.RENDERING]))
                | (Render.updated_at >= cutoff)
            )
            .order_by(Render.created_at)
            .all()
        )
        out = []
        for r in rows:
            d = _render_dict(r, list(r.clips), queue_position=_queue_position(s, r))
            d["job_name"] = r.job.video_name if r.job else ""
            out.append(d)
        return {"renders": out}


@router.post("/api/renders/{render_id}/cancel")
def cancel_render(render_id: str) -> dict[str, Any]:
    """Stops a request that is waiting or rendering. The editor notices
    within a second, stops ffmpeg and keeps nothing; a request still in the
    queue is skipped when its turn comes. A finished one cannot be cancelled
    -- delete it instead."""
    with session() as s:
        r = s.get(Render, render_id)
        if r is None:
            raise HTTPException(404, "request not found")
        if not r.is_active:
            raise HTTPException(409, f"this request is already {r.status}")
        r.status = RenderStatus.CANCELLED
        r.stage = RenderStage.CANCELLED
        return _render_dict(r, list(r.clips))


@router.delete("/api/renders/{render_id}", status_code=204)
def delete_render(render_id: str) -> Response:
    """Deletes a request and its videos. The montage stays saved: it can be
    requested again, with different music."""
    with session() as s:
        request_ = s.get(Render, render_id)
        if request_ is None:
            raise HTTPException(404, "request not found")
        s.delete(request_)
    return Response(status_code=204)


@router.get("/api/clips/{clip_id}/video")
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


@router.get("/api/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str, request: Request) -> Response:
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clip not found")
        key = (clip.meta or {}).get("thumb_key")
    if not key:
        raise HTTPException(404, "no thumbnail")
    return _serve_blob(key, request, "image/jpeg")


@router.get("/api/clips/{clip_id}/cuts.zip")
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
