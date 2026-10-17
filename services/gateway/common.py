"""What more than one part of the API uses: upload checks, `Range`
deliveries, the extension and MIME tables, and the JSON shapes of a job,
render, clip, media and montage."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from owcore.models import Clip, Media, Render, RenderStatus, utcnow
from owcore.models import Montage as MontageModel
from owcore.storage import get_storage

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts"}
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


def _render_eta_s(r: Render) -> float | None:
    """How long the render still needs, or None when it cannot be said yet.

    The rendering part of the bar runs from 0.1 to 0.95, and it advances with
    ffmpeg's position in each video -- in proportion to time, so the simplest
    maths holds: what is left, at the speed so far. It is measured from when
    the editor picked the request up, not from the request: the wait in the
    queue is not work. Below 5% of the work the error is bigger than the
    estimate.
    """
    if r.status != RenderStatus.RENDERING or r.started_at is None:
        return None
    done = (r.progress - 0.1) / 0.85
    if done < 0.05:
        return None
    elapsed = (utcnow() - _as_aware(r.started_at)).total_seconds()
    if elapsed <= 0:
        return None
    left = elapsed * (1 - min(done, 1.0)) / done
    return round(max(0.0, left), 1)


def _queue_position(s, r: Render) -> int | None:
    """How many requests are still ahead of this one, waiting or rendering
    -- None once it is being rendered itself."""
    if r.status != RenderStatus.PENDING:
        return None
    return (
        s.query(Render)
        .filter(
            Render.status.in_([RenderStatus.PENDING, RenderStatus.RENDERING]),
            Render.created_at < r.created_at,
        )
        .count()
    )


def _render_dict(
    r: Render, all_clips: list[Clip], *, queue_position: int | None = None
) -> dict[str, Any]:
    clips_of_render = [c for c in all_clips if c.render_id == r.id]
    return {
        "id": r.id,
        "job_id": r.job_id,
        "status": r.status,
        "stage": r.stage,
        "progress": round(r.progress, 3),
        "error": r.error,
        "created_at": _iso(r.created_at),
        "started_at": _iso(r.started_at) if r.started_at else None,
        "eta_s": _render_eta_s(r),
        "queue_position": queue_position,
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


# ── the JSON shapes the routers share ─────────────────────────────────────


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
        data["sfx_id"] = m.sfx_id or None
        data["voice"] = bool(m.voice)
        data |= {
            "bpm": round(m.bpm, 2),
            "beats": m.beats or [],
            "peaks": m.peaks or [],
            # the app still asks for the music through here
            "audio_url": f"/api/media/{m.id}/file",
        }
    else:
        data |= {"width": m.width, "height": m.height, "fps": round(m.fps, 3)}
        data["sticker_id"] = m.sticker_id or None
    return data


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
