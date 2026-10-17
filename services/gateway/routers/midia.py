"""Media: files the user adds to a match -- uploads, music and voice tracks,
and the sounds and stickers placed on it."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from owcore import sfx, stickers
from owcore.bus import get_bus
from owcore.db import session
from owcore.models import (
    STREAM_MEDIA,
    Job,
    Media,
    MediaKind,
    MediaUploaded,
    TrackStatus,
    new_id,
)
from owcore.storage import get_storage

from common import (
    VIDEO_EXTS,
    VIDEO_MIME,
    _media_dict,
    _safe_suffix,
    _serve_blob,
    _store_upload,
)

router = APIRouter()

#: `.weba` is what the editor's voice recorder sends from Chrome: WebM holding
#: only Opus audio, which `.webm` would have filed as video
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".weba"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

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
    ".weba": "audio/webm",
}


# ── the job's music: uploaded before any video exists ───────────────────────
#
# In the montage the music comes first. You cannot place a cut "on the turn of
# the chorus" without hearing the chorus, and you cannot snap a cut to the beat
# without knowing where the beats are. So it is uploaded, the system listens,
# and the app receives duration, BPM, beats and waveform to draw the ruler.
#
# The music belongs to the **job**, not to a request: the same track serves as
# many montages of that match as the user wants, with no re-upload.


def _store_media(
    job_id: str,
    upload: UploadFile,
    kind: MediaKind,
    expected_bytes: int = 0,
    voice: bool = False,
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
                voice=voice,
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


@router.post("/api/jobs/{job_id}/media", status_code=201)
def add_media(
    job_id: str,
    file: UploadFile = File(..., description="video, image or audio"),
    size: int = Form(0, description="file size, to verify the upload"),
    voice: bool = Form(False, description="a voice-over recorded in the editor"),
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
    if voice and kind != MediaKind.AUDIO:
        raise HTTPException(422, "a voice-over is audio")
    media_id = _store_media(job_id, file, kind, size, voice=voice)
    return {"id": media_id, "job_id": job_id, "kind": kind,
            "status": TrackStatus.PENDING}


@router.get("/api/media/{media_id}")
def get_media(media_id: str) -> dict[str, Any]:
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        return _media_dict(item)


@router.delete("/api/media/{media_id}", status_code=204)
def delete_media(media_id: str) -> Response:
    """Removes the item from the library. Videos already generated with it
    stay: the final mp4 already has what it needed inside."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        s.delete(item)
    return Response(status_code=204)


@router.get("/api/media/{media_id}/file")
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


@router.get("/api/media/{media_id}/thumb")
def media_thumb(media_id: str, request: Request) -> Response:
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        key = item.thumb_key
    if not key:
        raise HTTPException(404, "no thumbnail")
    # the worker's thumbnails are JPEG; a sticker's is its own transparent PNG
    mime = "image/png" if key.lower().endswith(".png") else "image/jpeg"
    response = _serve_blob(key, request, mime)
    response.headers["cache-control"] = "public, max-age=86400"
    return response


@router.get("/api/media/{media_id}/proxy")
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


@router.post("/api/jobs/{job_id}/sfx", status_code=201)
def add_sfx(job_id: str, body: dict = Body(...)) -> dict[str, Any]:
    """Brings an effect into the match's library, ready to go on the ruler.

    It comes out `ready` straight away: the server made the sound, so it
    already knows its duration and waveform, and there is nothing for the
    analyser to listen to. Adding the same effect again answers the item that
    is already there -- ten hits on the ruler are ten blocks of one item, not
    ten copies of the file.
    """
    sfx_id = str(body.get("sfx_id") or "")
    effect = sfx.catalog().get(sfx_id)
    if effect is None:
        raise HTTPException(404, f"sound effect {sfx_id!r} not found")

    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job not found")
        existing = s.scalars(
            select(Media).where(Media.job_id == job_id, Media.sfx_id == sfx_id)
        ).first()
        if existing is not None and existing.status == TrackStatus.READY:
            return _media_dict(existing)

    media_id = new_id()
    key = get_storage().put_stream(
        f"{job_id}/media/{media_id}.wav", io.BytesIO(sfx.wav_bytes(sfx_id))
    )
    with session() as s:
        item = Media(
            id=media_id,
            job_id=job_id,
            kind=MediaKind.AUDIO,
            status=TrackStatus.READY,
            name=effect.name,
            key=key,
            sfx_id=sfx_id,
            duration_s=sfx.duration_s(sfx_id),
            # no tempo: an effect is not something to cut to the beat of
            bpm=0.0,
            beats=[],
            peaks=list(sfx.peaks(sfx_id)),
        )
        s.add(item)
        s.flush()
        return _media_dict(item)


@router.post("/api/jobs/{job_id}/stickers", status_code=201)
def add_sticker(job_id: str, body: dict = Body(...)) -> dict[str, Any]:
    """Brings a sticker, in one colour, into the match's library.

    It comes out `ready` straight away: the server drew it, so it already
    knows its size, and there is nothing to analyse. Adding the same sticker in
    the same colour again answers the item already there.
    """
    sticker_id = str(body.get("sticker_id") or "")
    color = str(body.get("color") or stickers.DEFAULT_COLOR)
    sticker = stickers.catalog().get(sticker_id)
    if sticker is None:
        raise HTTPException(404, f"sticker {sticker_id!r} not found")
    if color not in stickers.COLORS:
        raise HTTPException(422, f"no colour {color!r}")
    library_id = stickers.library_id(sticker_id, color)

    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job not found")
        existing = s.scalars(
            select(Media).where(
                Media.job_id == job_id, Media.sticker_id == library_id
            )
        ).first()
        if existing is not None and existing.status == TrackStatus.READY:
            return _media_dict(existing)

    media_id = new_id()
    storage = get_storage()
    key = storage.put_stream(
        f"{job_id}/media/{media_id}.png",
        io.BytesIO(stickers.png_bytes(sticker_id, color)),
    )
    thumb_key = storage.put_stream(
        f"{job_id}/media/{media_id}.thumb.png",
        io.BytesIO(stickers.png_bytes(sticker_id, color, stickers.PREVIEW_SIZE)),
    )
    with session() as s:
        item = Media(
            id=media_id,
            job_id=job_id,
            kind=MediaKind.IMAGE,
            status=TrackStatus.READY,
            name=f"{sticker.name} ({color})",
            key=key,
            thumb_key=thumb_key,
            sticker_id=library_id,
            width=stickers.SIZE,
            height=stickers.SIZE,
        )
        s.add(item)
        s.flush()
        return _media_dict(item)


# ── the music routes, now thin shells over the library ──────────────────────
#
# They still exist because the app uses them and because "the job's music" is
# a useful name. Underneath it is all `Media` of kind audio.


@router.post("/api/jobs/{job_id}/tracks", status_code=201)
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


@router.get("/api/tracks/{track_id}")
def get_track(track_id: str) -> dict[str, Any]:
    return get_media(track_id)


@router.get("/api/tracks/{track_id}/audio")
def track_audio(track_id: str, request: Request) -> Response:
    return media_file(track_id, request)


@router.delete("/api/tracks/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    return delete_media(track_id)
