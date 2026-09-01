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
from sqlalchemy import select

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import init_db, session
from owcore.models import (
    STREAM_JOBS,
    STREAM_MEDIA,
    STREAM_RENDER_READY,
    STREAM_THUMBS,
    Clip,
    Job,
    JobCreated,
    JobParams,
    JobStatus,
    Media,
    MediaKind,
    MediaUploaded,
    Montage as MontageModel,
    MontageDraft,
    MontageVersion,
    Preset,
    Render,
    RenderRequested,
    Recipe,
    RenderStatus,
    ThumbsRequested,
    Timeline,
    TrackStatus,
    frame_key,
    new_id,
    utcnow,
)
from owcore.ffmpeg import probe
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
    description="Melhores momentos de partidas de Overwatch 2, automaticamente.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o app Flutter web roda em outra porta durante o dev
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────── helpers ────────────────────────────────────


def _safe_suffix(filename: str | None, allowed: set[str], default: str) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in allowed else default


def _store_upload(key: str, upload: UploadFile, expected_bytes: int) -> str:
    """Grava o upload conferindo se ele chegou inteiro.

    Um upload truncado nao se parece com erro nenhum: o multipart fecha
    direito, o `Content-Length` bate com o que de fato chegou, e o que sobra e
    meia gravacao guardada como se estivesse inteira. O estrago so aparecia
    fases depois, no preprocessador, como um `ffprobe saiu com 1` -- longe da
    tela de envio e sem dizer o que fazer.

    Conferir aqui custa um `stat` e devolve o problema onde ele nasceu, com a
    unica acao que resolve: enviar de novo.

    `expected_bytes` zero desliga a conferencia -- e um cliente antigo, que
    nao manda o tamanho.
    """
    storage = get_storage()
    stored = storage.put_stream(key, upload.file)
    got = storage.size(stored)
    if expected_bytes and got != expected_bytes:
        storage.delete(stored)
        raise HTTPException(
            400,
            f"o arquivo chegou incompleto: {got} de {expected_bytes} bytes. "
            "Envie de novo.",
        )
    return stored


def _json_list(raw: Any, field: str) -> list:
    """Reads a multipart field carrying a JSON list as text."""
    if raw is None or raw == "":
        return []
    if not isinstance(raw, str):
        raise HTTPException(422, f"'{field}' tem de ser um JSON em texto")
    try:
        valor = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"'{field}' nao e JSON valido: {exc}") from exc
    if not isinstance(valor, list):
        raise HTTPException(422, f"'{field}' tem de ser uma lista")
    return valor


def _job_dict(job: Job, *, full: bool = False) -> dict[str, Any]:
    has_cuts = any(
        (c.meta or {}).get("segments_zip_key") for c in job.clips
    )
    without_video = sum(1 for c in job.clips if not c.key)
    data = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
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
        # a gravacao em si: o preview da montage busca dentro dela
        "video_url": f"/api/jobs/{job.id}/video",
        # the reduced copy, when it exists. Jobs analysed before it came into
        # the world answer `null`, and the app falls back to the recording
        "proxy_url": f"/api/jobs/{job.id}/proxy" if job.proxy_key else None,
        # the whole match's package: it requires opening no video at all
        "zip_url": f"/api/jobs/{job.id}/cortes.zip" if job.clips else None,
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
        montages = sorted(job.montages, key=lambda m: _hora(m.updated_at), reverse=True)
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
                # um pedido antigo guarda `cuts`; um novo, camadas
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
            f"/api/clips/{c.id}/cortes.zip"
            if c.meta.get("segments_zip_key")
            else None
        ),
    }


_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def _serve_blob(key: str, request: Request, media_type: str) -> Response:
    """Serves a blob with Range support, so the player can seek."""
    storage = get_storage()
    if not storage.exists(key):
        raise HTTPException(404, "upload nao encontrado")
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


# ──────────────────────────────── rotas ─────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"ok": True, "mode": s.mode, "profile": s.profile}


@app.post("/api/jobs", status_code=201)
def create_job(
    video: UploadFile = File(..., description="gravacao da partida"),
    params: str = Form("{}", description="JobParams em JSON"),
    size: int = Form(0, description="tamanho do arquivo, para conferir o envio"),
) -> dict[str, Any]:
    """Primeira fase: so a gravacao.

    Nenhuma musica entra aqui. A analise descobre os momentos e monta a lista
    de videos possiveis; a escolha -- e a trilha de cada video -- vem depois,
    em `POST /api/jobs/{id}/renders`, quantas vezes o usuario quiser.
    """
    try:
        parsed = JobParams(**json.loads(params or "{}"))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(422, f"parametros invalidos: {exc}") from exc

    job_id = new_id()

    video_ext = _safe_suffix(video.filename, VIDEO_EXTS, ".mp4")
    video_key = _store_upload(f"{job_id}/source{video_ext}", video, size)

    with session() as s:
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            stage="na fila",
            video_key=video_key,
            video_name=video.filename or "gravacao",
            params=parsed.model_dump(),
        )
        s.add(job)

    get_bus().publish(STREAM_JOBS, JobCreated(job_id=job_id).model_dump())
    return {"id": job_id, "status": JobStatus.PENDING}


@app.post("/api/jobs/{job_id}/renders", status_code=201)
async def create_render(job_id: str, request: Request) -> dict[str, Any]:
    """Segunda fase: gerar os videos que o usuario montou.

    Recebe um multipart com o field `timelines` (JSON): cada montage com os
    seus blocos ja posicionados e, se tiver trilha, apontando para uma musica
    ja enviada a este job pela library.

    Ja houve um segundo field, `selections`, com as propostas que o sistema
    oferecia prontas. Nao ha mais propostas: o que vira video sai do editor.

    Pode ser chamado quantas vezes se quiser sobre o mesmo job -- usar um
    momento num video nao o consome para os outros.
    """
    form = await request.form()
    raw_timelines = _json_list(form.get("timelines"), "timelines")
    if not raw_timelines:
        raise HTTPException(422, "monte pelo menos uma linha do tempo")

    render_id = new_id()

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        if job.status != JobStatus.READY:
            raise HTTPException(
                409, f"a analise deste job ainda nao terminou (status: {job.status})"
            )
        # a montage points at music; the library holds more than that
        music_ids = {m.id: m.status for m in job.media if m.is_audio}
        library = {m.id for m in job.media}

    montages: list[Timeline] = []
    for item in raw_timelines:
        if not isinstance(item, dict):
            raise HTTPException(422, "cada linha do tempo tem de ser um objeto")
        try:
            spec = Timeline(**item)
        except ValidationError as exc:
            raise HTTPException(422, f"linha do tempo invalida: {exc}") from exc
        # a clip pointing at another job's media does not go in: the montage
        # would come out without it, and with no warning
        for clip in spec.clips:
            if clip.media_id and clip.media_id not in library:
                raise HTTPException(
                    422,
                    f"midia desconhecida neste job: {clip.media_id!r}",
                )
        _check_layers(spec, music_ids)
        # the watermark comes from the same library, and refusing it here is
        # better than letting the whole render fail later because of it
        if spec.export.watermark_id and spec.export.watermark_id not in library:
            raise HTTPException(
                422,
                f"marca desconhecida neste job: {spec.export.watermark_id!r}",
            )
        montages.append(spec)

    with session() as s:
        s.add(
            Render(
                id=render_id,
                job_id=job_id,
                status=RenderStatus.PENDING,
                stage="na fila",
                timelines=[m.model_dump() for m in montages],
            )
        )

    # straight to the editor: there is no rhythm stage in between any more.
    # This montage's music came up through the library, already analysed.
    get_bus().publish(
        STREAM_RENDER_READY, RenderRequested(render_id=render_id).model_dump()
    )
    return {"id": render_id, "job_id": job_id, "status": RenderStatus.PENDING}


@app.get("/api/renders/{render_id}")
def get_render(render_id: str) -> dict[str, Any]:
    with session() as s:
        pedido = s.get(Render, render_id)
        if pedido is None:
            raise HTTPException(404, "pedido nao encontrado")
        return _render_dict(pedido, list(pedido.clips))


@app.delete("/api/renders/{render_id}", status_code=204)
def delete_render(render_id: str) -> Response:
    """Deletes a request and its videos. The montage stays saved: it can be
    requested again, with different music."""
    with session() as s:
        pedido = s.get(Render, render_id)
        if pedido is None:
            raise HTTPException(404, "pedido nao encontrado")
        s.delete(pedido)
    return Response(status_code=204)


# ── music_ids do job: sobem antes de existir video ────────────────────────────
#
# In the montage the music comes first. You cannot place a cut "on the turn of
# the chorus" without hearing the chorus, and you cannot snap a cut to the beat
# without knowing where the beats are. So it is uploaded, the system listens,
# and the app receives duration, BPM, beats and waveform to draw the ruler.
#
# The music belongs to the **job**, not to a request: the same track serves as
# many montages of that match as the user wants, with no re-upload.


def _media_dict(m: Media) -> dict[str, Any]:
    """Um item da library de midia da partida.

    Audio vai completo -- batidas e forma de onda inclusive --, porque e com
    isso que a tela de montage desenha a musica e gruda os cortes na batida.
    Video e imagem vao com dimensions e os enderecos da miniatura e do proxy.
    """
    dados = {
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
        dados |= {
            "bpm": round(m.bpm, 2),
            "beats": m.beats or [],
            "peaks": m.peaks or [],
            # the app still asks for the music through here
            "audio_url": f"/api/media/{m.id}/file",
        }
    else:
        dados |= {"width": m.width, "height": m.height, "fps": round(m.fps, 3)}
    return dados


def _guardar_media(
    job_id: str, upload: UploadFile, kind: MediaKind, expected_bytes: int = 0
) -> str:
    """Grava o upload e manda analisa-lo. Devolve o id."""
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


def _tipo_de(filename: str | None) -> MediaKind | None:
    """What kind the file is, from its extension.

    Pela extensao e nao pelo `content-type` porque o navegador mente com
    frequencia -- manda `application/octet-stream` para tudo quando o upload
    veio de um lugar que ele nao conhece.
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
    file: UploadFile = File(..., description="video, imagem ou audio"),
    size: int = Form(0, description="tamanho do arquivo, para conferir o envio"),
) -> dict[str, Any]:
    """Brings a file into the match's library.

    Responde na hora, com o item ainda `pending`: a analise (dimensions,
    miniatura, proxy; batidas quando for audio) roda no worker, e o app
    acompanha por `GET /api/media/{id}`.
    """
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job nao encontrado")

    kind = _tipo_de(file.filename)
    if kind is None:
        raise HTTPException(
            422,
            f"nao sei o que fazer com {file.filename!r}: aceito video, imagem "
            "e audio",
        )
    media_id = _guardar_media(job_id, file, kind, size)
    return {"id": media_id, "job_id": job_id, "kind": kind,
            "status": TrackStatus.PENDING}


@app.get("/api/media/{media_id}")
def get_media(media_id: str) -> dict[str, Any]:
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
        return _media_dict(item)


@app.delete("/api/media/{media_id}", status_code=204)
def delete_media(media_id: str) -> Response:
    """Removes the item from the library. Videos already generated with it
    stay: the final mp4 already has what it needed inside."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
        s.delete(item)
    return Response(status_code=204)


@app.get("/api/media/{media_id}/file")
def media_file(media_id: str, request: Request) -> Response:
    """The file itself, with `Range`."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
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
            raise HTTPException(404, "midia nao encontrada")
        key = item.thumb_key
    if not key:
        raise HTTPException(404, "sem miniatura")
    resposta = _serve_blob(key, request, "image/jpeg")
    resposta.headers["cache-control"] = "public, max-age=86400"
    return resposta


@app.get("/api/media/{media_id}/proxy")
def media_proxy(media_id: str, request: Request) -> Response:
    """The reduced copy of the imported video -- what the monitor opens."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
        key = item.proxy_key
    if not key:
        raise HTTPException(404, "este item nao tem proxy")
    return _serve_blob(key, request, "video/mp4")


# ── as rotas de musica, agora cascas sobre a library ─────────────────────
#
# Elas continuam existindo porque o app as usa e porque "a musica do job" e um
# nome util. Por baixo e tudo `Media` de tipo audio.


@app.post("/api/jobs/{job_id}/tracks", status_code=201)
def add_track(
    job_id: str,
    audio: UploadFile = File(..., description="musica para montar em cima"),
    size: int = Form(0, description="tamanho do arquivo, para conferir o envio"),
) -> dict[str, Any]:
    """Uploads a track and has the system listen to it."""
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job nao encontrado")
    media_id = _guardar_media(job_id, audio, MediaKind.AUDIO, size)
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
        return {"jobs": [_job_dict(j) for j in jobs]}


#: while the analysis is not finished, the preprocessor is still going to
#: write the recording's fields -- there is nothing to patch up
_ANALISANDO = frozenset(
    {JobStatus.PENDING, JobStatus.PREPROCESSING, JobStatus.DETECTING}
)


def _remendar_o_tamanho(job: Job) -> None:
    """Finds the size of a recording analysed before this column existed.

    O reconciliador de esquema poe a coluna, mas nao tem como saber o que ela
    deveria valer -- so o upload sabe. Uma partida antiga abriria o editor sem
    poder dizer se um 9:16 corta o quadro dela. Custa um `ffprobe`, uma vez na
    vida de cada job. O ffmpeg le o upload onde ele esta -- por `Range`, se
    estiver no S3 --, entao medir uma gravacao de dois gigas custa o cabecalho
    dela, e nao os dois gigas.

    Nao vale a pena enquanto a analise esta rodando, e por dois motivos. Um:
    uma partida *sendo analisada agora* nao e uma partida antiga -- o
    preprocessador vai gravar o tamanho de verdade em segundos. Dois: o
    cabecalho e barato, mas le-lo pela rede enquanto o preprocessador baixa o
    mesmo upload disputa a mesma banda, e a tela consulta de dois em dois
    segundos. Medido: uma consulta que responde em 0,5s passou de 30s nessa
    janela, o que a tela mostra como "nao consegui falar com o servidor".
    """
    if job.width or not job.video_key or job.status in _ANALISANDO:
        return
    try:
        info = probe(get_storage().url(job.video_key))
    except Exception:  # noqa: BLE001 - um remendo nao derruba a tela do editor
        LOG.warning("nao deu para medir a gravacao de %s", job.id, exc_info=True)
        return
    job.width, job.height = info.width, info.height
    if not job.fps:
        job.fps = info.fps


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        _remendar_o_tamanho(job)
        _adotar_o_rascunho_antigo(s, job)
        return _job_dict(job, full=True)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, request: Request) -> Response:
    """The original recording, with `Range`.

    E o que o preview da tela de montage toca: em vez de renderizar o video a
    cada ajuste -- o que custaria uma volta inteira pelo ffmpeg por arrasto --,
    o app abre a propria gravacao e busca o instante do bloco sob a cabeca de
    leitura. O corte de verdade continua acontecendo no servidor; isto aqui e
    so para ver antes de pedir.
    """
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        key = job.video_key
    mime = VIDEO_MIME.get(Path(key).suffix.lower(), "video/mp4")
    return _serve_blob(key, request, mime)


@app.put("/api/jobs/{job_id}/draft")
def save_draft(job_id: str, draft: dict = Body(...)) -> dict[str, Any]:
    """Guarda a montage em andamento. **Legado da V1.**

    Desde a Fase 8 uma partida tem varias montages nomeadas, e o app salva pelo
    id de uma delas. Esta rota escreve na mais recente -- e cria a primeira, se
    nao houver nenhuma --, para que um app anterior continue funcionando em vez
    de perder o trabalho em silencio.
    """
    rascunho = _validate_montage(draft)

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        _adotar_o_rascunho_antigo(s, job)
        atual = max(job.montages, key=lambda m: _hora(m.updated_at), default=None)
        if atual is None:
            atual = MontageModel(job_id=job_id, name="Montagem 1")
            s.add(atual)
        atual.data = rascunho.model_dump()
    return {"job_id": job_id, "n_cuts": len(rascunho.clips)}


@app.delete("/api/jobs/{job_id}/draft", status_code=204)
def delete_draft(job_id: str) -> Response:
    """Joga as montages desta partida fora e comeca do zero. **Legado da V1.**"""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        job.draft = {}
        for m in list(job.montages):
            s.delete(m)
    return Response(status_code=204)


@app.get("/api/jobs/{job_id}/proxy")
def job_proxy(job_id: str, request: Request) -> Response:
    """The reduced copy of the recording, with `Range`.

    E o que o monitor do editor abre. A gravacao original tem centenas de
    megabytes, e buscar dentro dela dezenas de vezes por segundo enquanto se
    arrasta chegou a derrubar o elemento de video do navegador. Esta copia sai
    da mesma decodificacao dos recortes -- custa quase nada -- e o corte final
    continua vindo do upload original.
    """
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        key = job.proxy_key
    if not key:
        raise HTTPException(
            404, "esta partida foi analisada antes do proxy existir"
        )
    return _serve_blob(key, request, "video/mp4")


@app.get("/api/jobs/{job_id}/frame")
def job_frame(job_id: str, t: float, request: Request) -> Response:
    """A frame of the match at instant `t`, for the editor's sidebar.

    So entrega o que ja existe: quem extrai e o servico `thumbs`. Um 404 aqui
    quer dizer "ainda nao foi extraida", e o app mostra o lugar dela em vez de
    ficar sem item.
    """
    key = frame_key(job_id, t)
    if not get_storage().exists(key):
        raise HTTPException(404, "miniatura ainda nao extraida")
    resposta = _serve_blob(key, request, "image/jpeg")
    # o quadro de um instante nunca muda: vale a pena o navegador guardar
    resposta.headers["cache-control"] = "public, max-age=86400"
    return resposta


@app.post("/api/jobs/{job_id}/frames", status_code=202)
def request_frames(job_id: str) -> dict[str, Any]:
    """Requests extraction of the missing thumbnails.

    Jobs novos ja saem com elas -- o planejador pede assim que a analise
    termina. Isto aqui e para os antigos, e para o caso de alguma ter falhado:
    o app chama ao abrir o editor, e o servico pula o que ja esta no lugar.
    """
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job nao encontrado")
    get_bus().publish(STREAM_THUMBS, ThumbsRequested(job_id=job_id).model_dump())
    return {"job_id": job_id, "status": "pedido"}


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        s.delete(job)
    return Response(status_code=204)


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: str, request: Request) -> Response:
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clipe nao encontrado")
        key = clip.key
    if not key:
        raise HTTPException(
            404, "a montagem deste clipe falhou; baixe os cortes em cortes.zip"
        )
    return _serve_blob(key, request, "video/mp4")


@app.get("/api/clips/{clip_id}/thumb")
def clip_thumb(clip_id: str, request: Request) -> Response:
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clipe nao encontrado")
        key = (clip.meta or {}).get("thumb_key")
    if not key:
        raise HTTPException(404, "sem miniatura")
    return _serve_blob(key, request, "image/jpeg")


@app.get("/api/jobs/{job_id}/cortes.zip")
def job_zip(job_id: str, request: Request) -> Response:
    """Everything the match generated, in a single file.

    Montado na hora a partir do que ja esta no storage -- os videos finais e os
    cortes avulsos de cada montage --, em vez de guardar um terceiro package
    com os mesmos bytes. Como o zip so empacota (os mp4 ja estao comprimidos),
    o custo e basicamente o de copiar.
    """
    storage = get_storage()
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        if not job.clips:
            raise HTTPException(404, "esta partida ainda nao tem videos")
        base_name = Path(job.video_name or "partida").stem
        # um job rende varios pedidos ao longo do tempo; o package traz todos,
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
            for i, n_pedido, kind, video_key, cortes_key in items:
                folder = f"pedido_{n_pedido:02d}" if n_pedido else "videos"
                if video_key and storage.exists(video_key):
                    local = storage.get_file(video_key, tmp / f"v{i}.mp4")
                    zf.write(local, f"{folder}/videos/{i:02d}_{kind}.mp4")
                    local.unlink(missing_ok=True)
                if not cortes_key or not storage.exists(cortes_key):
                    continue
                local = storage.get_file(cortes_key, tmp / f"c{i}.zip")
                with zipfile.ZipFile(local) as origem:
                    for info in origem.infolist():
                        destino = f"{folder}/cortes/{i:02d}_{kind}/{info.filename}"
                        # copia chunk a chunk: `origem.read(nome)` punha um
                        # a whole cut in memory at a time
                        with origem.open(info) as entrada,                                 zf.open(destino, "w") as saida:
                            shutil.copyfileobj(entrada, saida, CHUNK)
                local.unlink(missing_ok=True)
        total = package.stat().st_size
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    def send_chunks() -> Iterator[bytes]:
        """Delivers the package in chunks and only then deletes the temp dir.

        O `read_bytes()` que estava aqui punha o package **inteiro** na memoria
        do gateway -- uma partida com alguns pedidos sao centenas de MB, por
        requisicao simultanea, e o processo que serve a API e o mesmo que serve
        o app. Streaming mantem o custo em um chunk de cada vez, e o upload
        temporario ja estava em disco de qualquer forma.
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
            "content-disposition": f'attachment; filename="{base_name}_cortes.zip"',
            "content-length": str(total),
        },
    )


@app.get("/api/clips/{clip_id}/cortes.zip")
def clip_segments_zip(clip_id: str, request: Request) -> Response:
    """Os cortes individuais da montage, num zip.

    Serve para reeditar por fora: cada upload e um trecho, nomeado pelo
    instante de onde saiu na gravacao original.
    """
    with session() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "clipe nao encontrado")
        key = (clip.meta or {}).get("segments_zip_key")
        nome = f"cortes_{clip.kind}_{clip.id}.zip"
    if not key:
        raise HTTPException(404, "este clipe nao tem cortes separados")
    response = _serve_blob(key, request, "application/zip")
    response.headers["content-disposition"] = f'attachment; filename="{nome}"'
    return response


@app.get("/api/profile")
def profile() -> dict[str, Any]:
    """Exposes the HUD profile for the app's calibration screen."""
    from owcore.profiles import load_profile

    return load_profile(get_settings().profile).data


# ── montages nomeadas ──────────────────────────────────────────────────────


def _hora(t: datetime | None) -> datetime:
    """A record's timestamp, always with a timezone.

    O SQLite guarda `datetime` sem fuso, entao uma linha lida do banco volta
    ingenua enquanto uma criada nesta mesma requisicao ainda esta com o fuso que
    `utcnow()` deu. Ordenar as duas juntas estoura -- e e exatamente o que
    acontece ao listar as montages logo depois de criar uma.
    """
    if t is None:
        return utcnow()
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _iso(t: datetime | None) -> str:
    """The timestamp as text, **with the timezone written out**.

    Sem o sufixo de fuso, quem le do outro lado trata a data como hora local --
    e o Dart faz exatamente isso. Como o que sai daqui e UTC, o app mostrava
    todo horario adiantado pelo fuso do usuario, e qualquer conta de "quanto
    falta" a partir de `created_at` dava tempo negativo.
    """
    return _hora(t).isoformat()


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


def _adotar_o_rascunho_antigo(s: Any, job: Job) -> None:
    """Brings the job's single montage into the list of named montages.

    Ate a Fase 8 havia uma so, numa coluna do proprio job. Fazer isto na
    leitura, e nao numa migracao de banco, e a mesma escolha do resto do
    sistema: quem sabe converter o formato velho e o codigo que le, e assim uma
    partida parada ha meses continua abrindo.
    """
    if not job.draft or job.montages:
        return
    # through the relationship, not through `s.add`: that way `job.montages`
    # already sees it within this same request, which is where it must appear
    job.montages.append(
        MontageModel(name=job.draft.get("title") or "Montagem 1", data=job.draft)
    )
    # the column is cleared so there are not two truths about the same montage
    job.draft = {}
    s.flush()


def _nome_livre(existentes: Sequence[Any], base: str) -> str:
    """A name that is not on the list yet.

    Nomes repetidos numa lista de escolher e o mesmo que nome nenhum.
    """
    nomes = {m.name for m in existentes}
    if base not in nomes:
        return base
    for i in range(2, 100):
        tentativa = f"{base} {i}"
        if tentativa not in nomes:
            return tentativa
    return f"{base} {new_id()[:4]}"


def _get_montage(s: Any, job_id: str, montage_id: str) -> MontageModel:
    m = s.get(MontageModel, montage_id)
    if m is None or m.job_id != job_id:
        raise HTTPException(404, "montagem nao encontrada nesta partida")
    return m


def _check_layers(spec: Any, sounds: dict[str, str]) -> None:
    """A layer either draws or plays -- and its content must match its kind.

    Uma musica numa camada de video faria o ffmpeg tentar redimensionar um fluxo
    de audio, e o render inteiro morreria com uma mensagem que nao explica nada.
    Uma imagem numa camada de audio seria pior: ela nao tem som, entao sairia em
    silencio, sem erro nenhum, e o usuario procuraria o problema na mixagem.
    """
    for camada in spec.layers:
        for clip in camada.clips:
            e_som = bool(clip.media_id) and clip.media_id in sounds
            if camada.is_audio and not e_som:
                raise HTTPException(
                    422,
                    "uma camada de audio so aceita musica da biblioteca",
                )
            if not camada.is_audio and e_som:
                raise HTTPException(
                    422,
                    f"a midia {clip.media_id!r} e som: ela vai numa camada "
                    "de audio",
                )
            # without the analysis finished there is neither beat nor
            # duration; it is also a sign the app sent it too early
            if e_som and sounds[clip.media_id] != TrackStatus.READY:
                raise HTTPException(
                    409,
                    f"a musica {clip.media_id} ainda nao foi analisada "
                    f"(status: {sounds[clip.media_id]})",
                )


def _validate_montage(data: dict) -> MontageDraft:
    try:
        return MontageDraft(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"montagem invalida: {exc}") from exc


@app.get("/api/jobs/{job_id}/montages")
def list_montages(job_id: str) -> dict[str, Any]:
    """This match's montages, most recent first."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        _adotar_o_rascunho_antigo(s, job)
        montages = sorted(job.montages, key=lambda m: _hora(m.updated_at), reverse=True)
        return {
            "job_id": job_id,
            "items": [_montage_dict(m, full=True) for m in montages],
        }


@app.post("/api/jobs/{job_id}/montages", status_code=201)
def create_montage(job_id: str, corpo: dict = Body(default={})) -> dict[str, Any]:
    """Starts a new montage, empty or from given content.

    Sao trabalhos diferentes sobre o mesmo material -- o corte de 30 s para o
    Shorts e a montage longa --, e ate aqui era preciso escolher um.
    """
    dados = corpo.get("data") or {}
    _validate_montage(dados)
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        _adotar_o_rascunho_antigo(s, job)
        nome = str(corpo.get("name") or "").strip()
        m = MontageModel(
            job_id=job_id,
            name=_nome_livre(job.montages, nome or f"Montagem {len(job.montages) + 1}"),
            data=dados,
        )
        s.add(m)
        s.flush()
        return _montage_dict(m, full=True)


@app.put("/api/jobs/{job_id}/montages/{montage_id}")
def save_montage(
    job_id: str, montage_id: str, corpo: dict = Body(...)
) -> dict[str, Any]:
    """Stores the montage. It is what the app calls by itself while editing.

    Um corte invalido e recusado aqui: guardar lixo agora seria devolver lixo na
    proxima abertura.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        if "data" in corpo:
            rascunho = _validate_montage(corpo["data"])
            m.data = rascunho.model_dump()
        if "name" in corpo:
            nome = str(corpo["name"] or "").strip()
            if not nome:
                raise HTTPException(422, "uma montagem sem nome nao da para achar")
            outras = [o for o in m.job.montages if o.id != m.id]
            m.name = _nome_livre(outras, nome)
        s.flush()
        return _montage_dict(m)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/duplicate", status_code=201)
def duplicate_montage(job_id: str, montage_id: str) -> dict[str, Any]:
    """A copy, to experiment without risking the one that is already good.

    A copia nao leva o historico da original: as snapshots dizem por onde *aquela*
    montage passou, e a copia ainda nao passou por lugar nenhum.
    """
    with session() as s:
        original = _get_montage(s, job_id, montage_id)
        copia = MontageModel(
            job_id=job_id,
            name=_nome_livre(original.job.montages, f"{original.name} (copia)"),
            data=dict(original.data or {}),
        )
        s.add(copia)
        s.flush()
        return _montage_dict(copia, full=True)


@app.delete("/api/jobs/{job_id}/montages/{montage_id}", status_code=204)
def delete_montage(job_id: str, montage_id: str) -> Response:
    with session() as s:
        s.delete(_get_montage(s, job_id, montage_id))
    return Response(status_code=204)


# ── historico de versoes ────────────────────────────────────────────────────

#: how many snapshots each montage keeps. Past that, the oldest goes.
#:
#: This is not undo -- that lives in the app. These are markers, and twenty
#: markers is already more history than anyone scrolls through in a list.
MAX_VERSOES = 20


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


def _guardar_foto(m: MontageModel, label: str) -> MontageVersion | None:
    """Takes a snapshot of the montage as it stands now.

    Recusa a snapshot identica a ultima: gerar o mesmo video duas vezes seguidas
    nao produziu version nenhuma, e uma lista de estados iguais nao ajuda
    ninguem a achar o "estava bom ontem".
    """
    if not m.data:
        return None
    ultima = max(m.versions, key=lambda v: _hora(v.created_at), default=None)
    if ultima is not None and ultima.data == m.data:
        return None

    # the timestamp comes from here, and not from the column's `default`:
    # without it the fresh snapshot joins the list with a null `created_at` and
    # cannot even be sorted
    snapshot = MontageVersion(label=label, data=dict(m.data), created_at=utcnow())
    m.versions.append(snapshot)
    velhas = sorted(m.versions, key=lambda v: _hora(v.created_at), reverse=True)
    for v in velhas[MAX_VERSOES:]:
        m.versions.remove(v)
    return snapshot


@app.get("/api/jobs/{job_id}/montages/{montage_id}/versions")
def list_versions(job_id: str, montage_id: str) -> dict[str, Any]:
    """This montage's snapshots, most recent first."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshots = sorted(m.versions, key=lambda v: _hora(v.created_at), reverse=True)
        return {"montage_id": montage_id, "items": [_version_dict(v) for v in snapshots]}


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions", status_code=201)
def create_version(
    job_id: str, montage_id: str, corpo: dict = Body(default={})
) -> dict[str, Any]:
    """Marks the montage as it stands: the "it was good like this"."""
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = _guardar_foto(m, str(corpo.get("label") or "marcada a mao"))
        if snapshot is None:
            raise HTTPException(409, "nao ha nada de novo para marcar")
        s.flush()
        return _version_dict(snapshot, full=True)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}/restore")
def restore_version(job_id: str, montage_id: str, version_id: str) -> dict[str, Any]:
    """Rolls the montage back to a snapshot.

    O estado de agora vira snapshot antes -- restaurar nunca apaga trabalho, so
    troca o que esta na frente.
    """
    with session() as s:
        m = _get_montage(s, job_id, montage_id)
        snapshot = s.get(MontageVersion, version_id)
        if snapshot is None or snapshot.montage_id != montage_id:
            raise HTTPException(404, "versao nao encontrada nesta montagem")
        _guardar_foto(m, "antes de restaurar")
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
            raise HTTPException(404, "versao nao encontrada nesta montagem")
        m.versions.remove(snapshot)
    return Response(status_code=204)


# ── predefinicoes ───────────────────────────────────────────────────────────


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
        raise HTTPException(422, f"receita invalida: {exc}") from exc


@app.get("/api/presets")
def list_presets() -> dict[str, Any]:
    """The presets. They belong to no match -- crossing from one to another is
    why they exist."""
    with session() as s:
        items = s.execute(select(Preset).order_by(Preset.created_at)).scalars().all()
        return {"items": [_preset_dict(p) for p in items]}


@app.post("/api/presets", status_code=201)
def create_preset(corpo: dict = Body(...)) -> dict[str, Any]:
    nome = str(corpo.get("name") or "").strip()
    if not nome:
        raise HTTPException(422, "uma predefinicao sem nome nao da para achar")
    receita = _validate_recipe(corpo.get("data") or {})
    with session() as s:
        existentes = s.execute(select(Preset)).scalars().all()
        p = Preset(
            name=_nome_livre(existentes, nome), data=receita.model_dump(mode="json")
        )
        s.add(p)
        s.flush()
        return _preset_dict(p)


@app.put("/api/presets/{preset_id}")
def update_preset(preset_id: str, corpo: dict = Body(...)) -> dict[str, Any]:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "predefinicao nao encontrada")
        if "data" in corpo:
            p.data = _validate_recipe(corpo["data"]).model_dump(mode="json")
        if "name" in corpo:
            nome = str(corpo["name"] or "").strip()
            if not nome:
                raise HTTPException(422, "uma predefinicao sem nome nao da para achar")
            outros = [
                o for o in s.execute(select(Preset)).scalars().all() if o.id != p.id
            ]
            p.name = _nome_livre(outros, nome)
        s.flush()
        return _preset_dict(p)


@app.delete("/api/presets/{preset_id}", status_code=204)
def delete_preset(preset_id: str) -> Response:
    with session() as s:
        p = s.get(Preset, preset_id)
        if p is None:
            raise HTTPException(404, "predefinicao nao encontrada")
        s.delete(p)
    return Response(status_code=204)


# -- the compiled Flutter app, when present (single-origin deploy) ----------
#
# With the app served by the gateway itself, Flutter calls the API on a
# relative path and the same build works on any host, with no recompiling and
# configurar CORS. Se ninguem rodou `flutter build web`, o mount simplesmente
# does not happen and the API stays available on its own. The check is on
# `index.html`, and not on the directory: in Docker the bind mount creates the
# folder empty even when nobody compiled the app, and mounting StaticFiles on
# transformaria a raiz do site em 404.

_web = Path(get_settings().web_dir)
if (_web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")


# O mount fica **por ultimo** de proposito: ele casa com qualquer caminho, e
# toda rota registrada depois dele fica inalcancavel -- um `POST` numa delas
# volta 405, porque quem responde e o servidor de arquivos.
