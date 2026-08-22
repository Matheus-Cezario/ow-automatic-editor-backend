"""API do sistema: upload, acompanhamento e entrega dos videos.

E o unico servico exposto ao mundo. Ele nao processa nada -- grava o arquivo,
cria o job e publica no barramento; o resto acontece nos workers.
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
    STREAM_RENDER,
    STREAM_THUMBS,
    Clip,
    ClipOptions,
    HighlightKind,
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
    Proposal,
    Render,
    RenderRequested,
    Receita,
    RenderStatus,
    Selection,
    ThumbsRequested,
    Timeline,
    TrackStatus,
    frame_key,
    new_id,
    utcnow,
)
from owcore.ffmpeg import probe
from owcore.storage import get_storage

#: propostas cujo video e montado em cortes -- so nelas a musica faz sentido.
#: Um trecho corrido da partida sai sempre com o audio original.
MONTAGE_KINDS = {
    HighlightKind.BEAT_MONTAGE,
    HighlightKind.ULT_MONTAGE,
    HighlightKind.SLEEP_MONTAGE,
    HighlightKind.STUN_MONTAGE,
}

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
CHUNK = 1024 * 256

#: A gravacao original tambem e servida ao app: e ela que o preview da tela de
#: montagem mostra, buscando o instante de cada bloco.
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

#: O player do app pede a musica por HTTP; sem o tipo certo alguns navegadores
#: se recusam a tocar (e sem tocar nao ha como posicionar corte nenhum).
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


def _lista_json(bruto: Any, campo: str) -> list:
    """Le um campo multipart que carrega uma lista JSON em texto."""
    if bruto is None or bruto == "":
        return []
    if not isinstance(bruto, str):
        raise HTTPException(422, f"'{campo}' tem de ser um JSON em texto")
    try:
        valor = json.loads(bruto)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"'{campo}' nao e JSON valido: {exc}") from exc
    if not isinstance(valor, list):
        raise HTTPException(422, f"'{campo}' tem de ser uma lista")
    return valor


def _job_dict(job: Job, *, full: bool = False) -> dict[str, Any]:
    tem_cortes = any(
        (c.meta or {}).get("segments_zip_key") for c in job.clips
    )
    sem_video = sum(1 for c in job.clips if not c.key)
    data = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": round(job.progress, 3),
        "error": job.error,
        "video_name": job.video_name,
        "duration_s": round(job.duration_s, 2),
        # `or 0` porque uma partida analisada antes desta coluna existir a le
        # como NULL ate o backfill passar por ela
        "fps": round(job.fps or 0.0, 3),
        "width": job.width or 0,
        "height": job.height or 0,
        "params": job.params,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "n_proposals": len(job.proposals),
        "n_renders": len(job.renders),
        # a listagem nao traz os pedidos inteiros, mas o app precisa saber se
        # vale continuar consultando
        "has_active_render": any(
            r.status in (RenderStatus.PENDING, RenderStatus.RENDERING)
            for r in job.renders
        ),
        "n_clips": len(job.clips),
        # a gravacao em si: o preview da montagem busca dentro dela
        "video_url": f"/api/jobs/{job.id}/video",
        # a copia reduzida, quando existir. Jobs analisados antes dela virem ao
        # mundo respondem `null`, e o app cai na gravacao original
        "proxy_url": f"/api/jobs/{job.id}/proxy" if job.proxy_key else None,
        # o pacote da partida inteira: nao exige abrir video nenhum
        "zip_url": f"/api/jobs/{job.id}/cortes.zip" if job.clips else None,
        "has_cuts": tem_cortes,
        #: clipes em que a montagem falhou mas os cortes sobreviveram
        "clips_only_cuts": sem_video,
    }
    if full:
        data["proposals"] = [
            _proposal_dict(p)
            for p in sorted(job.proposals, key=lambda p: (-p.score, p.start_s))
        ]
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
        biblioteca = sorted(job.media, key=lambda m: m.created_at)
        data["media"] = [_media_dict(m) for m in biblioteca]
        # `tracks` continua sendo so as musicas: e o que o seletor de trilha usa
        data["tracks"] = [_media_dict(m) for m in biblioteca if m.is_audio]
        # as montagens voltam com o job: e assim que a tela se reconstroi
        # depois de um F5, e e a lista que o seletor mostra
        montagens = sorted(job.montages, key=lambda m: _hora(m.updated_at), reverse=True)
        data["montages"] = [_montagem_dict(m, full=True) for m in montagens]
        # `draft` continua sendo a mais recente, para um app anterior a Fase 8
        data["draft"] = (montagens[0].data if montagens else job.draft) or {}
        # a onda do audio da partida, para a regua do editor. So no detalhe: sao
        # alguns milhares de numeros, e a listagem nao tem o que fazer com eles
        data["waveform"] = job.waveform or []
    return data


def _proposal_dict(p: Proposal) -> dict[str, Any]:
    """Um video que o sistema *pode* gerar. `accepts_music` diz se faz sentido
    oferecer uma trilha: montagens sao cortadas no ritmo, trechos corridos saem
    com o audio da partida."""
    return {
        "id": p.id,
        "job_id": p.job_id,
        "kind": p.kind,
        "title": p.title,
        "start_s": round(p.start_s, 2),
        "end_s": round(p.end_s, 2),
        "score": round(p.score, 2),
        "n_moments": len(p.moments or []),
        "moments": [round(float(t), 2) for t in (p.moments or [])],
        "accepts_music": p.kind in MONTAGE_KINDS,
        "meta": p.meta,
    }


def _render_dict(r: Render, todos_clipes: list[Clip]) -> dict[str, Any]:
    clipes = [c for c in todos_clipes if c.render_id == r.id]
    return {
        "id": r.id,
        "job_id": r.job_id,
        "status": r.status,
        "stage": r.stage,
        "progress": round(r.progress, 3),
        "error": r.error,
        "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(),
        "selections": [
            {
                "proposal_id": sel.get("proposal_id"),
                "music_name": sel.get("music_name"),
                "options": sel.get("options", {}),
            }
            for sel in (r.selections or [])
        ],
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
        "clips": [_clip_dict(c) for c in sorted(clipes, key=lambda c: -c.score)],
    }


def _clip_dict(c: Clip) -> dict[str, Any]:
    return {
        "id": c.id,
        "job_id": c.job_id,
        "render_id": c.render_id,
        "proposal_id": c.proposal_id,
        "kind": c.kind,
        "title": c.title,
        "start_s": round(c.start_s, 2),
        "end_s": round(c.end_s, 2),
        "score": round(c.score, 2),
        "meta": c.meta,
        # sem chave: a montagem falhou e so os cortes existem
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
    """Entrega um blob com suporte a Range, para o player poder buscar."""
    storage = get_storage()
    if not storage.exists(key):
        raise HTTPException(404, "arquivo nao encontrado")
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
    storage = get_storage()

    video_ext = _safe_suffix(video.filename, VIDEO_EXTS, ".mp4")
    video_key = storage.put_stream(f"{job_id}/source{video_ext}", video.file)

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
    """Segunda fase: gerar os videos escolhidos.

    Recebe um multipart com o campo `selections` (JSON) e, opcionalmente, um
    arquivo de musica por escolha, no campo `music_<proposal_id>` -- e assim que
    cada video ganha a sua trilha. Escolha sem musica vira video com o **audio
    original** da partida.

    O campo `timelines` (JSON) traz os videos que o usuario montou a mao: cada
    um com os seus blocos ja posicionados e, se quiser trilha, o `track_id` de
    uma musica ja enviada a este job. Os dois campos convivem no mesmo pedido --
    da para gerar uma proposta pronta e uma montagem manual de uma vez.

    Pode ser chamado quantas vezes se quiser sobre o mesmo job: as propostas
    continuam la, e usar um momento num video nao o consome para os outros.
    """
    form = await request.form()
    cru = _lista_json(form.get("selections"), "selections")
    cru_timelines = _lista_json(form.get("timelines"), "timelines")
    if not cru and not cru_timelines:
        raise HTTPException(
            422, "escolha pelo menos um video ou monte uma linha do tempo"
        )

    storage = get_storage()
    render_id = new_id()

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        if job.status != JobStatus.READY:
            raise HTTPException(
                409, f"a analise deste job ainda nao terminou (status: {job.status})"
            )
        validos = {p.id: p for p in job.proposals}
        # a montagem aponta para uma musica; a biblioteca guarda mais que isso
        musicas = {m.id: m.status for m in job.media if m.is_audio}
        biblioteca = {m.id for m in job.media}
        sons = {m.id for m in job.media if m.is_audio}

    escolhas: list[Selection] = []
    vistos: set[str] = set()
    for item in cru:
        if not isinstance(item, dict):
            raise HTTPException(422, "cada escolha tem de ser um objeto")
        pid = str(item.get("proposal_id") or "")
        if pid not in validos:
            raise HTTPException(422, f"proposta desconhecida: {pid!r}")
        if pid in vistos:
            raise HTTPException(422, f"proposta escolhida duas vezes: {pid!r}")
        vistos.add(pid)
        try:
            opcoes = ClipOptions(**(item.get("options") or {}))
        except ValidationError as exc:
            raise HTTPException(422, f"opcoes invalidas em {pid}: {exc}") from exc
        escolhas.append(Selection(proposal_id=pid, options=opcoes))

    # a musica vem num campo por escolha; guarda-se uma copia por pedido para
    # que apagar um pedido nao leve embora a trilha de outro
    for sel in escolhas:
        arquivo = form.get(f"music_{sel.proposal_id}")
        if arquivo is None or not getattr(arquivo, "filename", ""):
            continue
        if validos[sel.proposal_id].kind not in MONTAGE_KINDS:
            raise HTTPException(
                422,
                f"a proposta {sel.proposal_id} e um trecho corrido da partida e "
                "sai com o audio original; ela nao aceita trilha",
            )
        ext = _safe_suffix(arquivo.filename, AUDIO_EXTS, ".mp3")
        sel.music_key = storage.put_stream(
            f"{job_id}/music/{render_id}_{sel.proposal_id}{ext}", arquivo.file
        )
        sel.music_name = arquivo.filename

    montagens: list[Timeline] = []
    for item in cru_timelines:
        if not isinstance(item, dict):
            raise HTTPException(422, "cada linha do tempo tem de ser um objeto")
        try:
            spec = Timeline(**item)
        except ValidationError as exc:
            raise HTTPException(422, f"linha do tempo invalida: {exc}") from exc
        if spec.track_id is not None:
            if spec.track_id not in musicas:
                raise HTTPException(
                    422, f"musica desconhecida neste job: {spec.track_id!r}"
                )
            # sem a analise pronta nao ha batida nem duracao; e tambem sinal de
            # que o app mandou antes da hora
            if musicas[spec.track_id] != TrackStatus.READY:
                raise HTTPException(
                    409,
                    f"a musica {spec.track_id} ainda nao foi analisada "
                    f"(status: {musicas[spec.track_id]})",
                )
        # um clipe que aponta para midia de outro job nao entra: a montagem
        # sairia sem ele, e sem aviso
        for clip in spec.clips:
            if clip.media_id and clip.media_id not in biblioteca:
                raise HTTPException(
                    422,
                    f"midia desconhecida neste job: {clip.media_id!r}",
                )
        _conferir_as_camadas(spec, sons)
        # a marca d'agua vem da mesma biblioteca, e recusa-la aqui e melhor do
        # que deixar a renderizacao inteira falhar la na frente por causa dela
        if spec.export.watermark_id and spec.export.watermark_id not in biblioteca:
            raise HTTPException(
                422,
                f"marca desconhecida neste job: {spec.export.watermark_id!r}",
            )
        montagens.append(spec)

    with session() as s:
        s.add(
            Render(
                id=render_id,
                job_id=job_id,
                status=RenderStatus.PENDING,
                stage="na fila",
                selections=[sel.model_dump() for sel in escolhas],
                timelines=[m.model_dump() for m in montagens],
            )
        )

    get_bus().publish(STREAM_RENDER, RenderRequested(render_id=render_id).model_dump())
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
    """Apaga um pedido e os videos dele. As propostas ficam: da para pedir de
    novo, com outra musica."""
    with session() as s:
        pedido = s.get(Render, render_id)
        if pedido is None:
            raise HTTPException(404, "pedido nao encontrado")
        s.delete(pedido)
    return Response(status_code=204)


# ── musicas do job: sobem antes de existir video ────────────────────────────
#
# Na montagem manual a musica vem primeiro. Nao da para posicionar um corte
# "na virada do refrao" sem ouvir o refrao, e nao da para grudar um corte na
# batida sem saber onde as batidas estao. Entao ela sobe, o sistema ouve, e o
# app recebe duracao, BPM, batidas e forma de onda para desenhar a regua.
#
# A musica e do **job**, e nao de um pedido: a mesma trilha serve a quantas
# montagens o usuario quiser fazer daquela partida, sem subir de novo.


def _media_dict(m: Media) -> dict[str, Any]:
    """Um item da biblioteca de midia da partida.

    Audio vai completo -- batidas e forma de onda inclusive --, porque e com
    isso que a tela de montagem desenha a musica e gruda os cortes na batida.
    Video e imagem vao com dimensoes e os enderecos da miniatura e do proxy.
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
        "created_at": m.created_at.isoformat(),
    }
    if m.is_audio:
        dados |= {
            "bpm": round(m.bpm, 2),
            "beats": m.beats or [],
            "peaks": m.peaks or [],
            # o app ainda pede a musica por aqui
            "audio_url": f"/api/media/{m.id}/file",
        }
    else:
        dados |= {"width": m.width, "height": m.height, "fps": round(m.fps, 3)}
    return dados


def _guardar_media(job_id: str, arquivo: UploadFile, kind: MediaKind) -> str:
    """Grava o arquivo e manda analisa-lo. Devolve o id."""
    media_id = new_id()
    padrao = {
        MediaKind.AUDIO: (AUDIO_EXTS, ".mp3"),
        MediaKind.VIDEO: (VIDEO_EXTS, ".mp4"),
        MediaKind.IMAGE: (IMAGE_EXTS, ".png"),
    }[kind]
    ext = _safe_suffix(arquivo.filename, *padrao)
    key = get_storage().put_stream(f"{job_id}/media/{media_id}{ext}", arquivo.file)

    with session() as s:
        s.add(
            Media(
                id=media_id,
                job_id=job_id,
                kind=kind,
                status=TrackStatus.PENDING,
                name=arquivo.filename or "arquivo",
                key=key,
            )
        )
    get_bus().publish(STREAM_MEDIA, MediaUploaded(media_id=media_id).model_dump())
    return media_id


def _tipo_de(filename: str | None) -> MediaKind | None:
    """De que tipo e o arquivo, pela extensao.

    Pela extensao e nao pelo `content-type` porque o navegador mente com
    frequencia -- manda `application/octet-stream` para tudo quando o arquivo
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
) -> dict[str, Any]:
    """Traz um arquivo para a biblioteca da partida.

    Responde na hora, com o item ainda `pending`: a analise (dimensoes,
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
    media_id = _guardar_media(job_id, file, kind)
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
    """Tira o item da biblioteca. Os videos ja gerados com ele ficam: o mp4
    final ja tem o que precisava dentro."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
        s.delete(item)
    return Response(status_code=204)


@app.get("/api/media/{media_id}/file")
def media_file(media_id: str, request: Request) -> Response:
    """O arquivo em si, com `Range`."""
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
    """A copia reduzida do video importado -- o que o monitor abre."""
    with session() as s:
        item = s.get(Media, media_id)
        if item is None:
            raise HTTPException(404, "midia nao encontrada")
        key = item.proxy_key
    if not key:
        raise HTTPException(404, "este item nao tem proxy")
    return _serve_blob(key, request, "video/mp4")


# ── as rotas de musica, agora cascas sobre a biblioteca ─────────────────────
#
# Elas continuam existindo porque o app as usa e porque "a musica do job" e um
# nome util. Por baixo e tudo `Media` de tipo audio.


@app.post("/api/jobs/{job_id}/tracks", status_code=201)
def add_track(
    job_id: str,
    audio: UploadFile = File(..., description="musica para montar em cima"),
) -> dict[str, Any]:
    """Envia uma musica e manda o sistema ouvi-la."""
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job nao encontrado")
    media_id = _guardar_media(job_id, audio, MediaKind.AUDIO)
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


def _remendar_o_tamanho(job: Job) -> None:
    """Descobre o tamanho de uma gravacao analisada antes desta coluna existir.

    O reconciliador de esquema poe a coluna, mas nao tem como saber o que ela
    deveria valer -- so o arquivo sabe. Uma partida antiga abriria o editor sem
    poder dizer se um 9:16 corta o quadro dela. Custa um `ffprobe`, uma vez na
    vida de cada job. O ffmpeg le o arquivo onde ele esta -- por `Range`, se
    estiver no S3 --, entao medir uma gravacao de dois gigas custa o cabecalho
    dela, e nao os dois gigas.
    """
    if job.width or not job.video_key:
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
    """A gravacao original, com `Range`.

    E o que o preview da tela de montagem toca: em vez de renderizar o video a
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
    """Guarda a montagem em andamento. **Legado da V1.**

    Desde a Fase 8 uma partida tem varias montagens nomeadas, e o app salva pelo
    id de uma delas. Esta rota escreve na mais recente -- e cria a primeira, se
    nao houver nenhuma --, para que um app anterior continue funcionando em vez
    de perder o trabalho em silencio.
    """
    rascunho = _validar_montagem(draft)

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
    """Joga as montagens desta partida fora e comeca do zero. **Legado da V1.**"""
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
    """A copia reduzida da gravacao, com `Range`.

    E o que o monitor do editor abre. A gravacao original tem centenas de
    megabytes, e buscar dentro dela dezenas de vezes por segundo enquanto se
    arrasta chegou a derrubar o elemento de video do navegador. Esta copia sai
    da mesma decodificacao dos recortes -- custa quase nada -- e o corte final
    continua vindo do arquivo original.
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
    """Um quadro da partida no instante `t`, para a barra lateral do editor.

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
    """Manda extrair as miniaturas que faltam.

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
    """Tudo o que a partida gerou, num arquivo so.

    Montado na hora a partir do que ja esta no storage -- os videos finais e os
    cortes avulsos de cada montagem --, em vez de guardar um terceiro pacote
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
        nome_base = Path(job.video_name or "partida").stem
        # um job rende varios pedidos ao longo do tempo; o pacote traz todos,
        # cada um na sua pasta, para o mesmo tipo de video gerado duas vezes
        # com musicas diferentes nao se sobrescrever
        ordem = {r.id: n for n, r in enumerate(
            sorted(job.renders, key=lambda r: r.created_at), start=1
        )}
        itens = [
            (i, ordem.get(c.render_id, 0), c.kind, c.key,
             (c.meta or {}).get("segments_zip_key"))
            for i, c in enumerate(
                sorted(job.clips, key=lambda c: (ordem.get(c.render_id, 0), -c.score)),
                start=1,
            )
        ]

    tmp = Path(tempfile.mkdtemp(prefix="owzip-"))
    pacote = tmp / "pacote.zip"
    try:
        with zipfile.ZipFile(pacote, "w", zipfile.ZIP_STORED) as zf:
            for i, n_pedido, kind, video_key, cortes_key in itens:
                pasta = f"pedido_{n_pedido:02d}" if n_pedido else "videos"
                if video_key and storage.exists(video_key):
                    local = storage.get_file(video_key, tmp / f"v{i}.mp4")
                    zf.write(local, f"{pasta}/videos/{i:02d}_{kind}.mp4")
                    local.unlink(missing_ok=True)
                if not cortes_key or not storage.exists(cortes_key):
                    continue
                local = storage.get_file(cortes_key, tmp / f"c{i}.zip")
                with zipfile.ZipFile(local) as origem:
                    for nome in origem.namelist():
                        zf.writestr(f"{pasta}/cortes/{i:02d}_{kind}/{nome}",
                                    origem.read(nome))
                local.unlink(missing_ok=True)

        dados = pacote.read_bytes()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return Response(
        content=dados,
        media_type="application/zip",
        headers={
            "content-disposition": f'attachment; filename="{nome_base}_cortes.zip"',
            "content-length": str(len(dados)),
        },
    )


@app.get("/api/clips/{clip_id}/cortes.zip")
def clip_segments_zip(clip_id: str, request: Request) -> Response:
    """Os cortes individuais da montagem, num zip.

    Serve para reeditar por fora: cada arquivo e um trecho, nomeado pelo
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
    """Expoe o profile da HUD para a tela de calibracao do app."""
    from owcore.profiles import load_profile

    return load_profile(get_settings().profile).data


# ── montagens nomeadas ──────────────────────────────────────────────────────


def _hora(t: datetime | None) -> datetime:
    """A hora de um registro, sempre com fuso.

    O SQLite guarda `datetime` sem fuso, entao uma linha lida do banco volta
    ingenua enquanto uma criada nesta mesma requisicao ainda esta com o fuso que
    `utcnow()` deu. Ordenar as duas juntas estoura -- e e exatamente o que
    acontece ao listar as montagens logo depois de criar uma.
    """
    if t is None:
        return utcnow()
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _montagem_dict(m: MontageModel, *, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": m.id,
        "job_id": m.job_id,
        "name": m.name,
        "created_at": m.created_at.isoformat(),
        "updated_at": m.updated_at.isoformat(),
        "n_versions": len(m.versions),
        **m.resumo,
    }
    if full:
        d["data"] = m.data or {}
    return d


def _adotar_o_rascunho_antigo(s: Any, job: Job) -> None:
    """Traz a montagem unica do job para a lista de montagens nomeadas.

    Ate a Fase 8 havia uma so, numa coluna do proprio job. Fazer isto na
    leitura, e nao numa migracao de banco, e a mesma escolha do resto do
    sistema: quem sabe converter o formato velho e o codigo que le, e assim uma
    partida parada ha meses continua abrindo.
    """
    if not job.draft or job.montages:
        return
    # pela relacao, e nao por `s.add`: assim `job.montages` ja a enxerga nesta
    # mesma requisicao, que e onde ela precisa aparecer
    job.montages.append(
        MontageModel(name=job.draft.get("title") or "Montagem 1", data=job.draft)
    )
    # a coluna some para nao haver duas verdades sobre a mesma montagem
    job.draft = {}
    s.flush()


def _nome_livre(existentes: Sequence[Any], base: str) -> str:
    """Um nome que ainda nao esta na lista.

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


def _pegar_montagem(s: Any, job_id: str, montage_id: str) -> MontageModel:
    m = s.get(MontageModel, montage_id)
    if m is None or m.job_id != job_id:
        raise HTTPException(404, "montagem nao encontrada nesta partida")
    return m


def _conferir_as_camadas(spec: Any, sons: set[str]) -> None:
    """Uma camada desenha ou toca -- e o conteudo tem de ser do tipo dela.

    Uma musica numa camada de video faria o ffmpeg tentar redimensionar um fluxo
    de audio, e o render inteiro morreria com uma mensagem que nao explica nada.
    Uma imagem numa camada de audio seria pior: ela nao tem som, entao sairia em
    silencio, sem erro nenhum, e o usuario procuraria o problema na mixagem.
    """
    for camada in spec.layers:
        for clip in camada.clips:
            e_som = bool(clip.media_id) and clip.media_id in sons
            if camada.e_audio and not e_som:
                raise HTTPException(
                    422,
                    "uma camada de audio so aceita musica da biblioteca",
                )
            if not camada.e_audio and e_som:
                raise HTTPException(
                    422,
                    f"a midia {clip.media_id!r} e som: ela vai numa camada "
                    "de audio",
                )


def _validar_montagem(data: dict) -> MontageDraft:
    try:
        return MontageDraft(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"montagem invalida: {exc}") from exc


@app.get("/api/jobs/{job_id}/montages")
def list_montages(job_id: str) -> dict[str, Any]:
    """As montagens desta partida, da mais recente para a mais antiga."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        _adotar_o_rascunho_antigo(s, job)
        montagens = sorted(job.montages, key=lambda m: _hora(m.updated_at), reverse=True)
        return {
            "job_id": job_id,
            "items": [_montagem_dict(m, full=True) for m in montagens],
        }


@app.post("/api/jobs/{job_id}/montages", status_code=201)
def create_montage(job_id: str, corpo: dict = Body(default={})) -> dict[str, Any]:
    """Comeca uma montagem nova, vazia ou a partir de um conteudo dado.

    Sao trabalhos diferentes sobre o mesmo material -- o corte de 30 s para o
    Shorts e a montagem longa --, e ate aqui era preciso escolher um.
    """
    dados = corpo.get("data") or {}
    _validar_montagem(dados)
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
        return _montagem_dict(m, full=True)


@app.put("/api/jobs/{job_id}/montages/{montage_id}")
def save_montage(
    job_id: str, montage_id: str, corpo: dict = Body(...)
) -> dict[str, Any]:
    """Guarda a montagem. E o que o app chama sozinho enquanto se edita.

    Um corte invalido e recusado aqui: guardar lixo agora seria devolver lixo na
    proxima abertura.
    """
    with session() as s:
        m = _pegar_montagem(s, job_id, montage_id)
        if "data" in corpo:
            rascunho = _validar_montagem(corpo["data"])
            m.data = rascunho.model_dump()
        if "name" in corpo:
            nome = str(corpo["name"] or "").strip()
            if not nome:
                raise HTTPException(422, "uma montagem sem nome nao da para achar")
            outras = [o for o in m.job.montages if o.id != m.id]
            m.name = _nome_livre(outras, nome)
        s.flush()
        return _montagem_dict(m)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/duplicate", status_code=201)
def duplicate_montage(job_id: str, montage_id: str) -> dict[str, Any]:
    """Uma copia, para experimentar sem arriscar a que ja esta boa.

    A copia nao leva o historico da original: as fotos dizem por onde *aquela*
    montagem passou, e a copia ainda nao passou por lugar nenhum.
    """
    with session() as s:
        original = _pegar_montagem(s, job_id, montage_id)
        copia = MontageModel(
            job_id=job_id,
            name=_nome_livre(original.job.montages, f"{original.name} (copia)"),
            data=dict(original.data or {}),
        )
        s.add(copia)
        s.flush()
        return _montagem_dict(copia, full=True)


@app.delete("/api/jobs/{job_id}/montages/{montage_id}", status_code=204)
def delete_montage(job_id: str, montage_id: str) -> Response:
    with session() as s:
        s.delete(_pegar_montagem(s, job_id, montage_id))
    return Response(status_code=204)


# ── historico de versoes ────────────────────────────────────────────────────

#: quantas fotos cada montagem guarda. Passou disso, a mais velha sai.
#:
#: Nao e o desfazer -- esse vive no app. Sao marcos, e vinte marcos ja e mais
#: historia do que alguem percorre numa lista.
MAX_VERSOES = 20


def _versao_dict(v: MontageVersion, *, full: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": v.id,
        "montage_id": v.montage_id,
        "label": v.label,
        "created_at": v.created_at.isoformat(),
        **MontageModel(data=v.data).resumo,
    }
    if full:
        d["data"] = v.data or {}
    return d


def _guardar_foto(m: MontageModel, label: str) -> MontageVersion | None:
    """Tira uma foto da montagem como ela esta agora.

    Recusa a foto identica a ultima: gerar o mesmo video duas vezes seguidas
    nao produziu versao nenhuma, e uma lista de estados iguais nao ajuda
    ninguem a achar o "estava bom ontem".
    """
    if not m.data:
        return None
    ultima = max(m.versions, key=lambda v: _hora(v.created_at), default=None)
    if ultima is not None and ultima.data == m.data:
        return None

    # a hora vem daqui, e nao do `default` da coluna: sem ela a foto recem-feita
    # entra na lista com `created_at` nulo e nao da nem para ordenar
    foto = MontageVersion(label=label, data=dict(m.data), created_at=utcnow())
    m.versions.append(foto)
    velhas = sorted(m.versions, key=lambda v: _hora(v.created_at), reverse=True)
    for v in velhas[MAX_VERSOES:]:
        m.versions.remove(v)
    return foto


@app.get("/api/jobs/{job_id}/montages/{montage_id}/versions")
def list_versions(job_id: str, montage_id: str) -> dict[str, Any]:
    """As fotos desta montagem, da mais recente para a mais antiga."""
    with session() as s:
        m = _pegar_montagem(s, job_id, montage_id)
        fotos = sorted(m.versions, key=lambda v: _hora(v.created_at), reverse=True)
        return {"montage_id": montage_id, "items": [_versao_dict(v) for v in fotos]}


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions", status_code=201)
def create_version(
    job_id: str, montage_id: str, corpo: dict = Body(default={})
) -> dict[str, Any]:
    """Marca a montagem como ela esta: o "estava bom assim"."""
    with session() as s:
        m = _pegar_montagem(s, job_id, montage_id)
        foto = _guardar_foto(m, str(corpo.get("label") or "marcada a mao"))
        if foto is None:
            raise HTTPException(409, "nao ha nada de novo para marcar")
        s.flush()
        return _versao_dict(foto, full=True)


@app.post("/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}/restore")
def restore_version(job_id: str, montage_id: str, version_id: str) -> dict[str, Any]:
    """Volta a montagem para uma foto.

    O estado de agora vira foto antes -- restaurar nunca apaga trabalho, so
    troca o que esta na frente.
    """
    with session() as s:
        m = _pegar_montagem(s, job_id, montage_id)
        foto = s.get(MontageVersion, version_id)
        if foto is None or foto.montage_id != montage_id:
            raise HTTPException(404, "versao nao encontrada nesta montagem")
        _guardar_foto(m, "antes de restaurar")
        m.data = dict(foto.data or {})
        s.flush()
        return _montagem_dict(m, full=True)


@app.delete(
    "/api/jobs/{job_id}/montages/{montage_id}/versions/{version_id}", status_code=204
)
def delete_version(job_id: str, montage_id: str, version_id: str) -> Response:
    with session() as s:
        m = _pegar_montagem(s, job_id, montage_id)
        foto = s.get(MontageVersion, version_id)
        if foto is None or foto.montage_id != montage_id:
            raise HTTPException(404, "versao nao encontrada nesta montagem")
        m.versions.remove(foto)
    return Response(status_code=204)


# ── predefinicoes ───────────────────────────────────────────────────────────


def _preset_dict(p: Preset) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "data": p.data or {},
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def _validar_receita(data: dict) -> Receita:
    try:
        return Receita(**(data or {}))
    except ValidationError as exc:
        raise HTTPException(422, f"receita invalida: {exc}") from exc


@app.get("/api/presets")
def list_presets() -> dict[str, Any]:
    """As predefinicoes. Nao pertencem a partida nenhuma -- e para atravessar de
    uma para outra que elas existem."""
    with session() as s:
        itens = s.execute(select(Preset).order_by(Preset.created_at)).scalars().all()
        return {"items": [_preset_dict(p) for p in itens]}


@app.post("/api/presets", status_code=201)
def create_preset(corpo: dict = Body(...)) -> dict[str, Any]:
    nome = str(corpo.get("name") or "").strip()
    if not nome:
        raise HTTPException(422, "uma predefinicao sem nome nao da para achar")
    receita = _validar_receita(corpo.get("data") or {})
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
            p.data = _validar_receita(corpo["data"]).model_dump(mode="json")
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


# ── app Flutter compilado, quando existir (deploy de origem unica) ──────────
#
# Com o app servido pelo proprio gateway, o Flutter chama a API em caminho
# relativo e a mesma build funciona em qualquer host, sem recompilar nem
# configurar CORS. Se ninguem rodou `flutter build web`, o mount simplesmente
# nao acontece e a API segue disponivel sozinha. A checagem e pelo
# `index.html`, e nao pelo diretorio: no Docker o bind mount cria a pasta
# vazia mesmo quando ninguem compilou o app, e montar StaticFiles nela
# transformaria a raiz do site em 404.

_web = Path(get_settings().web_dir)
if (_web / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")


# O mount fica **por ultimo** de proposito: ele casa com qualquer caminho, e
# toda rota registrada depois dele fica inalcancavel -- um `POST` numa delas
# volta 405, porque quem responde e o servidor de arquivos.
