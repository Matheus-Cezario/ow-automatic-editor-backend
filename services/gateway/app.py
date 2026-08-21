"""API do sistema: upload, acompanhamento e entrega dos videos.

E o unico servico exposto ao mundo. Ele nao processa nada -- grava o arquivo,
cria o job e publica no barramento; o resto acontece nos workers.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

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
    STREAM_RENDER,
    STREAM_THUMBS,
    STREAM_TRACK,
    Clip,
    ClipOptions,
    HighlightKind,
    Job,
    JobCreated,
    JobParams,
    JobStatus,
    MontageDraft,
    Proposal,
    Render,
    RenderRequested,
    RenderStatus,
    Selection,
    ThumbsRequested,
    Timeline,
    Track,
    TrackStatus,
    TrackUploaded,
    frame_key,
    new_id,
)
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
        data["tracks"] = [
            _track_dict(t) for t in sorted(job.tracks, key=lambda t: t.created_at)
        ]
        # a montagem em andamento volta com o job: e assim que a tela de
        # montagem se reconstroi depois de um F5
        data["draft"] = job.draft or {}
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


def _track_dict(t: Track) -> dict[str, Any]:
    """Uma musica ja ouvida pelo sistema.

    Vai completa para o app -- batidas e forma de onda inclusive -- porque e
    com isso que a tela de montagem desenha a musica e gruda os cortes na
    batida. Sao alguns milhares de numeros, na ordem de dezenas de KB: menos do
    que uma miniatura, e evita o app baixar o audio so para desenhar.
    """
    return {
        "id": t.id,
        "job_id": t.job_id,
        "status": t.status,
        "error": t.error,
        "name": t.name,
        "duration_s": round(t.duration_s, 3),
        "bpm": round(t.bpm, 2),
        "beats": t.beats or [],
        "peaks": t.peaks or [],
        "audio_url": f"/api/tracks/{t.id}/audio",
        "created_at": t.created_at.isoformat(),
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
                "n_cuts": len(tl.get("cuts") or []),
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
        musicas = {t.id: t.status for t in job.tracks}

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


@app.post("/api/jobs/{job_id}/tracks", status_code=201)
def add_track(
    job_id: str,
    audio: UploadFile = File(..., description="musica para montar em cima"),
) -> dict[str, Any]:
    """Envia uma musica e manda o sistema ouvi-la.

    Responde na hora, com a musica ainda `pending`: a analise roda no worker de
    ritmo e o app acompanha por `GET /api/tracks/{id}`.
    """
    with session() as s:
        if s.get(Job, job_id) is None:
            raise HTTPException(404, "job nao encontrado")

    track_id = new_id()
    ext = _safe_suffix(audio.filename, AUDIO_EXTS, ".mp3")
    key = get_storage().put_stream(f"{job_id}/tracks/{track_id}{ext}", audio.file)

    with session() as s:
        s.add(
            Track(
                id=track_id,
                job_id=job_id,
                status=TrackStatus.PENDING,
                name=audio.filename or "musica",
                key=key,
            )
        )

    get_bus().publish(STREAM_TRACK, TrackUploaded(track_id=track_id).model_dump())
    return {"id": track_id, "job_id": job_id, "status": TrackStatus.PENDING}


@app.get("/api/tracks/{track_id}")
def get_track(track_id: str) -> dict[str, Any]:
    with session() as s:
        track = s.get(Track, track_id)
        if track is None:
            raise HTTPException(404, "musica nao encontrada")
        return _track_dict(track)


@app.get("/api/tracks/{track_id}/audio")
def track_audio(track_id: str, request: Request) -> Response:
    """O arquivo em si, com Range -- e o que o player do app toca enquanto o
    usuario arrasta os cortes."""
    with session() as s:
        track = s.get(Track, track_id)
        if track is None:
            raise HTTPException(404, "musica nao encontrada")
        key = track.key
    mime = AUDIO_MIME.get(Path(key).suffix.lower(), "audio/mpeg")
    return _serve_blob(key, request, mime)


@app.delete("/api/tracks/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    """Tira a musica do job. Os videos ja gerados com ela ficam: o mp4 final ja
    tem a trilha dentro."""
    with session() as s:
        track = s.get(Track, track_id)
        if track is None:
            raise HTTPException(404, "musica nao encontrada")
        s.delete(track)
    return Response(status_code=204)


@app.get("/api/jobs")
def list_jobs(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    with session() as s:
        jobs = s.scalars(
            select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return {"jobs": [_job_dict(j) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
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
    """Guarda a montagem em andamento.

    O app chama sozinho enquanto o usuario edita. E um rascunho, entao aceita
    zero cortes -- mas cada corte e validado, porque guardar lixo agora seria
    devolver lixo na proxima abertura.
    """
    try:
        rascunho = MontageDraft(**draft)
    except ValidationError as exc:
        raise HTTPException(422, f"rascunho invalido: {exc}") from exc

    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        job.draft = rascunho.model_dump()
    return {"job_id": job_id, "n_cuts": len(rascunho.cuts)}


@app.delete("/api/jobs/{job_id}/draft", status_code=204)
def delete_draft(job_id: str) -> Response:
    """Joga a montagem em andamento fora e comeca do zero."""
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job nao encontrado")
        job.draft = {}
    return Response(status_code=204)


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
