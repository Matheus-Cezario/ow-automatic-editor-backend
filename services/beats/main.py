"""Microsservico de analise ritmica.

Entra na **segunda fase**: quando o usuario pede a geracao, cada video escolhido
pode vir com uma musica diferente. Este servico analisa cada uma, guarda a
grade de batidas no pedido e so entao libera o editor.

Videos sem musica passam direto: nao ha ritmo a extrair, e o corte sai com o
audio original da partida.

O mesmo processo ouve dois streams. O segundo e o da **midia recem-enviada** --
musica, clipe ou imagem que o usuario trouxe. Ela sobe antes de existir video
nenhum, porque e olhando para ela (as batidas de uma musica, a miniatura de um
clipe) que o usuario decide o que montar.

Sao dois lacos porque sao dois momentos do fluxo. A analise de audio e a mesma
dos dois lados, e a de video e imagem cabe aqui porque este processo ja tem
ffmpeg -- um servico a mais no compose custaria uma imagem inteira para trinta
linhas de trabalho.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import fail_render, set_render_status
from owcore.models import (
    STREAM_MEDIA,
    STREAM_RENDER,
    STREAM_RENDER_READY,
    Media,
    MediaKind,
    Render,
    RenderRequested,
    RenderStatus,
    Selection,
    TrackStatus,
)
from owcore import ffmpeg
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

from detect import analyze_music, analyze_track


class BeatsService(Worker):
    name = "beats"
    stream = STREAM_RENDER
    group = "beats"

    def handle(self, payload: dict[str, Any]) -> None:
        render_id = payload["render_id"]

        with session() as s:
            render = s.get(Render, render_id)
            if render is None:
                self.log.warning("pedido %s sumiu; ignorando", render_id)
                return
            job_id = render.job_id
            selections = [Selection(**d) for d in (render.selections or [])]

        com_musica = [sel for sel in selections if sel.music_key]
        if com_musica:
            set_render_status(
                render_id, RenderStatus.RENDERING,
                stage=f"ouvindo {len(com_musica)} musica(s)", progress=0.05,
            )

        work = Path(get_settings().work_dir) / job_id / "beats" / render_id
        work.mkdir(parents=True, exist_ok=True)

        # duas escolhas podem apontar para o mesmo arquivo; analisa uma vez
        por_arquivo: dict[str, dict] = {}
        grids: dict[str, dict] = {}
        for sel in com_musica:
            assert sel.music_key is not None
            if sel.music_key not in por_arquivo:
                por_arquivo[sel.music_key] = self._analyze(sel.music_key, work)
            grid = por_arquivo[sel.music_key]
            if grid:
                grids[sel.proposal_id] = grid

        with session() as s:
            render = s.get(Render, render_id)
            if render is None:
                return
            render.beats = grids

        get_bus().publish(
            STREAM_RENDER_READY, RenderRequested(render_id=render_id).model_dump()
        )
        self.log.info(
            "pedido %s: %d musica(s) analisada(s), editor avisado",
            render_id, len(grids),
        )

    def _analyze(self, music_key: str, work: Path) -> dict:
        """Uma musica ilegivel nao cancela a geracao: o video sai com corte de
        duracao fixa em vez de nao sair."""
        try:
            music = local_copy(music_key, work)
            grid = analyze_music(music, work)
        except Exception as exc:
            self.log.warning("musica %s nao pode ser analisada: %s", music_key, exc)
            return {}
        self.log.info("%s: %.1f BPM, %d batidas", music_key, grid.bpm, len(grid.beats))
        return grid.model_dump()

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        render_id = payload.get("render_id")
        if render_id:
            fail_render(render_id, f"{self.name}: {exc}")


class MediaAnalyzer(Worker):
    """Olha o arquivo que o usuario acabou de trazer.

    O resultado nao gera video nenhum: ele volta para a tela de montagem, que
    precisa saber o que fazer com o arquivo antes de deixar montar com ele.

    O que cada tipo ganha:

    * **audio** -- duracao, BPM, batidas e forma de onda. E o que a regua
      desenha e o que o ima usa;
    * **video** -- duracao, dimensoes, fps, uma miniatura e um **proxy**, pelo
      mesmo motivo do proxy da gravacao: o monitor nao pode arrastar o arquivo
      cheio a cada busca;
    * **imagem** -- dimensoes e uma miniatura.
    """

    name = "media"
    stream = STREAM_MEDIA
    group = "media"

    def handle(self, payload: dict[str, Any]) -> None:
        media_id = payload["media_id"]

        with session() as s:
            item = s.get(Media, media_id)
            if item is None:
                self.log.warning("midia %s sumiu; ignorando", media_id)
                return
            if item.status == TrackStatus.READY:
                self.log.info("midia %s ja analisada; ignorando", media_id)
                return
            job_id, key, kind = item.job_id, item.key, item.kind

        work = Path(get_settings().work_dir) / job_id / "media" / media_id
        work.mkdir(parents=True, exist_ok=True)

        try:
            local = local_copy(key, work)
            campos = (
                self._audio(local, work)
                if kind == MediaKind.AUDIO
                else self._visual(local, work, job_id, media_id, kind)
            )
        except Exception as exc:
            self._falhou(media_id, str(exc))
            raise

        with session() as s:
            item = s.get(Media, media_id)
            if item is None:
                return
            item.status = TrackStatus.READY
            item.error = None
            for campo, valor in campos.items():
                setattr(item, campo, valor)

        self.log.info("midia %s (%s) analisada: %s", media_id, kind,
                      ", ".join(f"{k}={v}" for k, v in campos.items()
                                if k not in ("beats", "peaks")))

    # ── por tipo ────────────────────────────────────────────────────────────

    def _audio(self, local: Path, work: Path) -> dict[str, Any]:
        analise = analyze_track(local, work)
        return {
            "duration_s": analise.duration_s,
            "bpm": analise.grid.bpm,
            "beats": analise.grid.beats,
            "peaks": analise.peaks,
        }

    def _visual(
        self, local: Path, work: Path, job_id: str, media_id: str, kind: str
    ) -> dict[str, Any]:
        storage = get_storage()
        info = ffmpeg.probe(local)
        campos: dict[str, Any] = {
            "width": info.width,
            "height": info.height,
            # imagem nao tem duracao: quanto ela fica na tela e escolha da
            # montagem, nao propriedade do arquivo
            "duration_s": info.duration_s if kind == MediaKind.VIDEO else 0.0,
            "fps": info.fps if kind == MediaKind.VIDEO else 0.0,
        }

        try:
            thumb = ffmpeg.thumbnail(local, work / "thumb.jpg", at=0.0, width=320)
            campos["thumb_key"] = storage.put_file(
                f"{job_id}/media/{media_id}_thumb.jpg", thumb
            )
        except ffmpeg.FFmpegError:
            # um item sem miniatura ainda serve para montar; a lista mostra o
            # lugar dela
            self.log.warning("sem miniatura para a midia %s", media_id)

        if kind == MediaKind.VIDEO:
            try:
                proxy = ffmpeg.proxy(local, work / "proxy.mp4")
                campos["proxy_key"] = storage.put_file(
                    f"{job_id}/media/{media_id}_proxy.mp4", proxy
                )
            except ffmpeg.FFmpegError:
                self.log.warning("sem proxy para a midia %s", media_id)

        return campos

    def _falhou(self, media_id: str, motivo: str) -> None:
        with session() as s:
            item = s.get(Media, media_id)
            if item is not None:
                item.status = TrackStatus.FAILED
                item.error = motivo[:2000]

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """Um arquivo ilegivel e problema dele, nao do job: a analise da partida
        continua valendo e o usuario so precisa mandar outro."""
        media_id = payload.get("media_id")
        if media_id:
            self._falhou(media_id, f"{self.name}: {exc}")


if __name__ == "__main__":
    # dois lacos, um processo: a analise de musica solta e a do pedido usam o
    # mesmo codigo e as mesmas dependencias pesadas (librosa, numpy)
    threading.Thread(target=MediaAnalyzer().run, daemon=True).start()
    sys.exit(run_worker(BeatsService))
