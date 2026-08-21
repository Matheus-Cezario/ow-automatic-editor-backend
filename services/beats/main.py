"""Microsservico de analise ritmica.

Entra na **segunda fase**: quando o usuario pede a geracao, cada video escolhido
pode vir com uma musica diferente. Este servico analisa cada uma, guarda a
grade de batidas no pedido e so entao libera o editor.

Videos sem musica passam direto: nao ha ritmo a extrair, e o corte sai com o
audio original da partida.

O mesmo processo ouve dois streams. O segundo e o da **musica recem-enviada**:
na montagem manual a musica sobe antes de existir video nenhum, porque e
ouvindo ela -- com as batidas e a forma de onda na tela -- que o usuario decide
onde cada corte cai. Sao dois lacos porque sao dois momentos do fluxo, mas a
analise por tras e a mesma, e nao vale um servico a mais no compose para isso.
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
    STREAM_RENDER,
    STREAM_RENDER_READY,
    STREAM_TRACK,
    Render,
    RenderRequested,
    RenderStatus,
    Selection,
    Track,
    TrackStatus,
)
from owcore.storage import local_copy
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


class TrackAnalyzer(Worker):
    """Ouve a musica que o usuario acabou de enviar.

    O resultado nao gera video nenhum: ele volta para a tela de montagem, que
    precisa da duracao para desenhar a regua, das batidas para grudar os cortes
    e da forma de onda para o usuario achar o refrao a olho.
    """

    name = "tracks"
    stream = STREAM_TRACK
    group = "tracks"

    def handle(self, payload: dict[str, Any]) -> None:
        track_id = payload["track_id"]

        with session() as s:
            track = s.get(Track, track_id)
            if track is None:
                self.log.warning("musica %s sumiu; ignorando", track_id)
                return
            if track.status == TrackStatus.READY:
                self.log.info("musica %s ja analisada; ignorando", track_id)
                return
            job_id, key = track.job_id, track.key

        work = Path(get_settings().work_dir) / job_id / "tracks" / track_id
        work.mkdir(parents=True, exist_ok=True)

        try:
            analise = analyze_track(local_copy(key, work), work)
        except Exception as exc:
            self._falhou(track_id, str(exc))
            raise

        with session() as s:
            track = s.get(Track, track_id)
            if track is None:
                return
            track.status = TrackStatus.READY
            track.error = None
            track.duration_s = analise.duration_s
            track.bpm = analise.grid.bpm
            track.beats = analise.grid.beats
            track.peaks = analise.peaks

        self.log.info(
            "musica %s: %.1fs, %.1f BPM, %d batidas",
            track_id, analise.duration_s, analise.grid.bpm, len(analise.grid.beats),
        )

    def _falhou(self, track_id: str, motivo: str) -> None:
        with session() as s:
            track = s.get(Track, track_id)
            if track is not None:
                track.status = TrackStatus.FAILED
                track.error = motivo[:2000]

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """Uma musica ilegivel e problema dela, nao do job: a analise da partida
        continua valendo e o usuario so precisa mandar outro arquivo."""
        track_id = payload.get("track_id")
        if track_id:
            self._falhou(track_id, f"{self.name}: {exc}")


if __name__ == "__main__":
    # dois lacos, um processo: a analise de musica solta e a do pedido usam o
    # mesmo codigo e as mesmas dependencias pesadas (librosa, numpy)
    threading.Thread(target=TrackAnalyzer().run, daemon=True).start()
    sys.exit(run_worker(BeatsService))
