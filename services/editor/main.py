"""Microsservico de edicao.

Segunda fase do sistema. Nao decide mais *o que* vale a pena virar video --
isso o planejador ja fez e o usuario ja escolheu. Aqui so se corta: para cada
escolha do pedido, com as opcoes e a musica daquela escolha.

Um pedido traz dois tipos de video, e eles convivem:

* **propostas escolhidas** (`selections`) -- o editor calcula os cortes a partir
  dos momentos da proposta e da grade de batidas da musica;
* **montagens manuais** (`timelines`) -- nao ha o que calcular: o usuario ja
  disse que trecho entra, em que ponto do video e por quanto tempo. Ao editor
  sobra cortar aquilo e preencher de preto o que ele deixou vazio.

Sem musica o video sai com o audio original da partida.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import fail_render, set_render_status
from owcore.models import (
    STREAM_RENDER_READY,
    BeatGrid,
    Clip,
    HighlightKind,
    Job,
    Proposal,
    Render,
    RenderStatus,
    Media,
    Selection,
    Timeline,
)
from owcore.compose import MidiaNoDisco
from owcore.rules import Highlight
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

import render


class Editor(Worker):
    name = "editor"
    stream = STREAM_RENDER_READY
    group = "editor"

    def handle(self, payload: dict[str, Any]) -> None:
        render_id = payload["render_id"]
        settings = get_settings()
        storage = get_storage()

        with session() as s:
            pedido = s.get(Render, render_id)
            if pedido is None:
                self.log.warning("pedido %s sumiu; ignorando", render_id)
                return
            if pedido.clips:
                self.log.info("pedido %s ja foi gerado; ignorando", render_id)
                return
            job = s.get(Job, pedido.job_id)
            if job is None:
                return
            job_id, video_key, duration = job.id, job.video_key, job.duration_s
            selections = [Selection(**d) for d in (pedido.selections or [])]
            timelines = [Timeline(**d) for d in (pedido.timelines or [])]
            grids = dict(pedido.beats or {})

        if not selections and not timelines:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="nada escolhido",
                error="o pedido nao trouxe proposta nem linha do tempo",
            )
            return

        set_render_status(
            render_id, RenderStatus.RENDERING,
            stage="preparando os cortes", progress=0.1,
        )

        work = Path(settings.work_dir) / job_id / "renders" / render_id
        work.mkdir(parents=True, exist_ok=True)
        source = local_copy(video_key, work)

        items = self._items(job_id, selections, grids, work)
        items += self._timeline_items(job_id, timelines, work)
        if not items:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="propostas nao encontradas",
                error="as escolhas nao correspondem a nenhuma proposta deste job",
            )
            return

        def progress(done: float) -> None:
            set_render_status(
                render_id, progress=0.1 + 0.85 * done,
                stage=f"renderizando ({int(done * 100)}%)",
            )

        clips = render.render_all(
            source, items, duration, work / "clips",
            on_progress=progress, seed=render_id,
        )

        with session() as s:
            for c in clips:
                clip = Clip(
                    job_id=job_id,
                    render_id=render_id,
                    proposal_id=c.proposal_id or None,
                    kind=str(c.highlight.kind),
                    title=c.highlight.title,
                    start_s=c.highlight.start,
                    end_s=c.highlight.end,
                    score=c.highlight.score,
                    key="",
                    meta={**c.meta, "duration_s": round(c.duration_s, 2)},
                )
                s.add(clip)
                s.flush()  # precisa do id para montar a chave no storage
                # sem video: a montagem falhou, mas os cortes existem e vao
                # junto assim mesmo
                clip.key = (
                    storage.put_file(f"{job_id}/clips/{clip.id}.mp4", c.video)
                    if c.video is not None
                    else ""
                )
                extras: dict = {}
                if c.thumb is not None:
                    extras["thumb_key"] = storage.put_file(
                        f"{job_id}/clips/{clip.id}.jpg", c.thumb
                    )
                if c.segments_zip is not None:
                    extras["segments_zip_key"] = storage.put_file(
                        f"{job_id}/clips/{clip.id}_cortes.zip", c.segments_zip
                    )
                if extras:
                    clip.meta = {**clip.meta, **extras}

        com_video = sum(1 for c in clips if c.video is not None)
        so_cortes = len(clips) - com_video
        if not clips:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="nada pode ser cortado", progress=1.0,
                error="nenhum dos videos pedidos pode ser gerado",
            )
            return

        estagio = f"{com_video} video(s) prontos"
        if so_cortes:
            estagio += f" + {so_cortes} so com os cortes"
        set_render_status(render_id, RenderStatus.DONE, stage=estagio, progress=1.0)
        self.log.info(
            "pedido %s concluido: %d video(s), %d apenas com os cortes",
            render_id, com_video, so_cortes,
        )

    # ── das escolhas do usuario para o que o renderizador entende ───────────

    def _items(
        self,
        job_id: str,
        selections: list[Selection],
        grids: dict[str, Any],
        work: Path,
    ) -> list[render.RenderItem]:
        items: list[render.RenderItem] = []
        with session() as s:
            for sel in selections:
                proposta = s.get(Proposal, sel.proposal_id)
                if proposta is None or proposta.job_id != job_id:
                    self.log.warning(
                        "proposta %s nao e deste job; pulando", sel.proposal_id
                    )
                    continue
                grid_data = grids.get(sel.proposal_id)
                items.append(
                    render.RenderItem(
                        highlight=Highlight(
                            kind=HighlightKind(proposta.kind),
                            start=proposta.start_s,
                            end=proposta.end_s,
                            score=proposta.score,
                            title=proposta.title,
                            beats_at=list(proposta.moments or []),
                            meta=dict(proposta.meta or {}),
                        ),
                        proposal_id=proposta.id,
                        options=sel.options,
                        music=(
                            local_copy(sel.music_key, work) if sel.music_key else None
                        ),
                        beats=BeatGrid(**grid_data) if grid_data else None,
                        music_name=sel.music_name,
                    )
                )
        return items

    def _timeline_items(
        self, job_id: str, timelines: list[Timeline], work: Path
    ) -> list[render.TimelineItem]:
        """As montagens feitas a mao. Nao passam por proposta nenhuma -- o que
        cortar e onde por ja veio decidido da tela."""
        items: list[render.TimelineItem] = []
        with session() as s:
            for i, spec in enumerate(timelines, start=1):
                music = None
                music_name = None
                if spec.track_id:
                    track = s.get(Media, spec.track_id)
                    if track is None or track.job_id != job_id:
                        # a musica sumiu, o video nao: sai com o audio da partida
                        self.log.warning(
                            "musica %s nao e deste job; monto sem trilha",
                            spec.track_id,
                        )
                    else:
                        music = local_copy(track.key, work)
                        music_name = track.name
                # a biblioteca de midia que esta montagem usa, ja em disco
                midias: dict[str, object] = {}
                for clip in spec.clips:
                    if not clip.media_id or clip.media_id in midias:
                        continue
                    item = s.get(Media, clip.media_id)
                    if item is None or item.job_id != job_id:
                        self.log.warning(
                            "midia %s nao e deste job; a montagem vai sem ela",
                            clip.media_id,
                        )
                        continue
                    midias[clip.media_id] = MidiaNoDisco(
                        caminho=local_copy(item.key, work), kind=item.kind
                    )

                items.append(
                    render.TimelineItem(
                        timeline=spec,
                        title=spec.title or f"Montagem {i}",
                        music=music,
                        music_name=music_name,
                        midias=midias,
                    )
                )
        return items

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        render_id = payload.get("render_id")
        if render_id:
            fail_render(render_id, f"{self.name}: {exc}")


if __name__ == "__main__":
    sys.exit(run_worker(Editor))
