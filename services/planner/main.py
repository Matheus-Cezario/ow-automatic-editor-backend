"""Microsservico de planejamento.

Fecha a **primeira fase** do sistema: quando todos os detectores terminam, este
servico cruza o que cada um achou e escreve a lista de videos que da para
gerar. Nada de video, nada de musica -- so regras sobre instantes.

E aqui que moram as decisoes que dependem de mais de um detector, como
reconhecer uma ultimate anulada (ultimate inimiga + eliminacao logo em
seguida): nenhum detector sozinho enxerga os dois tipos de evento.

Terminado o planejamento o job fica em `ready`, parado, esperando o usuario
escolher o que quer. A geracao e outra fase, disparada quantas vezes ele
quiser -- e por isso ela nao acontece aqui.
"""

from __future__ import annotations

import sys
from typing import Any

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import (
    all_detectors_done,
    claim_for_planning,
    expected_detectors,
    load_events,
    reported_detectors,
    set_status,
    stale_detecting_jobs,
)
from owcore.models import (
    STREAM_EDIT,
    STREAM_THUMBS,
    Job,
    JobParams,
    JobStatus,
    Proposal,
    ThumbsRequested,
)
from owcore.rules import build_highlights
from owcore.worker import Worker, run_worker


class Planner(Worker):
    name = "planner"
    stream = STREAM_EDIT
    group = "planner"

    # ── entrada normal: um detector terminou ────────────────────────────────

    def handle(self, payload: dict[str, Any]) -> None:
        job_id = payload["job_id"]
        expected = expected_detectors(job_id)
        if not all_detectors_done(job_id, expected):
            faltam = set(expected) - reported_detectors(job_id)
            self.log.info("job %s ainda espera: %s", job_id, ", ".join(sorted(faltam)))
            return
        self._plan(job_id)

    # ── resgate: algum detector morreu e nunca reportou ─────────────────────

    def idle(self) -> None:
        timeout = get_settings().detector_timeout_s
        for job_id in stale_detecting_jobs(timeout):
            faltam = set(expected_detectors(job_id)) - reported_detectors(job_id)
            self.log.warning(
                "job %s parado ha mais de %.0fs sem %s; planejo com o que tenho",
                job_id, timeout, ", ".join(sorted(faltam)) or "ninguem",
            )
            try:
                self._plan(job_id)
            except Exception:
                self.log.exception("resgate do job %s falhou", job_id)

    # ── planejamento ────────────────────────────────────────────────────────

    def _plan(self, job_id: str) -> None:
        # os detectores terminam quase juntos e todos avisam; a reivindicacao
        # atomica garante uma lista de propostas so
        if not claim_for_planning(job_id):
            self.log.debug("job %s ja foi planejado", job_id)
            return

        with session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            params = JobParams(**(job.params or {}))
            duration = job.duration_s

        events = load_events(job_id)
        highlights = build_highlights(events, params, duration)
        self.log.info(
            "job %s: %d evento(s) -> %d proposta(s)",
            job_id, len(events), len(highlights),
        )

        if not highlights:
            set_status(
                job_id, JobStatus.READY,
                stage="nenhum momento encontrado", progress=1.0,
            )
            # sem proposta ainda pode haver momento: a montagem manual vive dos
            # eventos, nao das propostas, e a barra lateral quer as miniaturas
            self._pedir_miniaturas(job_id)
            return

        with session() as s:
            for h in highlights:
                s.add(
                    Proposal(
                        job_id=job_id,
                        kind=str(h.kind),
                        title=h.title,
                        start_s=h.start,
                        end_s=h.end,
                        score=h.score,
                        moments=[round(t, 3) for t in h.beats_at],
                        meta=h.meta,
                    )
                )

        set_status(
            job_id, JobStatus.READY,
            stage=f"{len(highlights)} video(s) possiveis — escolha o que gerar",
            progress=1.0,
        )
        self._pedir_miniaturas(job_id)

    def _pedir_miniaturas(self, job_id: str) -> None:
        """Avisa quem extrai as miniaturas dos momentos.

        Vai daqui porque e daqui que se sabe que a analise acabou -- e nao passa
        de publicar uma mensagem: o planejador continua sem abrir video nenhum.
        """
        get_bus().publish(
            STREAM_THUMBS, ThumbsRequested(job_id=job_id).model_dump()
        )


if __name__ == "__main__":
    sys.exit(run_worker(Planner))
