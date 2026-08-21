"""Microsservico das miniaturas dos momentos.

A barra lateral do editor mostra cada momento da partida com um quadro dele --
sem imagem, escolher entre trinta eliminacoes e escolher entre trinta relogios.
Este servico e quem tira esses quadros.

Roda **depois** da analise e **fora** dela: escuta o fim do planejamento, mas
nao segura o job em `ready`. Se as miniaturas demorarem, ou nem sairem, tudo o
mais continua funcionando -- a barra lateral so fica sem imagem.

Nao guarda nada no banco: a chave de cada quadro sai do instante
(`frame_key`), entao quem escreve e quem le chegam nela sozinhos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore import ffmpeg
from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import load_events
from owcore.models import STREAM_THUMBS, THUMB_KINDS, Job, frame_key
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

#: Largura da miniatura. Cabe numa lista lateral em tela cheia e continua
#: legivel no dobro da densidade de um celular; maior que isso e so byte.
WIDTH = 240

#: Teto de quadros por partida. Uma gravacao longa pode ter centenas de
#: momentos, e ninguem rola uma lista dessas -- o app mostra os primeiros.
MAX_FRAMES = 300


class Thumbs(Worker):
    name = "thumbs"
    stream = STREAM_THUMBS
    group = "thumbs"

    def handle(self, payload: dict[str, Any]) -> None:
        job_id = payload["job_id"]
        storage = get_storage()

        with session() as s:
            job = s.get(Job, job_id)
            if job is None:
                self.log.warning("job %s sumiu; ignorando", job_id)
                return
            video_key = job.video_key

        kinds = {str(k) for k in THUMB_KINDS}
        instantes: list[float] = []
        vistos: set[str] = set()
        for e in load_events(job_id):
            if str(e.kind) not in kinds:
                continue
            chave = frame_key(job_id, e.t)
            if chave in vistos:
                continue  # dois detectores no mesmo instante dao o mesmo quadro
            vistos.add(chave)
            if not storage.exists(chave):
                instantes.append(e.t)
            if len(vistos) >= MAX_FRAMES:
                break

        if not instantes:
            self.log.info("job %s: nada a extrair", job_id)
            return

        work = Path(get_settings().work_dir) / job_id / "frames"
        work.mkdir(parents=True, exist_ok=True)
        source = local_copy(video_key, work)

        feitos = 0
        for t in instantes:
            destino = work / f"{t:.2f}.jpg"
            try:
                # `-ss` antes do input: o ffmpeg pula ate o keyframe mais
                # proximo em vez de decodificar tudo desde o comeco. Um quadro
                # sai em dezenas de milissegundos, e por isso extrair um por vez
                # sai mais barato do que uma passagem unica pelo video inteiro.
                ffmpeg.thumbnail(source, destino, at=t, width=WIDTH)
            except ffmpeg.FFmpegError:
                # um quadro que nao sai nao pode custar os outros: a barra
                # lateral aguenta um item sem imagem
                self.log.warning("sem miniatura para %.2fs do job %s", t, job_id)
                continue
            storage.put_file(frame_key(job_id, t), destino)
            destino.unlink(missing_ok=True)
            feitos += 1

        self.log.info("job %s: %d miniatura(s) de %d momento(s)",
                      job_id, feitos, len(instantes))

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """Miniatura que falha nao derruba o job: a analise ja terminou e o
        video continua podendo ser montado, so que sem as imagens."""
        self.log.warning(
            "nao consegui extrair as miniaturas de %s: %s",
            payload.get("job_id"), exc,
        )


if __name__ == "__main__":
    sys.exit(run_worker(Thumbs))
