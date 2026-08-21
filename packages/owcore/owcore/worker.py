"""Laço base de um microsserviço consumidor.

Um worker declara o stream que escuta e o grupo ao qual pertence; o laço cuida
de reconexão, ack, encerramento limpo e de marcar o job como falho quando o
handler estoura.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
from abc import ABC, abstractmethod
from typing import Any

from .bus import Message, get_bus
from .db import init_db
from .jobs import fail
from .logging import setup_logging


class Worker(ABC):
    #: stream que este serviço consome
    stream: str
    #: consumer group — um por serviço, para que cada serviço veja toda mensagem
    group: str
    name: str = "worker"

    def __init__(self) -> None:
        self.log = setup_logging(self.name)
        self.consumer = f"{self.name}-{socket.gethostname()}-{os.getpid()}"
        self._running = True

    # ── contrato ────────────────────────────────────────────────────────────

    @abstractmethod
    def handle(self, payload: dict[str, Any]) -> None: ...

    def accepts(self, payload: dict[str, Any]) -> bool:
        """Filtro opcional — usado pelos detectores para pegar só a sua ROI."""
        return True

    # ── laço ────────────────────────────────────────────────────────────────

    def stop(self, *_a: object) -> None:
        self.log.info("encerrando…")
        self._running = False

    def run(self) -> None:
        init_db()
        bus = get_bus()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.stop)
            except (ValueError, OSError):  # sem sinais fora da thread principal
                pass
        self.log.info("ouvindo '%s' como grupo '%s'", self.stream, self.group)
        while self._running:
            try:
                got = False
                for msg in bus.consume(self.stream, self.group, self.consumer):
                    got = True
                    self._dispatch(bus, msg)
                if not got:
                    self.idle()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                self.stop()
            except Exception:
                self.log.exception("erro no laço do worker; tentando de novo em 2s")
                time.sleep(2.0)
        self.log.info("parado.")

    def _dispatch(self, bus, msg: Message) -> None:
        payload = msg.payload
        if not self.accepts(payload):
            bus.ack(self.stream, self.group, msg.id)
            return
        job_id = payload.get("job_id", "?")
        started = time.monotonic()
        try:
            self.handle(payload)
            self.log.info(
                "job %s processado em %.1fs", job_id, time.monotonic() - started
            )
        except Exception as exc:
            self.log.exception("falha no job %s", job_id)
            self.on_error(payload, exc)
        finally:
            bus.ack(self.stream, self.group, msg.id)

    def idle(self) -> None:
        """Chamado quando nao havia mensagem. O editor usa para resgatar jobs
        cujo detector morreu e nunca avisou."""
        return None

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """O que fazer quando o handler estoura. Por padrao o job vira FAILED;
        os detectores sobrescrevem, porque a falha de um deles nao deve
        impedir que o editor monte com o que os outros acharam."""
        job_id = payload.get("job_id")
        if job_id:
            fail(job_id, f"{self.name}: {exc}")


def run_worker(cls: type[Worker]) -> None:  # pragma: no cover - entrypoint
    logging.captureWarnings(True)
    cls().run()
