"""Base loop of a consumer microservice.

A worker declares the stream it listens to and the group it belongs to; the
loop takes care of reconnection, acking, clean shutdown, and marking the job as
failed when the handler blows up.
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
    #: the stream this service consumes
    stream: str
    #: consumer group -- one per service, so that every service sees every
    #: message
    group: str
    name: str = "worker"

    def __init__(self) -> None:
        self.log = setup_logging(self.name)
        self.consumer = f"{self.name}-{socket.gethostname()}-{os.getpid()}"
        self._running = True

    # -- contract -----------------------------------------------------------

    @abstractmethod
    def handle(self, payload: dict[str, Any]) -> None: ...

    def accepts(self, payload: dict[str, Any]) -> bool:
        """Optional filter -- used by the detectors to take only their own ROI."""
        return True

    # -- loop ---------------------------------------------------------------

    def stop(self, *_a: object) -> None:
        self.log.info("encerrando…")
        self._running = False

    def run(self) -> None:
        init_db()
        bus = get_bus()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.stop)
            except (ValueError, OSError):  # no signals outside the main thread
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
        """Called when there was no message. The planner uses it to rescue jobs
        whose detector died and never reported."""
        return None

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """What to do when the handler blows up. By default the job becomes
        FAILED; the detectors override this, because one of them failing must
        not stop the editor from working with what the others found."""
        job_id = payload.get("job_id")
        if job_id:
            fail(job_id, f"{self.name}: {exc}")


def run_worker(cls: type[Worker]) -> None:  # pragma: no cover - entrypoint
    logging.captureWarnings(True)
    cls().run()
