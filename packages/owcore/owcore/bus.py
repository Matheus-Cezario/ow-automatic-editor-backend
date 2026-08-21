"""Barramento de mensagens com semântica de *consumer group*.

Duas implementações:

* ``RedisBus``  — Redis Streams (XADD/XREADGROUP/XACK), usado no modo docker.
* ``LocalBus``  — fila durável em disco, usada no modo local. Cada mensagem é
  um arquivo JSON; cada grupo tem uma pasta de marcadores e a reivindicação é
  feita com ``open(..., "x")``, que é atômico também no Windows. Isso dá o
  mesmo contrato do Redis (fan-out entre grupos, competição dentro do grupo)
  sem precisar de nenhum servidor rodando.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings


@dataclass(slots=True)
class Message:
    id: str
    payload: dict[str, Any]


class Bus(ABC):
    @abstractmethod
    def publish(self, stream: str, payload: dict[str, Any]) -> str: ...

    @abstractmethod
    def consume(
        self, stream: str, group: str, consumer: str, block_ms: int = 2000
    ) -> Iterator[Message]: ...

    @abstractmethod
    def ack(self, stream: str, group: str, msg_id: str) -> None: ...


# ────────────────────────────── Redis Streams ───────────────────────────────


class RedisBus(Bus):
    def __init__(self, url: str):
        import redis  # import tardio

        self.r = redis.Redis.from_url(url, decode_responses=True)

    def _ensure_group(self, stream: str, group: str) -> None:
        import redis

        try:
            self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as exc:  # BUSYGROUP: já existe
            if "BUSYGROUP" not in str(exc):
                raise

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        return self.r.xadd(stream, {"data": json.dumps(payload)})

    def consume(
        self, stream: str, group: str, consumer: str, block_ms: int = 2000
    ) -> Iterator[Message]:
        self._ensure_group(stream, group)
        while True:
            resp = self.r.xreadgroup(
                group, consumer, {stream: ">"}, count=1, block=block_ms
            )
            if not resp:
                yield from ()
                return
            for _stream, entries in resp:
                for msg_id, fields in entries:
                    yield Message(id=msg_id, payload=json.loads(fields["data"]))

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        self.r.xack(stream, group, msg_id)


# ────────────────────────────── fila em disco ───────────────────────────────


class LocalBus(Bus):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _stream_dir(self, stream: str) -> Path:
        d = self.root / stream
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _group_dir(self, stream: str, group: str) -> Path:
        d = self._stream_dir(stream) / "_groups" / group
        d.mkdir(parents=True, exist_ok=True)
        return d

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        d = self._stream_dir(stream)
        # nome ordenável no tempo + sufixo aleatório contra colisão
        msg_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        tmp = d / f".{msg_id}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, d / f"{msg_id}.json")  # publicação atômica
        return msg_id

    def consume(
        self, stream: str, group: str, consumer: str, block_ms: int = 2000
    ) -> Iterator[Message]:
        sd = self._stream_dir(stream)
        gd = self._group_dir(stream, group)
        deadline = time.monotonic() + block_ms / 1000.0
        while True:
            for f in sorted(sd.glob("*.json")):
                marker = gd / f.name
                try:
                    # criação exclusiva == reivindicação atômica da mensagem
                    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, consumer.encode())
                os.close(fd)
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                yield Message(id=f.name, payload=payload)
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(0.15)

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        # o marcador já registra a entrega; ack só o carimba como concluído
        marker = self._group_dir(stream, group) / msg_id
        try:
            marker.write_text("acked", encoding="utf-8")
        except OSError:
            pass

    def purge(self) -> None:
        """Apaga todas as mensagens — usado nos testes."""
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


_bus: Bus | None = None


def get_bus() -> Bus:
    global _bus
    if _bus is None:
        s = get_settings()
        _bus = RedisBus(s.redis_url) if s.mode == "docker" else LocalBus(s.bus_dir)
    return _bus


def reset_bus() -> None:
    """Descarta o singleton (testes que trocam de diretório)."""
    global _bus
    _bus = None
