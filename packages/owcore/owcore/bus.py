"""Message bus with *consumer group* semantics.

Two implementations:

* ``RedisBus``  -- Redis Streams (XADD/XREADGROUP/XACK), used in docker mode.
* ``LocalBus``  -- a durable on-disk queue, used in local mode. Each message is
  a JSON file; each group has a folder of markers and the claim is made with
  ``open(..., "x")``, which is atomic on Windows too. That gives the same
  contract as Redis (fan-out across groups, competition within a group) without
  needing any server running.

**Both delete what has already been consumed**, and that is not housekeeping
detail: neither of them deleted anything before, and the consequences differed
on each side. In Redis, `XACK` only removes the message from the group's
pending list -- the entry stays in the stream, in server memory, forever; Redis
RAM grew with the number of matches already processed. In the on-disk queue,
every consumer re-read the **entire** `sorted(glob("*.json"))` every 150 ms, so
the cost of waiting for work grew with all the work already done.
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
    def __init__(self, url: str, maxlen: int = 0):
        import redis  # late import

        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.maxlen = int(maxlen or 0)

    def _ensure_group(self, stream: str, group: str) -> None:
        import redis

        try:
            self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as exc:  # BUSYGROUP: already exists
            if "BUSYGROUP" not in str(exc):
                raise

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        # `approximate=True` lets Redis trim at the internal node boundary,
        # which is amortised O(1); an exact trim would mean walking the stream
        # on every publish to save a few dozen entries
        if self.maxlen > 0:
            return self.r.xadd(
                stream, {"data": json.dumps(payload)},
                maxlen=self.maxlen, approximate=True,
            )
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


#: Marker of a finished message. Delivered-but-unfinished carries the
#: consumer's name in the file; only this text means "safe to sweep".
_ACKED = "acked"


class LocalBus(Bus):
    def __init__(self, root: Path, retention_s: float = 0.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_s = float(retention_s or 0.0)
        #: when the next sweep may happen, per stream
        self._next_sweep: dict[str, float] = {}

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
        # time-sortable name + random suffix against collisions
        msg_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        tmp = d / f".{msg_id}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, d / f"{msg_id}.json")  # atomic publish
        return msg_id

    # -- sweep --------------------------------------------------------------

    def _sweep(self, stream: str) -> int:
        """Deletes the messages **every** group has already finished.

        Without this the folder only grows, and the wait loop below re-lists
        everything that ever passed through it every 150 ms -- the cost of
        sitting idle starts to depend on how many matches the system has
        already processed.

        Two guards against deleting too early: a message only counts as
        consumed once **every existing group** has stamped it finished, and
        even then it waits `bus_retention_s` before disappearing, so a service
        coming up for the first time still finds what was published before it.
        """
        if self.retention_s <= 0:
            return 0
        sd = self._stream_dir(stream)
        groups = [d for d in (sd / "_groups").glob("*") if d.is_dir()]
        if not groups:
            return 0
        cutoff = time.time() - self.retention_s
        deleted = 0
        for f in sd.glob("*.json"):
            try:
                if f.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            markers = [g / f.name for g in groups]
            if not all(m.exists() for m in markers):
                continue
            try:
                if not all(m.read_text(encoding="utf-8") == _ACKED
                           for m in markers):
                    continue
            except OSError:
                continue
            try:
                f.unlink()
                for m in markers:
                    m.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        return deleted

    def consume(
        self, stream: str, group: str, consumer: str, block_ms: int = 2000
    ) -> Iterator[Message]:
        sd = self._stream_dir(stream)
        gd = self._group_dir(stream, group)
        # the sweep goes here, and not in a separate task, because this is the
        # one point every worker passes through all the time. Once a minute per
        # process is enough: what it deletes is of no use to anyone any more
        now = time.monotonic()
        if now >= self._next_sweep.get(stream, 0.0):
            self._next_sweep[stream] = now + 60.0
            self._sweep(stream)
        deadline = time.monotonic() + block_ms / 1000.0
        while True:
            for f in sorted(sd.glob("*.json")):
                marker = gd / f.name
                try:
                    # exclusive creation == atomic claim on the message
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
        # the marker already records delivery; ack just stamps it finished --
        # and it is that stamp which authorises the sweep to delete it
        marker = self._group_dir(stream, group) / msg_id
        try:
            marker.write_text(_ACKED, encoding="utf-8")
        except OSError:
            pass

    def purge(self) -> None:
        """Deletes every message -- used by the tests."""
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


_bus: Bus | None = None


def get_bus() -> Bus:
    global _bus
    if _bus is None:
        s = get_settings()
        _bus = (
            RedisBus(s.redis_url, maxlen=s.stream_maxlen)
            if s.mode == "docker"
            else LocalBus(s.bus_dir, retention_s=s.bus_retention_s)
        )
    return _bus


def reset_bus() -> None:
    """Drops the singleton (tests that switch directories)."""
    global _bus
    _bus = None
