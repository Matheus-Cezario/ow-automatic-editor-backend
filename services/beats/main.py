"""The microservice that looks at the file the user has just brought in.

Music, clip or image: everything entering the editor's library passes through
here before it can be built with. It is by looking at the result (a track's
beats, a clip's thumbnail) that the user decides what to do with the file.

There used to be a second loop in this process, at the start of generation: when
the user chose ready-made proposals, each could come with its own music, and
somebody had to analyse them before the editor could cut. There are no proposals
any more, and the editor's music arrives through the library -- already analysed
here, long before any render request exists. The request is born ready and goes
straight to the editor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore.config import get_settings
from owcore.db import session
from owcore.models import (
    STREAM_MEDIA,
    Media,
    MediaKind,
    TrackStatus,
)
from owcore import ffmpeg
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

from detect import analyze_track


class MediaAnalyzer(Worker):
    """Looks at the file the user has just brought in.

    The result generates no video: it goes back to the editing screen, which
    needs to know what to do with the file before it will let you build with it.

    What each kind gets:

    * **audio** -- duration, BPM, beats and waveform. It is what the ruler draws
      and what the magnet uses;
    * **video** -- duration, dimensions, fps, a thumbnail and a **proxy**, for
      the same reason as the recording's proxy: the monitor cannot drag the full
      file on every seek;
    * **image** -- dimensions and a thumbnail.
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
            fields = (
                self._audio(local, work)
                if kind == MediaKind.AUDIO
                else self._visual(local, work, job_id, media_id, kind)
            )
        except Exception as exc:
            self._failed(media_id, str(exc))
            raise

        with session() as s:
            item = s.get(Media, media_id)
            if item is None:
                return
            item.status = TrackStatus.READY
            item.error = None
            for field, value in fields.items():
                setattr(item, field, value)

        self.log.info("midia %s (%s) analisada: %s", media_id, kind,
                      ", ".join(f"{k}={v}" for k, v in fields.items()
                                if k not in ("beats", "peaks")))

    # -- by kind -------------------------------------------------------------

    def _audio(self, local: Path, work: Path) -> dict[str, Any]:
        analysis = analyze_track(local, work)
        return {
            "duration_s": analysis.duration_s,
            "bpm": analysis.grid.bpm,
            "beats": analysis.grid.beats,
            "peaks": analysis.peaks,
        }

    def _visual(
        self, local: Path, work: Path, job_id: str, media_id: str, kind: str
    ) -> dict[str, Any]:
        storage = get_storage()
        info = ffmpeg.probe(local)
        fields: dict[str, Any] = {
            "width": info.width,
            "height": info.height,
            # an image has no duration: how long it stays on screen is the
            # montage's choice, not a property of the file
            "duration_s": info.duration_s if kind == MediaKind.VIDEO else 0.0,
            "fps": info.fps if kind == MediaKind.VIDEO else 0.0,
        }

        try:
            thumb = ffmpeg.thumbnail(local, work / "thumb.jpg", at=0.0, width=320)
            fields["thumb_key"] = storage.put_file(
                f"{job_id}/media/{media_id}_thumb.jpg", thumb
            )
        except ffmpeg.FFmpegError:
            # an item with no thumbnail can still be built with; the list shows
            # the space where it would be
            self.log.warning("sem miniatura para a midia %s", media_id)

        if kind == MediaKind.VIDEO:
            try:
                proxy = ffmpeg.proxy(local, work / "proxy.mp4")
                fields["proxy_key"] = storage.put_file(
                    f"{job_id}/media/{media_id}_proxy.mp4", proxy
                )
            except ffmpeg.FFmpegError:
                self.log.warning("sem proxy para a midia %s", media_id)

        return fields

    def _failed(self, media_id: str, reason: str) -> None:
        with session() as s:
            item = s.get(Media, media_id)
            if item is not None:
                item.status = TrackStatus.FAILED
                item.error = reason[:2000]

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """An unreadable file is its own problem, not the job's: the match
        analysis still stands and the user only has to send another."""
        media_id = payload.get("media_id")
        if media_id:
            self._failed(media_id, f"{self.name}: {exc}")


if __name__ == "__main__":
    sys.exit(run_worker(MediaAnalyzer))
