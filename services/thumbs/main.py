"""Microservice for the moments' thumbnails.

The editor's sidebar shows each moment of the match with a frame of it --
without a picture, choosing between thirty kills is choosing between thirty
clocks. This service is what takes those frames.

It runs **after** the analysis and **outside** it: it listens for the end of the
analysis, but does not hold the job back from `ready`. If the thumbnails are
slow, or never come, everything else keeps working -- the sidebar just has no
pictures.

It stores nothing in the database: each frame's key derives from the instant
(`frame_key`), so writer and reader arrive at it on their own.
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

#: Thumbnail width. It fits a sidebar list at full screen and stays legible at
#: twice a phone's density; larger than that is just bytes.
WIDTH = 240

#: Ceiling of frames per match. A long recording can have hundreds of moments,
#: and nobody scrolls a list like that -- the app shows the first ones.
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
        instants: list[float] = []
        seen: set[str] = set()
        for e in load_events(job_id):
            if str(e.kind) not in kinds:
                continue
            key = frame_key(job_id, e.t)
            if key in seen:
                continue  # two detectors at the same instant give the same frame
            seen.add(key)
            if not storage.exists(key):
                instants.append(e.t)
            if len(seen) >= MAX_FRAMES:
                break

        if not instants:
            self.log.info("job %s: nada a extrair", job_id)
            return

        work = Path(get_settings().work_dir) / job_id / "frames"
        work.mkdir(parents=True, exist_ok=True)
        source = local_copy(video_key, work)

        done = 0
        for t in instants:
            dest = work / f"{t:.2f}.jpg"
            try:
                # `-ss` before the input: ffmpeg jumps to the nearest keyframe
                # instead of decoding everything from the start. One frame comes
                # out in tens of milliseconds, which is why extracting them one
                # at a time is cheaper than a single pass over the whole video.
                ffmpeg.thumbnail(source, dest, at=t, width=WIDTH)
            except ffmpeg.FFmpegError:
                # a frame that fails must not cost the others: the sidebar can
                # live with one item having no picture
                self.log.warning("sem miniatura para %.2fs do job %s", t, job_id)
                continue
            storage.put_file(frame_key(job_id, t), dest)
            dest.unlink(missing_ok=True)
            done += 1

        self.log.info("job %s: %d miniatura(s) de %d momento(s)",
                      job_id, done, len(instants))

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """A failed thumbnail does not bring the job down: the analysis is over
        and the video can still be built, just without the pictures."""
        self.log.warning(
            "nao consegui extrair as miniaturas de %s: %s",
            payload.get("job_id"), exc,
        )


if __name__ == "__main__":
    sys.exit(run_worker(Thumbs))
