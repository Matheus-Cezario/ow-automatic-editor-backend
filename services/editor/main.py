"""Editing microservice.

The system's second phase. It decides nothing: whoever built the montage
decided. The request carries the `timelines` -- which stretch comes in, at which
point of the video and for how long -- and here exactly that is cut, filling
with black whatever was left empty.

There used to be a second kind of video in the same request: the `selections`,
proposals the system assembled on its own from the rules. They no longer exist;
the system is an editor, and what it generates is what was edited.

With no music the video comes out with the match's original audio.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import fail_render, set_render_status
from owcore.models import (
    CLIP_KIND_CUSTOM,
    STREAM_RENDER_READY,
    Clip,
    Job,
    Render,
    RenderStatus,
    Media,
    Timeline,
)
from owcore.compose import LibraryFile
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
            job_id, video_key = job.id, job.video_key
            timelines = [Timeline(**d) for d in (pedido.timelines or [])]

        if not timelines:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="nada escolhido",
                error="o pedido nao trouxe linha do tempo nenhuma",
            )
            return

        set_render_status(
            render_id, RenderStatus.RENDERING,
            stage="preparando os cortes", progress=0.1,
        )

        work = Path(settings.work_dir) / job_id / "renders" / render_id
        work.mkdir(parents=True, exist_ok=True)
        source = local_copy(video_key, work)

        items = self._timeline_items(job_id, timelines, work)
        if not items:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="montagens nao encontradas",
                error="o pedido nao trouxe nenhuma montagem que se possa cortar",
            )
            return

        def progress(done: float) -> None:
            set_render_status(
                render_id, progress=0.1 + 0.85 * done,
                stage=f"renderizando ({int(done * 100)}%)",
            )

        clips = render.render_all(
            source, items, work / "clips", on_progress=progress,
        )

        with session() as s:
            for c in clips:
                clip = Clip(
                    job_id=job_id,
                    render_id=render_id,
                    kind=CLIP_KIND_CUSTOM,
                    title=c.title,
                    start_s=c.start_s,
                    end_s=c.end_s,
                    key="",
                    meta={**c.meta, "duration_s": round(c.duration_s, 2)},
                )
                s.add(clip)
                s.flush()  # the id is needed to build the storage key
                # no video: the montage failed, but the cuts exist and go
                # along anyway
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

        with_video = sum(1 for c in clips if c.video is not None)
        cuts_only = len(clips) - with_video
        if not clips:
            set_render_status(
                render_id, RenderStatus.FAILED,
                stage="nada pode ser cortado", progress=1.0,
                error="nenhum dos videos pedidos pode ser gerado",
            )
            return

        stage_text = f"{with_video} video(s) prontos"
        if cuts_only:
            stage_text += f" + {cuts_only} so com os cortes"
        set_render_status(render_id, RenderStatus.DONE, stage=stage_text, progress=1.0)
        self.log.info(
            "pedido %s concluido: %d video(s), %d apenas com os cortes",
            render_id, with_video, cuts_only,
        )

    # -- from the user's montages to what the renderer understands ----------

    def _timeline_items(
        self, job_id: str, timelines: list[Timeline], work: Path
    ) -> list[render.TimelineItem]:
        """The hand-built montages. They go through no proposal -- what to cut
        and where to put it arrived already decided from the screen."""
        items: list[render.TimelineItem] = []
        with session() as s:
            for i, spec in enumerate(timelines, start=1):
                # the media library this montage uses, already on disk
                library: dict[str, object] = {}
                for clip in spec.clips:
                    if not clip.media_id or clip.media_id in library:
                        continue
                    item = s.get(Media, clip.media_id)
                    if item is None or item.job_id != job_id:
                        self.log.warning(
                            "midia %s nao e deste job; a montagem vai sem ela",
                            clip.media_id,
                        )
                        continue
                    library[clip.media_id] = LibraryFile(
                        path=local_copy(item.key, work), kind=item.kind
                    )

                # the track name is only a label -- it lets the video list say
                # which music that one came out with. It comes from the first
                # sound block, which is the one that starts playing
                music_name = None
                for camada in spec.layers:
                    if not camada.is_audio:
                        continue
                    for clip in camada.clips:
                        item = s.get(Media, clip.media_id) if clip.media_id else None
                        if item is not None:
                            music_name = item.name
                            break
                    if music_name:
                        break

                items.append(
                    render.TimelineItem(
                        timeline=spec,
                        title=spec.title or f"Montagem {i}",
                        music_name=music_name,
                        library=library,
                    )
                )
        return items

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        render_id = payload.get("render_id")
        if render_id:
            fail_render(render_id, f"{self.name}: {exc}")


if __name__ == "__main__":
    sys.exit(run_worker(Editor))
