"""Exact-preview microservice.

Renders a stretch of one montage through the same filter graph the final video
uses (`owcore.compose.compose_graph`), on a small frame and with the fastest
encoder settings, so the user can check what the server will really produce
without paying for the full render.

It reads the **proxies** -- the match's and each imported video's -- and not the
originals: they are already the preview's size, and decoding a 1440p60
recording to shrink it again would be most of the wait.

It listens on a stream of its own, so a preview never queues behind a full
render in the editor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore import ffmpeg
from owcore.compose import LibraryFile, compose_graph
from owcore.config import get_settings
from owcore.db import session
from owcore.models import (
    STREAM_PREVIEW,
    Job,
    Media,
    Preview,
    RenderStatus,
    Timeline,
)
from owcore.preview import preview_timeline
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker


def cached_copy(key: str, dest_dir: Path) -> Path:
    """`local_copy`, skipping the download when the same blob is already
    there -- a match is previewed over and over while it is being edited."""
    st = get_storage()
    dest = dest_dir / Path(key).name
    if dest.is_file() and dest.stat().st_size == st.size(key):
        return dest
    return local_copy(key, dest_dir)


def playable(proxy_key: str, original_key: str, dest_dir: Path) -> Path:
    """The file the preview reads for one source: the proxy's picture with the
    original's sound, built once and kept; the original itself when there is
    no proxy (an image, a song, a match analysed before proxies existed)."""
    if not proxy_key:
        return cached_copy(original_key, dest_dir)
    joined = dest_dir / f"{Path(proxy_key).stem}_av.mp4"
    if joined.is_file():
        return joined
    proxy = cached_copy(proxy_key, dest_dir)
    original = cached_copy(original_key, dest_dir)
    # written aside and renamed: a run that dies halfway must not leave a
    # broken file that every later preview would trust
    partial = joined.with_suffix(".partial.mp4")
    ffmpeg.with_audio(proxy, original, partial)
    return partial.replace(joined)


def set_preview(preview_id: str, **fields: Any) -> None:
    with session() as s:
        p = s.get(Preview, preview_id)
        if p is None:
            return
        for k, v in fields.items():
            setattr(p, k, v)


class Previewer(Worker):
    name = "previewer"
    stream = STREAM_PREVIEW
    group = "previewer"

    def handle(self, payload: dict[str, Any]) -> None:
        preview_id = payload["preview_id"]
        settings = get_settings()

        with session() as s:
            p = s.get(Preview, preview_id)
            if p is None:
                # replaced by a newer preview before its turn came
                self.log.info("preview %s is gone; skipping", preview_id)
                return
            if p.status != RenderStatus.PENDING:
                return
            job = s.get(Job, p.job_id)
            if job is None:
                return
            job_id = job.id
            source_keys = (job.proxy_key, job.video_key)
            spec = Timeline(**p.timeline)
            from_s, to_s = p.from_s, p.to_s
            media_ids = {c.media_id for c in spec.clips if c.media_id}
            if spec.export.watermark_id:
                media_ids.add(spec.export.watermark_id)
            media = {
                m.id: ((m.proxy_key, m.key), m.kind)
                for m in (s.get(Media, i) for i in media_ids)
                if m is not None and m.job_id == job_id
            }

        set_preview(preview_id, status=RenderStatus.RENDERING, progress=0.1)

        # the sources stay between previews: the same match is previewed
        # again and again while editing
        work = Path(settings.work_dir) / job_id / "previews"
        sources = work / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        source = playable(*source_keys, sources)
        library = {
            mid: LibraryFile(path=playable(*keys, sources), kind=kind)
            for mid, (keys, kind) in media.items()
        }
        info = ffmpeg.probe(source)
        set_preview(preview_id, progress=0.3)

        small = preview_timeline(
            spec,
            from_s=from_s,
            to_s=to_s,
            source_width=info.width,
            source_height=info.height,
            source_fps=info.fps,
        )
        comp = compose_graph(
            small,
            source=source,
            width=info.width,
            height=info.height,
            fps=info.fps,
            source_duration_s=info.duration_s,
            library=library,
        )
        dest = work / f"{preview_id}.mp4"
        ffmpeg.compose(comp, dest, preset="ultrafast", audio_kbps=96)

        key = get_storage().put_file(f"{job_id}/previews/{preview_id}.mp4", dest)
        dest.unlink(missing_ok=True)
        set_preview(
            preview_id, status=RenderStatus.DONE, progress=1.0, video_key=key
        )
        self.log.info("preview %s done (%.1fs-%.1fs)", preview_id, from_s, to_s)

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        preview_id = payload.get("preview_id")
        if preview_id:
            set_preview(
                preview_id,
                status=RenderStatus.FAILED,
                error=f"{self.name}: {exc}"[:500],
            )


if __name__ == "__main__":
    sys.exit(run_worker(Previewer))
