"""Assembly of the final videos.

It receives the finished montage (it sees no pixel until this point) and goes
back to the original video -- at full quality -- to cut only the stretches that
matter.

Each requested video is independent. A stretch used in one video stays available
for the others -- nothing is "spent".

There used to be a second path here, the proposals one: given a handful of
instants and a beat grid, this module decided on its own where each micro-clip
started and ended. That decision now belongs to the editor, and what arrives
here already comes with its start, duration and position settled.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from owcore import ffmpeg, timeline as tl
from owcore.compose import LibraryFile, compose_graph
from owcore.models import Timeline

log = logging.getLogger(__name__)

@dataclass(slots=True)
class TimelineItem:
    """A video the **user** built, block by block.

    There is no rule behind it: it already says which stretch comes in, where in
    the music and for how long. What is left for the editor is cutting and
    joining exactly that -- the black of the spaces the user left empty
    included.
    """

    timeline: Timeline
    title: str = "Montagem"
    #: only to name the video in the list: the music is in the blocks, as media
    music_name: str | None = None
    #: the library items this montage uses, already on disk
    library: dict = field(default_factory=dict)


@dataclass(slots=True)
class RenderedClip:
    """Um video pronto (ou os cortes dele, quando a juncao falhou).

    It carried a `Highlight` while the system generated videos by rule: that was
    what said the kind, the title and the score of the proposed video. Now every
    video comes out of the editor, so the title is whatever the user gave and
    there is no score at all -- the fields that still matter live here, with no
    intermediary.
    """

    title: str
    #: the stretch of the recording the montage covers, first cut to last
    start_s: float
    end_s: float
    #: None when the cuts came out but the final assembly failed
    video: Path | None
    thumb: Path | None
    duration_s: float
    meta: dict
    #: zip of the individual cuts that made up the montage, for whoever wants
    #: to re-edit on their own
    segments_zip: Path | None = None


def render_all(
    source: Path,
    items: list[TimelineItem],
    out_dir: Path,
    *,
    on_progress=None,
) -> list[RenderedClip]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedClip] = []

    for i, item in enumerate(items):
        try:
            clip = _render_timeline(source, item, out_dir, i)
        except ffmpeg.FFmpegError:
            # a problematic clip must not cost the rest of the delivery
            log.exception(
                "falha ao renderizar '%s'; sigo com os demais", item.title
            )
            continue
        rendered.append(clip)
        if on_progress:
            on_progress((i + 1) / max(1, len(items)))

    return rendered


def _render_timeline(
    source: Path, item: TimelineItem, out_dir: Path, index: int
) -> RenderedClip:
    """Assembles exactly what the user drew on the timeline.

    Nothing here is calculated: each cut's duration and the point of the music
    where it comes in arrived ready-made. The only decision left is about the
    gaps -- space the user left empty becomes black with the music playing, and
    not a shortening of the video, or every block after it would leave the place
    where it was put.
    """
    spec = item.timeline
    media = ffmpeg.probe(source)

    # A layer, a transform, adjusted sound or imported media do not fit
    # cut-and-splice: they require two pieces existing at the same time. There
    # the montage becomes a filter graph -- more powerful and less forgiving,
    # because one error in it brings down the whole render instead of costing
    # one cut.
    if not spec.single_layer:
        return _render_composition(source, item, media, out_dir, index)

    pieces = tl.plan(spec.cuts, source_duration_s=media.duration_s)
    if not any(p.is_cut for p in pieces):
        raise ffmpeg.FFmpegError(
            "nenhum dos cortes cai dentro da gravacao"
        )

    # this path only takes montages with no music -- a sound layer already
    # sends the montage to the filter graph -- so the match audio is all the
    # video has
    muted = False
    audio_rate = media.audio_rate

    parts_dir = out_dir / f"{index:02d}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    #: everything that goes into the video, in order -- cuts and blacks
    parts: list[Path] = []
    #: only the real cuts, which are what goes into the zip
    cut_files: list[tuple[Path, tuple[float, float]]] = []

    for j, piece in enumerate(pieces):
        part = parts_dir / f"{j:03d}.mp4"
        if piece.is_cut:
            try:
                ffmpeg.cut(source, piece.start_s, piece.end_s, part, mute=muted)
                parts.append(part)
                cut_files.append((part, (piece.start_s, piece.end_s)))
                continue
            except ffmpeg.FFmpegError:
                # the cut failed, but its slot stays reserved: turning black
                # keeps every following block at the point of the music where
                # the user put it
                log.warning(
                    "corte em %.1fs falhou; o lugar dele fica preto", piece.start_s
                )
        try:
            ffmpeg.black_clip(
                part, piece.duration_s,
                width=media.width, height=media.height, fps=media.fps,
                audio_rate=audio_rate,
            )
        except ffmpeg.FFmpegError:
            log.warning("nao consegui gerar o preto de %.2fs", piece.duration_s)
            continue
        parts.append(part)

    if not cut_files:
        raise ffmpeg.FFmpegError("nenhum corte pode ser feito")

    # the zip comes out BEFORE the assembly: if joining or adding the
    # soundtrack fails, the cut material is not lost. Only the cuts go in -- the
    # black of the gaps is nobody's material
    zip_path = _zip_segments(
        [p for p, _ in cut_files],
        [span for _, span in cut_files],
        out_dir / f"{index:02d}_cortes.zip",
    )

    dest: Path | None = out_dir / f"{index:02d}_custom.mp4"
    assembly_error: str | None = None
    try:
        joined = out_dir / f"{index:02d}_custom_raw.mp4"
        ffmpeg.concat(parts, joined, mute=muted)
        joined.replace(dest)
    except ffmpeg.FFmpegError as exc:
        log.exception("montagem de '%s' falhou; entrego so os cortes", item.title)
        assembly_error = str(exc)[:500]
        dest = None

    for p in parts:
        p.unlink(missing_ok=True)
    parts_dir.rmdir()

    # the thumbnail comes from the first cut, and not from the first second:
    # whoever started the montage with an empty space would get a black cover
    until_first_cut = 0.0
    for piece in pieces:
        if piece.is_cut:
            break
        until_first_cut += piece.duration_s
    thumb = (
        _thumb(dest, out_dir, index, at=until_first_cut + 0.2)
        if dest is not None else None
    )
    return RenderedClip(
        title=item.title or "Montagem",
        start_s=min(c.start_s for c in spec.cuts),
        end_s=max(c.end_s for c in spec.cuts),
        video=dest,
        thumb=thumb,
        duration_s=tl.total_duration_s(pieces),
        segments_zip=zip_path,
        meta={
            "segments": len(cut_files),
            "blackfill_s": round(
                sum(p.duration_s for p in pieces if p.black), 2
            ),
            "hand_made": True,
            "music_name": None,
            "original_audio": True,
            **({"render_error": assembly_error} if assembly_error else {}),
        },
    )


def _render_composition(
    source: Path,
    item: TimelineItem,
    media: ffmpeg.MediaInfo,
    out_dir: Path,
    index: int,
) -> RenderedClip:
    """Assembles in layers, through a filter graph.

    A background canvas covers the whole video and each clip is overlaid onto it
    at the right moment. The gap between clips stops being a special case: it is
    simply where nobody covered the background.

    Unlike the cut-and-splice path, here there is **no zip of cuts**: the pieces
    never come to exist as files, and cutting them out just for the zip would be
    paying for the assembly twice.
    """
    spec = item.timeline

    comp = compose_graph(
        spec,
        source=source,
        width=media.width,
        height=media.height,
        fps=media.fps,
        source_duration_s=media.duration_s,
        library=item.library,
    )

    dest: Path | None = out_dir / f"{index:02d}_custom.mp4"
    error_text: str | None = None
    try:
        ffmpeg.compose(comp, dest)
    except ffmpeg.FFmpegError as exc:
        log.exception("composicao de '%s' falhou", item.title)
        error_text = str(exc)[:500]
        dest = None

    clips = spec.clips
    layers = [l for l in spec.layers if not l.hidden]
    thumb = _thumb(dest, out_dir, index) if dest is not None else None
    return RenderedClip(
        title=item.title or "Montagem",
        start_s=min((c.start_s for c in clips), default=0.0),
        end_s=max((c.end_s for c in clips), default=0.0),
        video=dest,
        thumb=thumb,
        duration_s=spec.duration_s,
        meta={
            "segments": len(clips),
            "layers": len(layers),
            "composed": True,
            "hand_made": True,
            "media": len({c.media_id for c in clips if c.media_id}),
            "music_name": item.music_name,
            "original_audio": not spec.has_music,
            **({"render_error": error_text} if error_text else {}),
        },
    )


def _zip_segments(
    parts: list[Path], segments: list[tuple[float, float]], dest: Path
) -> Path | None:
    """Packs the individual cuts for download.

    It stores each cut **once**, in chronological order: when the montage repeats
    stretches, the same material would appear several times in the zip without
    adding anything for whoever re-edits. The name carries the instant the cut
    came from in the original recording.

    The files are already H.264 and compress no further, so the zip is only a
    packaging (ZIP_STORED) -- recompressing would just burn CPU.
    """
    # Keyed by the start alone, keeping the longest version: the montage's last
    # stretch is usually a trim of the same cut, and delivering both versions
    # would mean delivering the same material twice, one of them halved.
    unique: dict[float, tuple[float, Path]] = {}
    for part, (start, end) in zip(parts, segments):
        key = round(start, 2)
        current = unique.get(key)
        if current is None or (end - start) > current[0]:
            unique[key] = (end - start, part)
    if not unique:
        return None

    try:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
            for i, (start, (_dur, part)) in enumerate(sorted(unique.items()), start=1):
                minutes = int(start) // 60
                seconds = start - minutes * 60
                zf.write(part, f"{i:02d}_{minutes:02d}m{seconds:04.1f}s.mp4")
    except OSError:
        log.warning("nao consegui montar o zip dos cortes em %s", dest)
        return None
    return dest


def _thumb(video: Path, out_dir: Path, index: int, at: float = 0.2) -> Path | None:
    try:
        dest = out_dir / f"{index:02d}.jpg"
        return ffmpeg.thumbnail(video, dest, at=at)
    except ffmpeg.FFmpegError:
        log.warning("sem miniatura para %s", video.name)
        return None
