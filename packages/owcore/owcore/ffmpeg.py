"""Thin wrappers around ffmpeg/ffprobe."""

from __future__ import annotations

import json
import logging
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import get_settings
from .models import RoiSpec

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


def _run(cmd: Sequence[str]) -> str:
    log.debug("exec: %s", " ".join(cmd))
    proc = subprocess.run(
        list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        raise FFmpegError(
            f"{cmd[0]} saiu com {proc.returncode}:\n" + "\n".join(tail)
        )
    return proc.stdout


def _run_acompanhado(
    cmd: Sequence[str], duracao_s: float, avanco: Callable[[float], None]
) -> None:
    """Runs ffmpeg reporting how far along it is, from 0 to 1.

    ffmpeg only says where it is when asked: `-progress pipe:1` makes it write
    `out_time_us=<microseconds>` twice a second. The alternative is guessing by
    the wall clock, and on an 11-minute recording that is minutes off -- the
    cropping does not advance at a constant rate.

    stderr is merged into stdout on purpose: reading two pipes with a single
    loop deadlocks when the second one fills up, and a damaged recording fills
    it up.
    """
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    log.debug("exec: %s", " ".join(cmd))
    proc = subprocess.Popen(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    # only the last lines matter, and only they are kept: a damaged recording
    # makes ffmpeg complain **per frame**, and an uncapped list grew with the
    # video duration only to hand over 25 lines in the end
    resto: deque[str] = deque(maxlen=25)
    assert proc.stdout is not None
    for linha in proc.stdout:
        linha = linha.strip()
        chave, _, valor = linha.partition("=")
        if chave == "out_time_us" and valor.lstrip("-").isdigit():
            if duracao_s > 0:
                avanco(max(0.0, min(1.0, int(valor) / 1e6 / duracao_s)))
        elif "=" not in linha and linha:
            # not a progress line: this is ffmpeg complaining
            resto.append(linha)
    if proc.wait() != 0:
        raise FFmpegError(
            f"{cmd[0]} saiu com {proc.returncode}:\n" + "\n".join(resto)
        )


@dataclass(slots=True)
class MediaInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    #: audio sample rate, 0 when there is no track. The montage's black
    #: filler needs it: the `concat` demuxer refuses to join pieces whose audio
    #: does not match.
    audio_rate: int = 0


def probe(path: Path | str) -> MediaInfo:
    """Measures a file. Accepts an `http` address too: ffprobe pulls only the
    header, via `Range`."""
    s = get_settings()
    try:
        out = _run(
            [
                s.ffprobe, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ]
        )
    except FFmpegError as exc:
        # ffprobe's own wording is cryptic ("moov atom not found"), and what it
        # means almost every time is a file that arrived incomplete. Saying so
        # here is what turns a report nobody can act on into one that names the
        # thing to do.
        raise FFmpegError(
            f"nao consegui ler o video em {path}: o arquivo parece corrompido "
            f"ou incompleto.\n{exc}"
        ) from exc
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((x for x in streams if x.get("codec_type") == "video"), None)
    audio = next((x for x in streams if x.get("codec_type") == "audio"), None)
    if video is None:
        raise FFmpegError(f"nenhum stream de vídeo em {path}")

    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration = float(data.get("format", {}).get("duration") or 0.0)
    if not duration:
        duration = float(video.get("duration") or 0.0)

    return MediaInfo(
        duration_s=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        has_audio=audio is not None,
        audio_rate=int(float((audio or {}).get("sample_rate") or 0)),
    )


def extract_rois(
    src: Path,
    rois: Sequence[RoiSpec],
    out_dir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, Path]:
    """A single decode of the video producing every crop.

    This is the centre of the system's economics: the heavy video is read once
    and each detector receives only the band of pixels that matters, already at
    low resolution and low FPS.

    `on_progress` recebe 0..1 conforme o recorte anda. Vale passar: numa
    match recording this call alone is ~3/4 of the total analysis time, and
    without it the screen sits on the same number for minutes.
    """
    if not rois:
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = get_settings()

    labels = [f"s{i}" for i in range(len(rois))]
    chains = [f"[0:v]split={len(rois)}" + "".join(f"[{l}]" for l in labels)]
    outputs: dict[str, Path] = {}
    args: list[str] = []

    for i, (roi, label) in enumerate(zip(rois, labels)):
        if roi.fullscreen:
            crop = ""
        else:
            crop = (
                f"crop=w=iw*{roi.w:.6f}:h=ih*{roi.h:.6f}"
                f":x=iw*{roi.x:.6f}:y=ih*{roi.y:.6f},"
            )
        # `min(width_px, iw)`: never *upscale*. On a 360p recording the
        # native crop is ~100px wide, and stretching it to 320 only invents
        # pixels and triples the file without adding information.
        # (a virgula precisa de escape: no filtergraph ela separa filtros)
        chains.append(
            f"[{label}]{crop}scale=min({roi.width_px}\\,iw):-2:flags=bilinear,"
            f"fps={roi.fps}[o{i}]"
        )
        dest = out_dir / f"{roi.name}.mp4"
        outputs[roi.name] = dest
        args += [
            "-map", f"[o{i}]", "-an",
            # CRF 22: measured on real gameplay, kill detection is identical
            # from CRF 16 to 28 -- not worth paying for quality no detector
            # uses. Higher than that starts eating into saturation, which is
            # precisely the discriminator.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            # Explicit BT.601 tags -- and this is NOT cosmetic. The detectors
            # decide by saturation, and the YUV->RGB matrix changes exactly
            # that. The same file tagged BT.709 was read with saturation 231 on
            # the host machine and 205 inside the container (one decoder
            # honours the tag, the other assumes 601 because the crop is
            # small) -- enough of a difference for the detector to find 20
            # kills in one place and 10 in the other, with identical code.
            # Tagged BT.601, which is what both assume for frames this size,
            # the two read exactly 220.
            "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
            "-colorspace", "smpte170m", "-color_range", "tv",
            "-movflags", "+faststart",
            str(dest),
        ]

    cmd = [s.ffmpeg, "-y", "-v", "error", "-i", str(src),
           "-filter_complex", ";".join(chains), *args]
    if on_progress is None:
        _run(cmd)
    else:
        _run_acompanhado(cmd, probe(src).duration_s, on_progress)
    return outputs


def extract_audio(src: Path, dest: Path, sample_rate: int = 22050) -> Path | None:
    """Extracts mono WAV audio. Returns None if the media has no audio track."""
    if not probe(src).has_audio:
        return None
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([s.ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn",
          "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dest)])
    return dest


def cut(
    src: Path,
    start: float,
    end: float,
    dest: Path,
    *,
    fade: float = 0.0,
    scale_width: int | None = None,
    mute: bool = False,
) -> Path:
    """Cuts [start, end) re-encoding, so the cut lands on the exact frame."""
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, start)
    dur = max(0.05, end - start)

    vf: list[str] = []
    if scale_width:
        vf.append(f"scale={scale_width}:-2:flags=lanczos")
    if fade > 0:
        vf.append(f"fade=t=in:st=0:d={fade:.3f}")
        vf.append(f"fade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}")

    cmd = [s.ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}",
           "-i", str(src), "-t", f"{dur:.3f}"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p"]
    cmd += ["-an"] if mute else ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(dest)]
    _run(cmd)
    return dest


def concat(parts: Sequence[Path], dest: Path, *, mute: bool = False) -> Path:
    """Concatenates via the `concat` demuxer, re-encoding (immune to PTS drift)."""
    if not parts:
        raise FFmpegError("nada para concatenar")
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.with_suffix(".txt")
    listing.write_text(
        "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in parts),
        encoding="utf-8",
    )
    cmd = [s.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
           "-i", str(listing), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += ["-an"] if mute else ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(dest)]
    _run(cmd)
    listing.unlink(missing_ok=True)
    return dest


def add_music(
    video: Path, music: Path, dest: Path, *, music_start: float = 0.0,
    fade_out: float = 1.5,
) -> Path:
    """Swaps the video audio for the music, trimmed to the video length."""
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = probe(video).duration_s
    af = f"afade=t=out:st={max(0.0, dur - fade_out):.3f}:d={fade_out:.3f}"
    _run([
        s.ffmpeg, "-y", "-v", "error",
        "-i", str(video),
        "-ss", f"{music_start:.3f}", "-i", str(music),
        "-map", "0:v:0", "-map", "1:a:0",
        "-af", af, "-shortest",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-movflags", "+faststart", str(dest),
    ])
    return dest


def black_clip(
    dest: Path,
    duration: float,
    *,
    width: int,
    height: int,
    fps: float,
    audio_rate: int = 0,
) -> Path:
    """A stretch of black screen, to fill a gap in the montage.

    When the user leaves space between two blocks, this is what plays there:
    black with the music over it. It comes out with the same parameters as the
    neighbouring cuts because `concat` joins before re-encoding and does not
    forgive divergence -- audio included, hence the silence at the same rate
    when the cuts have sound.
    """
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.02, duration)
    fps = fps if fps > 0 else 30.0

    cmd = [s.ffmpeg, "-y", "-v", "error",
           "-f", "lavfi", "-i",
           f"color=c=black:s={int(width)}x{int(height)}:r={fps:.3f}"]
    if audio_rate > 0:
        cmd += ["-f", "lavfi", "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={audio_rate}"]
    cmd += ["-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p"]
    cmd += (["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", str(audio_rate)]
            if audio_rate > 0 else ["-an"])
    cmd += ["-movflags", "+faststart", str(dest)]
    _run(cmd)
    return dest


def proxy(src: Path, dest: Path, *, width: int = 640, fps: float = 24.0) -> Path:
    """A small copy of a video, for the editor's monitor.

    The match recording gets its own for free, inside the decode that already
    extracts the crops. An **imported** video does not go through that pass,
    and
    por isso precisa da sua aqui — ainda vale a pena: buscar dentro do arquivo
    dragging the full file on every seek is what brings the player down.
    """
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([
        s.ffmpeg, "-y", "-v", "error", "-i", str(src),
        # the comma needs escaping: in the filtergraph it separates filters
        "-vf", f"scale=min({width}\\,iw):-2:flags=bilinear,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dest),
    ])
    return dest


def compose(comp, dest: Path) -> Path:
    """Runs the graph built by `owcore.compose`.

    This is the layered-montage path. The cut-and-splice one still exists for
    single-layer montages, and is more resilient: there a cut that fails costs
    only itself, here an error in the graph brings the whole render down.
    """
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [s.ffmpeg, "-y", "-v", "error"]
    cmd += comp.input_args()
    cmd += ["-filter_complex", comp.filter_complex, "-map", comp.video_map]
    if comp.audio_map:
        cmd += ["-map", comp.audio_map, "-c:a", "aac", "-b:a", "192k", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(comp.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    _run(cmd)
    return dest


def thumbnail(src: Path, dest: Path, at: float = 0.0, width: int = 480) -> Path:
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([s.ffmpeg, "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(src),
          "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)])
    return dest
