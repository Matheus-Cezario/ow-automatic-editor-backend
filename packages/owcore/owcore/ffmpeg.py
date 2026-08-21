"""Wrappers finos em cima do ffmpeg/ffprobe."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


@dataclass(slots=True)
class MediaInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    #: taxa de amostragem do audio, 0 quando nao ha faixa. O preenchimento
    #: preto da montagem manual precisa dela: o demuxer `concat` recusa juntar
    #: pedacos cujo audio nao bate.
    audio_rate: int = 0


def probe(path: Path) -> MediaInfo:
    s = get_settings()
    out = _run(
        [
            s.ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
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


def extract_rois(src: Path, rois: Sequence[RoiSpec], out_dir: Path) -> dict[str, Path]:
    """Uma única decodificação do vídeo produzindo todos os recortes.

    É o ponto central da economia do sistema: o vídeo pesado é lido uma vez e
    cada detector recebe só a faixa de pixels que interessa, já em baixa
    resolução e baixo FPS.
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
        # `min(width_px, iw)`: nunca *ampliar*. Numa gravacao 360p o recorte
        # nativo tem ~100px de largura, e estica-lo ate 320 so inventa pixels e
        # triplica o arquivo sem acrescentar informacao.
        # (a virgula precisa de escape: no filtergraph ela separa filtros)
        chains.append(
            f"[{label}]{crop}scale=min({roi.width_px}\\,iw):-2:flags=bilinear,"
            f"fps={roi.fps}[o{i}]"
        )
        dest = out_dir / f"{roi.name}.mp4"
        outputs[roi.name] = dest
        args += [
            "-map", f"[o{i}]", "-an",
            # CRF 22: medido em gameplay real, a deteccao de eliminacoes e
            # identica de CRF 16 a 28 -- nao vale pagar por qualidade que
            # detector nenhum usa. Mais alto que isso comeca a comer a
            # saturacao, que e justamente o discriminador.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            # Tags BT.601 explicitas -- e isto NAO e cosmetico. Os detectores
            # decidem por saturacao, e a matriz YUV->RGB muda justamente a
            # saturacao. O mesmo arquivo marcado como BT.709 era lido com
            # saturacao 231 na maquina host e 205 dentro do container (um
            # decodificador honra a tag, o outro assume 601 porque o recorte e
            # pequeno) -- diferenca suficiente para o detector achar 20
            # eliminacoes num lugar e 10 no outro, com o mesmo codigo. Marcado
            # como BT.601, que e o que ambos assumem para quadros deste tamanho,
            # os dois leem exatamente 220.
            "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
            "-colorspace", "smpte170m", "-color_range", "tv",
            "-movflags", "+faststart",
            str(dest),
        ]

    _run([s.ffmpeg, "-y", "-v", "error", "-i", str(src),
          "-filter_complex", ";".join(chains), *args])
    return outputs


def extract_audio(src: Path, dest: Path, sample_rate: int = 22050) -> Path | None:
    """Extrai áudio mono WAV. Devolve None se a mídia não tiver faixa de áudio."""
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
    """Corta [start, end) reencodando, para que o corte caia no frame exato."""
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
    """Concatena via demuxer `concat`, reencodando (imune a diferenças de PTS)."""
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
    """Troca o áudio do vídeo pela música, cortando no tamanho do vídeo."""
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
    """Um trecho de tela preta, para tapar buraco na montagem manual.

    Quando o usuário deixa espaço entre dois blocos, é isto que toca ali: preto
    com a música por cima. Sai com os mesmos parâmetros dos cortes vizinhos
    porque o `concat` junta antes de reencodar e não perdoa divergência —
    inclusive no áudio, daí o silêncio na mesma taxa quando os cortes têm som.
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


def thumbnail(src: Path, dest: Path, at: float = 0.0, width: int = 480) -> Path:
    s = get_settings()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([s.ffmpeg, "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(src),
          "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)])
    return dest
