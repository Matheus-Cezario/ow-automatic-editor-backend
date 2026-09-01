"""Extraction of the beat grid from the music the user chose.

Uses `librosa` when available. If it is not installed (or fails), it falls back
to an estimator of its own, in pure numpy: energy envelope -> autocorrelation ->
dominant period -> the phase that best matches the onsets. The system goes on
building to the beat instead of simply giving up.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from owcore.audio import peaks_for, read_wav, waveform
from owcore.config import get_settings
from owcore.models import BeatGrid

log = logging.getLogger(__name__)

SR = 22050
HOP_FPS = 100.0  # 10 ms windows for the envelope


def _decode_to_wav(src: Path, dest: Path) -> Path:
    """Any audio format -> mono PCM WAV, via ffmpeg.

    The destination can coincide with the source: with S3 storage the blob is
    downloaded into the same working directory where the decode was going to
    write, and ffmpeg refuses input and output being the same file. In that case
    the output is renamed instead of blowing up.
    """
    s = get_settings()
    src = Path(src).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.resolve() == src:
        dest = dest.with_name(f"{dest.stem}_decodificado{dest.suffix}")
    proc = subprocess.run(
        [s.ffmpeg, "-y", "-v", "error", "-i", str(src), "-vn",
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg nao decodificou a musica: {proc.stderr[-500:]}")
    return dest


# ------------------------------ numpy fallback -------------------------------


def _onset_envelope(data: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr / HOP_FPS))
    n = data.size // hop
    if n < 16:
        return np.zeros(0, np.float32), HOP_FPS
    energy = np.sqrt((data[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(energy, 1e-6))
    # poor man's spectral flux: only the rising part matters for the onset
    flux = np.diff(db, prepend=db[0])
    return np.maximum(flux, 0.0), sr / hop


def _estimate_beats(data: np.ndarray, sr: int, duration: float) -> BeatGrid:
    env, fps = _onset_envelope(data, sr)
    if env.size < 16:
        return BeatGrid(bpm=120.0, beats=list(np.arange(0, duration, 0.5)))

    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1 :]
    lo = int(fps * 60.0 / 200.0)  # 200 BPM
    hi = int(fps * 60.0 / 60.0)   # 60 BPM
    hi = min(hi, ac.size - 1)
    if hi <= lo:
        return BeatGrid(bpm=120.0, beats=list(np.arange(0, duration, 0.5)))

    lag = int(np.argmax(ac[lo:hi])) + lo
    period = lag / fps
    bpm = 60.0 / period

    # phase: shift the grid until it sums the most onset energy
    step = max(1, lag // 24)
    offsets = np.arange(0, lag, step)
    scores = [env[int(o) :: lag].sum() for o in offsets]
    phase = float(offsets[int(np.argmax(scores))]) / fps

    beats = list(np.arange(phase, duration, period))
    return BeatGrid(bpm=round(bpm, 2), beats=[round(float(b), 3) for b in beats])


# --------------------------------- publico ----------------------------------


@dataclass(slots=True)
class TrackAnalysis:
    """Everything the app needs to know about a track to build on top of it."""

    grid: BeatGrid
    duration_s: float
    #: waveform reduced to a few thousand peaks in 0..1
    peaks: list[float]


def analyze_track(src: Path, work_dir: Path) -> TrackAnalysis:
    """Full analysis of the track: beats, duration and waveform.

    It runs **before** any video exists -- it is what the editing screen uses to
    draw the music and snap the cuts to the beat. A single decode serves all
    three.
    """
    wav = _decode_to_wav(Path(src), Path(work_dir) / "track.wav")
    data, sr = read_wav(wav)
    duration = data.size / sr if sr else 0.0
    if duration <= 0:
        raise ValueError("musica vazia ou ilegivel")
    return TrackAnalysis(
        grid=_beats(data, sr, duration),
        duration_s=round(duration, 3),
        peaks=waveform(data, peaks_for(duration)),
    )


def _beats(data: np.ndarray, sr: int, duration: float) -> BeatGrid:
    try:
        import librosa

        tempo, frames = librosa.beat.beat_track(y=data, sr=sr, units="frames")
        beats = librosa.frames_to_time(frames, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if len(beats) >= 2:
            log.info("librosa: %.1f BPM, %d batidas", bpm, len(beats))
            return BeatGrid(
                bpm=round(bpm, 2), beats=[round(float(b), 3) for b in beats]
            )
        log.warning("librosa devolveu batidas de menos; usando o estimador proprio")
    except Exception as exc:
        log.warning("librosa indisponivel (%s); usando o estimador proprio", exc)

    grid = _estimate_beats(data, sr, duration)
    log.info("estimador proprio: %.1f BPM, %d batidas", grid.bpm, len(grid.beats))
    return grid
