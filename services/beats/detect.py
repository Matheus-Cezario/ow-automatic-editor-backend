"""Extracao da grade de batidas da musica escolhida pelo usuario.

Usa `librosa` quando disponivel. Se nao estiver instalado (ou falhar), cai num
estimador proprio, em numpy puro: envelope de energia -> autocorrelacao ->
periodo dominante -> fase que melhor casa com os picos. O sistema continua
montando no ritmo em vez de simplesmente desistir.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from owcore.config import get_settings
from owcore.models import BeatGrid

log = logging.getLogger(__name__)

SR = 22050
HOP_FPS = 100.0  # janelas de 10 ms para o envelope


def _decode_to_wav(src: Path, dest: Path) -> Path:
    """Qualquer formato de audio -> WAV mono PCM, via ffmpeg.

    O destino pode coincidir com a origem: com storage S3 o blob e baixado
    para o mesmo diretorio de trabalho onde a decodificacao ia escrever, e o
    ffmpeg recusa entrada e saida no mesmo arquivo. Nesse caso renomeia-se a
    saida em vez de estourar.
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


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


# ------------------------------ fallback numpy -------------------------------


def _onset_envelope(data: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr / HOP_FPS))
    n = data.size // hop
    if n < 16:
        return np.zeros(0, np.float32), HOP_FPS
    energy = np.sqrt((data[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(energy, 1e-6))
    # fluxo espectral do pobre: so a parte que sobe importa para o ataque
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

    # fase: desloca a grade ate somar o maximo de energia de ataque
    step = max(1, lag // 24)
    offsets = np.arange(0, lag, step)
    scores = [env[int(o) :: lag].sum() for o in offsets]
    phase = float(offsets[int(np.argmax(scores))]) / fps

    beats = list(np.arange(phase, duration, period))
    return BeatGrid(bpm=round(bpm, 2), beats=[round(float(b), 3) for b in beats])


# --------------------------------- publico ----------------------------------


@dataclass(slots=True)
class TrackAnalysis:
    """Tudo o que o app precisa saber de uma musica para montar em cima dela."""

    grid: BeatGrid
    duration_s: float
    #: forma de onda reduzida a alguns milhares de picos em 0..1
    peaks: list[float]


#: Quantos picos por segundo de musica a forma de onda guarda. 40 e o
#: suficiente para o desenho parecer a musica num celular; guardar a onda em
#: resolucao de amostra seria mandar o audio inteiro em JSON.
PEAKS_PER_S = 40
MAX_PEAKS = 6000


def _waveform(data: np.ndarray, n: int) -> list[float]:
    """Envelope em `n` pontos, normalizado pelo pico da musica.

    Normalizar pelo maximo, e nao por um valor absoluto, e o que faz uma musica
    gravada baixo desenhar igual a uma gravada alta -- o desenho serve para
    achar o refrao, nao para medir volume.
    """
    if data.size == 0 or n <= 0:
        return []
    n = min(n, data.size)
    corte = (data.size // n) * n
    if corte == 0:
        return []
    blocos = np.abs(data[:corte]).reshape(n, -1).max(axis=1)
    topo = float(blocos.max())
    if topo <= 0:
        return [0.0] * n
    return [round(float(v), 3) for v in (blocos / topo)]


def analyze_track(src: Path, work_dir: Path) -> TrackAnalysis:
    """Analise completa da musica: batidas, duracao e forma de onda.

    Roda **antes** de existir video nenhum -- e o que a tela de montagem usa
    para desenhar a musica e grudar os cortes na batida. Uma unica decodificacao
    serve as tres coisas.
    """
    wav = _decode_to_wav(Path(src), Path(work_dir) / "track.wav")
    data, sr = _read_wav(wav)
    duration = data.size / sr if sr else 0.0
    if duration <= 0:
        raise ValueError("musica vazia ou ilegivel")
    n = int(min(MAX_PEAKS, max(64, duration * PEAKS_PER_S)))
    return TrackAnalysis(
        grid=_beats(data, sr, duration),
        duration_s=round(duration, 3),
        peaks=_waveform(data, n),
    )


def analyze_music(src: Path, work_dir: Path) -> BeatGrid:
    wav = _decode_to_wav(Path(src), Path(work_dir) / "music.wav")
    data, sr = _read_wav(wav)
    duration = data.size / sr if sr else 0.0
    if duration <= 0:
        raise ValueError("musica vazia ou ilegivel")
    return _beats(data, sr, duration)


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
