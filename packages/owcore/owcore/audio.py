"""Leitura de áudio e a forma de onda reduzida que o app desenha.

Mora aqui, e não no serviço de ritmo, porque agora são **dois** os áudios que o
editor desenha: a música que o usuário escolheu e o áudio da própria partida —
o tiro, a explosão, o barulho da jogada. Os dois viram a mesma coisa: alguns
milhares de picos entre 0 e 1.

Mandar a onda em vez do áudio é o ponto: o app precisa dela para *desenhar*, e
baixar minutos de som só para descobrir onde está o refrão sairia caro num
celular.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

#: Quantos picos por segundo a forma de onda guarda. 40 é o suficiente para o
#: desenho parecer o som num celular; guardar em resolução de amostra seria
#: mandar o áudio inteiro em JSON.
PEAKS_PER_S = 40

#: Teto de picos por faixa. Uma partida de 20 minutos não precisa de 48 mil
#: pontos para caber numa régua de mil pixels.
MAX_PEAKS = 6000


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """WAV PCM -> amostras em -1..1, mono."""
    with wave.open(str(path), "rb") as w:
        sr, ch, _width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def peaks_para(duration_s: float) -> int:
    """Quantos pontos usar para uma faixa desta duração."""
    return int(min(MAX_PEAKS, max(64, duration_s * PEAKS_PER_S)))


def waveform(data: np.ndarray, n: int) -> list[float]:
    """Envelope em `n` pontos, normalizado pelo pico da faixa.

    Normalizar pelo máximo, e não por um valor absoluto, é o que faz uma
    gravação baixa desenhar igual a uma alta — o desenho serve para achar o
    momento a olho, não para medir volume.
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


def waveform_de(path: Path) -> tuple[list[float], float]:
    """A onda de um WAV, com a duração que ela cobre.

    Devolve `([], 0.0)` para arquivo vazio ou ilegível: uma faixa sem onda é um
    desenho a menos, não um erro que valha derrubar o processamento.
    """
    try:
        data, sr = read_wav(Path(path))
    except (OSError, wave.Error, ValueError):
        return [], 0.0
    if sr <= 0 or data.size == 0:
        return [], 0.0
    duracao = data.size / sr
    return waveform(data, peaks_para(duracao)), round(duracao, 3)
