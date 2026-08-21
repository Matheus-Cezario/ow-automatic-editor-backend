"""Deteccao de ultimates inimigas.

Duas pistas independentes, cada uma fraca sozinha e razoavel juntas:

1. **Killfeed** -- casamento de template contra os icones de ultimate que o
   usuario colocar em `templates/ults/`. Sem templates o detector nao inventa
   nada: ele avisa no log e devolve zero eventos por essa via.
2. **Audio** -- toda ultimate inimiga vem acompanhada de uma fala alta e
   proxima. **Desligada por padrao**: num video sintetico, em que a fala e o
   unico som alto, ela acerta; em partida real, tiro, explosao e habilidade
   produzem picos iguais. Em 19 minutos de gameplay real ela rendeu 88
   "ultimates" -- e, sem gabarito de quais ultimates realmente aconteceram,
   nao ha como calibrar um limiar honesto. Ligue em `ults.audio_enabled` se
   quiser experimentar na sua gravacao.

O detector emite apenas `ULT_USED`. Decidir que uma ultimate foi *anulada*
exige cruzar com as eliminacoes, que sao de outro microsservico -- essa
correlacao e feita no editor, que e quem enxerga todos os eventos.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import TemplateBank, find_pulses, iter_frames

log = logging.getLogger(__name__)


# ------------------------------- killfeed -----------------------------------


def _detect_from_killfeed(
    killfeed_video: Path, profile: Profile, templates_dir: Path
) -> list[DetectionEvent]:
    cfg = profile.section("ults")
    bank = TemplateBank.from_dir(templates_dir)
    if not bank:
        log.warning(
            "nenhum template em %s -- a via de killfeed fica desligada. "
            "Recorte os icones de ultimate do seu proprio gameplay e salve ali.",
            templates_dir,
        )
        return []
    log.info("%d template(s) de ultimate carregado(s)", len(bank))

    roi = profile.roi("killfeed")
    times: list[float] = []
    scores: list[float] = []
    names: list[str] = []
    for frame in iter_frames(killfeed_video, fps_hint=roi.fps):
        name, score = bank.best_match(frame.bgr)
        times.append(frame.t)
        scores.append(score)
        names.append(name or "")

    threshold = float(cfg.get("template_threshold", 0.68))
    pulses = find_pulses(
        times,
        scores,
        rise=threshold,
        fall=threshold * 0.85,
        min_duration=0.0,
        min_gap=float(cfg.get("min_gap_s", 3.0)),
    )

    events: list[DetectionEvent] = []
    for p in pulses:
        window = [(t, s, n) for t, s, n in zip(times, scores, names) if p.start <= t <= p.end]
        best = max(window, key=lambda x: x[1]) if window else (p.start, p.peak, "")
        events.append(
            DetectionEvent(
                kind=EventKind.ULT_USED,
                t=round(p.start, 3),
                confidence=round(min(1.0, float(best[1])), 3),
                meta={"source": "killfeed", "ult": best[2], "score": round(float(best[1]), 3)},
            )
        )
    return events


# --------------------------------- audio ------------------------------------


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"esperado PCM 16 bits, recebi {width * 8} bits")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _detect_from_audio(audio_path: Path, profile: Profile) -> list[DetectionEvent]:
    """Pico de volume medido como *contraste local*, em dB.

    Um limiar absoluto nao serve: a mixagem varia de gravacao para gravacao.
    O que caracteriza uma fala de ultimate e ela ser muito mais alta do que os
    segundos em volta dela -- entao compara-se cada janela com a mediana movel
    da vizinhanca. O numero resultante e em dB, que e interpretavel e nao
    depende de escala.
    """
    cfg = profile.section("ults")
    data, sr = _read_wav_mono(audio_path)
    if data.size == 0:
        return []

    fps = 20  # janelas de 50 ms
    hop = max(1, sr // fps)
    n_frames = data.size // hop
    if n_frames < 4 * fps:
        return []

    frames = data[: n_frames * hop].reshape(n_frames, hop)
    db = 20.0 * np.log10(np.maximum(np.sqrt((frames**2).mean(axis=1)), 1e-6))

    window = max(3, int(float(cfg.get("audio_context_s", 5.0)) * fps)) | 1
    pad = np.pad(db, (window // 2, window // 2), mode="edge")
    baseline = np.median(sliding_window_view(pad, window), axis=-1)[:n_frames]
    excess = db - baseline
    times = np.arange(n_frames) / fps

    threshold = float(cfg.get("audio_spike_db", 8.0))
    pulses = find_pulses(
        [float(t) for t in times],
        [float(v) for v in excess],
        rise=threshold,
        fall=threshold * 0.5,
        min_duration=0.0,
        min_gap=float(cfg.get("audio_min_gap_s", 8.0)),
    )
    return [
        DetectionEvent(
            kind=EventKind.ULT_USED,
            t=round(p.start, 3),
            confidence=0.55,
            meta={"source": "audio", "excess_db": round(float(p.peak), 2)},
        )
        for p in pulses
    ]


# --------------------------------- fusao ------------------------------------


def detect_ults(
    killfeed_video: Path | None,
    audio_path: Path | None,
    profile: Profile,
    templates_dir: Path,
) -> list[DetectionEvent]:
    cfg = profile.section("ults")
    events: list[DetectionEvent] = []

    if killfeed_video and Path(killfeed_video).exists():
        events += _detect_from_killfeed(Path(killfeed_video), profile, templates_dir)
    if not bool(cfg.get("audio_enabled", False)):
        if audio_path:
            log.info(
                "via de audio desligada (ults.audio_enabled=false): em partida "
                "real ela confunde tiro e explosao com fala de ultimate"
            )
    elif audio_path and Path(audio_path).exists():
        try:
            events += _detect_from_audio(Path(audio_path), profile)
        except Exception as exc:  # audio ruim nao pode derrubar o detector
            log.warning("via de audio indisponivel: %s", exc)

    events.sort(key=lambda e: e.t)
    merged = _merge_sources(events, float(cfg.get("min_gap_s", 3.0)))
    log.info(
        "%d ultimate(s) inimiga(s) -- %d apos fundir as duas pistas",
        len(events),
        len(merged),
    )
    return merged


def _merge_sources(
    events: list[DetectionEvent], window: float
) -> list[DetectionEvent]:
    """Duas pistas apontando o mesmo instante viram um evento so, com mais
    confianca do que qualquer uma delas teria sozinha."""
    merged: list[DetectionEvent] = []
    for e in events:
        if merged and e.t - merged[-1].t <= window:
            prev = merged[-1]
            sources = set(str(prev.meta.get("source", "")).split("+")) | {
                str(e.meta.get("source", ""))
            }
            prev.meta["source"] = "+".join(sorted(s for s in sources if s))
            prev.confidence = round(
                min(1.0, 1.0 - (1.0 - prev.confidence) * (1.0 - e.confidence)), 3
            )
            continue
        merged.append(e.model_copy(deep=True))
    return merged
