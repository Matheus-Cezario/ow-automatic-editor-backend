"""Motor de regras: transforma eventos detectados em highlights.

Função pura (eventos → highlights), sem I/O, para ser testável sozinha.
"""

from __future__ import annotations

import random
from typing import Sequence

from pydantic import BaseModel, Field

from .models import BeatGrid, DetectionEvent, EventKind, HighlightKind, JobParams


class Highlight(BaseModel):
    kind: HighlightKind
    start: float
    end: float
    score: float = 0.0
    title: str = ""
    #: instantes que originaram o highlight (usado pela montagem na batida)
    beats_at: list[float] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _times(events: Sequence[DetectionEvent], kind: EventKind) -> list[float]:
    return sorted(e.t for e in events if e.kind == kind)


def _cluster(times: Sequence[float], window: float, minimum: int) -> list[list[float]]:
    """Agrupa instantes em rajadas: ``minimum`` ocorrências dentro de ``window``,
    estendendo a rajada enquanto os intervalos continuarem curtos."""
    clusters: list[list[float]] = []
    i = 0
    n = len(times)
    while i < n:
        j = i
        while j + 1 < n and times[j + 1] - times[i] <= window:
            j += 1
        if j - i + 1 >= minimum:
            # a rajada continua enquanto as mortes seguirem se encadeando
            while j + 1 < n and times[j + 1] - times[j] <= window:
                j += 1
            clusters.append(list(times[i : j + 1]))
            i = j + 1
        else:
            i += 1
    return clusters


def _has_death_between(deaths: Sequence[float], a: float, b: float) -> bool:
    return any(a <= d <= b for d in deaths)


def derive_negated_ults(
    events: Sequence[DetectionEvent], window_s: float
) -> list[DetectionEvent]:
    """Cruza `ULT_USED` com `KILL` para produzir `ULT_NEGATED`.

    Fica aqui, e nao no detector de ults, de proposito: nenhum detector sozinho
    ve os dois tipos de evento. Correlacao entre microsservicos e trabalho de
    quem agrega -- o editor.
    """
    kills = _times(events, EventKind.KILL)
    out: list[DetectionEvent] = []
    for ult in (e for e in events if e.kind == EventKind.ULT_USED):
        after = [k for k in kills if ult.t <= k <= ult.t + window_s]
        if not after:
            continue
        out.append(
            DetectionEvent(
                kind=EventKind.ULT_NEGATED,
                t=round(after[0], 3),
                confidence=round(min(1.0, ult.confidence * 0.9), 3),
                meta={
                    "ult_at": ult.t,
                    "delay_s": round(after[0] - ult.t, 2),
                    "ult": ult.meta.get("ult"),
                    "source": ult.meta.get("source"),
                },
            )
        )
    return out


def build_highlights(
    events: Sequence[DetectionEvent],
    params: JobParams,
    duration_s: float,
) -> list[Highlight]:
    """Lista o que dá para gerar com os eventos encontrados.

    Roda logo depois da análise, **antes** de existir qualquer música: o ritmo
    entra só na hora de cortar, e cada vídeo pode ter o seu.
    """
    events = list(events)
    # ultimates anuladas so existem cruzando duas fontes; derive antes de tudo
    if not any(e.kind == EventKind.ULT_NEGATED for e in events):
        events += derive_negated_ults(events, params.ult_negate_window_s)

    kills = _times(events, EventKind.KILL)
    deaths = _times(events, EventKind.DEATH)
    escapes = _times(events, EventKind.ESCAPE)
    negated = _times(events, EventKind.ULT_NEGATED)
    sleeps = _times(events, EventKind.SLEEP)
    stuns = _times(events, EventKind.STUN)

    highlights: list[Highlight] = []

    # ── rajadas de eliminação ───────────────────────────────────────────────
    for cluster in _cluster(kills, params.multikill_window_s, params.multikill_min):
        span = cluster[-1] - cluster[0]
        solo = (
            len(cluster) >= params.solo_wipe_min
            and span <= params.solo_wipe_window_s
            and not _has_death_between(deaths, cluster[0], cluster[-1])
        )
        kind = HighlightKind.SOLO_WIPE if solo else HighlightKind.MULTIKILL
        title = (
            f"{len(cluster)} eliminações sozinho"
            if solo
            else f"{len(cluster)} eliminações em {span:.0f}s"
        )
        highlights.append(
            Highlight(
                kind=kind,
                start=max(0.0, cluster[0] - params.pre_roll_s),
                end=min(duration_s, cluster[-1] + params.post_roll_s),
                score=len(cluster) * (2.0 if solo else 1.0)
                + (1.0 / (span + 1.0)) * len(cluster),
                title=title,
                beats_at=list(cluster),
                meta={"kills": len(cluster), "span_s": round(span, 2), "solo": solo},
            )
        )

    # ── fugas (sobreviver com vida baixa, várias vezes seguidas) ────────────
    for cluster in _cluster(escapes, params.escape_window_s, params.escape_min_events):
        if _has_death_between(deaths, cluster[0], cluster[-1]):
            continue
        highlights.append(
            Highlight(
                kind=HighlightKind.ESCAPE,
                start=max(0.0, cluster[0] - params.pre_roll_s),
                end=min(duration_s, cluster[-1] + params.post_roll_s),
                score=len(cluster) * 1.5,
                title=f"Sobreviveu {len(cluster)}x na corda bamba",
                beats_at=list(cluster),
                meta={"escapes": len(cluster)},
            )
        )

    # ── montagem no ritmo ───────────────────────────────────────────────────
    # Usa *todas* as eliminações, inclusive as que já viraram rajada: um trecho
    # aproveitado num vídeo continua disponível para os outros. Cada vídeo é uma
    # montagem independente, não uma partilha do material.
    if params.make_beat_montage:
        if kills:
            highlights.append(
                _montage(
                    HighlightKind.BEAT_MONTAGE,
                    f"Montagem no ritmo — {len(kills)} eliminações",
                    kills,
                    params,
                    duration_s,
                )
            )
        if negated:
            highlights.append(
                _montage(
                    HighlightKind.ULT_MONTAGE,
                    f"Ultimates anuladas — {len(negated)}",
                    negated,
                    params,
                    duration_s,
                )
            )
        if sleeps:
            highlights.append(
                _montage(
                    HighlightKind.SLEEP_MONTAGE,
                    f"Dardos no alvo — {len(sleeps)}",
                    sleeps,
                    params,
                    duration_s,
                )
            )
        if stuns:
            highlights.append(
                _montage(
                    HighlightKind.STUN_MONTAGE,
                    f"Pedradas certeiras — {len(stuns)}",
                    stuns,
                    params,
                    duration_s,
                )
            )

    highlights.sort(key=lambda h: (-h.score, h.start))
    return highlights


def _montage(
    kind: HighlightKind,
    title: str,
    moments: Sequence[float],
    params: JobParams,
    duration_s: float,
) -> Highlight:
    """Um highlight-guarda-chuva: `beats_at` guarda os instantes e o editor
    corta um micro-clipe por instante, com duração alinhada às batidas."""
    return Highlight(
        kind=kind,
        start=max(0.0, min(moments) - params.pre_roll_s),
        end=min(duration_s, max(moments) + params.post_roll_s),
        score=len(moments) * 0.8,
        title=title,
        beats_at=list(moments),
        meta={"moments": len(moments)},
    )


def montage_segments(
    moments: Sequence[float],
    beats: BeatGrid | None,
    clip_beats: int,
    duration_s: float,
) -> list[tuple[float, float]]:
    """Converte instantes em pares (início, fim) de micro-clipes.

    Com grade de batidas, a duração de cada clipe vira exatamente N intervalos
    entre batidas — assim, concatenados, as trocas de cena caem na percussão.
    Sem música, cai num tamanho fixo razoável.
    """
    if beats and len(beats.beats) >= 2:
        intervals = [b - a for a, b in zip(beats.beats, beats.beats[1:])]
        beat_len = sorted(intervals)[len(intervals) // 2]  # mediana
    else:
        beat_len = 60.0 / 120.0
    clip_len = max(0.6, beat_len * max(1, clip_beats))

    segments: list[tuple[float, float]] = []
    for t in sorted(moments):
        # o instante da eliminação fica a ~70% do clipe: sobra antecipação e um respiro
        start = max(0.0, t - clip_len * 0.7)
        end = min(duration_s, start + clip_len)
        if end - start < 0.3:
            continue
        if segments and start < segments[-1][1]:
            # sobreposição: estende o clipe anterior em vez de repetir imagem
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, end)
            continue
        segments.append((start, end))
    return segments


def fit_to_window(
    segments: Sequence[tuple[float, float]],
    target_s: float | None,
    *,
    loop: bool,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Ajusta a montagem à janela de música escolhida pelo usuário.

    * sem janela (``target_s`` None): devolve os segmentos como estão;
    * com ``loop``: repete os trechos até a montagem ter **exatamente** a
      duração pedida, aparando o último;
    * sem ``loop``: acrescenta trechos enquanto couberem, e para -- o vídeo sai
      do tamanho que der, **sem nunca passar** da duração pedida.

    Ao repetir, a ordem é **sorteada**: os trechos são embaralhados a cada
    passada, então a montagem não vira a mesma sequência repetida em loop, e um
    trecho não reaparece antes de todos os outros terem entrado. Sem repetição a
    ordem cronológica é mantida.

    Como cada trecho dura um número inteiro de intervalos entre batidas, repetir
    não estraga a sincronia: as trocas de cena continuam caindo na percussão. Só
    o corte final pode ficar fora do tempo, e é onde a música acaba de qualquer
    jeito.
    """
    segments = list(segments)
    if target_s is None or target_s <= 0 or not segments:
        return segments

    rng = rng or random.Random(0)
    out: list[tuple[float, float]] = []
    total = 0.0
    fila: list[tuple[float, float]] = []
    entregues = 0

    while True:
        if not fila:
            if loop:
                fila = segments[:]
                rng.shuffle(fila)
            else:
                fila = segments[entregues:]
                if not fila:
                    break
        start, end = fila.pop(0)
        length = end - start

        if total + length <= target_s + 1e-6:
            out.append((start, end))
            total += length
            entregues += 1
            if loop and total >= target_s - 1e-6:
                break
            continue

        if not loop:
            break
        # apara o último trecho para fechar a duração exata
        remaining = target_s - total
        if remaining > 0.1:
            out.append((start, start + remaining))
        break

    return out
