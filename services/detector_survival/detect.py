"""Deteccao de sobrevivencia: vida baixa, interrupcao e fuga.

Entrada: uma tira fina do canto inferior esquerdo -- so a barra de vida.

**Como isto era antes, e por que mudou.** A primeira versao inferia vida baixa
pela vinheta vermelha nas bordas da tela. Medido em 19 minutos de gameplay
real, esse sinal aparecia em **32% dos quadros** e rendia 122 "fugas" -- porque
aquela vinheta e o indicador de *dano recebido*, que pisca o tempo todo numa
partida, e nao o aviso de vida baixa. A morte era inferida pela queda de
saturacao da killcam, e encontrou **zero** mortes: a killcam do OW2 nao e
dessaturada.

A versao atual le a barra de vida diretamente. A barra e desenhada como uma
sequencia de tracinhos verticais claros, e o OW2 normaliza a largura dela --
entao a fracao preenchida e a fracao de vida, seja o heroi de 200 ou de 700 de
vida. A leitura nao usa brilho (o cenario atras da HUD pode ser claro): usa a
**alternancia** clara/escura dos tracinhos, que so existe na parte preenchida.
Contra valores lidos na tela, o erro ficou em ate 0.05 (0.56 -> 0.55,
0.53 -> 0.49, 0.95 -> 0.92).

**Sobre `DEATH`.** Ao morrer no OW2 voce passa a espectar um companheiro, e a
vida *dele* aparece na HUD -- por isso a assinatura da morte e a vida ir a zero
e voltar cheia no quadro seguinte, e nao a barra ficar zerada. A barra sumir de
vez (menu, selecao de heroi, troca de round) e tratada igual, de proposito:
para as regras os dois casos significam a mesma coisa -- a sequencia de acao do
jogador foi interrompida, entao a rajada nao vale como "sozinho contra todos" e
a fuga nao vale como sobrevivencia. Por isso o evento nao promete ser "morte"
no sentido estrito, e o `meta` diz o que disparou.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import find_pulses, iter_frames

log = logging.getLogger(__name__)

#: o perfil horizontal da barra e reamostrado para este tamanho antes da
#: analise, para que os limiares valham em qualquer resolucao de gravacao
PROFILE_SAMPLES = 256


def read_health_fraction(
    bgr: np.ndarray, *, energy_floor: float, tick_threshold: float
) -> float | None:
    """Fracao preenchida da barra de vida, ou None se a barra nao esta na tela.

    Mede a alternancia clara/escura dos tracinhos ao longo da tira: onde a
    barra esta preenchida o perfil horizontal oscila, e onde esta vazia ele e
    liso. Assim um fundo claro atras da HUD nao vira "vida cheia".
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=0)
    if profile.size < 16:
        return None

    # As janelas abaixo sao contadas em amostras, entao dependeriam da largura
    # da tira -- e a tira sai com a largura nativa do video, que vai de ~100px
    # em 360p a ~300px em 1080p. Reamostrar o perfil para um tamanho fixo torna
    # a leitura igual em qualquer resolucao, sem precisar ampliar o video (o que
    # so engordaria o recorte sem acrescentar informacao).
    profile = np.interp(
        np.linspace(0, profile.size - 1, PROFILE_SAMPLES),
        np.arange(profile.size),
        profile,
    )

    gradient = np.abs(np.diff(profile))
    energy = np.convolve(gradient, np.ones(5) / 5, mode="same")
    if float(energy.max()) < energy_floor:
        return None  # sem barra na tela

    normalized = energy / energy.max()
    hot = (normalized > tick_threshold).astype(np.float32)

    # Nao basta achar a coluna mais a direita com gradiente alto: a *borda* do
    # trilho vazio tambem e um degrau forte, e uma barra zerada seria lida como
    # cheia. O que caracteriza a parte preenchida e haver varios tracinhos
    # seguidos -- ou seja, densidade de alternancia numa vizinhanca, e nao um
    # degrau isolado.
    window = max(5, normalized.size // 10)
    density = np.convolve(hot, np.ones(window) / window, mode="same")
    filled = np.flatnonzero(density > 0.25)
    if filled.size == 0:
        return 0.0  # barra na tela, porem vazia
    return float(filled.max() + 1) / float(normalized.size)


def _median3(series: list[float | None]) -> list[float | None]:
    """Mediana movel de 3, tratando None (barra ausente) como categoria propria."""
    out: list[float | None] = []
    for i in range(len(series)):
        window = series[max(0, i - 1) : i + 2]
        nones = sum(1 for x in window if x is None)
        if nones > len(window) // 2:
            out.append(None)
            continue
        vals = sorted(x for x in window if x is not None)
        out.append(vals[len(vals) // 2])
    return out


def detect_survival(health_video: Path, profile: Profile) -> list[DetectionEvent]:
    cfg = profile.section("survival")
    death_cfg = profile.section("death")
    roi = profile.roi("health")

    energy_floor = float(cfg.get("bar_energy_floor", 2.0))
    tick_threshold = float(cfg.get("tick_threshold", 0.25))
    low_frac = float(cfg.get("low_hp_frac", 0.30))

    times: list[float] = []
    health: list[float | None] = []
    for frame in iter_frames(health_video, fps_hint=roi.fps):
        times.append(frame.t)
        health.append(
            read_health_fraction(
                frame.bgr, energy_floor=energy_floor, tick_threshold=tick_threshold
            )
        )

    if not times:
        return []
    if all(h is None for h in health):
        log.warning(
            "a barra de vida nunca foi encontrada -- confira a ROI 'health' do "
            "profile com tools/calibrate.py; sem ela nao ha eventos de sobrevivencia"
        )
        return []

    death_frac = float(death_cfg.get("dead_hp_frac", 0.06))
    events: list[DetectionEvent] = []

    # As duas leituras usam series diferentes de proposito. A morte e um evento
    # *transitorio* -- as vezes um unico quadro com a vida zerada --, entao ela
    # tem de sair da serie crua; suavizar aqui apagaria exatamente o que se quer
    # ver. Vida baixa e o oposto: dura segundos, e uma mediana de 3 tira o ruido
    # de leitura sem encurtar episodio nenhum.
    smooth = _median3(health)

    # ── interrupcoes ────────────────────────────────────────────────────────
    # A vida cair a zero e voltar ao topo no quadro seguinte e a assinatura da
    # morte: ao morrer voce passa a espectar um companheiro, com a vida *dele*
    # na tela. Nenhuma cura sobe assim. A barra sumir de vez (menu, troca de
    # round) conta igual, porque significa a mesma coisa para as regras.
    down = [1.0 if (h is None or h <= death_frac) else 0.0 for h in health]
    absent_pulses = find_pulses(
        times,
        down,
        rise=0.5,
        fall=0.5,
        min_duration=float(death_cfg.get("min_duration_s", 0.1)),
        min_gap=float(death_cfg.get("min_gap_s", 3.0)),
    )
    interruptions = [p.start for p in absent_pulses]
    for p in absent_pulses:
        events.append(
            DetectionEvent(
                kind=EventKind.DEATH,
                t=round(p.start, 3),
                confidence=0.7,
                meta={"reason": "vida_zerada_ou_hud_ausente",
                      "duration_s": round(p.duration, 2)},
            )
        )

    # ── vida baixa: so onde a barra existe e ainda ha vida ──────────────────
    danger = [
        0.0
        if (h is None or h <= death_frac or h >= low_frac)
        else (low_frac - h) / max(1e-6, low_frac)
        for h in smooth
    ]
    low_pulses = find_pulses(
        times,
        danger,
        rise=0.02,
        fall=0.005,
        min_duration=float(cfg.get("min_duration_s", 1.0)),
        min_gap=float(cfg.get("min_gap_s", 3.0)),
    )

    safe_after = float(cfg.get("safe_after_s", 4.0))
    for p in low_pulses:
        lowest = low_frac * (1.0 - p.peak)
        meta = {
            "hp_min": round(max(0.0, lowest), 3),
            "duration_s": round(p.duration, 2),
        }
        events.append(
            DetectionEvent(
                kind=EventKind.LOW_HP, t=round(p.start, 3), confidence=0.85, meta=meta
            )
        )
        survived = not any(p.start <= d <= p.end + safe_after for d in interruptions)
        if survived:
            events.append(
                DetectionEvent(
                    kind=EventKind.ESCAPE,
                    t=round(p.end, 3),
                    confidence=0.8,
                    meta={**meta, "low_hp_at": round(p.start, 3)},
                )
            )

    events.sort(key=lambda e: e.t)
    log.info(
        "%d interrupcao(oes), %d episodio(s) de vida baixa, %d fuga(s)",
        len(interruptions),
        len(low_pulses),
        sum(1 for e in events if e.kind == EventKind.ESCAPE),
    )
    return events
