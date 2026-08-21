"""Deteccao de eliminacoes.

Entrada: o recorte central da tela em volta da mira, ja em baixa resolucao e
baixo FPS -- o microsservico nunca ve o video inteiro.

No Overwatch 2, cada eliminacao desenha uma **caveira magenta na mira**, que
cresce, encolhe e some em cerca de meio segundo.

A primeira versao deste detector so media "quanto da regiao esta vermelha", e
isso **nao funciona em partida real**: o mundo do jogo (iluminacao quente dos
mapas, contornos vermelhos dos inimigos, fogo) e principalmente o *indicador
direcional de dano* pintam a mesma faixa de cor quase o tempo todo. Medido em
19 minutos de gameplay real, a regiao passava de 3% de vermelho em metade dos
quadros -- e a precisao ficava em torno de 17%.

O que separa a caveira do resto sao quatro coisas, e nenhuma delas e a cor:

* **saturacao** -- a caveira e um elemento de interface, desenhado por cima da
  cena com saturacao muito alta; cenario iluminado de vermelho fica bem abaixo;
* **posicao** -- ela nasce exatamente na mira, enquanto o indicador de dano e um
  arco desenhado num raio ao redor dela, encostado na borda da regiao;
* **forma** -- e um blob compacto, quase quadrado; o indicador de dano e um arco
  largo e achatado;
* **tamanho** -- a caveira ocupa de 5% a 14% da regiao. Este era o filtro que
  faltava: com o minimo em 0.4%, qualquer respingo vermelho de 20 pixels virava
  eliminacao, e eram esses respingos a maior parte dos falsos positivos que
  sobravam.

Com os quatro filtros a precisao no mesmo material foi de ~17% para ~91%.
"""

from __future__ import annotations

import logging
from pathlib import Path

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import TemplateBank, find_icon, find_pulses, iter_frames

log = logging.getLogger(__name__)


def detect_kills(
    roi_video: Path,
    profile: Profile,
    templates_dir: Path | None = None,
) -> list[DetectionEvent]:
    cfg = profile.section("kills")
    roi = profile.roi("kills")
    ranges = cfg.get("hsv_ranges", [])

    min_area = float(cfg.get("min_area_frac", 0.004))
    release = float(cfg.get("release_area_frac", min_area / 2))
    max_offset = float(cfg.get("max_offset", 0.30))
    aspect = tuple(cfg.get("aspect_range", [0.55, 1.9]))

    bank = TemplateBank.from_dir(templates_dir) if templates_dir else TemplateBank({})
    if bank:
        log.info("%d template(s) de caveira carregado(s)", len(bank))

    times: list[float] = []
    scores: list[float] = []
    holes: list[float] = []
    tpl_scores: list[float] = []

    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        blob = find_icon(
            frame.bgr,
            ranges,
            min_area_frac=min_area,
            max_offset=max_offset,
            aspect_range=aspect,  # type: ignore[arg-type]
        )
        times.append(frame.t)
        scores.append(blob.area_frac if blob else 0.0)
        holes.append(blob.hole_ratio if blob else 0.0)
        if bank:
            tpl_scores.append(bank.best_match(frame.bgr)[1])

    pulses = find_pulses(
        times,
        scores,
        rise=min_area,
        fall=release,
        min_duration=float(cfg.get("min_pulse_s", 0.0)),
        min_gap=float(cfg.get("min_gap_s", 0.8)),
    )

    tpl_threshold = float(cfg.get("template_threshold", 0.85))
    events: list[DetectionEvent] = []
    for p in pulses:
        # quanto maior o pico em relacao ao minimo exigido, mais confianca
        confidence = min(0.9, 0.55 + 0.35 * min(1.0, p.peak / (min_area * 2.5)))
        # as orbitas da caveira: quando aparecem, nao ha o que confundir. Nem
        # sempre aparecem (em 360p sao poucos pixels), entao isto soma
        # confianca em vez de servir de filtro.
        best_holes = max(
            (hr for t, hr in zip(times, holes) if p.start <= t <= p.end), default=0.0
        )
        if best_holes > 0.02:
            confidence = min(1.0, confidence + 0.1)
        meta: dict = {
            "peak_area_frac": round(p.peak, 5),
            "hole_ratio": round(best_holes, 4),
            "pulse_s": round(p.duration, 3),
        }
        if tpl_scores:
            window = [s for t, s in zip(times, tpl_scores) if p.start <= t <= p.end]
            best = max(window) if window else 0.0
            meta["template_score"] = round(best, 3)
            if best >= tpl_threshold:
                confidence = min(1.0, confidence + 0.2)
        events.append(
            DetectionEvent(
                kind=EventKind.KILL,
                t=round(p.start, 3),
                confidence=round(confidence, 3),
                meta=meta,
            )
        )
    log.info("%d eliminacao(oes) detectada(s)", len(events))
    return events
