"""Deteccao das habilidades anunciadas na faixa do rodape.

Entrada: uma tira fina do rodape, onde o OW2 empilha os avisos de acao.

Quando uma habilidade acerta, aparece uma faixa: "PUT <HEROI> (<JOGADOR>) TO
SLEEP" para o dardo da Ana, "<HEROI> (<JOGADOR>) STUNNED BY ACCRETION" para a
pedrada do Sigma. So que o rodape mostra uma **familia** de avisos com a mesma
cara -- mesma cor, mesma forma, mesma posicao: "SAVED BY ...", "ORB OF HARMONY
FROM ...", "GAINED FROM ...". Cor e geometria acham a faixa, mas nao dizem qual
e.

Quem diz e o **icone** na ponta esquerda, e e ele que este detector compara --
nao o texto. O texto muda de idioma; o icone nao. Cada habilidade configurada
tem o seu molde, e num mesmo quadro so o molde vencedor pontua: uma faixa
anuncia **uma** habilidade, entao deixar dois moldes marcarem a mesma faixa
geraria dois eventos para o mesmo acontecimento.

Medido em duas gravacoes reais, com os moldes treinados so na primeira metade
de cada uma:

* Ana, 16 min: 11/11 dardos, nenhum falso;
* Sigma, 11 min: 23/23 pedradas, nenhum falso (12 delas nunca vistas no treino).

A pedrada tambem foi encontrada uma vez na gravacao da Ana -- conferida a olho,
era real: um Sigma aliado acertando uma Accretion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import Banner, find_banners, find_pulses, iter_frames

log = logging.getLogger(__name__)

#: lado do molde do icone, em pixels
ICON_SIZE = 24
#: a janela recortada da faixa e maior que o molde, para o casamento poder
#: deslizar. Sem essa folga o recorte tem exatamente o tamanho do molde, o
#: `matchTemplate` fica com uma unica posicao possivel e um desalinhamento de
#: um pixel derruba o score de uma habilidade legitima.
ICON_WINDOW_W = 34
ICON_WINDOW_H = 28

#: habilidades reconhecidas quando o profile nao diz nada (compatibilidade)
HABILIDADES_PADRAO = [
    {"key": "ana_sleep", "icon": "ana_sleep_icon.png", "event": "sleep"},
    {"key": "sigma_accretion", "icon": "sigma_accretion_icon.png", "event": "stun"},
]

#: usado quando nem a habilidade nem a secao dizem qual limiar usar
LIMIAR_PADRAO = 0.90


@dataclass(slots=True)
class Habilidade:
    key: str
    event: EventKind
    template: np.ndarray
    #: cada molde separa a sua habilidade num ponto diferente -- um icone cheio
    #: e contrastado casa mais alto que um de tracos finos --, entao o limiar e
    #: por habilidade. Um numero unico obrigaria a escolher entre perder dardos
    #: e aceitar pedradas falsas.
    limiar: float


def _icon_of(bgr: np.ndarray, banner: Banner) -> np.ndarray | None:
    """Recorta a ponta esquerda da faixa, onde mora o icone, normalizada.

    O recorte acompanha a faixa (posicao e altura relativas a ela), então vale
    para qualquer resolucao de gravacao e para textos de qualquer tamanho.

    A janela sai maior que o molde, em cima e dos lados, para o casamento poder
    deslizar: a caixa da faixa varia um ou dois pixels de um quadro para o
    outro, e sem folga esse desvio bastava para derrubar o score.
    """
    lado = banner.h
    folga_x = int(round(lado * (ICON_WINDOW_W / ICON_SIZE - 1) / 2))
    folga_y = int(round(lado * (ICON_WINDOW_H / ICON_SIZE - 1) / 2))
    x0 = banner.x + int(lado * 0.18) - folga_x
    y0 = banner.y - folga_y
    x1, y1 = x0 + lado + 2 * folga_x, y0 + lado + 2 * folga_y
    h_roi, w_roi = bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_roi, x1), min(h_roi, y1)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    gray = cv2.cvtColor(
        cv2.resize(
            crop, (ICON_WINDOW_W, ICON_WINDOW_H), interpolation=cv2.INTER_AREA
        ),
        cv2.COLOR_BGR2GRAY,
    )
    # normalizar o contraste tira a influencia do cenario atras da faixa, e
    # torna o molde independente da cor da faixa -- a mesma habilidade aparece
    # em ciano numa gravacao e em verde noutra
    return cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)


def _carregar(cfg: dict, shapes_dir: Path) -> list[Habilidade]:
    limiar_secao = float(cfg.get("match_threshold", LIMIAR_PADRAO))
    out: list[Habilidade] = []
    for spec in cfg.get("abilities", HABILIDADES_PADRAO):
        caminho = Path(shapes_dir) / spec["icon"]
        template = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        if template is None:
            log.warning(
                "molde de '%s' nao encontrado em %s -- sem ele nao da para "
                "distinguir este aviso dos outros do rodape",
                spec["key"], caminho,
            )
            continue
        out.append(
            Habilidade(
                key=spec["key"],
                event=EventKind(spec["event"]),
                template=template,
                limiar=float(spec.get("match_threshold", limiar_secao)),
            )
        )
    return out


def detect_abilities(
    roi_video: Path, profile: Profile, shapes_dir: Path
) -> list[DetectionEvent]:
    cfg = profile.section("banner")
    roi = profile.roi("banner")

    habilidades = _carregar(cfg, shapes_dir)
    if not habilidades:
        log.warning("nenhum molde de icone disponivel; nada a detectar")
        return []

    ranges = cfg.get("hsv_ranges", [])

    times: list[float] = []
    curvas: dict[str, list[float]] = {h.key: [] for h in habilidades}

    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        times.append(frame.t)
        melhor = {h.key: 0.0 for h in habilidades}
        for banner in find_banners(
            frame.bgr,
            ranges,
            height_range=tuple(cfg.get("height_range", [0.18, 0.55])),
            width_range=tuple(cfg.get("width_range", [0.25, 0.98])),
            min_aspect=float(cfg.get("min_aspect", 3.0)),
            max_offset=float(cfg.get("max_offset", 0.25)),
            min_fill=float(cfg.get("min_fill", 0.55)),
        ):
            icone = _icon_of(frame.bgr, banner)
            if icone is None:
                continue
            icone = icone.astype(np.uint8)
            scores = {
                h.key: float(
                    cv2.matchTemplate(icone, h.template, cv2.TM_CCOEFF_NORMED).max()
                )
                for h in habilidades
            }
            # a faixa anuncia UMA habilidade: so o molde vencedor pontua nela
            vencedor = max(scores, key=scores.__getitem__)
            melhor[vencedor] = max(melhor[vencedor], scores[vencedor])
        for key, valor in melhor.items():
            curvas[key].append(valor)

    events: list[DetectionEvent] = []
    for h in habilidades:
        pulses = find_pulses(
            times,
            curvas[h.key],
            rise=h.limiar,
            fall=h.limiar * 0.8,
            min_duration=float(cfg.get("min_pulse_s", 0.0)),
            min_gap=float(cfg.get("min_gap_s", 3.0)),
        )
        events += [
            DetectionEvent(
                kind=h.event,
                t=round(p.start, 3),
                confidence=round(min(1.0, 0.5 + 0.5 * p.peak), 3),
                meta={"ability": h.key, "icon_score": round(float(p.peak), 3)},
            )
            for p in pulses
        ]
        log.info("%s: %d ocorrencia(s)", h.key, len(pulses))

    events.sort(key=lambda e: e.t)
    return events
