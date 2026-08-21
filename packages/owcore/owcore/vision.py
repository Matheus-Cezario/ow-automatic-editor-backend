"""Utilidades de visão computacional compartilhadas pelos detectores.

Nenhuma delas conhece Overwatch: são primitivas (razão de pixels numa faixa
HSV, casamento de template, detecção de pulsos numa série temporal) que os
detectores combinam segundo o profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ──────────────────────────── leitura de frames ─────────────────────────────


@dataclass(slots=True)
class Frame:
    index: int
    t: float
    bgr: np.ndarray


def iter_frames(path: Path, fps_hint: float | None = None) -> Iterator[Frame]:
    """Percorre um vídeo já recortado. Os recortes são gerados com o filtro
    ``fps``, ou seja, CFR — então ``t = index / fps``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"não consegui abrir {path}")
    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 10.0
    if fps <= 0:
        fps = fps_hint or 10.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield Frame(index=i, t=i / fps, bgr=frame)
            i += 1
    finally:
        cap.release()


# ──────────────────────────── métricas por frame ────────────────────────────


def hsv_ratio(bgr: np.ndarray, ranges: Sequence[dict]) -> float:
    """Fração de pixels dentro de qualquer uma das faixas HSV dadas."""
    if not ranges:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return float(np.count_nonzero(mask)) / float(mask.size)


def mean_saturation(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def color_mask(bgr: np.ndarray, ranges: Sequence[dict]) -> np.ndarray:
    """Máscara binária dos pixels dentro de qualquer uma das faixas HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


@dataclass(slots=True)
class Banner:
    """Uma faixa horizontal da HUD -- os avisos do rodapé do OW2."""

    x: int
    y: int
    w: int
    h: int
    #: altura da faixa como fração da ROI
    height_frac: float
    offset_x: float


def find_banners(
    bgr: np.ndarray,
    ranges: Sequence[dict],
    *,
    height_range: tuple[float, float],
    width_range: tuple[float, float],
    min_aspect: float,
    max_offset: float,
    min_fill: float,
) -> list[Banner]:
    """Todas as faixas de aviso do recorte, de cima para baixo.

    O céu, a água e superfícies claras do cenário caem nas mesmas faixas de cor
    da HUD, e cobrem áreas bem maiores. O que caracteriza a faixa é a **forma**:
    um retângulo cheio, largo e **baixo**, centrado na horizontal. O limite de
    altura é o filtro que descarta cenário -- uma ROI tomada pelo céu vira um
    blob que ocupa a altura inteira.

    O texto branco parte a máscara em pedaços, então ela é fechada com um
    elemento largo antes de medir: o que interessa é o retângulo, não as letras.

    **Uma máscara por cor, nunca a união.** Uma faixa tem uma cor só, e as cores
    da HUD mudam de gravação para gravação -- há partidas com o aviso em ciano e
    partidas com ele em verde. Somando as cores numa máscara única, o cenário de
    uma cor cola na faixa da outra durante o fechamento horizontal e o retângulo
    deixa de existir: medido numa gravação de Sigma (faixa verde, cenário
    azulado), a máscara somada perdia 5 das 23 pedradas; separada por cor, achou
    as 23.

    Devolve **todas** as candidatas porque o OW2 empilha avisos -- num mesmo
    quadro pode haver "ORB OF HARMONY" em cima e "STUNNED BY ACCRETION" embaixo.
    Ficar só com a maior perderia os casos empilhados.
    """
    h_roi, w_roi = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    lo_h, hi_h = height_range
    lo_w, hi_w = width_range

    achadas: list[Banner] = []
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        if not mask.any():
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, count):
            x, y, w, h, area = stats[k]
            hf, wf = h / h_roi, w / w_roi
            if not (lo_h < hf < hi_h) or not (lo_w < wf < hi_w):
                continue
            if w / max(1, h) < min_aspect:
                continue
            if area / max(1, w * h) < min_fill:
                continue
            off_x = (centroids[k][0] - w_roi / 2) / (w_roi / 2)
            if abs(off_x) > max_offset:
                continue
            achadas.append(
                Banner(
                    x=int(x), y=int(y), w=int(w), h=int(h),
                    height_frac=float(hf), offset_x=float(off_x),
                )
            )
    achadas.sort(key=lambda b: b.y)
    return achadas


def find_banner(
    bgr: np.ndarray,
    ranges: Sequence[dict],
    *,
    height_range: tuple[float, float],
    width_range: tuple[float, float],
    min_aspect: float,
    max_offset: float,
    min_fill: float,
) -> Banner | None:
    """A maior faixa do recorte, ou None. Conveniência sobre `find_banners`."""
    achadas = find_banners(
        bgr, ranges,
        height_range=height_range, width_range=width_range,
        min_aspect=min_aspect, max_offset=max_offset, min_fill=min_fill,
    )
    return max(achadas, key=lambda b: b.w * b.h, default=None)


@dataclass(slots=True)
class Blob:
    """Um candidato a ícone da HUD dentro da ROI."""

    area_frac: float
    offset_x: float  # -1..1 em relação ao centro da ROI
    offset_y: float
    aspect: float
    #: fração da área do blob que são buracos internos (as órbitas da caveira);
    #: 0 numa mancha sólida
    hole_ratio: float


def _hole_ratio(component: np.ndarray) -> float:
    """Quanto do blob, já preenchido, era buraco interno.

    Preenche a partir de fora com flood fill: o que sobrar preenchido e não
    estava no blob original é buraco.
    """
    padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    ff_mask = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, ff_mask, (0, 0), 255)
    filled = padded | cv2.bitwise_not(flooded)
    filled_area = int((filled > 0).sum())
    if filled_area == 0:
        return 0.0
    return (filled_area - int((padded > 0).sum())) / filled_area


def find_icon(
    bgr: np.ndarray,
    ranges: Sequence[dict],
    *,
    min_area_frac: float,
    max_offset: float,
    aspect_range: tuple[float, float],
) -> Blob | None:
    """Procura um *ícone* da HUD centrado na ROI.

    Medir só "quanto da região está vermelha" não funciona em partida real: o
    cenário do jogo, os contornos de inimigos e o indicador direcional de dano
    pintam a mesma faixa de cor o tempo todo. Um ícone de HUD se distingue por
    três coisas, e nenhuma delas é a cor:

    * é um blob **compacto e de proporção quase quadrada** -- o indicador de
      dano é um arco largo e achatado;
    * nasce **centrado na mira** -- o indicador de dano fica num raio acima dela;
    * tem um **tamanho característico**. Este último era o filtro que faltava:
      com o mínimo em 0.4% da região, qualquer respingo vermelho de 20 pixels
      passava, e era isso que fazia "todo vermelho virar eliminação". A caveira
      ocupa de 5% a 14% da ROI; respingos ficam abaixo de 2.5%.

    Devolve None quando nada na ROI se parece com um ícone.
    """
    mask = color_mask(bgr, ranges)
    if not mask.any():
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    h, w = mask.shape
    total = h * w
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best: Blob | None = None
    lo_ar, hi_ar = aspect_range
    for k in range(1, count):
        bx, by, bw, bh, area = stats[k]
        frac = area / total
        if frac < min_area_frac:
            continue
        aspect = bw / max(1, bh)
        if not (lo_ar < aspect < hi_ar):
            continue
        off_x = (centroids[k][0] - w / 2) / (w / 2)
        off_y = (centroids[k][1] - h / 2) / (h / 2)
        if abs(off_x) > max_offset or abs(off_y) > max_offset:
            continue
        if best is not None and frac <= best.area_frac:
            continue
        component = ((labels[by : by + bh, bx : bx + bw] == k) * 255).astype(np.uint8)
        best = Blob(
            area_frac=float(frac),
            offset_x=float(off_x),
            offset_y=float(off_y),
            aspect=float(aspect),
            hole_ratio=_hole_ratio(component),
        )
    return best


def border_mask(shape: tuple[int, int], frac: float) -> np.ndarray:
    """Máscara booleana da moldura externa (a vinheta de vida baixa fica ali)."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    bh = max(1, int(h * frac))
    bw = max(1, int(w * frac))
    m[:bh, :] = m[-bh:, :] = True
    m[:, :bw] = m[:, -bw:] = True
    return m


def hsv_ratio_masked(bgr: np.ndarray, ranges: Sequence[dict], mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hit = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        hit = cv2.bitwise_or(hit, cv2.inRange(hsv, lo, hi))
    sel = hit.astype(bool) & mask
    denom = int(mask.sum()) or 1
    return float(sel.sum()) / denom


# ──────────────────────────── casamento de template ─────────────────────────


class TemplateBank:
    """Coleção de templates em escala de cinza carregada de um diretório."""

    def __init__(self, templates: dict[str, np.ndarray]):
        self.templates = templates

    def __bool__(self) -> bool:
        return bool(self.templates)

    def __len__(self) -> int:
        return len(self.templates)

    @classmethod
    def from_dir(cls, path: Path, max_width: int = 96) -> "TemplateBank":
        path = Path(path)
        out: dict[str, np.ndarray] = {}
        if not path.exists():
            return cls(out)
        for f in sorted(path.iterdir()):
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warning("template ilegível, ignorando: %s", f.name)
                continue
            if img.shape[1] > max_width:
                scale = max_width / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            out[f.stem] = img
        return cls(out)

    #: o template vem de um recorte do usuario, que raramente esta na mesma
    #: escala do que o pipeline entrega; testar algumas escalas evita que um
    #: template correto passe batido por ser 20% maior ou menor
    SCALES = (0.7, 0.85, 1.0, 1.2, 1.45)

    def best_match(self, bgr: np.ndarray) -> tuple[str | None, float]:
        """Melhor (nome, score) entre todos os templates, em varias escalas."""
        if not self.templates:
            return None, 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        best_name, best_score = None, 0.0
        for name, tpl in self.templates.items():
            for scale in self.SCALES:
                if scale != 1.0:
                    scaled = cv2.resize(
                        tpl, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                    )
                else:
                    scaled = tpl
                if scaled.shape[0] > gray.shape[0] or scaled.shape[1] > gray.shape[1]:
                    continue
                if min(scaled.shape[:2]) < 6:
                    continue
                score = float(cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED).max())
                if score > best_score:
                    best_name, best_score = name, score
        return best_name, best_score


# ─────────────────────── detecção de pulsos numa série ──────────────────────


@dataclass(slots=True)
class Pulse:
    start: float
    end: float
    peak: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_pulses(
    times: Sequence[float],
    values: Sequence[float],
    *,
    rise: float,
    fall: float,
    min_duration: float = 0.0,
    min_gap: float = 0.0,
) -> list[Pulse]:
    """Histerese de Schmitt: sobe acima de ``rise``, só termina abaixo de
    ``fall``. Isso evita contar um ícone piscando como várias ocorrências."""
    pulses: list[Pulse] = []
    active = False
    start = 0.0
    peak = 0.0
    for t, v in zip(times, values):
        if not active and v >= rise:
            active, start, peak = True, t, v
        elif active:
            peak = max(peak, v)
            if v < fall:
                active = False
                if t - start >= min_duration:
                    pulses.append(Pulse(start=start, end=t, peak=peak))
    if active and times and times[-1] - start >= min_duration:
        pulses.append(Pulse(start=start, end=times[-1], peak=peak))

    if min_gap <= 0 or not pulses:
        return pulses
    merged = [pulses[0]]
    for p in pulses[1:]:
        if p.start - merged[-1].start < min_gap:
            merged[-1] = Pulse(
                start=merged[-1].start, end=p.end, peak=max(merged[-1].peak, p.peak)
            )
        else:
            merged.append(p)
    return merged
