"""Computer-vision utilities shared by the detectors.

None of them knows anything about Overwatch: they are primitives (ratio of
pixels within an HSV range, template matching, pulse detection over a time
series) that the detectors combine according to the profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------- frame reading --------------------------------


@dataclass(slots=True)
class Frame:
    index: int
    t: float
    bgr: np.ndarray


def iter_frames(path: Path, fps_hint: float | None = None) -> Iterator[Frame]:
    """Walks an already-cropped video. The crops are generated with the
    ``fps`` filter, i.e. CFR -- so ``t = index / fps``."""
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


# --------------------------- per-frame metrics -----------------------------


def hsv_ratio(bgr: np.ndarray, ranges: Sequence[dict]) -> float:
    """Fraction of pixels inside any of the given HSV ranges."""
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
    """Binary mask of the pixels inside any of the HSV ranges."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array(r["lo"], dtype=np.uint8)
        hi = np.array(r["hi"], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


@dataclass(slots=True)
class Banner:
    """A horizontal HUD banner -- the OW2 footer notices."""

    x: int
    y: int
    w: int
    h: int
    #: banner height as a fraction of the ROI
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
    """Every notice banner in the crop, top to bottom.

    Sky, water and bright scenery fall into the same colour ranges as the HUD,
    and cover much larger areas. What characterises the banner is its **shape**:
    a filled rectangle, wide and **short**, horizontally centred. The height
    limit is the filter that discards scenery -- a ROI taken over by the sky
    becomes a blob occupying the full height.

    White text breaks the mask into pieces, so it is closed with a wide element
    before measuring: what matters is the rectangle, not the letters.

    **One mask per colour, never the union.** A banner has a single colour, and
    the HUD colours change from recording to recording -- some matches show the
    notice in cyan and some in green. Summing the colours into a single mask,
    the scenery of one colour sticks to the banner of the other during the
    horizontal closing and the rectangle stops existing: measured on a Sigma
    recording (green banner, bluish scenery), the summed mask lost 5 of the 23
    rock throws; split by colour, it found all 23.

    Returns **all** the candidates because OW2 stacks notices -- a single frame
    may hold "ORB OF HARMONY" on top and "STUNNED BY ACCRETION" below. Keeping
    only the largest would lose the stacked cases.
    """
    h_roi, w_roi = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    lo_h, hi_h = height_range
    lo_w, hi_w = width_range

    found: list[Banner] = []
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
            found.append(
                Banner(
                    x=int(x), y=int(y), w=int(w), h=int(h),
                    height_frac=float(hf), offset_x=float(off_x),
                )
            )
    found.sort(key=lambda b: b.y)
    return found


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
    """The largest banner in the crop, or None. Convenience over `find_banners`."""
    found = find_banners(
        bgr, ranges,
        height_range=height_range, width_range=width_range,
        min_aspect=min_aspect, max_offset=max_offset, min_fill=min_fill,
    )
    return max(found, key=lambda b: b.w * b.h, default=None)


@dataclass(slots=True)
class Blob:
    """A candidate HUD icon inside the ROI."""

    area_frac: float
    offset_x: float  # -1..1 relative to the ROI centre
    offset_y: float
    aspect: float
    #: fraction of the blob's area that is internal holes (the skull's eye
    #: sockets); 0 on a solid patch
    hole_ratio: float


def _hole_ratio(component: np.ndarray) -> float:
    """How much of the blob, once filled, was an internal hole.

    Fills from the outside with a flood fill: whatever ends up filled and was
    not in the original blob is a hole.
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
    """Looks for a HUD *icon* centred in the ROI.

    Measuring only "how much of the region is red" does not work on a real
    match: the game scenery, enemy outlines and the directional damage
    indicator paint the same colour range all the time. A HUD icon is
    distinguished by three things, and none of them is colour:

    * it is a **compact blob with a near-square aspect** -- the damage
      indicator is a wide, flattened arc;
    * nasce **centrado na mira** -- o indicador de dano fica num raio acima dela;
    * it has a **characteristic size**. That last one was the missing filter:
      with the minimum at 0.4% of the region, any 20-pixel red splash got
      through, and that is what made "all red become a kill". The skull
      ocupa de 5% a 14% da ROI; respingos ficam abaixo de 2.5%.

    Returns None when nothing in the ROI looks like an icon.
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
    """Boolean mask of the outer frame (the low-health vignette lives there)."""
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
    """A collection of greyscale templates loaded from a directory."""

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


# ------------------- pulse detection over a time series --------------------


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
    ``fall``. That avoids counting a flickering icon as several occurrences."""
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


# --------------------- glyphs: the mark inside an icon ---------------------
#
# O casamento de template de `TemplateBank` desliza o molde pela imagem em
# several scales. It works when you do not know *where* the icon is -- but
# when you do (the ultimate button, the killfeed box), it is expensive and
# fragile: the game icon shows up sometimes black on a white disc, sometimes
# white on a dark box, at sizes that change with the recording's resolution.
#
# Para esse caso vale mais recortar a **marca** — os pixels que formam o
# drawing, without background -- fit it into a square and always compare at
# the same size. Position, scale and polarity then stop mattering, and the
# comparison becomes an inner product: the whole bank fits in one matrix.

#: lado do glifo normalizado, em pixels. 56 separa bem as ~270 habilidades do
#: jogo e ainda deixa o banco inteiro numa matriz de poucos megabytes.
GLYPH_SIDE = 56


def normalized_glyph(mask: np.ndarray) -> np.ndarray | None:
    """Crops the mark out of a binary mask, centres it and normalises its size.

    The square is built from the **longest** side of the crop, so the drawing's
    aspect is preserved: a tall, narrow icon does not become a wide, short one.
    Returns None when there is not enough mark to compare.

    The mark is taken **whole**, fragments and all. Discarding small pieces
    looks like cheap cleanup -- it would deal with three pixels of a
    neighbouring element that fall into the window -- but half the game's icons
    are made of loose parts (dots, sparks, separate arrows), and cutting them
    changes the box from one frame to the next: on a real recording that turned
    one kill into four. Making sure only the mark enters the window is the job
    of whoever crops it.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if int(np.count_nonzero(mask)) < 20:
        return None
    ys, xs = np.nonzero(mask)
    crop = (mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] > 0).astype(np.uint8)
    side = max(crop.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    square[oy: oy + crop.shape[0], ox: ox + crop.shape[1]] = crop * 255
    return cv2.resize(square, (GLYPH_SIDE, GLYPH_SIDE), interpolation=cv2.INTER_AREA)


def glyph_on_dark(bgr: np.ndarray, *, max_sat: int = 90, min_val: int = 170) -> np.ndarray | None:
    """The bright mark of an icon drawn over a dark background.

    É assim que o killfeed mostra a habilidade que matou: desenho branco numa
    caixinha cinza.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return normalized_glyph((hsv[:, :, 1] < max_sat) & (hsv[:, :, 2] > min_val))


def glyph_in_disc(
    bgr: np.ndarray,
    *,
    min_disc_frac: float = 0.20,
    max_dark: int = 130,
    max_sat: int = 70,
    min_val: int = 185,
) -> np.ndarray | None:
    """A marca escura desenhada dentro de um disco branco.

    It is the shape the game uses for *ultimates*: in the footer button and in
    the killfeed box, an ultimate comes as a white disc with the drawing in
    black.

    The disc is found first so its rim does not enter the mark -- and whatever
    falls outside the circle is erased, otherwise the disc's own outline would
    become part of the drawing.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < max_sat) & (hsv[:, :, 2] > min_val)).astype(np.uint8)
    if white.mean() < min_disc_frac:
        return None
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(white, 8)
    if count < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, 4]))
    x, y, w, h = (int(v) for v in stats[k, :4])
    if min(w, h) < 8:
        return None
    gray = cv2.cvtColor(bgr[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return normalized_glyph((gray < max_dark) & (r < 0.88))


class IconBank:
    """Ícones de habilidade do jogo, prontos para comparar com um glifo.

    Each file is a black mark on a white background, and its name -- hero
    folder plus ability name -- becomes the event's label. The templates sit in
    a matrix already centred and normalised, so comparing a glyph with the bank
    is **one** matrix-vector product, not 270 template matches.
    """

    def __init__(self, keys: list[str], matrix: np.ndarray):
        self.keys = keys
        self.matrix = matrix

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def _vector(glyph: np.ndarray) -> np.ndarray:
        v = glyph.astype(np.float32).ravel()
        v -= v.mean()
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    @classmethod
    def from_dir(cls, path: Path) -> "IconBank":
        """Reads `<path>/<hero>/<ability>.png`. A missing folder gives an empty bank."""
        path = Path(path)
        keys: list[str] = []
        vectors: list[np.ndarray] = []
        if path.exists():
            for f in sorted(path.rglob("*.png")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning("ícone ilegível, ignorando: %s", f)
                    continue
                glyph = normalized_glyph(img < 128)
                if glyph is None:
                    log.warning("ícone sem marca legível, ignorando: %s", f)
                    continue
                keys.append(f"{f.parent.name}/{f.stem}")
                vectors.append(cls._vector(glyph))
        return cls(keys, np.stack(vectors) if vectors else np.zeros((0, GLYPH_SIDE ** 2), np.float32))

    def best_match(self, glyph: np.ndarray) -> tuple[str | None, float]:
        """Best (key, correlation) in the bank for this glyph. -1..1."""
        if not self.keys:
            return None, 0.0
        scores = self.matrix @ self._vector(glyph)
        k = int(np.argmax(scores))
        return self.keys[k], float(scores[k])
