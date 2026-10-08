"""The stickers library: arrows, rings, skulls, stars -- the marker pen of a
montage.

Like the sound effects (`owcore.sfx`), they are **drawn here**, not bundled
as files from the internet: a ring around the enemy or an arrow at the
killfeed is a few circles and lines, the same pixels on every machine, and
no question about who owns them. The game's own hero portraits are left out
for that same reason -- they are Blizzard's; whoever wants one uploads the
picture, and an image in the library places exactly like a sticker.

Each sticker is a shape described as a signed distance (negative inside,
positive outside), so one description gives the fill, a dark outline that
keeps it readable over any scene, and smooth edges at any size. It is pure
Python on purpose: the gateway serves this, and its image has no numpy or
Pillow. Drawing only the edge cells pixel by pixel (a shape's distance tells
how far the nearest edge is, so a cell far from every edge is all inside or
all outside) keeps a 512px sticker well under a second.

A sticker enters a montage the way any picture does -- as an image item of
the match library (see the gateway's `POST /api/jobs/{id}/stickers`), placed
whole on the frame (`TimelineClip.fit`), at the size and place the user drags
it to.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

#: the side of the picture a sticker is added as, in pixels: a sticker at a
#: third of a 1080p frame's height is drawn at its own size or smaller
SIZE = 512

#: the size the shelf shows
PREVIEW_SIZE = 128

#: how thick the outline is, as a share of half the sticker's side
OUTLINE = 0.055

#: the shapes are drawn at this share of the sticker, so an outline at the
#: tip of an arrow is never cut by the picture's edge
FIT = 0.9

#: what the outline is painted in
OUTLINE_RGB = (17, 17, 17)

#: the colours a sticker comes in, in the order the app offers them
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 59, 48),
    "yellow": (255, 204, 0),
    "orange": (255, 149, 0),
    "green": (52, 199, 89),
    "blue": (10, 132, 255),
    "purple": (175, 82, 222),
    "pink": (255, 55, 145),
    "white": (255, 255, 255),
}

DEFAULT_COLOR = "red"

Point = tuple[float, float]
#: a shape: (distance to the fill, distance to the details) at a point. The
#: details are drawn in the outline's colour inside the fill (a skull's eyes)
Shape = Callable[[float, float], tuple[float, float]]

_FAR = 1e9


# ── distances ───────────────────────────────────────────────────────────────
#
# Coordinates go from -1 to 1 across the sticker, y downwards, the same way
# the frame's transform counts.


def _circle(x: float, y: float, cx: float, cy: float, r: float) -> float:
    return math.hypot(x - cx, y - cy) - r


def _ring(x: float, y: float, cx: float, cy: float, r: float, w: float) -> float:
    return abs(_circle(x, y, cx, cy, r)) - w


def _capsule(x: float, y: float, a: Point, b: Point, r: float) -> float:
    """A line from `a` to `b` with round ends, `r` thick on each side."""
    px, py = x - a[0], y - a[1]
    ex, ey = b[0] - a[0], b[1] - a[1]
    h = max(0.0, min(1.0, (px * ex + py * ey) / (ex * ex + ey * ey)))
    return math.hypot(px - ex * h, py - ey * h) - r


def _box(x: float, y: float, cx: float, cy: float, hw: float, hh: float, r: float) -> float:
    """A box centred on (cx, cy), half `hw` wide and `hh` tall, corners round by `r`."""
    qx = abs(x - cx) - hw + r
    qy = abs(y - cy) - hh + r
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r


def _polygon(x: float, y: float, pts: tuple[Point, ...]) -> float:
    """The exact distance to a closed polygon, negative inside."""
    d = (x - pts[0][0]) ** 2 + (y - pts[0][1]) ** 2
    s = 1.0
    j = len(pts) - 1
    for i in range(len(pts)):
        vi, vj = pts[i], pts[j]
        ex, ey = vj[0] - vi[0], vj[1] - vi[1]
        wx, wy = x - vi[0], y - vi[1]
        h = max(0.0, min(1.0, (wx * ex + wy * ey) / (ex * ex + ey * ey)))
        bx, by = wx - ex * h, wy - ey * h
        d = min(d, bx * bx + by * by)
        c1, c2, c3 = y >= vi[1], y < vj[1], ex * wy > ey * wx
        if (c1 and c2 and c3) or (not c1 and not c2 and not c3):
            s = -s
        j = i
    return s * math.sqrt(d)


def _star_points(n: int, outer: float, inner: float, turn: float = -math.pi / 2) -> tuple[Point, ...]:
    pts = []
    for k in range(2 * n):
        r = outer if k % 2 == 0 else inner
        a = turn + k * math.pi / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return tuple(pts)


# ── the stickers ────────────────────────────────────────────────────────────


def _arrow(x: float, y: float) -> tuple[float, float]:
    shaft = _capsule(x, y, (-0.8, 0.0), (0.1, 0.0), 0.17)
    head = _polygon(x, y, ((0.0, -0.55), (0.82, 0.0), (0.0, 0.55))) - 0.05
    return min(shaft, head), _FAR


def _ring_mark(x: float, y: float) -> tuple[float, float]:
    return _ring(x, y, 0.0, 0.0, 0.78, 0.09), _FAR


def _cross(x: float, y: float) -> tuple[float, float]:
    return min(
        _capsule(x, y, (-0.62, -0.62), (0.62, 0.62), 0.17),
        _capsule(x, y, (-0.62, 0.62), (0.62, -0.62), 0.17),
    ), _FAR


def _check(x: float, y: float) -> tuple[float, float]:
    return min(
        _capsule(x, y, (-0.65, 0.02), (-0.2, 0.47), 0.16),
        _capsule(x, y, (-0.2, 0.47), (0.68, -0.55), 0.16),
    ), _FAR


def _exclamation(x: float, y: float) -> tuple[float, float]:
    return min(
        _capsule(x, y, (0.0, -0.72), (0.0, 0.22), 0.17),
        _circle(x, y, 0.0, 0.67, 0.18),
    ), _FAR


def _crosshair(x: float, y: float) -> tuple[float, float]:
    ticks = min(
        _capsule(x, y, (0.0, -0.92), (0.0, -0.3), 0.07),
        _capsule(x, y, (0.0, 0.3), (0.0, 0.92), 0.07),
        _capsule(x, y, (-0.92, 0.0), (-0.3, 0.0), 0.07),
        _capsule(x, y, (0.3, 0.0), (0.92, 0.0), 0.07),
    )
    return min(_ring(x, y, 0.0, 0.0, 0.62, 0.07), ticks, _circle(x, y, 0.0, 0.0, 0.09)), _FAR


def _target(x: float, y: float) -> tuple[float, float]:
    return min(
        _ring(x, y, 0.0, 0.0, 0.8, 0.1),
        _ring(x, y, 0.0, 0.0, 0.45, 0.1),
        _circle(x, y, 0.0, 0.0, 0.15),
    ), _FAR


def _skull(x: float, y: float) -> tuple[float, float]:
    head = _circle(x, y, 0.0, -0.18, 0.68)
    jaw = _box(x, y, 0.0, 0.5, 0.4, 0.28, 0.12)
    eyes = min(
        _circle(x, y, -0.27, -0.12, 0.18),
        _circle(x, y, 0.27, -0.12, 0.18),
    )
    nose = _polygon(x, y, ((0.0, 0.12), (0.1, 0.3), (-0.1, 0.3)))
    teeth = min(
        _capsule(x, y, (-0.14, 0.55), (-0.14, 0.74), 0.035),
        _capsule(x, y, (0.14, 0.55), (0.14, 0.74), 0.035),
        _capsule(x, y, (0.0, 0.55), (0.0, 0.74), 0.035),
    )
    return min(head, jaw), min(eyes, nose, teeth)


_STAR = _star_points(5, 0.95, 0.42, turn=-math.pi / 2)


def _star(x: float, y: float) -> tuple[float, float]:
    return _polygon(x, y + 0.06, _STAR) - 0.03, _FAR


_BURST = tuple(
    (px * (1.0 if k % 4 else 0.9), py * (1.0 if k % 4 else 0.9))
    for k, (px, py) in enumerate(_star_points(12, 0.95, 0.62, turn=-math.pi / 2))
)


def _burst(x: float, y: float) -> tuple[float, float]:
    return _polygon(x, y, _BURST), _FAR


def _heart(x: float, y: float) -> tuple[float, float]:
    # Inigo Quilez's exact heart, which stands on its tip at the origin, y up,
    # about 1.1 tall: turned over and stretched onto the sticker
    k = 0.65
    px, py = abs(x) * k, (0.85 - y) * k
    if px + py > 1.0:
        d = math.hypot(px - 0.25, py - 0.75) - math.sqrt(2.0) / 4.0
    else:
        m = 0.5 * max(px + py, 0.0)
        d = math.sqrt(min(px * px + (py - 1.0) ** 2, (px - m) ** 2 + (py - m) ** 2))
        d *= 1.0 if px > py else -1.0
    return d / k, _FAR


_BOLT = ((0.22, -0.95), (-0.55, 0.12), (-0.04, 0.12), (-0.22, 0.95), (0.58, -0.18), (0.06, -0.18))


def _lightning(x: float, y: float) -> tuple[float, float]:
    return _polygon(x, y, _BOLT) - 0.02, _FAR


_CROWN = ((-0.8, 0.55), (0.8, 0.55), (0.82, -0.4), (0.4, -0.02), (0.0, -0.58), (-0.4, -0.02), (-0.82, -0.4))


def _crown(x: float, y: float) -> tuple[float, float]:
    tips = min(
        _circle(x, y, -0.82, -0.45, 0.11),
        _circle(x, y, 0.0, -0.63, 0.11),
        _circle(x, y, 0.82, -0.45, 0.11),
    )
    band = _capsule(x, y, (-0.55, 0.32), (0.55, 0.32), 0.035)
    return min(_polygon(x, y, _CROWN) - 0.03, tips), band


def _bubble(x: float, y: float) -> tuple[float, float]:
    body = _box(x, y, 0.0, -0.15, 0.9, 0.6, 0.3)
    tail = _polygon(x, y, ((-0.45, 0.3), (-0.05, 0.3), (-0.55, 0.85))) - 0.02
    return min(body, tail), _FAR


@dataclass(frozen=True, slots=True)
class Sticker:
    id: str
    name: str
    category: str
    shape: Shape


#: What each group is called and the order the app offers them in.
CATEGORIES: tuple[str, ...] = ("point", "mark", "game", "fun")

STICKERS: tuple[Sticker, ...] = (
    Sticker("arrow", "Arrow", "point", _arrow),
    Sticker("ring", "Ring", "point", _ring_mark),
    Sticker("crosshair", "Crosshair", "point", _crosshair),
    Sticker("target", "Target", "point", _target),
    Sticker("cross", "Cross", "mark", _cross),
    Sticker("check", "Check", "mark", _check),
    Sticker("exclamation", "Exclamation", "mark", _exclamation),
    Sticker("skull", "Skull", "game", _skull),
    Sticker("crown", "Crown", "game", _crown),
    Sticker("lightning", "Lightning", "game", _lightning),
    Sticker("star", "Star", "fun", _star),
    Sticker("burst", "Burst", "fun", _burst),
    Sticker("heart", "Heart", "fun", _heart),
    Sticker("bubble", "Speech bubble", "fun", _bubble),
)


@lru_cache(maxsize=1)
def catalog() -> dict[str, Sticker]:
    return {s.id: s for s in STICKERS}


def library_id(sticker_id: str, color: str) -> str:
    """What the library item made from this sticker and colour is known by:
    adding the same pair again reuses it."""
    return f"{sticker_id}:{color}"


# ── drawing ─────────────────────────────────────────────────────────────────

#: the side of a cell checked as a whole before drawing it pixel by pixel
_CELL = 8


@lru_cache(maxsize=64)
def _masks(sticker_id: str, size: int) -> tuple[bytes, bytes, bytes]:
    """(fill, outline-and-fill, details) coverage, one byte a pixel, row by row.

    Colour-free, so the eight colours of a sticker share one drawing.
    """
    drawn = catalog()[sticker_id].shape

    def shape(x: float, y: float) -> tuple[float, float]:
        d, dd = drawn(x / FIT, y / FIT)
        return d * FIT, dd * FIT

    half = size / 2
    o = OUTLINE * half  # the outline, in pixels
    fill = bytearray(size * size)
    outer = bytearray(size * size)
    detail = bytearray(size * size)

    def cover(d_px: float) -> int:
        # a pixel-wide ramp across the edge: smooth, and exact at 0.5
        v = 0.5 - d_px
        return 0 if v <= 0 else 255 if v >= 1 else int(v * 255 + 0.5)

    def pixel(px: int, py: int) -> None:
        d, dd = shape((px + 0.5) / half - 1, (py + 0.5) / half - 1)
        i = py * size + px
        fill[i] = cover(d * half)
        outer[i] = cover(d * half - o)
        detail[i] = cover(dd * half)

    # a distance is never more than the true distance to the nearest edge, so
    # a cell whose centre is further from every edge than its own reach is all
    # one thing, and only its centre needs working out
    reach = _CELL * math.sqrt(2) / 2 + 1
    for cy in range(0, size, _CELL):
        for cx in range(0, size, _CELL):
            w = min(_CELL, size - cx)
            h = min(_CELL, size - cy)
            mx, my = cx + w / 2, cy + h / 2
            d, dd = shape(mx / half - 1, my / half - 1)
            d_px, dd_px = d * half, dd * half
            if (
                abs(d_px) > reach + o
                and abs(dd_px) > reach
            ):
                f, u, t = cover(d_px), cover(d_px - o), cover(dd_px)
                for py in range(cy, cy + h):
                    row = py * size
                    fill[row + cx:row + cx + w] = bytes([f]) * w
                    outer[row + cx:row + cx + w] = bytes([u]) * w
                    detail[row + cx:row + cx + w] = bytes([t]) * w
                continue
            for py in range(cy, cy + h):
                for px in range(cx, cx + w):
                    pixel(px, py)
    return bytes(fill), bytes(outer), bytes(detail)


def _png(size: int, rgba: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    stride = size * 4
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


@lru_cache(maxsize=64)
def png_bytes(sticker_id: str, color: str = DEFAULT_COLOR, size: int = SIZE) -> bytes:
    """The sticker in one colour, as a transparent PNG."""
    if sticker_id not in catalog():
        raise KeyError(sticker_id)
    if color not in COLORS:
        raise KeyError(color)
    fill, outer, detail = _masks(sticker_id, size)
    fr, fg, fb = COLORS[color]
    lr, lg, lb = OUTLINE_RGB
    # each fill coverage, mixed between outline and colour, worked out once
    mix = [
        (
            (lr * (255 - v) + fr * v) // 255,
            (lg * (255 - v) + fg * v) // 255,
            (lb * (255 - v) + fb * v) // 255,
        )
        for v in range(256)
    ]
    out = bytearray(size * size * 4)
    for i in range(size * size):
        a = outer[i]
        if a == 0:
            continue
        r, g, b = mix[fill[i]]
        # details only show on the fill: an eye never pokes out of the head
        t = detail[i] * fill[i] // 255
        if t:
            r = (r * (255 - t) + lr * t) // 255
            g = (g * (255 - t) + lg * t) // 255
            b = (b * (255 - t) + lb * t) // 255
        j = i * 4
        out[j] = r
        out[j + 1] = g
        out[j + 2] = b
        out[j + 3] = a
    return _png(size, bytes(out))
