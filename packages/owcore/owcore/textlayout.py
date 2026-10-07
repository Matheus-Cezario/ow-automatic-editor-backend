"""How wide a line of text comes out, and where it breaks.

`drawtext` places one line at a time and knows only its own width, so laying
out several lines -- breaking them inside a box, lining them up on the left or
the right, drawing one box behind all of them -- needs the widths *before*
ffmpeg runs. They come from the font file itself: the sum of each character's
advance, as `drawtext` adds them up. Measured against `drawtext` on DejaVu Sans
Bold at 60px: 116.7 against 117, 128.5 against 128, 198.5 against 199.

The editor's monitor breaks lines with Flutter's own measurement, from the same
font file. Both break between words, greedily, at the same width; a word ending
within a pixel of the edge is the only place they can disagree.
"""

from __future__ import annotations

from functools import lru_cache

from fontTools.ttLib import TTFont


@lru_cache(maxsize=32)
def _face(path: str) -> tuple[dict[int, int], float, float, float]:
    """(advance per code point, in em; ascent; descent; the fallback advance)"""
    font = TTFont(path, lazy=True)
    em = float(font["head"].unitsPerEm)
    cmap = font.getBestCmap() or {}
    hmtx = font["hmtx"]
    advances = {cp: hmtx[name][0] / em for cp, name in cmap.items() if name in hmtx.metrics}
    hhea = font["hhea"]
    # what drawtext falls back to for a character the font lacks: .notdef
    notdef = hmtx[".notdef"][0] / em if ".notdef" in hmtx.metrics else 0.5
    return advances, hhea.ascent / em, -hhea.descent / em, notdef


def width(font: str, text: str, px: float) -> float:
    """How wide `text` comes out at `px` letters, in pixels."""
    advances, _, _, notdef = _face(font)
    return px * sum(advances.get(ord(c), notdef) for c in text)


def ascent_share(font: str) -> float:
    """Where the baseline sits in a line, as a share of the line's height.

    Flutter spreads a taller line's extra room in proportion to the font's
    ascent and descent; doing the same keeps the baselines of both sides on the
    same row.
    """
    _, asc, desc, _ = _face(font)
    return asc / (asc + desc) if asc + desc > 0 else 0.8


def wrap(font: str, text: str, px: float, max_width: float | None) -> list[str]:
    """The lines `text` is drawn in.

    A typed line break always breaks. With `max_width`, each typed line also
    breaks between words wherever the next word would cross it; a single word
    wider than the box stays whole on its own line rather than being cut.
    """
    lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if max_width is None or max_width <= 0:
            lines.append(paragraph)
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if current and width(font, candidate, px) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines
