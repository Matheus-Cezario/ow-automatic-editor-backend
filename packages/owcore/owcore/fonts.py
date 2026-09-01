"""Where to find the font that `drawtext` will use.

ffmpeg ships no built-in font: without a `.ttf` on disk, every text clip fails
at render time. This module finds one, and fails loudly when it cannot --
discovering that halfway through a render would be worse than at setup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import get_settings

#: Where to look, in order. DejaVu ships with ffmpeg in the Docker image; the
#: others cover people running the system outside it.
CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)


@lru_cache
def default_font() -> str:
    """The font path to use when nobody picked one.

    `OW_FONT` wins when set. Without it, the first candidate that exists on
    disk is used.
    """
    chosen = get_settings().font
    if chosen:
        return chosen
    for path in CANDIDATES:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(
        "nenhuma fonte encontrada para o texto; aponte uma em OW_FONT"
    )


def available() -> bool:
    """Can this machine draw text at all?"""
    try:
        default_font()
    except FileNotFoundError:
        return False
    return True
