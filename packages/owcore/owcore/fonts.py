"""Where to find the font that `drawtext` will use.

ffmpeg ships no built-in font: without a `.ttf` on disk, every text clip fails
at render time. This module finds one, and fails loudly when it cannot --
discovering that halfway through a render would be worse than at setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
        "no font found for text; point one at OW_FONT"
    )


def available() -> bool:
    """Can this machine draw text at all?"""
    try:
        default_font()
    except FileNotFoundError:
        return False
    return True


# -- the catalogue: the fonts the editor offers -----------------------------
#
# A text clip stores a font **id**, not a path: the same montage renders on
# any machine, and the app can ask for the same file to draw its preview.

#: the system's own faces, offered alongside the bundled ones when present.
#: `dejavu-sans-bold` is what text always used, so it stays the default
_SYSTEM = (
    ("dejavu-sans-bold", "DejaVu Sans Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("dejavu-sans", "DejaVu Sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ("dejavu-serif", "DejaVu Serif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ("dejavu-sans-mono", "DejaVu Sans Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
)

DEFAULT_ID = "dejavu-sans-bold"


@dataclass(frozen=True)
class Font:
    id: str
    name: str
    path: Path
    category: str


@lru_cache
def catalog() -> dict[str, Font]:
    """Every font on offer, by id: the bundled ones, then the system's."""
    out: dict[str, Font] = {}
    listing = get_settings().fonts_dir / "catalog.json"
    if listing.is_file():
        for f in json.loads(listing.read_text())["fonts"]:
            path = get_settings().fonts_dir / f["file"]
            if path.is_file():
                out[f["id"]] = Font(f["id"], f["name"], path, f["category"])
    for fid, name, path in _SYSTEM:
        if Path(path).is_file():
            out[fid] = Font(fid, name, Path(path), "system")
    return out


def resolve(font: str) -> str:
    """The file for a text clip's `font`: an id from the catalogue, a path
    (montages saved before the catalogue stored one), or empty for the
    default."""
    if not font:
        return default_font()
    known = catalog().get(font)
    if known is not None:
        return str(known.path)
    if "/" in font or "\\" in font:
        return font
    raise ValueError(f"unknown font {font!r}")
