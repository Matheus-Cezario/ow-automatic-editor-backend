"""Catalogue: the read-only libraries -- fonts, sound effects, stickers --
and the detection profile."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from owcore import fonts, sfx, stickers
from owcore.config import get_settings

from common import _RANGE

router = APIRouter()


# ── the text fonts ────────────────────────────────────────────────────────


@router.get("/api/fonts")
def list_fonts() -> list[dict[str, Any]]:
    """The fonts a text can use. The app loads each file to draw its preview
    with the same face the render uses."""
    return [
        {
            "id": f.id,
            "name": f.name,
            "category": f.category,
            "url": f"/api/fonts/{f.id}",
            "default": f.id == fonts.DEFAULT_ID,
        }
        for f in fonts.catalog().values()
    ]


@router.get("/api/fonts/{font_id}")
def font_file(font_id: str) -> Response:
    font = fonts.catalog().get(font_id)
    if font is None:
        raise HTTPException(404, "font not found")
    return Response(
        content=font.path.read_bytes(),
        media_type="font/ttf",
        # a font file never changes under its id
        headers={"cache-control": "public, max-age=604800"},
    )


# ── the sound effects library ───────────────────────────────────────────────
#
# A catalogue the server synthesises (`owcore.sfx`), not files the user brings.
# Browsing it costs nothing; adding an effect to a match turns it into an
# ordinary audio item of that match's library, and from there on it is a block
# like the music -- same layers, same render, same mix.


def _sfx_dict(effect: sfx.Effect) -> dict[str, Any]:
    return {
        "id": effect.id,
        "name": effect.name,
        "category": effect.category,
        "duration_s": sfx.duration_s(effect.id),
        "peaks": list(sfx.peaks(effect.id)),
        "audio_url": f"/api/sfx/{effect.id}/audio",
    }


@router.get("/api/sfx")
def list_sfx() -> dict[str, Any]:
    """Every effect on offer, grouped by `category` in the order to show."""
    return {
        "categories": list(sfx.CATEGORIES),
        "effects": [_sfx_dict(e) for e in sfx.EFFECTS],
    }


@router.get("/api/sfx/{sfx_id}/audio")
def sfx_audio(sfx_id: str, request: Request) -> Response:
    """The effect itself, to listen to before adding it."""
    if sfx_id not in sfx.catalog():
        raise HTTPException(404, "sound effect not found")
    body = sfx.wav_bytes(sfx_id)
    total = len(body)
    # an effect never changes under its id
    headers = {"accept-ranges": "bytes", "cache-control": "public, max-age=604800"}
    match = _RANGE.match(request.headers.get("range") or "")
    if not match:
        return Response(content=body, media_type="audio/wav", headers=headers)
    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else total - 1
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return Response(
        content=body[start : end + 1],
        status_code=206,
        media_type="audio/wav",
        headers=headers | {"content-range": f"bytes {start}-{end}/{total}"},
    )


# ── the stickers library ────────────────────────────────────────────────────
#
# The same idea as the sound effects, for pictures: a catalogue the server
# draws (`owcore.stickers`); adding one, in a colour, turns it into an ordinary
# image item of the match's library.


def _sticker_dict(sticker: stickers.Sticker) -> dict[str, Any]:
    return {
        "id": sticker.id,
        "name": sticker.name,
        "category": sticker.category,
        "preview_url": f"/api/stickers/{sticker.id}.png",
    }


@router.get("/api/stickers")
def list_stickers() -> dict[str, Any]:
    """Every sticker on offer, grouped by `category` in the order to show, and
    the colours each one comes in."""
    return {
        "categories": list(stickers.CATEGORIES),
        "colors": [
            {"id": c, "hex": "#%02x%02x%02x" % rgb}
            for c, rgb in stickers.COLORS.items()
        ],
        "stickers": [_sticker_dict(st) for st in stickers.STICKERS],
    }


@router.get("/api/stickers/{sticker_id}.png")
def sticker_png(sticker_id: str, color: str = stickers.DEFAULT_COLOR) -> Response:
    """A small copy of the sticker, for the shelf."""
    if sticker_id not in stickers.catalog():
        raise HTTPException(404, "sticker not found")
    if color not in stickers.COLORS:
        raise HTTPException(422, f"no colour {color!r}")
    body = stickers.png_bytes(sticker_id, color, stickers.PREVIEW_SIZE)
    # a sticker never changes under its id
    return Response(
        content=body,
        media_type="image/png",
        headers={"cache-control": "public, max-age=604800"},
    )


@router.get("/api/profile")
def profile() -> dict[str, Any]:
    """Exposes the HUD profile for the app's calibration screen."""
    from owcore.profiles import load_profile

    return load_profile(get_settings().profile).data
