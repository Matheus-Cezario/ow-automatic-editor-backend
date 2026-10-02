#!/usr/bin/env python
"""Generates a synthetic video that imitates the Overwatch 2 HUD, with a ground
truth JSON (the exact timestamp of every event).

It serves two purposes:

1. testing the whole pipeline end to end without real gameplay;
2. giving the user a way to see the system working before recording.

Usage:
    python tools/make_sample.py --out data/sample/match.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import wave
from typing import NamedTuple
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
SR = 44100

# ground truth (seconds)
KILLS = [
    5.0, 6.2, 7.4,            # burst of 3
    20.0, 21.0, 22.2, 23.1,   # burst of 4
    34.0,                     # single
    41.5,                     # single
    52.0,                     # single
]
# a destroyed deployable -- a Symmetra turret: the crosshair draws the same
# skull as a kill, and NOTHING appears in the killfeed. It must not count.
# Inside the first 12 s, so the short sample of the pipeline tests has it too.
OBJECT_KILLS = [2.5]
# When you die in OW2 you start spectating a teammate: health drops to zero for
# an instant and comes back full. That is the signature the detector looks for.
DEATHS = [28.0]
LOW_HP = [(12.0, 2.5), (15.5, 2.0), (45.0, 2.2)]  # survived episodes
LOW_HP_FRAC = 0.18
FULL_HP_FRAC = 0.95
ULTS = [40.0, 51.0]                               # enemy ult (audio + icon)
# abilities announced in the footer, and decoy banners with the same format
SLEEPS = [18.0, 30.0, 48.0]          # Ana's dart, cyan banner
STUNS = [22.0, 38.0]                 # Sigma's rock, GREEN banner on purpose:
                                     # the banner colour changes from recording
                                     # to recording, and the pipeline must cope
DECOY_BANNERS = [25.0, 55.0]
BANNER_S = 2.5
SKULL_DURATION = 0.9

# the PLAYER's ultimate: the instant it is used. The button stays charged for
# the seconds before and goes out here -- the falling edge is what becomes the
# event.
SELF_ULTS = [24.0, 47.0]
SELF_ULT_CHARGE_S = 6.0
# trap: the button shows up charged for an instant and disappears. In the game
# that is the kill cam (a disc with the killer's face, in the same place) or an
# explosion flash; in 27 min of real recordings every short stretch like that
# was false, and a real ultimate stays charged for seconds before being used.
# Far enough from `SELF_ULTS` that `min_after_s` is not what discards it: it is
# `min_charged_s` that must refuse it.
ULT_FLASHES = [30.0]
ULT_FLASH_S = 0.4
# critical hit: a red X marker on the crosshair. Away from the kills on purpose
# -- the skull covers the same diagonals, and the detector notes that in that
# clash the skull wins.
HEADSHOTS = [10.0, 33.0, 44.0]
HEADSHOT_S = 0.35
# a kill with an ability announced in the killfeed. The line stays on screen
# for several seconds: the event is it APPEARING, not it being there.
# Away from the LOW_HP windows: the synthetic video's damage vignette is much
# stronger than the game's and covers the killfeed corner, erasing the red
# plate. That is a limitation of the test drawing, not of the detector.
ABILITY_KILLS = [26.0, 36.0]
ABILITY_ROW_S = 6.0
# the same killfeed line, but with ANOTHER name on the killer's plate: a
# teammate killing with an ability. The killfeed announces all ten people in
# the match, and only the player's become montage material -- these exist so
# the detector has to refuse them.
#
# `PATRICK` has the same 7 letters as `PLAYER_NAME` on purpose: with names of
# different lengths the comparison would get it right by the count, without
# ever looking at the letter shapes, and the test would pass even with the
# drawing broken. No letter of the two sits in the same position either.
#
# Not every 7-letter name survives the killfeed's small font: in some, two
# letters touch and the name is read with 6. `HUNTER7` was checked to read whole.
TEAMMATE_KILLS = [16.0, 51.0]
PLAYER_NAME = "HUNTER7"
TEAMMATE_NAME = "PATRICK"
# every kill of `KILLS` puts the player's line in the killfeed too, the way a
# gun kill does in the game: no icon between the plates. It is that line that
# tells the kill from the turret of `OBJECT_KILLS`.
KILL_ROW_S = 4.0
#: how much shorter each kill's victim plate is. In the game a plate is as long
#: as the name on it, and the tracker tells lines apart by that width: two
#: lines less than `hold_s` apart with the same width would be one line to it.
#: These differ by at least 20 px from every other line alive near them, and
#: none goes past -100: a shorter plate falls under `killfeed.min_aspect` and
#: is no plate at all to the detector.
KILL_VICTIMS = [-20, -80, -100, -20, -100, -40, -80, -100, -20, -60]

DURATION = 60.0


# ------------------------------- drawing ------------------------------------


def draw_skull(img: np.ndarray, cx: int, cy: int, r: int, alpha: float) -> None:
    """Kill skull, on the crosshair.

    Colour and position measured in real gameplay: a very saturated magenta
    (HSV ~167, 230, 235) centred at (0.50, 0.485) of the screen -- slightly
    *above* the middle.
    """
    layer = img.copy()
    red = (115, 23, 235)  # BGR of the HUD magenta
    cv2.circle(layer, (cx, cy - r // 5), r, red, -1)
    cv2.rectangle(layer, (cx - r // 2, cy + r // 2), (cx + r // 2, cy + r), red, -1)
    dark = (10, 10, 40)
    eye = max(2, r // 4)
    cv2.circle(layer, (cx - r // 2, cy - r // 4), eye, dark, -1)
    cv2.circle(layer, (cx + r // 2, cy - r // 4), eye, dark, -1)
    cv2.rectangle(layer, (cx - eye // 2, cy + r // 3), (cx + eye // 2, cy + r), dark, -1)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def draw_ult_icon(img: np.ndarray, x: int, y: int) -> None:
    """Killfeed icon used as an ultimate: an orange diamond with a ring."""
    pts = np.array([[x, y - 18], [x + 18, y], [x, y + 18], [x - 18, y]], np.int32)
    cv2.fillPoly(img, [pts], (30, 150, 255))
    cv2.circle(img, (x, y), 8, (255, 255, 255), 2)


def ult_template() -> np.ndarray:
    tpl = np.zeros((44, 44, 3), np.uint8)
    draw_ult_icon(tpl, 22, 22)
    return tpl


# ---------------------- ultimate button and killfeed ------------------------
#
# These three things -- charged ultimate button, critical hit marker and
# killfeed line -- are what the newer detectors read. Here they are drawn with
# the same geometry the real HUD uses, measured on the reference recordings;
# what is tested is the plumbing (find the region, crop the glyph, match, turn
# into an event), not the matching accuracy, which was measured on real
# gameplay.

#: the marks used in the sample video. They are polygons and not files because
#: the same drawing has to come out in two places -- on screen and in the icon
#: bank the detector compares against -- and drawing it guarantees both match
#: without depending on any game asset.
GLYPHS: dict[str, list[tuple[float, float]]] = {
    # wide arrow pointing up, with a notch: asymmetric vertically and
    # horizontally, so it does not match itself rotated
    "self_ult": [(0.5, 0.05), (0.95, 0.55), (0.68, 0.55), (0.68, 0.95),
                 (0.32, 0.95), (0.32, 0.55), (0.05, 0.55)],
    # hourglass lying down
    "ability_kill": [(0.05, 0.08), (0.05, 0.92), (0.5, 0.5),
                     (0.95, 0.92), (0.95, 0.08), (0.5, 0.5)],
}


def glyph_mask(key: str, side: int) -> np.ndarray:
    """The mark, white on black, at the requested size."""
    m = np.zeros((side, side), np.uint8)
    pts = np.array([[int(x * side), int(y * side)] for x, y in GLYPHS[key]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def glyph_template(key: str, side: int = 128) -> np.ndarray:
    """The same mark in the icon bank's format: black on white."""
    return 255 - glyph_mask(key, side)


def _stamp(img: np.ndarray, mask: np.ndarray, x: int, y: int,
           colour: tuple[int, int, int]) -> None:
    """Paints the mark through its mask -- instead of pasting an opaque square."""
    h, w = mask.shape
    target = img[y:y + h, x:x + w]
    if target.shape[:2] != mask.shape:
        return
    target[mask > 0] = colour


#: ultimate button geometry, the same as the `ow2_default` profile's
ULT_CX, ULT_CY = 0.5, 0.86
ULT_DISC_R = 26
ULT_RING_R0, ULT_RING_R1 = 30, 38


def draw_ult_button(img: np.ndarray, charged: bool) -> None:
    """The footer button in its two states.

    Charged is a WHITE disc with the hero's mark in black, surrounded by a CYAN
    ring; not charged is just a dark ring. The detector requires both things
    together, so drawing only one of them would produce no event -- which is
    exactly what makes the uncharged state a useful trap.
    """
    cx, cy = int(ULT_CX * W), int(ULT_CY * H)
    if not charged:
        cv2.circle(img, (cx, cy), ULT_DISC_R + 4, (210, 210, 210), 2)
        return
    cv2.circle(img, (cx, cy), ULT_RING_R1, (235, 190, 60), -1)   # cyan BGR
    cv2.circle(img, (cx, cy), ULT_RING_R0, (40, 45, 50), -1)
    cv2.circle(img, (cx, cy), ULT_DISC_R, (250, 250, 250), -1)
    side = int(ULT_DISC_R * 1.15)
    _stamp(img, glyph_mask("self_ult", side),
           cx - side // 2, cy - side // 2, (20, 20, 20))


def draw_crit_marker(img: np.ndarray) -> None:
    """Critical hit marker: four red strokes in an X on the crosshair.

    The four STRAIGHT directions stay clear on purpose -- that is the
    difference between this marker and the kill skull, which fills all eight.
    """
    cx, cy = W // 2, H // 2
    r0, r1 = 14, 40
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        p0 = (cx + int(dx * r0 * 0.71), cy + int(dy * r0 * 0.71))
        p1 = (cx + int(dx * r1 * 0.71), cy + int(dy * r1 * 0.71))
        cv2.line(img, p0, p1, (60, 40, 235), 5)


#: geometry of a killfeed line, inside the `killfeed` ROI
KF_Y0, KF_H = 30, 26
#: distance between two stacked lines
KF_STEP = 34
KF_ALLY = (900, 1060)     # cyan plate: the killer
KF_ENEMY = (1120, 1270)   # red plate: the victim


def draw_killfeed_row(img: np.ndarray, victim: int = 0,
                      killer: str = PLAYER_NAME, row: int = 0,
                      icon: bool = True) -> None:
    """`[ killer plate ] [ icon ] > [ victim plate ]`.

    `victim` shortens the red plate. In the game the length of each plate is
    that of the name written on it, so two kills only have the same width if
    they are by the same player **on the same victim**; drawing them all
    identical would make the sample test a killfeed that does not exist. And it
    is by that width that the detector recognises a line from one frame to the
    next.

    `killer` is the name written on the blue plate. The colour does NOT say
    whose kill it was -- blue is the killer and red the victim, on both sides
    of the match -- so it is this name, and only it, that separates the
    player's kill from a teammate's.

    `row` stacks the line under the ones already on screen. `icon` False is a
    gun kill: the gap between the plates holds only the `>`.
    """
    y0 = KF_Y0 + row * KF_STEP
    y1 = y0 + KF_H
    cv2.rectangle(img, (KF_ALLY[0], y0), (KF_ALLY[1], y1), (190, 140, 70), -1)
    _draw_name(img, killer, KF_ALLY[0] + 6, y0, KF_H, 0.5, 2)
    cv2.rectangle(img, (KF_ENEMY[0], y0), (KF_ENEMY[1] + victim, y1),
                  (90, 60, 200), -1)
    if icon:
        # the icon box and the chevron fill the gap between the two plates
        cv2.rectangle(img, (KF_ALLY[1] + 4, y0 - 3), (KF_ALLY[1] + 36, y1 + 3),
                      (55, 52, 50), -1)
        side = 22
        _stamp(img, glyph_mask("ability_kill", side),
               KF_ALLY[1] + 9, (y0 + y1) // 2 - side // 2, (240, 240, 240))
    # the `>` takes the LAST quarter of the gap, as on the real HUD: measured on
    # the reference recording, the icon box goes up to ~0.73 of the gap and the
    # chevron starts there. `killfeed.icon_span` stops at 0.66 precisely so it
    # does not touch it -- drawing the chevron further left than in the game
    # would make the sample test a layout that does not exist.
    cv2.putText(img, ">", (KF_ALLY[1] + int(0.76 * (KF_ENEMY[0] - KF_ALLY[1])), y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)


#: banner geometry, the same as the `ow2_default` profile's
BANNER_Y0, BANNER_Y1 = 0.695, 0.727
BANNER_W = 0.167


#: banner colours, in BGR, inside the profile's two HSV ranges
CYAN = (196, 150, 74)
GREEN = (110, 165, 60)


def _icon(name: str) -> np.ndarray:
    """Loads a real template used by the detector, in greyscale.

    The synthetic video exists to exercise the *pipeline* -- find the banner,
    crop the icon in the right place, match and turn into an event. The
    matching accuracy itself was measured on real gameplay; what is tested here
    is the plumbing, and for that the icon must be the one the detector looks
    for.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "shapes" / name
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"icon template not found: {path}")
    return img


def draw_banner(
    img: np.ndarray,
    text: str,
    icon: np.ndarray | None,
    colour: tuple[int, int, int] = CYAN,
) -> None:
    """Footer banner. With `icon` None it draws a decoy: same banner, same
    colour, same position -- a different symbol."""
    y0, y1 = int(BANNER_Y0 * H), int(BANNER_Y1 * H)
    width = int(BANNER_W * W)
    x0 = W // 2 - width // 2
    height = y1 - y0
    cv2.rectangle(img, (x0, y0), (x0 + width, y1), colour, -1)
    side = height
    ax = x0 + int(height * 0.18)
    if icon is not None:
        # composited through its alpha, and not pasted as an opaque square: on
        # the OW2 HUD the icon is a light drawing ON TOP of the coloured banner.
        # Pasting the whole template put a dark frame around it, and the
        # detector's contrast normalisation started seeing that frame instead
        # of the drawing -- the crop stopped looking like the template itself.
        alpha = (
            cv2.resize(icon, (side, side), interpolation=cv2.INTER_AREA)
            .astype(np.float32) / 255.0
        )[:, :, None]
        under = img[y0:y0 + side, ax:ax + side].astype(np.float32)
        img[y0:y0 + side, ax:ax + side] = (
            under * (1.0 - alpha) + 255.0 * alpha
        ).astype(np.uint8)
    else:
        cv2.circle(img, (ax + side // 2, y0 + side // 2), side // 3,
                   (255, 255, 255), 2)
    cv2.putText(img, text, (ax + side + 6, y1 - int(height * 0.28)),
                cv2.FONT_HERSHEY_SIMPLEX, height / 46.0, (255, 255, 255), 1)


def background(t: float) -> np.ndarray:
    """Playable scenery: a greenish gradient + moving blobs.

    The palette is chosen to stay away from the two colours the detectors look
    for: the kill skull's magenta (hue 156-178) and the footer banner's cyan
    (hue 90-115). That way, an event detected in the synthetic video can only
    have come from the drawn HUD, never from the scenery.
    """
    yy = np.linspace(60, 150, H, dtype=np.float32)[:, None]
    xx = np.linspace(40, 120, W, dtype=np.float32)[None, :]
    img = np.zeros((H, W, 3), np.float32)
    img[:, :, 0] = 30 + xx * 0.12               # low blue
    img[:, :, 1] = yy * 0.85 + xx * 0.35        # dominant green
    img[:, :, 2] = 50 + xx * 0.30               # medium red
    img = np.clip(img, 0, 255).astype(np.uint8)

    for k in range(6):
        px = int((math.sin(t * 0.6 + k) * 0.4 + 0.5) * W)
        py = int((math.cos(t * 0.45 + k * 1.7) * 0.4 + 0.5) * H)
        cv2.circle(img, (px, py), 60 + k * 12, (55, 130 + k * 12, 85), -1)
    rng = np.random.default_rng(int(t * FPS))
    noise = rng.integers(-8, 8, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


#: health bar geometry, the same as the `ow2_default` profile's
BAR_X0, BAR_X1 = 0.09, 0.225
BAR_Y0, BAR_Y1 = 0.855, 0.895
N_TICKS = 26


def draw_hud(img: np.ndarray, hp: float | None) -> None:
    # crosshair
    cv2.line(img, (W // 2 - 12, H // 2), (W // 2 + 12, H // 2), (230, 230, 230), 2)
    cv2.line(img, (W // 2, H // 2 - 12), (W // 2, H // 2 + 12), (230, 230, 230), 2)
    if hp is None:
        return  # no HUD: menu, round change

    # the player's card follows the rest of the HUD: it disappears with it in
    # the menu and on round changes, as in the game
    draw_player_card(img)

    # Health bar as OW2 draws it: light vertical ticks over a dark track, with
    # the total width normalised. It is the *alternation* of those ticks that
    # the detector reads -- that is why they must really exist, and not be a
    # solid rectangle.
    x0, x1 = int(BAR_X0 * W), int(BAR_X1 * W)
    y0, y1 = int(BAR_Y0 * H), int(BAR_Y1 * H)
    cv2.rectangle(img, (x0, y0), (x1, y1), (38, 34, 30), -1)
    step = (x1 - x0) / N_TICKS
    filled = int(round(N_TICKS * max(0.0, min(1.0, hp))))
    for i in range(filled):
        tx = int(x0 + i * step)
        cv2.rectangle(img, (tx + 1, y0 + 2), (int(tx + step) - 1, y1 - 2),
                      (235, 240, 240), -1)


#: the player's own card, the same as the `ow2_default` profile's (roi
#: `player`). Slightly inside the ROI: in the game the card does not touch the
#: crop's edge, and a letter glued to the edge comes out cut by the crop.
CARD_X0, CARD_X1 = 0.095, 0.232
CARD_Y0, CARD_Y1 = 0.899, 0.929


def draw_player_card(img: np.ndarray, name: str = PLAYER_NAME) -> None:
    """The card with the player's name.

    It is the only place on screen that says who is playing. The killfeed
    detector reads it to know which of the announced kills are the user's.

    The letter scale is larger than the killfeed's on purpose: on the real HUD
    the two writings have different sizes and spacings, and comparing two
    identical writings would make the sample test a comparison that does not
    exist.
    """
    x0, x1 = int(CARD_X0 * W), int(CARD_X1 * W)
    y0, y1 = int(CARD_Y0 * H), int(CARD_Y1 * H)
    cv2.rectangle(img, (x0, y0), (x1, y1), (120, 62, 44), -1)
    _draw_name(img, name, x0 + int((x1 - x0) * 0.22), y0, y1 - y0, 0.62, 2)


def _draw_name(img: np.ndarray, name: str, x: int, y: int, height: int,
               scale: float, thickness: int) -> None:
    """Writes the name on a plate, vertically centred.

    Below a 0.5 scale the Hershey letters touch and merge into one -- which is
    already a different name. Measured at the killfeed plate's size: at 0.45
    and thickness 1 a 7-letter name came out with 6 letters; at 0.5 and
    thickness 2 it comes out with all 7.
    """
    (_tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(img, name, (x, y + (height + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (240, 240, 240), thickness, cv2.LINE_AA)


def draw_damage_vignette(img: np.ndarray, strength: float) -> None:
    """Red vignette on the edges -- in OW2 this is *damage taken*, not low
    health. It is here on purpose: in a real match it flashes all the time, and
    the detector must keep getting it right with it on screen."""
    band_y, band_x = int(H * 0.11), int(W * 0.11)
    layer = img.copy()
    cv2.rectangle(layer, (0, 0), (W, band_y), (30, 30, 245), -1)
    cv2.rectangle(layer, (0, H - band_y), (W, H), (30, 30, 245), -1)
    cv2.rectangle(layer, (0, 0), (band_x, H), (30, 30, 245), -1)
    cv2.rectangle(layer, (W - band_x, 0), (W, H), (30, 30, 245), -1)
    cv2.addWeighted(layer, 0.75 * strength, img, 1 - 0.75 * strength, 0, img)


def health_at(t: float) -> float | None:
    """Health at instant t. None = no HUD on screen."""
    for d in DEATHS:
        if d <= t < d + 0.35:
            return 0.0
    for start, length in LOW_HP:
        if start <= t < start + length:
            return LOW_HP_FRAC
    return FULL_HP_FRAC


def frame_at(t: float) -> np.ndarray:
    img = background(t)
    hp = health_at(t)

    # The damage vignette follows the low-health moments, as in the game --
    # and serves as a trap: it is red and covers the edges, but must not
    # produce any event on its own.
    for s, ln in LOW_HP:
        if s <= t < s + ln:
            draw_damage_vignette(img, 0.6 + 0.4 * abs(math.sin((t - s) * 6.0)))
    draw_hud(img, hp)

    for k in KILLS + OBJECT_KILLS:
        if k <= t < k + SKULL_DURATION:
            phase = (t - k) / SKULL_DURATION
            alpha = min(1.0, (1.0 - phase) * 2.2)
            # centred at (0.50, 0.485), as measured in real gameplay
            draw_skull(img, W // 2, int(H * 0.485), 34, alpha)

    for s0 in SLEEPS:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "PUT MERCY (TEST) TO SLEEP",
                        _icon("ana_sleep_icon.png"), CYAN)
    for s0 in STUNS:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "ORISA (TEST) STUNNED BY ACCRETION",
                        _icon("sigma_accretion_icon.png"), GREEN)
    for s0 in DECOY_BANNERS:
        if s0 <= t < s0 + BANNER_S:
            draw_banner(img, "ORB OF HARMONY FROM TEST", None, CYAN)

    for u in ULTS:
        if u <= t < u + 2.5:
            # below the stack of killfeed lines: at 0.12 the lines of
            # `KILLFEED` covered it
            draw_ult_icon(img, int(W * 0.80), int(H * 0.27))
            cv2.putText(img, "ULTIMATE", (int(W * 0.66), int(H * 0.20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

    draw_ult_button(img, ult_charged(t))
    for hs in HEADSHOTS:
        if hs <= t < hs + HEADSHOT_S:
            draw_crit_marker(img)
    for ln in KILLFEED:
        if ln.start <= t < ln.start + ln.duration:
            draw_killfeed_row(img, victim=ln.victim, killer=ln.killer,
                              row=ln.row, icon=ln.icon)
    return img


class _FeedLine(NamedTuple):
    start: float
    duration: float
    victim: int
    killer: str
    icon: bool
    row: int = 0


def _stack(lines: list[_FeedLine]) -> list[_FeedLine]:
    """Gives each line the first row free when it appears, and keeps it there.

    In the game the lines stack and slide; here they only stack. The tracker
    does not follow a line by its height, so the slide adds nothing to test.
    """
    placed: list[_FeedLine] = []
    for ln in sorted(lines, key=lambda x: x.start):
        busy = {
            p.row for p in placed
            if p.start <= ln.start < p.start + p.duration
        }
        row = next(r for r in range(len(lines) + 1) if r not in busy)
        placed.append(ln._replace(row=row))
    return placed


#: every killfeed line of the sample, already stacked
KILLFEED = _stack(
    # ability kills: different victims, plates of different lengths
    [_FeedLine(ak, ABILITY_ROW_S, -40 * i, PLAYER_NAME, True)
     for i, ak in enumerate(ABILITY_KILLS)]
    # a teammate's, with widths that do not coincide with ABILITY_KILLS': in
    # the game the plate's length is that of the name on it, and two lines
    # only have the same width when they are the same pair. Drawing two
    # different kills with the same width, less than `hold_s` apart, would
    # make the tracker see them as one line that vanished and came back --
    # which is what it exists not to confuse.
    + [_FeedLine(tk, ABILITY_ROW_S, -60 - 35 * i, TEAMMATE_NAME, True)
       for i, tk in enumerate(TEAMMATE_KILLS)]
    # the player's gun kills, which confirm the crosshair's skulls
    + [_FeedLine(k, KILL_ROW_S, v, PLAYER_NAME, False)
       for k, v in zip(KILLS, KILL_VICTIMS, strict=True)]
)


def ult_charged(t: float) -> bool:
    """The button stays charged for the seconds before each use.

    And also during the `ULT_FLASHES` blinks, which are no use at all: they are
    the trap the detector must refuse by duration.
    """
    if any(u - SELF_ULT_CHARGE_S <= t < u for u in SELF_ULTS):
        return True
    return any(f <= t < f + ULT_FLASH_S for f in ULT_FLASHES)


# -------------------------------- audio -------------------------------------


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def game_audio(duration: float) -> np.ndarray:
    """Low ambience + loud bursts at the ults (the detectors' audio source)."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    sig = 0.02 * rng.standard_normal(n) + 0.02 * np.sin(2 * np.pi * 110 * t)
    for u in ULTS:
        i0 = int(u * SR)
        if i0 >= n:
            continue  # video shorter than the full ground truth
        i1 = min(n, i0 + int(1.2 * SR))
        env = np.exp(-np.linspace(0, 5, i1 - i0))
        sig[i0:i1] += 0.8 * env * np.sin(2 * np.pi * 220 * t[i0:i1])
    return sig


def click_track(duration: float, bpm: float = 120.0) -> np.ndarray:
    """Test music: a kick on the beat + bass. Known BPM."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    sig = 0.10 * np.sin(2 * np.pi * 55 * t)
    period = 60.0 / bpm
    beat = 0.0
    while beat < duration:
        i0 = int(beat * SR)
        i1 = min(n, i0 + int(0.12 * SR))
        env = np.exp(-np.linspace(0, 14, i1 - i0))
        tt = t[i0:i1] - beat
        sig[i0:i1] += 0.9 * env * np.sin(2 * np.pi * 65 * tt)
        sig[i0:i1] += 0.35 * env * np.sin(2 * np.pi * 2500 * tt)
        beat += period
    return sig


# ------------------------------ generation ----------------------------------


def render(out: Path, duration: float, ffmpeg: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_path = out.with_name("game_audio.wav")
    write_wav(audio_path, game_audio(duration))

    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(duration * FPS)
    assert proc.stdin is not None
    for i in range(total):
        proc.stdin.write(frame_at(i / FPS).tobytes())
        if i % (FPS * 5) == 0:
            print(f"  {i / FPS:5.1f}s / {duration:.0f}s", file=sys.stderr)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed to build the sample video")
    audio_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/sample/match.mp4"))
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--music", type=Path, default=None,
                    help="also generates test music (a kick at 120 BPM)")
    ap.add_argument("--ult-templates", type=Path, default=None,
                    help="saves the ult icon used, to calibrate the detector")
    ap.add_argument("--ability-icons", type=Path, default=None,
                    help="saves the ult button and killfeed marks in the "
                         "templates/abilities/ format, so the detector can "
                         "say WHICH ability it was")
    args = ap.parse_args()

    print(f"generating {args.out} ({args.duration:.0f}s)...", file=sys.stderr)
    render(args.out, args.duration, args.ffmpeg)

    truth = {
        "duration_s": args.duration,
        "fps": FPS,
        "size": [W, H],
        "kills": KILLS,
        # skulls with no killfeed line: they must not end up as kills
        "object_kills": OBJECT_KILLS,
        "deaths": list(DEATHS),
        "low_hp": [d[0] for d in LOW_HP],
        "ults": ULTS,
        "sleeps": SLEEPS,
        "stuns": STUNS,
        "decoy_banners": DECOY_BANNERS,
        "self_ults": SELF_ULTS,
        "ult_flashes": ULT_FLASHES,
        "headshots": HEADSHOTS,
        "ability_kills": ABILITY_KILLS,
        # a teammate's: the killfeed announces them the same way, and the
        # detector must refuse them by the name written on the plate
        "teammate_kills": TEAMMATE_KILLS,
        "player_name": PLAYER_NAME,
    }
    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    if args.music:
        args.music.parent.mkdir(parents=True, exist_ok=True)
        write_wav(args.music, click_track(args.duration, 120.0))
        print(f"test music: {args.music}", file=sys.stderr)

    if args.ult_templates:
        args.ult_templates.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.ult_templates / "sample_ult.png"), ult_template())
        print(f"ult template: {args.ult_templates}", file=sys.stderr)

    if args.ability_icons:
        # the structure is the same as templates/abilities/: one folder per hero
        hero = args.ability_icons / "sample"
        hero.mkdir(parents=True, exist_ok=True)
        for key in GLYPHS:
            cv2.imwrite(str(hero / f"{key}.png"), glyph_template(key))
        print(f"ability icons: {hero}", file=sys.stderr)

    print(f"done. ground truth in {truth_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
