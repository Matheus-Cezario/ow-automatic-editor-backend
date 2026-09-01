"""Kill detection.

Input: the central crop of the screen around the crosshair, already at low
resolution and low FPS -- the microservice never sees the whole video.

In Overwatch 2, every kill draws a **magenta skull on the crosshair**, which
grows, shrinks and disappears in about half a second.

The first version of this detector only measured "how much of the region is
red", and that **does not work on a real match**: the game world (warm map
lighting, red enemy outlines, fire) and above all the *directional damage
indicator* paint the same colour range almost all the time. Measured over 19
minutes of real gameplay, the region went over 3% red in half the frames -- and
precision sat at around 17%.

What separates the skull from the rest is four things, and none of them is
colour:

* **saturation** -- the skull is an interface element, drawn over the scene with
  very high saturation; scenery lit red sits well below it;
* **position** -- it is born exactly on the crosshair, while the damage
  indicator is an arc drawn at a radius around it, up against the region's edge;
* **shape** -- it is a compact, near-square blob; the damage indicator is a
  wide, flattened arc;
* **size** -- the skull occupies 5% to 14% of the region. This was the missing
  filter: with the minimum at 0.4%, any 20-pixel red splash became a kill, and
  those splashes were most of the remaining false positives.

With all four filters, precision on the same material went from ~17% to ~91%.

The same crop answers a second question -- **was the shot a headshot?** --
because the critical-hit marker is born on that same crosshair. See
`detect_headshots`, at the end of the file.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

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
        # the higher the peak relative to the required minimum, the more confidence
        confidence = min(0.9, 0.55 + 0.35 * min(1.0, p.peak / (min_area * 2.5)))
        # the skull's eye sockets: when they show, there is nothing to confuse
        # it with. They do not always show (at 360p they are a few pixels), so
        # this adds confidence rather than acting as a filter.
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


# ------------------------------ critical hits -------------------------------
#
# The same crosshair crop answers a second question: was the shot a headshot?
#
# OW2 draws the normal hit marker in WHITE and the critical one in RED. They are
# four thick strokes in an X, centred on the crosshair, which start small and
# grow over about 0.3s.
#
# Deciding by colour is not enough -- the kill skull is also red and is also
# born on the crosshair. What separates the two is SHAPE: in the X the four
# diagonals are painted and the four straight directions are not; the skull
# fills all eight. Hence the score is `min of the diagonals - max of the
# straights`.
#
# Measured on a real recording (2558x1438, crosshair with a red marker): the X
# scores 0.81 at its peak and the rest of the video sits at 0.06 in the 99th
# percentile. On a recording with kills and no headshots at all, the maximum
# was 0.06.


def _redness(bgr: np.ndarray, min_redness: int, min_value: int) -> np.ndarray:
    """Mask of the marker's red, measured as channel *dominance*.

    A hue threshold does not serve: recordings with a colour filter (Ashe's
    scope, for instance) push the whole scene towards magenta, and then
    everything falls inside red's hue range. How far red exceeds the average of
    the other two channels does not move with the filter: the marker gives ~107
    and tinted scenery ~20.
    """
    i = bgr.astype(np.int16)
    b, g, r = i[:, :, 0], i[:, :, 1], i[:, :, 2]
    mask = ((r - (g + b) // 2) > min_redness) & (r > min_value)
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))


def _ray_coverage(
    mask: np.ndarray,
    center: tuple[float, float],
    radius: float,
    angles: Sequence[float],
    band: tuple[float, float],
    samples: int,
) -> list[float]:
    """How much of each ray, leaving the centre, falls inside the mask."""
    h, w = mask.shape
    lo, hi = band
    out: list[float] = []
    for angle in angles:
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        hits = 0
        for k in range(samples):
            rr = radius * (lo + (hi - lo) * k / max(1, samples - 1))
            x, y = int(round(center[0] + dx * rr)), int(round(center[1] + dy * rr))
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                hits += 1
        out.append(hits / samples)
    return out


#: the four tips of the X and the four directions that must stay clean
_DIAGONALS = (45.0, 135.0, 225.0, 315.0)
_STRAIGHTS = (0.0, 90.0, 180.0, 270.0)


def _headshot_score(bgr: np.ndarray, center: tuple[float, float], cfg: dict) -> float:
    """How much the frame looks like the red critical-hit X, in 0..1.

    The marker **grows** while it lives, and a single ring of rays would only
    catch the size it happened to be at that instant -- in a headshot that
    became a kill, the X is still small when the skull starts. So the
    measurement is taken over several radius bands and the best one is kept.
    """
    h, w = bgr.shape[:2]
    cx, cy = center[0] * w, center[1] * h
    radius = min(cx, cy, w - cx, h - cy)
    if radius <= 4:
        return 0.0
    mask = _redness(
        bgr, int(cfg.get("min_redness", 70)), int(cfg.get("min_value", 150))
    )
    samples = int(cfg.get("ray_samples", 16))
    best = 0.0
    for band in cfg.get("radius_bands", [[0.08, 0.30], [0.15, 0.50], [0.25, 0.70]]):
        band_range = (float(band[0]), float(band[1]))
        diagonal = min(_ray_coverage(mask, (cx, cy), radius, _DIAGONALS, band_range, samples))
        straight = max(_ray_coverage(mask, (cx, cy), radius, _STRAIGHTS, band_range, samples))
        best = max(best, diagonal - straight)
    return best


def detect_headshots(roi_video: Path, profile: Profile) -> list[DetectionEvent]:
    """Critical hits, read from the same crosshair crop as the kills."""
    cfg = profile.section("kills").get("headshot", {})
    roi = profile.roi("kills")
    # the crosshair is the centre of the SCREEN; inside the crop it is not centred
    center = roi.relative(0.5, 0.5)

    times: list[float] = []
    scores: list[float] = []
    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        times.append(frame.t)
        scores.append(_headshot_score(frame.bgr, center, cfg))

    threshold = float(cfg.get("threshold", 0.55))
    pulses = find_pulses(
        times,
        scores,
        rise=threshold,
        fall=threshold * 0.6,
        min_duration=0.0,
        min_gap=float(cfg.get("min_gap_s", 1.0)),
    )
    events = [
        DetectionEvent(
            kind=EventKind.HEADSHOT,
            t=round(p.start, 3),
            confidence=round(min(1.0, 0.5 + 0.5 * p.peak), 3),
            meta={"x_score": round(float(p.peak), 3)},
        )
        for p in pulses
    ]
    log.info("%d acerto(s) critico(s)", len(events))
    return events
