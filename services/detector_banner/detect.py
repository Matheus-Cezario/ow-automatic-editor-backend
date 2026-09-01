"""Detection of the abilities announced in the footer banner.

Input: a thin strip of the footer, where OW2 stacks its action notices.

When an ability lands, a banner appears: "PUT <HERO> (<PLAYER>) TO SLEEP" for
Ana's dart, "<HERO> (<PLAYER>) STUNNED BY ACCRETION" for Sigma's rock. Except
the footer shows a whole **family** of notices with the same look -- same
colour, same shape, same position: "SAVED BY ...", "ORB OF HARMONY FROM ...",
"GAINED FROM ...". Colour and geometry find the banner, but do not say which one
it is.

What says it is the **icon** at the left end, and that is what this detector
compares -- not the text. Text changes with the language; the icon does not.
Each configured ability has its own template, and within one frame only the
winning template scores: a banner announces **one** ability, so letting two
templates mark the same banner would produce two events for the same happening.

Measured on two real recordings, with the templates trained only on the first
half of each:

* Ana, 16 min: 11/11 darts, no false positives;
* Sigma, 11 min: 23/23 rocks, no false positives (12 of them never seen in
  training).

The rock was also found once in Ana's recording -- checked by eye, it was real:
a friendly Sigma landing an Accretion.
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

#: side of the icon template, in pixels
ICON_SIZE = 24
#: the window cropped from the banner is larger than the template, so the
#: match can slide. Without that slack the crop is exactly the template's size,
#: `matchTemplate` is left with a single possible position, and a one-pixel
#: misalignment sinks the score of a legitimate ability.
ICON_WINDOW_W = 34
ICON_WINDOW_H = 28

#: abilities recognised when the profile says nothing (compatibility)
DEFAULT_ABILITIES = [
    {"key": "ana_sleep", "icon": "ana_sleep_icon.png", "event": "sleep"},
    {"key": "sigma_accretion", "icon": "sigma_accretion_icon.png", "event": "stun"},
]

#: used when neither the ability nor the section says which threshold to use
DEFAULT_THRESHOLD = 0.90


@dataclass(slots=True)
class Ability:
    key: str
    event: EventKind
    template: np.ndarray
    #: each template separates its ability at a different point -- a full,
    #: contrasty icon matches higher than one made of thin strokes -- so the
    #: threshold is per ability. A single number would force a choice between
    #: losing darts and accepting false rocks.
    threshold: float


def _icon_of(bgr: np.ndarray, banner: Banner) -> np.ndarray | None:
    """Crops the left end of the banner, where the icon lives, normalised.

    The crop follows the banner (position and height relative to it), so it
    holds at any recording resolution and for text of any size.

    The window comes out larger than the template, above and to the sides, so
    the match can slide: the banner's box varies by a pixel or two from one
    frame to the next, and without slack that drift was enough to sink the
    score.
    """
    side = banner.h
    slack_x = int(round(side * (ICON_WINDOW_W / ICON_SIZE - 1) / 2))
    slack_y = int(round(side * (ICON_WINDOW_H / ICON_SIZE - 1) / 2))
    x0 = banner.x + int(side * 0.18) - slack_x
    y0 = banner.y - slack_y
    x1, y1 = x0 + side + 2 * slack_x, y0 + side + 2 * slack_y
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
    # normalising the contrast removes the influence of the scenery behind the
    # banner, and makes the template independent of the banner's colour -- the
    # same ability appears in cyan in one recording and green in another
    return cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)


def _load_abilities(cfg: dict, shapes_dir: Path) -> list[Ability]:
    section_threshold = float(cfg.get("match_threshold", DEFAULT_THRESHOLD))
    out: list[Ability] = []
    for spec in cfg.get("abilities", DEFAULT_ABILITIES):
        path = Path(shapes_dir) / spec["icon"]
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            log.warning(
                "molde de '%s' nao encontrado em %s -- sem ele nao da para "
                "distinguir este aviso dos outros do rodape",
                spec["key"], path,
            )
            continue
        out.append(
            Ability(
                key=spec["key"],
                event=EventKind(spec["event"]),
                template=template,
                threshold=float(spec.get("match_threshold", section_threshold)),
            )
        )
    return out


def detect_abilities(
    roi_video: Path, profile: Profile, shapes_dir: Path
) -> list[DetectionEvent]:
    cfg = profile.section("banner")
    roi = profile.roi("banner")

    abilities = _load_abilities(cfg, shapes_dir)
    if not abilities:
        log.warning("nenhum molde de icon disponivel; nada a detectar")
        return []

    ranges = cfg.get("hsv_ranges", [])

    times: list[float] = []
    curves: dict[str, list[float]] = {h.key: [] for h in abilities}

    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        times.append(frame.t)
        best = {h.key: 0.0 for h in abilities}
        for banner in find_banners(
            frame.bgr,
            ranges,
            height_range=tuple(cfg.get("height_range", [0.18, 0.55])),
            width_range=tuple(cfg.get("width_range", [0.25, 0.98])),
            min_aspect=float(cfg.get("min_aspect", 3.0)),
            max_offset=float(cfg.get("max_offset", 0.25)),
            min_fill=float(cfg.get("min_fill", 0.55)),
        ):
            icon = _icon_of(frame.bgr, banner)
            if icon is None:
                continue
            icon = icon.astype(np.uint8)
            scores = {
                h.key: float(
                    cv2.matchTemplate(icon, h.template, cv2.TM_CCOEFF_NORMED).max()
                )
                for h in abilities
            }
            # the banner announces ONE ability: only the winning template scores
            winner = max(scores, key=scores.__getitem__)
            best[winner] = max(best[winner], scores[winner])
        for key, value in best.items():
            curves[key].append(value)

    events: list[DetectionEvent] = []
    for h in abilities:
        pulses = find_pulses(
            times,
            curves[h.key],
            rise=h.threshold,
            fall=h.threshold * 0.8,
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
