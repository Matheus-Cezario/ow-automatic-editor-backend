"""Survival detection: low health, interruption and escape.

Input: a thin strip of the bottom-left corner -- the health bar alone.

**How this used to work, and why it changed.** The first version inferred low
health from the red vignette at the screen edges. Measured over 19 minutes of
real gameplay, that signal appeared in **32% of frames** and produced 122
"escapes" -- because that vignette is the *damage taken* indicator, which
flashes constantly in a match, and not the low-health warning. Death was
inferred from the killcam's drop in saturation, and found **zero** deaths: the
OW2 killcam is not desaturated.

The current version reads the health bar directly. The bar is drawn as a run of
bright vertical ticks, and OW2 normalises its width -- so the filled fraction is
the health fraction, whether the hero has 200 or 700 health. The reading does
not use brightness (the scenery behind the HUD can be bright): it uses the
bright/dark **alternation** of the ticks, which only exists in the filled part.
Against values read off the screen, the error stayed within 0.05 (0.56 -> 0.55,
0.53 -> 0.49, 0.95 -> 0.92).

**About `DEATH`.** The first version assumed that on death the HUD moves to the
teammate being spectated, so death would be one frame at zero and then someone
else's full bar. In full real matches the HUD stays on the player's own card
until the respawn, and the bar turns into a dim track with no ticks -- the
"dead bar", which the reader alone takes for health (see `detect_survival`).
Both are recognised. The bar disappearing entirely (menu, hero select, round
change, killcam) is treated the same, on purpose: for the rules both cases mean the same thing --
the player's run of action was interrupted, so a streak does not count as a solo
wipe and an escape does not count as survival. That is why the event does not
promise to be "death" in the strict sense, and `meta` says what triggered it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import Pulse, find_pulses, iter_frames

log = logging.getLogger(__name__)

#: the bar's horizontal profile is resampled to this size before analysis, so
#: the thresholds hold at any recording resolution
PROFILE_SAMPLES = 256


class BarReading(NamedTuple):
    #: filled fraction of the bar; None when there is no bar on screen
    fraction: float | None
    #: how strong the bar's strongest step is, in grey levels
    strength: float
    #: how regular the alternation over the "filled" part is, 0 to 1
    regularity: float


def read_health(
    bgr: np.ndarray, *, energy_floor: float, tick_threshold: float
) -> BarReading:
    """Filled fraction of the health bar, plus what tells a lit bar from a
    dead one.

    It measures the bright/dark alternation of the ticks along the strip: where
    the bar is filled the horizontal profile oscillates, and where it is empty
    the profile is flat. That way a bright background behind the HUD does not
    become "full health".

    The fraction alone cannot tell a lit bar from a dead one: it is measured
    against the strongest step in the strip, so whatever is there -- bright
    ticks or the scenery seen through a dead bar -- becomes the scale. Hence
    the other two: lit ticks are strong steps, and they repeat at a fixed pitch
    (one tick per 25 health), which scenery does not. `detect_survival` weighs
    them against the rest of the recording.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=0)
    if profile.size < 16:
        return BarReading(None, 0.0, 0.0)

    # The windows below are counted in samples, so they would depend on the
    # strip's width -- and the strip comes out at the video's native width,
    # which runs from ~100px at 360p to ~300px at 1080p. Resampling the profile
    # to a fixed size makes the reading identical at any resolution, without
    # having to upscale the video (which would only fatten the crop without
    # adding information).
    profile = np.interp(
        np.linspace(0, profile.size - 1, PROFILE_SAMPLES),
        np.arange(profile.size),
        profile,
    )

    gradient = np.abs(np.diff(profile))
    energy = np.convolve(gradient, np.ones(5) / 5, mode="same")
    strength = float(energy.max())
    if strength < energy_floor:
        return BarReading(None, strength, 0.0)  # no bar on screen

    normalized = energy / strength
    hot = (normalized > tick_threshold).astype(np.float32)

    # Finding the rightmost column with a high gradient is not enough: the
    # *edge* of the empty track is also a strong step, and an empty bar would be
    # read as full. What characterises the filled part is several ticks in a row
    # -- that is, a density of alternation over a neighbourhood, not an isolated
    # step.
    window = max(5, normalized.size // 10)
    density = np.convolve(hot, np.ones(window) / window, mode="same")
    filled = np.flatnonzero(density > 0.25)
    if filled.size == 0:
        return BarReading(0.0, strength, 0.0)  # bar on screen, but empty
    end = int(filled.max() + 1)
    return BarReading(end / float(normalized.size), strength, _regularity(profile[:end]))


def _regularity(profile: np.ndarray) -> float:
    """Best autocorrelation of the profile's detail over the tick pitches a
    bar can have (from ~6 ticks across the bar to ~30), 0 to 1.

    The ticks repeat at a fixed pitch, so the profile matches itself shifted by
    one pitch: lit bars measured 0.75-0.92. Scenery behind a dead bar has no
    pitch, and measured 0.1-0.36. A bar with only one or two ticks left is too
    short to measure, and reads low too -- which is why this only decides
    together with the strength.
    """
    if profile.size < 24:
        return 0.0
    detail = profile - np.convolve(profile, np.ones(9) / 9, mode="same")
    detail = detail[4:-4]
    best = 0.0
    for lag in range(4, min(41, detail.size // 2 + 1)):
        a, b = detail[:-lag], detail[lag:]
        norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if norm > 0:
            best = max(best, float((a * b).sum()) / norm)
    return best


def read_health_fraction(
    bgr: np.ndarray, *, energy_floor: float, tick_threshold: float
) -> float | None:
    """Filled fraction of the health bar, or None if the bar is off screen."""
    return read_health(
        bgr, energy_floor=energy_floor, tick_threshold=tick_threshold
    ).fraction


def _median3(series: list[float | None]) -> list[float | None]:
    """Rolling median of 3, treating None (bar absent) as its own category."""
    out: list[float | None] = []
    for i in range(len(series)):
        window = series[max(0, i - 1) : i + 2]
        nones = sum(1 for x in window if x is None)
        if nones > len(window) // 2:
            out.append(None)
            continue
        vals = sorted(x for x in window if x is not None)
        out.append(vals[len(vals) // 2])
    return out


def detect_survival(health_video: Path, profile: Profile) -> list[DetectionEvent]:
    cfg = profile.section("survival")
    death_cfg = profile.section("death")
    roi = profile.roi("health")

    energy_floor = float(cfg.get("bar_energy_floor", 2.0))
    tick_threshold = float(cfg.get("tick_threshold", 0.25))
    low_frac = float(cfg.get("low_hp_frac", 0.30))

    times: list[float] = []
    health: list[float | None] = []
    readings: list[BarReading] = []
    for frame in iter_frames(health_video, fps_hint=roi.fps):
        times.append(frame.t)
        r = read_health(
            frame.bgr, energy_floor=energy_floor, tick_threshold=tick_threshold
        )
        readings.append(r)
        health.append(r.fraction)

    if not times:
        return []
    if all(h is None for h in health):
        log.warning(
            "the health bar was never found -- check the profile's 'health' ROI "
            "with tools/calibrate.py; without it there are no survival events"
        )
        return []

    death_frac = float(death_cfg.get("dead_hp_frac", 0.06))
    events: list[DetectionEvent] = []

    # -- the dead bar --------------------------------------------------------
    # After dying in OW2 the HUD stays on the *player's* card, health at 0,
    # until the respawn: the bar becomes a dark translucent track with no
    # ticks, and the scenery shows through it. The reader measures the
    # alternation against the strongest step in the strip, so with no ticks
    # the scenery became the scale and the dead bar was read as half or nearly
    # full -- which turned deaths into recoveries, and the low-health stretch
    # before them into escapes (d1).
    #
    # Two things give it away, and it takes both. The strength: lit ticks are
    # bright on dark, the scenery through the dead track is dimmed -- on two
    # real 1080p matches (PC and PS5) the lit bar ran at 23-34 and the dead one
    # at 3-11. And the pitch: the scoreboard (Tab) dims a lit bar just as much,
    # but its ticks are still there, still regular (0.8), and the dead bar's
    # scenery is not (under 0.4).
    #
    # The strength's scale changes with resolution and compression, so the
    # reference is the recording's own: the bar is lit most of the time, and an
    # upper percentile of the strength is what a lit bar looks like here.
    present = [r.strength for r in readings if r.fraction is not None]
    reference = float(np.percentile(present, 75)) if present else 0.0
    dim_below = reference * float(death_cfg.get("dim_bar_ratio", 0.45))
    irregular = float(death_cfg.get("dead_bar_regularity", 0.5))
    dim = [
        1.0
        if r.fraction is not None and r.strength < dim_below
        and r.regularity < irregular
        else 0.0
        for r in readings
    ]

    # The two readings use different series on purpose. Death is a *transient*
    # event -- sometimes a single frame with zeroed health -- so it has to come
    # out of the raw series; smoothing here would erase exactly what we want to
    # see. Low health is the opposite: it lasts seconds, and a median of 3
    # removes reading noise without shortening any episode. For low health, a
    # dead bar is no health at all.
    health = [0.0 if d else h for h, d in zip(health, dim)]
    smooth = _median3(health)

    # -- interruptions -------------------------------------------------------
    # Health dropping to zero is the signature of death; the bar disappearing
    # entirely (menu, round change, killcam) counts the same, because it means
    # the same thing for the rules. A bar read at zero for a single frame and
    # then full is death too: depending on the recording, the HUD can move to
    # the teammate being spectated, with *their* health on screen.
    #
    # A dead bar has to last before it is a death: a lone dim, irregular frame
    # can be a lit bar under a flash or a compression smear, and a false death
    # throws away the escape and the streak around it. A real dead bar stays
    # until the respawn or the killcam: 4 s or more on the real matches.
    zero_or_absent = [
        1.0 if (r.fraction is None or r.fraction <= death_frac) else 0.0
        for r in readings
    ]
    pulses = [
        (p, "zero_health_or_hud_absent")
        for p in find_pulses(
            times, zero_or_absent, rise=0.5, fall=0.5,
            min_duration=float(death_cfg.get("min_duration_s", 0.1)),
        )
    ] + [
        (p, "dead_bar")
        # averaged over ~5 frames: through a dead bar, a frame of bright
        # scenery can read as lit, and it must not cut the stretch in two
        for p in find_pulses(
            times, np.convolve(dim, np.ones(5) / 5, mode="same").tolist(),
            rise=0.6, fall=0.2,
            min_duration=float(death_cfg.get("dim_min_duration_s", 2.0)),
        )
    ]
    # One death per stretch: the two readings overlap on the same death, and a
    # dead bar lasts seconds, so what starts before the last one has ended
    # (plus the gap) is the same death.
    pulses.sort(key=lambda x: x[0].start)
    gap = float(death_cfg.get("min_gap_s", 3.0))
    merged: list[tuple[Pulse, str]] = []
    for p, why in pulses:
        if merged and p.start - merged[-1][0].end < gap:
            q, first = merged[-1]
            merged[-1] = (Pulse(start=q.start, end=max(q.end, p.end), peak=1.0), first)
        else:
            merged.append((p, why))
    interruptions = [p.start for p, _ in merged]
    for p, why in merged:
        events.append(
            DetectionEvent(
                kind=EventKind.DEATH,
                t=round(p.start, 3),
                confidence=0.7,
                meta={"reason": why, "duration_s": round(p.duration, 2)},
            )
        )

    # -- low health: only where the bar exists and there is health left -------
    danger = [
        0.0
        if (h is None or h <= death_frac or h >= low_frac)
        else (low_frac - h) / max(1e-6, low_frac)
        for h in smooth
    ]
    low_pulses = find_pulses(
        times,
        danger,
        rise=0.02,
        fall=0.005,
        min_duration=float(cfg.get("min_duration_s", 1.0)),
        min_gap=float(cfg.get("min_gap_s", 3.0)),
    )

    safe_after = float(cfg.get("safe_after_s", 4.0))
    for p in low_pulses:
        lowest = low_frac * (1.0 - p.peak)
        meta = {
            "hp_min": round(max(0.0, lowest), 3),
            "duration_s": round(p.duration, 2),
        }
        events.append(
            DetectionEvent(
                kind=EventKind.LOW_HP, t=round(p.start, 3), confidence=0.85, meta=meta
            )
        )
        survived = not any(p.start <= d <= p.end + safe_after for d in interruptions)
        if survived:
            events.append(
                DetectionEvent(
                    kind=EventKind.ESCAPE,
                    t=round(p.end, 3),
                    confidence=0.8,
                    meta={**meta, "low_hp_at": round(p.start, 3)},
                )
            )

    events.sort(key=lambda e: e.t)
    log.info(
        "%d interruption(s), %d low-health episode(s), %d escape(s)",
        len(interruptions),
        len(low_pulses),
        sum(1 for e in events if e.kind == EventKind.ESCAPE),
    )
    return events
