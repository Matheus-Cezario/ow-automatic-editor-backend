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

**About `DEATH`.** When you die in OW2 you start spectating a teammate, and
*their* health appears on the HUD -- which is why the signature of death is
health going to zero and coming back full on the next frame, rather than the bar
staying at zero. The bar disappearing entirely (menu, hero select, round change)
is treated the same, on purpose: for the rules both cases mean the same thing --
the player's run of action was interrupted, so a streak does not count as a solo
wipe and an escape does not count as survival. That is why the event does not
promise to be "death" in the strict sense, and `meta` says what triggered it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import find_pulses, iter_frames

log = logging.getLogger(__name__)

#: the bar's horizontal profile is resampled to this size before analysis, so
#: the thresholds hold at any recording resolution
PROFILE_SAMPLES = 256


def read_health_fraction(
    bgr: np.ndarray, *, energy_floor: float, tick_threshold: float
) -> float | None:
    """Filled fraction of the health bar, or None if the bar is off screen.

    It measures the bright/dark alternation of the ticks along the strip: where
    the bar is filled the horizontal profile oscillates, and where it is empty
    the profile is flat. That way a bright background behind the HUD does not
    become "full health".
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=0)
    if profile.size < 16:
        return None

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
    if float(energy.max()) < energy_floor:
        return None  # no bar on screen

    normalized = energy / energy.max()
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
        return 0.0  # bar on screen, but empty
    return float(filled.max() + 1) / float(normalized.size)


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
    for frame in iter_frames(health_video, fps_hint=roi.fps):
        times.append(frame.t)
        health.append(
            read_health_fraction(
                frame.bgr, energy_floor=energy_floor, tick_threshold=tick_threshold
            )
        )

    if not times:
        return []
    if all(h is None for h in health):
        log.warning(
            "a barra de vida nunca foi encontrada -- confira a ROI 'health' do "
            "profile com tools/calibrate.py; sem ela nao ha eventos de sobrevivencia"
        )
        return []

    death_frac = float(death_cfg.get("dead_hp_frac", 0.06))
    events: list[DetectionEvent] = []

    # The two readings use different series on purpose. Death is a *transient*
    # event -- sometimes a single frame with zeroed health -- so it has to come
    # out of the raw series; smoothing here would erase exactly what we want to
    # see. Low health is the opposite: it lasts seconds, and a median of 3
    # removes reading noise without shortening any episode.
    smooth = _median3(health)

    # -- interruptions -------------------------------------------------------
    # Health dropping to zero and returning to the top on the next frame is the
    # signature of death: when you die you start spectating a teammate, with
    # *their* health on screen. No heal climbs like that. The bar disappearing
    # entirely (menu, round change) counts the same, because it means the same
    # thing for the rules.
    down = [1.0 if (h is None or h <= death_frac) else 0.0 for h in health]
    absent_pulses = find_pulses(
        times,
        down,
        rise=0.5,
        fall=0.5,
        min_duration=float(death_cfg.get("min_duration_s", 0.1)),
        min_gap=float(death_cfg.get("min_gap_s", 3.0)),
    )
    interruptions = [p.start for p in absent_pulses]
    for p in absent_pulses:
        events.append(
            DetectionEvent(
                kind=EventKind.DEATH,
                t=round(p.start, 3),
                confidence=0.7,
                meta={"reason": "zero_health_or_hud_absent",
                      "duration_s": round(p.duration, 2)},
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
        "%d interrupcao(oes), %d episodio(s) de vida baixa, %d fuga(s)",
        len(interruptions),
        len(low_pulses),
        sum(1 for e in events if e.kind == EventKind.ESCAPE),
    )
    return events
