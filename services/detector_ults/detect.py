"""Ultimate detection.

Three routes, and they do not answer the same question. `meta["side"]` says
whose ultimate it was: `"self"` for the player's, `"enemy"` for the other two.

1. **Footer button** (`side="self"`) -- the *player's* ultimate. The button has
   two states that look nothing alike, and using the ultimate wipes the charged
   one: the event is the falling edge. It is the only route that works without
   the user preparing anything, and the only one that says **which** ultimate it
   was, by comparing the disc's icon against `templates/abilities/`.
2. **Killfeed** (`side="enemy"`) -- template matching against the ultimate icons
   the user puts in `templates/ults/`. With no templates the detector invents
   nothing: it warns in the log and returns zero events by this route.
3. **Audio** (`side="enemy"`) -- every enemy ultimate comes with a loud, close
   voice line. **Off by default**: on a synthetic video, where the voice line is
   the only loud sound, it is right; on a real match, gunfire, explosions and
   abilities produce identical peaks. Over 19 minutes of real gameplay it
   yielded 88 "ultimates" -- and without ground truth for which ultimates
   actually happened, there is no way to calibrate an honest threshold. Turn it
   on with `ults.audio_enabled` if you want to experiment on your own recording.

The detector emits only `ULT_USED`. Deciding that an ultimate was *negated*
requires crossing it with the kills, which belong to another microservice --
that correlation happens where all the events are visible at once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from owcore.audio import read_wav
from owcore.models import DetectionEvent, EventKind
from owcore.profiles import Profile
from owcore.vision import (
    IconBank,
    Pulse,
    TemplateBank,
    find_icon,
    find_pulses,
    glyph_in_disc,
    hsv_ratio,
    iter_frames,
)

log = logging.getLogger(__name__)


# ------------------------------- killfeed -----------------------------------


def _detect_from_killfeed(
    killfeed_video: Path, profile: Profile, templates_dir: Path
) -> list[DetectionEvent]:
    cfg = profile.section("ults")
    bank = TemplateBank.from_dir(templates_dir)
    if not bank:
        log.warning(
            "nenhum template em %s -- a via de killfeed fica desligada. "
            "Recorte os icones de ultimate do seu proprio gameplay e salve ali.",
            templates_dir,
        )
        return []
    log.info("%d template(s) de ultimate carregado(s)", len(bank))

    roi = profile.roi("killfeed")
    times: list[float] = []
    scores: list[float] = []
    names: list[str] = []
    for frame in iter_frames(killfeed_video, fps_hint=roi.fps):
        name, score = bank.best_match(frame.bgr)
        times.append(frame.t)
        scores.append(score)
        names.append(name or "")

    threshold = float(cfg.get("template_threshold", 0.68))
    pulses = find_pulses(
        times,
        scores,
        rise=threshold,
        fall=threshold * 0.85,
        min_duration=0.0,
        min_gap=float(cfg.get("min_gap_s", 3.0)),
    )

    events: list[DetectionEvent] = []
    for p in pulses:
        window = [(t, s, n) for t, s, n in zip(times, scores, names) if p.start <= t <= p.end]
        best = max(window, key=lambda x: x[1]) if window else (p.start, p.peak, "")
        events.append(
            DetectionEvent(
                kind=EventKind.ULT_USED,
                t=round(p.start, 3),
                confidence=round(min(1.0, float(best[1])), 3),
                meta={"side": "enemy", "source": "killfeed", "ult": best[2],
                      "score": round(float(best[1]), 3)},
            )
        )
    return events


# --------------------------------- audio ------------------------------------


def _detect_from_audio(audio_path: Path, profile: Profile) -> list[DetectionEvent]:
    """A volume spike measured as *local contrast*, in dB.

    An absolute threshold does not serve: the mix varies from recording to
    recording. What characterises an ultimate voice line is being far louder
    than the seconds around it -- so each window is compared against the rolling
    median of its neighbourhood. The resulting number is in dB, which is
    interpretable and does not depend on scale.
    """
    cfg = profile.section("ults")
    # the shared reader works in blocks: the single copy left is the signal
    # itself, no longer three of the whole file at once
    data, sr = read_wav(audio_path)
    if data.size == 0:
        return []

    fps = 20  # janelas de 50 ms
    hop = max(1, sr // fps)
    n_frames = data.size // hop
    if n_frames < 4 * fps:
        return []

    frames = data[: n_frames * hop].reshape(n_frames, hop)
    db = 20.0 * np.log10(np.maximum(np.sqrt((frames**2).mean(axis=1)), 1e-6))

    window = max(3, int(float(cfg.get("audio_context_s", 5.0)) * fps)) | 1
    pad = np.pad(db, (window // 2, window // 2), mode="edge")
    baseline = np.median(sliding_window_view(pad, window), axis=-1)[:n_frames]
    excess = db - baseline
    times = np.arange(n_frames) / fps

    threshold = float(cfg.get("audio_spike_db", 8.0))
    pulses = find_pulses(
        [float(t) for t in times],
        [float(v) for v in excess],
        rise=threshold,
        fall=threshold * 0.5,
        min_duration=0.0,
        min_gap=float(cfg.get("audio_min_gap_s", 8.0)),
    )
    return [
        DetectionEvent(
            kind=EventKind.ULT_USED,
            t=round(p.start, 3),
            confidence=0.55,
            meta={"side": "enemy", "source": "audio",
                  "excess_db": round(float(p.peak), 2)},
        )
        for p in pulses
    ]


# --------------------------------- fusao ------------------------------------


def detect_ults(
    killfeed_video: Path | None,
    audio_path: Path | None,
    profile: Profile,
    templates_dir: Path,
) -> list[DetectionEvent]:
    cfg = profile.section("ults")
    events: list[DetectionEvent] = []

    if killfeed_video and Path(killfeed_video).exists():
        events += _detect_from_killfeed(Path(killfeed_video), profile, templates_dir)
    if not bool(cfg.get("audio_enabled", False)):
        if audio_path:
            log.info(
                "via de audio desligada (ults.audio_enabled=false): em partida "
                "real ela confunde tiro e explosao com fala de ultimate"
            )
    elif audio_path and Path(audio_path).exists():
        try:
            events += _detect_from_audio(Path(audio_path), profile)
        except Exception as exc:  # audio ruim nao pode derrubar o detector
            log.warning("via de audio indisponivel: %s", exc)

    events.sort(key=lambda e: e.t)
    merged = _merge_sources(events, float(cfg.get("min_gap_s", 3.0)))
    log.info(
        "%d ultimate(s) inimiga(s) -- %d apos fundir as duas pistas",
        len(events),
        len(merged),
    )
    return merged


def _merge_sources(
    events: list[DetectionEvent], window: float
) -> list[DetectionEvent]:
    """Two routes pointing at the same instant become one event, with more
    confidence than either would have alone."""
    merged: list[DetectionEvent] = []
    for e in events:
        if merged and e.t - merged[-1].t <= window:
            prev = merged[-1]
            sources = set(str(prev.meta.get("source", "")).split("+")) | {
                str(e.meta.get("source", ""))
            }
            prev.meta["source"] = "+".join(sorted(s for s in sources if s))
            prev.confidence = round(
                min(1.0, 1.0 - (1.0 - prev.confidence) * (1.0 - e.confidence)), 3
            )
            continue
        merged.append(e.model_copy(deep=True))
    return merged


# ---------------------- the player's own ultimate ---------------------------
#
# This is the route that depends on the user preparing nothing.
#
# The footer's ultimate button has two states, and they look nothing alike:
# charged, it is a WHITE disc with the hero's icon in black, ringed by a lit
# CYAN circle; discharged, it is a dark ring with the percentage inside. Using
# the ultimate wipes both at once -- so the event is not a peak, it is the
# **falling edge** of the charged stretch.
#
# The two conditions hold together, never alone. D.Va's explosion fills the
# region with white and no cyan at all; the sky on some maps does the opposite.
#
# Which ultimate it was comes from the disc itself: the black drawing inside is
# the ability's official icon, and comparing it with `templates/abilities/`
# gives hero and name. Without those files the event still comes out -- just
# without a label.


def _charged_score(bgr, cfg: dict) -> tuple[float, float]:
    """(white disc fraction, cyan fraction) of the frame."""
    blob = find_icon(
        bgr,
        cfg.get("hsv_white", []),
        min_area_frac=float(cfg.get("min_disc_frac", 0.08)) * 0.5,
        max_offset=float(cfg.get("max_offset", 0.30)),
        aspect_range=tuple(cfg.get("aspect_range", [0.70, 1.45])),  # type: ignore[arg-type]
    )
    return (blob.area_frac if blob else 0.0), hsv_ratio(bgr, cfg.get("hsv_cyan", []))


def _merge_flicker(pulses: list, max_gap: float) -> list:
    """Stitches together stretches separated by a HUD flicker.

    A frame with no button in the middle of a charged ultimate is not an
    ultimate being used -- it is the HUD disappearing. `find_pulses` does not
    serve here: its `min_gap` compares *starts*, and two long stretches
    separated by one frame have very distant starts.
    """
    if not pulses:
        return []
    out = [pulses[0]]
    for p in pulses[1:]:
        if p.start - out[-1].end <= max_gap:
            out[-1] = Pulse(start=out[-1].start, end=p.end,
                            peak=max(out[-1].peak, p.peak))
        else:
            out.append(p)
    return out


def detect_self_ults(
    roi_video: Path, profile: Profile, icons_dir: Path
) -> list[DetectionEvent]:
    cfg = profile.section("ults").get("self", {})
    roi = profile.roi("ult")
    min_disc = float(cfg.get("min_disc_frac", 0.08))
    min_cyan = float(cfg.get("min_cyan_frac", 0.04))

    bank = IconBank.from_dir(icons_dir)
    if bank:
        log.info("%d icone(s) de habilidade carregado(s)", len(bank))
    else:
        log.warning(
            "sem icones em %s -- as ultimates continuam sendo detectadas, mas "
            "sem dizer de qual heroi. Rode tools/fetch_ability_icons.py",
            icons_dir,
        )

    times: list[float] = []
    scores: list[float] = []
    #: (instant, disc size, glyph) -- one per charged *run*, and not per
    #: frame. It is where you can read which ultimate it is.
    #:
    #: Only the best of each run is kept. Keeping them all really was wasted
    #: memory: an ultimate stays charged for seconds or minutes before being
    #: used, so "only the charged frames" could be half the match -- thousands
    #: of matrices, to end up using **one** per stretch. Since a run of
    #: `score >= 1.0` always fits inside a stretch (the stretch opens when the
    #: score crosses 1.0 upwards, and `_merge_flicker` only extends them), the
    #: largest disc in the stretch is the largest among the runs' bests.
    glyphs: list[tuple[float, float, "np.ndarray"]] = []
    best_of_run: tuple[float, float, "np.ndarray"] | None = None

    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        disc, cyan = _charged_score(frame.bgr, cfg)
        score = min(disc / min_disc, cyan / min_cyan) if min_disc and min_cyan else 0.0
        times.append(frame.t)
        scores.append(min(2.0, score))
        if score >= 1.0 and bank:
            glyph = glyph_in_disc(frame.bgr, min_disc_frac=min_disc)
            if glyph is not None and (best_of_run is None or disc > best_of_run[1]):
                best_of_run = (frame.t, disc, glyph)
        elif best_of_run is not None:
            # the button went dark: close the run, keeping only its best frame
            glyphs.append(best_of_run)
            best_of_run = None
    if best_of_run is not None:
        glyphs.append(best_of_run)

    if not times:
        return []

    pulses = _merge_flicker(
        find_pulses(
            times, scores,
            rise=1.0, fall=0.7,
            # No minimum duration HERE, on purpose: the stretch is still to be
            # stitched by `_merge_flicker`, and cutting by duration before that
            # would measure a piece of a stretch instead of the stretch. The cut
            # comes later, with `min_charged_s`.
            min_duration=0.0,
            min_gap=0.0,
        ),
        float(cfg.get("flicker_s", 2.0)),
    )

    min_after = float(cfg.get("min_after_s", 6.0))
    min_charged = float(cfg.get("min_charged_s", 2.0))
    limiar_icone = float(cfg.get("icon_threshold", 0.55))

    events: list[DetectionEvent] = []
    for i, p in enumerate(pulses):
        if p.duration < min_charged:
            # Bright, round and centred in this window is not only the button:
            # the killcam draws a disc with the killer's face, and the glare of
            # an explosion or of the sky passes too. What separates them is the
            # clock -- those last a handful of frames, while an ultimate stays
            # charged for seconds before being used.
            continue
        if p.end >= times[-1]:
            # The stretch runs to the last frame, and there the two cases
            # become identical: `find_pulses` closes at `times[-1]` both when
            # the button goes dark on the last frame and when it stays lit and
            # the video ends. With no way to tell them apart, no event is
            # invented.
            continue
        proxima = pulses[i + 1].start if i + 1 < len(pulses) else float("inf")
        if proxima - p.end < min_after:
            # it recharged far too quickly to have been spent: that was the
            # HUD disappearing (killcam, scoreboard) and not an ultimate
            continue

        meta: dict = {"side": "self", "charged_s": round(p.duration, 2)}
        confidence = 0.9
        if bank:
            janela = [g for g in glyphs if p.start <= g[0] <= p.end]
            if janela:
                # the frame with the largest disc: the best-framed of the stretch
                _t, _disc, glyph = max(janela, key=lambda g: g[1])
                key, score = bank.best_match(glyph)
                meta["icon_score"] = round(score, 3)
                if key and score >= limiar_icone:
                    hero, _, ability = key.partition("/")
                    meta["hero"], meta["ability"] = hero, ability
                    confidence = round(min(1.0, 0.8 + 0.2 * score), 3)
        events.append(
            DetectionEvent(
                kind=EventKind.ULT_USED,
                t=round(p.end, 3),
                confidence=confidence,
                meta=meta,
            )
        )
    log.info("%d ultimate(s) do jogador", len(events))
    return events
