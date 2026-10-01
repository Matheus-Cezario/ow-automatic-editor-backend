"""Detector accuracy against the synthetic video's ground truth.

It is the test that backs the project's thesis: cropping a tiny band of the
screen, at low FPS, is enough to recover the match's events.
"""

from __future__ import annotations

import json

import pytest

from conftest import (
    ABILITY_ICONS,
    MUSIC,
    SAMPLE,
    TRUTH,
    ULT_TEMPLATES,
    needs_sample,
    service_module,
)
from owcore.ffmpeg import extract_audio, extract_rois, probe
from owcore.models import EventKind
from owcore.profiles import load_profile

TOLERANCE_S = 0.35

pytestmark = needs_sample


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads(TRUTH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rois(tmp_path_factory):
    prof = load_profile("ow2_default")
    out = tmp_path_factory.mktemp("rois")
    crops = extract_rois(
        SAMPLE,
        prof.rois(["kills", "health", "killfeed", "banner", "ult", "player"]),
        out,
    )
    crops["audio"] = extract_audio(SAMPLE, out / "audio.wav")
    return crops


def match(detected: list[float], expected: list[float]) -> bool:
    if len(detected) != len(expected):
        return False
    return all(
        abs(d - e) <= TOLERANCE_S
        for d, e in zip(sorted(detected), sorted(expected))
    )


def test_the_crop_is_much_smaller_than_the_original(rois):
    """The point of the architecture: the detector receives a fraction of the
    bytes."""
    assert rois["kills"].stat().st_size < SAMPLE.stat().st_size / 50


def test_kills_match_the_ground_truth(rois, truth):
    detect = service_module("detector_kills")
    ev = detect.detect_kills(rois["kills"], load_profile("ow2_default"))
    assert [e.kind for e in ev] == [EventKind.KILL] * len(ev)
    assert match([e.t for e in ev], truth["kills"])


def test_survival_matches_the_ground_truth(rois, truth):
    detect = service_module("detector_survival")
    ev = detect.detect_survival(rois["health"], load_profile("ow2_default"))

    low = [e.t for e in ev if e.kind == EventKind.LOW_HP]
    deaths = [e.t for e in ev if e.kind == EventKind.DEATH]
    escapes = [e.t for e in ev if e.kind == EventKind.ESCAPE]

    assert match(low, truth["low_hp"])
    assert match(deaths, truth["deaths"])
    # no ground-truth episode ends in death, so they are all escapes
    assert len(escapes) == len(truth["low_hp"])


def test_the_damage_vignette_alone_is_not_an_event(rois, truth):
    """Regression of the bug that prompted the rewrite: the red vignette on the
    edges is *damage taken*, not low health. It covers the edges at the same
    instants as the ground truth, but the health bar is what decides."""
    detect = service_module("detector_survival")
    ev = detect.detect_survival(rois["health"], load_profile("ow2_default"))
    low = [e.t for e in ev if e.kind == EventKind.LOW_HP]
    assert len(low) == len(truth["low_hp"]), (
        "more episodes than the ground truth means the vignette is leaking in"
    )


def test_reading_the_health_bar(rois):
    """The bar is read by the alternation of its ticks, so a light background
    behind the HUD must not turn into 'full health'."""
    import numpy as np

    from owcore.vision import iter_frames

    detect = service_module("detector_survival")
    prof = load_profile("ow2_default")
    readings = [
        detect.read_health_fraction(f.bgr, energy_floor=2.0, tick_threshold=0.25)
        for f in iter_frames(rois["health"], prof.roi("health").fps)
    ]
    full = [v for v in readings if v is not None and v > 0.7]
    low = [v for v in readings if v is not None and 0.05 < v < 0.4]
    assert full, "never read the bar full"
    assert low, "never read the bar low"
    assert float(np.median(full)) > 0.85


def test_without_templates_or_audio_the_detector_does_not_invent(rois, tmp_path):
    """Explicit contract: without the game icons and with the audio path off
    (the default), the detector returns nothing -- instead of faking detection."""
    detect = service_module("detector_ults")
    ev = detect.detect_ults(
        rois["killfeed"], rois["audio"], load_profile("ow2_default"), tmp_path / "empty"
    )
    assert ev == []


def test_audio_path_when_explicitly_enabled(rois, truth, tmp_path):
    """When on, it finds the synthetic video's ultimates -- where the voice line
    is the only loud sound. It ships off because in a real match it does not
    hold."""
    detect = service_module("detector_ults")
    profile = load_profile("ow2_default")
    profile.data["ults"] = {**profile.data["ults"], "audio_enabled": True,
                            "audio_spike_db": 8.0}
    try:
        ev = detect.detect_ults(
            rois["killfeed"], rois["audio"], profile, tmp_path / "empty"
        )
        assert match([e.t for e in ev], truth["ults"])
        assert all(e.meta["source"] == "audio" for e in ev)
    finally:
        load_profile.cache_clear()


@pytest.mark.skipif(not ULT_TEMPLATES.exists(), reason="no sample templates")
def test_ult_templates_find_the_ultimates(rois, truth):
    """With the icons the killfeed counts, and it does not depend on audio."""
    detect = service_module("detector_ults")
    ev = detect.detect_ults(
        rois["killfeed"], rois["audio"], load_profile("ow2_default"), ULT_TEMPLATES
    )
    assert match([e.t for e in ev], truth["ults"])
    assert all(e.meta["source"] == "killfeed" for e in ev)


@pytest.mark.skipif(not MUSIC.exists(), reason="no sample music")
def test_bpm_of_the_test_music(tmp_path):
    detect = service_module("beats")
    grid = detect.analyze_track(MUSIC, tmp_path).grid
    assert 110 <= grid.bpm <= 130  # the track was generated at 120 BPM
    assert len(grid.beats) > 50


@pytest.mark.skipif(not MUSIC.exists(), reason="no sample music")
def test_own_estimator_finds_the_bpm_without_librosa(tmp_path):
    """Fallback path: it must keep cutting on the beat without librosa."""
    from owcore.audio import read_wav

    detect = service_module("beats")
    wav = detect._decode_to_wav(MUSIC, tmp_path / "m.wav")
    # reading the WAV lives in `owcore.audio` since the match got a waveform
    # too: two tracks are drawn, and the computation is one
    data, sr = read_wav(wav)
    grid = detect._estimate_beats(data, sr, data.size / sr)
    assert 110 <= grid.bpm <= 130


def test_probe_reads_the_video(truth):
    info = probe(SAMPLE)
    assert info.duration_s == pytest.approx(truth["duration_s"], abs=0.5)
    assert [info.width, info.height] == truth["size"]
    assert info.has_audio


# ── abilities announced in the footer ──────────────────────────────────────


def banners(rois):
    from conftest import ROOT

    detect = service_module("detector_banner")
    return detect.detect_abilities(
        rois["banner"], load_profile("ow2_default"), ROOT / "config" / "shapes"
    )


def test_anas_darts_match_the_ground_truth(rois, truth):
    ev = [e for e in banners(rois) if e.kind == EventKind.SLEEP]
    assert match([e.t for e in ev], truth["sleeps"])


def test_sigmas_rocks_match_the_ground_truth(rois, truth):
    """The rock's banner is drawn in **green**, and the dart's in cyan: the
    banner colour changes from recording to recording, and the same detector
    has to find both."""
    ev = [e for e in banners(rois) if e.kind == EventKind.STUN]
    assert match([e.t for e in ev], truth["stuns"])


def test_each_ability_keeps_its_own_event(rois, truth):
    """The risk of having two templates is one firing on the other's banner:
    the same banner would become two events."""
    ev = banners(rois)
    assert {e.kind for e in ev} == {EventKind.SLEEP, EventKind.STUN}
    for e in ev:
        expected = truth["sleeps"] if e.kind == EventKind.SLEEP else truth["stuns"]
        other = truth["stuns"] if e.kind == EventKind.SLEEP else truth["sleeps"]
        assert any(abs(e.t - x) <= TOLERANCE_S for x in expected)
        assert not any(abs(e.t - x) <= TOLERANCE_S for x in other), (
            f"{e.kind} at {e.t}s landed on the other ability"
        )


def test_banners_from_the_same_footer_do_not_become_abilities(rois, truth):
    """The footer shows several banners with the same colour, shape and
    position -- only the icon tells them apart. The synthetic video draws
    decoys precisely to prove the banner alone is not enough."""
    detected = [e.t for e in banners(rois)]
    for decoy in truth["decoy_banners"]:
        assert not any(abs(t - decoy) <= TOLERANCE_S for t in detected), (
            f"the decoy banner at {decoy}s became an ability"
        )


def test_without_the_template_the_detector_does_not_invent(rois, tmp_path):
    detect = service_module("detector_banner")
    ev = detect.detect_abilities(
        rois["banner"], load_profile("ow2_default"), tmp_path / "empty"
    )
    assert ev == []


# ── critical hits, in the same crosshair crop ──────────────────────────────


def test_headshots_match_the_ground_truth(rois, truth):
    detect = service_module("detector_kills")
    ev = detect.detect_headshots(rois["kills"], load_profile("ow2_default"))
    assert [e.kind for e in ev] == [EventKind.HEADSHOT] * len(ev)
    assert match([e.t for e in ev], truth["headshots"])


def test_the_kill_skull_is_not_a_headshot(rois, truth):
    """The skull is red and is born on the same crosshair as the critical
    marker. What separates them is the shape: the X leaves the four straight
    directions clear, and the skull fills all eight. Without this second check
    every kill would also become a headshot."""
    detect = service_module("detector_kills")
    detected = [e.t for e in detect.detect_headshots(
        rois["kills"], load_profile("ow2_default")
    )]
    for k in truth["kills"]:
        if any(abs(k - h) <= 1.5 for h in truth["headshots"]):
            continue  # there is a real headshot nearby; nothing to prove here
        assert not any(abs(t - k) <= 0.5 for t in detected), (
            f"the skull at {k}s became a headshot"
        )


# ── the player's own ultimate, read on the footer button ───────────────────


def player_ults(rois, icons):
    detect = service_module("detector_ults")
    return detect.detect_self_ults(rois["ult"], load_profile("ow2_default"), icons)


def test_the_players_ultimate_matches_the_ground_truth(rois, truth):
    ev = player_ults(rois, ABILITY_ICONS)
    assert [e.kind for e in ev] == [EventKind.ULT_USED] * len(ev)
    assert match([e.t for e in ev], truth["self_ults"])
    assert all(e.meta["side"] == "self" for e in ev)


def test_the_event_is_the_instant_the_ultimate_is_USED(rois, truth):
    """The button stays charged for several seconds before -- it is the falling
    edge that marks the instant, not the presence of the white disc."""
    ev = player_ults(rois, ABILITY_ICONS)
    assert ev, "no ultimate detected"
    for e in ev:
        assert e.meta["charged_s"] > 1.0, (
            "the charged button lasted less than the drawn window: the event "
            "probably came from the wrong place in the stretch"
        )


@pytest.mark.skipif(not ABILITY_ICONS.exists(), reason="no sample icons")
def test_the_disc_icon_says_which_ultimate_it_was(rois):
    ev = player_ults(rois, ABILITY_ICONS)
    assert ev
    for e in ev:
        assert e.meta["hero"] == "sample"
        assert e.meta["ability"] == "self_ult"


def test_a_flashing_button_is_not_an_ultimate(rois, truth):
    """Not everything bright, round and centred in that window is the button:
    the kill cam draws a disc with the killer's face, and explosion flashes
    pass through there. What separates them is the clock -- they last a handful
    of frames, and an ultimate stays charged for seconds before being used."""
    ev = player_ults(rois, ABILITY_ICONS)
    for flash in truth["ult_flashes"]:
        assert not any(abs(e.t - flash) < 1.5 for e in ev)
    assert match([e.t for e in ev], truth["self_ults"])


def test_without_icons_the_ultimate_is_still_detected(rois, truth, tmp_path):
    """Contract: the icons say *which* ultimate it was, and nothing more.
    Without them the event still comes out -- just without a label."""
    ev = player_ults(rois, tmp_path / "empty")
    assert match([e.t for e in ev], truth["self_ults"])
    assert all("hero" not in e.meta for e in ev)


# ── kills with an ability, read in the killfeed ────────────────────────────


def ability_kills(rois, icons, player=...):
    detect = service_module("detector_killfeed")
    return detect.detect_ability_kills(
        rois["killfeed"],
        rois["player"] if player is ... else player,
        load_profile("ow2_default"),
        icons,
    )


def test_ability_kills_match_the_ground_truth(rois, truth):
    ev = ability_kills(rois, ABILITY_ICONS)
    assert [e.kind for e in ev] == [EventKind.ABILITY_KILL] * len(ev)
    assert match([e.t for e in ev], truth["ability_kills"])
    assert all(e.meta["ability"] == "sample/ability_kill" for e in ev)


def test_a_teammates_kill_does_not_count(rois, truth):
    """The killfeed announces all ten people in the match, not just the player.

    The plate colour does not settle it: measured on a real recording, blue is
    the killer and red the victim, on both sides. What separates them is the
    name written on the blue plate -- and in the sample video the teammate
    kills with the SAME ability, with a name of the SAME length, so it cannot
    be got right by chance.
    """
    ev = ability_kills(rois, ABILITY_ICONS)
    for t in truth["teammate_kills"]:
        assert not any(abs(e.t - t) <= TOLERANCE_S for e in ev), (
            f"the teammate's kill at {t}s counted as the player's"
        )


def test_without_the_players_card_nothing_can_be_attributed(rois):
    """Without knowing who the player is there is no kill to report: returning
    all of them would hand over other people's as if they were the player's."""
    assert ability_kills(rois, ABILITY_ICONS, player=None) == []


def test_a_killfeed_line_counts_as_ONE_kill(rois, truth):
    """The line stays on screen for seconds. Counting frames above the
    threshold would give one kill per frame; what counts is it appearing."""
    ev = ability_kills(rois, ABILITY_ICONS)
    assert len(ev) == len(truth["ability_kills"])


def test_without_icons_the_killfeed_does_not_invent(rois, tmp_path):
    """Without the bank there is no saying WHICH ability it was -- and a kill
    without that answer is already reported by the crosshair detector."""
    assert ability_kills(rois, tmp_path / "empty") == []


# ── reading the written name, without reading the letters ──────────────────


def _written(text: str, scale: float, thickness: int,
             background: tuple[int, int, int] = (190, 140, 70)) -> "np.ndarray":
    """A HUD plate with a name written on it."""
    import cv2
    import numpy as np

    (width, letter_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    height = int(letter_h * 2.6)
    img = np.full((height, width + 16, 3), background, np.uint8)
    cv2.putText(img, text, (8, (height + letter_h) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (240, 240, 240), thickness, cv2.LINE_AA)
    return img


def test_the_same_name_at_different_sizes_is_the_same_name():
    """The HUD's two writings do not have the same size or spacing: on the
    reference recording the same name comes out 50% wider in the killfeed than
    on the footer card, normalised by height. That is why the comparison is
    letter by letter, with each letter normalised on its own."""
    from owcore.nameplate import read_name, same_name

    large = read_name(_written("SHOOTER", 0.9, 2, (120, 62, 44)))
    small = read_name(_written("SHOOTER", 0.5, 2))
    assert large is not None and small is not None
    assert len(large) == len(small) == 7
    assert same_name(large, small) > 0.5


def test_a_name_of_another_length_is_refused_outright():
    from owcore.nameplate import read_name, same_name

    assert same_name(read_name(_written("SHOOTER", 0.9, 2)),
                     read_name(_written("TEAMMATE", 0.9, 2))) == 0.0


def test_a_name_of_the_same_length_is_still_another_name():
    """The length shortcut settles most cases and hides the rest: in a real
    match there were two nine-letter names. What separates them is the shape of
    each letter."""
    from owcore.nameplate import read_name, same_name

    player = read_name(_written("SHOOTER", 0.9, 2, (120, 62, 44)))
    other = read_name(_written("PATRICK", 0.5, 2))
    assert len(player) == len(other) == 7
    assert same_name(player, other) < 0.4


def test_a_plate_with_nothing_written_does_not_invent_a_name():
    import numpy as np
    from owcore.nameplate import read_name

    assert read_name(np.full((30, 160, 3), (190, 140, 70), np.uint8)) is None


#: a killfeed line, already tracked since t=10.0. The plates it came from: the
#: killer at x 100..250 and the victim at x 290..410.
def _tracked_line():
    kf = service_module("detector_killfeed")
    line = kf._Line(
        inner_left=250, inner_right=290, outer_left=100, outer_right=410, h=30,
        start=10.0, last_seen=10.0, key="a/b", score=0.8, style="ability",
    )
    return kf, line


def test_a_line_sliding_in_does_not_become_a_second_kill():
    """While the line slides in, the plates are still opening: the outer edge
    moves dozens of pixels from one frame to the next. Requiring it there
    would make the same kill come out twice."""
    kf, line = _tracked_line()
    ally = kf.Plate(x=100, y=10, w=150, h=30)
    growing = kf.Plate(x=290, y=10, w=90, h=30)  # not fully open yet
    assert line.same_as(ally, growing, 10.2, slide=0.6)
    # once the entrance is over, the same difference is no longer the same line
    assert not line.same_as(ally, growing, 12.0, slide=0.6)


def test_two_kills_by_the_same_player_are_two_lines():
    """The inner edges surround the icon and are the same in both -- that is
    why the outer ones, which are the length of the names, must count."""
    kf, line = _tracked_line()
    ally = kf.Plate(x=100, y=10, w=150, h=30)          # same killer
    other_victim = kf.Plate(x=290, y=10, w=60, h=30)   # shorter name
    assert not line.same_as(ally, other_victim, 12.0, slide=0.6)


def test_a_line_is_recognised_even_sliding_down():
    """When a new kill arrives, the whole stack moves down. The line's identity
    must not depend on its height, or it would switch lines right then."""
    kf, line = _tracked_line()
    ally = kf.Plate(x=100, y=70, w=150, h=30)
    enemy = kf.Plate(x=290, y=70, w=120, h=30)
    assert line.same_as(ally, enemy, 12.0, slide=0.6)
