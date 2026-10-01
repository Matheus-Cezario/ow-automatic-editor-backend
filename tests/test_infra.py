"""Bus, storage and vision -- the infrastructure pieces the microservices
assume work."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from owcore.audio import peaks_for, read_wav, waveform, waveform_of
from owcore.bus import LocalBus
from owcore.storage import LocalStorage
from owcore.vision import (
    GLYPH_SIDE,
    IconBank,
    TemplateBank,
    border_mask,
    find_pulses,
    glyph_in_disc,
    glyph_on_dark,
    hsv_ratio,
    normalized_glyph,
)


# ─────────────────────────────────── bus ────────────────────────────────────


def test_a_published_message_is_delivered(tmp_path):
    bus = LocalBus(tmp_path)
    bus.publish("s", {"job_id": "a"})
    got = list(bus.consume("s", "g", "c1", block_ms=0))
    assert [m.payload for m in got] == [{"job_id": "a"}]


def test_each_group_receives_the_same_message(tmp_path):
    """Fan-out: the preprocessor sends once and every detector must see it."""
    bus = LocalBus(tmp_path)
    bus.publish("s", {"n": 1})
    a = list(bus.consume("s", "group-a", "c", block_ms=0))
    b = list(bus.consume("s", "group-b", "c", block_ms=0))
    assert len(a) == len(b) == 1


def test_within_a_group_only_one_consumer_takes_it(tmp_path):
    """Competition: two replicas of the same detector do not process twice."""
    bus = LocalBus(tmp_path)
    bus.publish("s", {"n": 1})
    first = list(bus.consume("s", "g", "c1", block_ms=0))
    second = list(bus.consume("s", "g", "c2", block_ms=0))
    assert len(first) == 1
    assert second == []


def test_publication_order_is_preserved(tmp_path):
    bus = LocalBus(tmp_path)
    for i in range(5):
        bus.publish("s", {"n": i})
    seen = []
    for _ in range(5):
        seen += [m.payload["n"] for m in bus.consume("s", "g", "c", block_ms=0)]
    assert seen == [0, 1, 2, 3, 4]


def test_consume_without_a_message_returns_empty(tmp_path):
    bus = LocalBus(tmp_path)
    assert list(bus.consume("empty", "g", "c", block_ms=0)) == []


def test_a_message_consumed_by_everyone_is_swept(tmp_path):
    """The disk queue must forget what has gone through it.

    Nothing used to be deleted: each `consume` re-listed the whole
    `sorted(glob("*.json"))` every 150 ms, so the cost of an **idle** worker grew
    with all the work the system had ever done.
    """
    bus = LocalBus(tmp_path, retention_s=0.0001)
    for i in range(4):
        bus.publish("s", {"n": i})
    for _ in range(4):
        for m in bus.consume("s", "g", "c1"):
            bus.ack("s", "g", m.id)

    time.sleep(0.01)
    assert bus._sweep("s") == 4
    assert list((tmp_path / "s").glob("*.json")) == []


def test_the_sweep_respects_a_group_that_has_not_finished(tmp_path):
    """Deleting too early would cost the message of a service that is down.

    Only what **every** existing group stamped as done goes away.
    """
    bus = LocalBus(tmp_path, retention_s=0.0001)
    bus.publish("s", {"n": 1})
    for m in bus.consume("s", "a", "c1"):
        bus.ack("s", "a", m.id)
    bus._group_dir("s", "b")  # exists, but never consumed

    time.sleep(0.01)
    assert bus._sweep("s") == 0
    assert len(list((tmp_path / "s").glob("*.json"))) == 1

    # and the late group still receives the message
    assert [m.payload["n"] for m in bus.consume("s", "b", "c2")] == [1]


def test_a_delivered_but_unfinished_message_is_not_swept(tmp_path):
    """Delivered != done: a worker that died mid-handler must not see the
    message vanish from under it."""
    bus = LocalBus(tmp_path, retention_s=0.0001)
    bus.publish("s", {"n": 1})
    for _m in bus.consume("s", "g", "c1"):
        pass  # delivered, no ack -- like a process that crashed

    time.sleep(0.01)
    assert bus._sweep("s") == 0


# ──────────────────────────────── storage ───────────────────────────────────


def test_writes_and_reads_a_file(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    src = tmp_path / "x.bin"
    src.write_bytes(b"contents")
    st.put_file("a/b/x.bin", src)
    assert st.exists("a/b/x.bin")
    assert st.size("a/b/x.bin") == 8
    dest = st.get_file("a/b/x.bin", tmp_path / "out.bin")
    assert dest.read_bytes() == b"contents"


def test_reading_by_byte_range(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    src = tmp_path / "x.bin"
    src.write_bytes(bytes(range(256)))
    st.put_file("x.bin", src)
    assert st.open_range("x.bin", 10, 5) == bytes(range(10, 15))


def test_a_key_cannot_escape_the_root(tmp_path):
    st = LocalStorage(tmp_path / "blobs")
    with pytest.raises(ValueError):
        st.put_file("../outside.bin", tmp_path / "x.bin")


# ────────────────────────────────── audio ───────────────────────────────────
#
# What is covered here is the **memory ceiling**, and not just the result. The
# previous version put the whole WAV in RAM three times (raw bytes, a float32
# copy, the mono mix) to end up handing over a few thousand numbers: a 20-min
# audio cost a ~370 MB peak in the preprocessor. The result was right; the
# cost was not.


def _wav(path, *, seconds: float, sr: int = 22050, channels: int = 1):
    import wave

    n = int(seconds * sr)
    signal = (np.sin(np.arange(n * channels) / 40.0) * 20000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(signal.tobytes())
    return path


def _reference_waveform(data, n):
    """The naive implementation, to check the result did not change."""
    n = min(n, data.size)
    cut = (data.size // n) * n
    blocks = np.abs(data[:cut]).reshape(n, -1).max(axis=1)
    top = float(blocks.max())
    return [round(float(v), 3) for v in (blocks / top)]


@pytest.mark.parametrize("channels", [1, 2])
def test_the_waveform_read_in_blocks_equals_the_naive_one(tmp_path, channels):
    wav = _wav(tmp_path / "a.wav", seconds=12.0, channels=channels)
    data, sr = read_wav(wav)
    expected = _reference_waveform(data, peaks_for(data.size / sr))

    assert waveform(data, peaks_for(data.size / sr)) == expected
    wave_, duration = waveform_of(wav)
    assert wave_ == expected
    assert duration == pytest.approx(12.0, abs=0.01)


def test_the_waveform_does_not_load_the_whole_file(tmp_path):
    """`waveform_of` must cost the same with 10 s and with 10 min of audio.

    It is the preprocessor's path: it only wants the waveform for the editor
    to draw, and has no reason to hold the match audio in memory for that.
    """
    short = _wav(tmp_path / "short.wav", seconds=5.0)
    long_ = _wav(tmp_path / "long.wav", seconds=600.0)
    assert long_.stat().st_size > 20 * short.stat().st_size

    tracemalloc.start()
    try:
        waveform_of(short)
        _, short_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        waveform_of(long_)
        _, long_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # the long file is 120x the size of the short one; the peak must not
    # follow. The margin is generous on purpose: what is pinned here is the
    # order of magnitude -- constant instead of proportional to the file.
    assert long_peak < short_peak + 32 * 1024 * 1024, (
        f"peak went from {short_peak/1e6:.1f} MB to {long_peak/1e6:.1f} MB: "
        "the waveform is loading the whole file again"
    )
    # and the file size must not become the peak either
    assert long_peak < long_.stat().st_size


def test_read_wav_does_not_multiply_the_signal_in_memory(tmp_path):
    """Whoever needs the whole signal (the beat tracker) pays for **one** copy
    of it, not three."""
    wav = _wav(tmp_path / "m.wav", seconds=300.0)
    tracemalloc.start()
    try:
        data, _sr = read_wav(wav)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < data.nbytes * 1.75, (
        f"peak {peak/1e6:.1f} MB for a {data.nbytes/1e6:.1f} MB signal"
    )


def test_an_unreadable_wav_becomes_an_empty_waveform_instead_of_an_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    assert waveform_of(bad) == ([], 0.0)


# ───────────────────────────────── vision ───────────────────────────────────


def test_a_pulse_must_cross_the_rise_threshold():
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 5 + [1.0] * 5 + [0.0] * 10
    p = find_pulses(t, v, rise=0.5, fall=0.2)
    assert len(p) == 1
    assert p[0].start == pytest.approx(0.5)


def test_hysteresis_avoids_counting_a_flicker_as_two_events():
    """The icon flickers; with hysteresis that is still a single event."""
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 3 + [1.0, 0.3, 1.0, 0.3, 1.0] + [0.0] * 12
    assert len(find_pulses(t, v, rise=0.5, fall=0.2)) == 1


def test_min_gap_merges_adjacent_pulses():
    t = [i * 0.1 for i in range(30)]
    v = [0.0] * 3 + [1.0] * 2 + [0.0] * 2 + [1.0] * 2 + [0.0] * 21
    assert len(find_pulses(t, v, rise=0.5, fall=0.2)) == 2
    assert len(find_pulses(t, v, rise=0.5, fall=0.2, min_gap=1.0)) == 1


def test_a_pulse_too_short_is_dropped():
    t = [i * 0.1 for i in range(20)]
    v = [0.0] * 5 + [1.0] + [0.0] * 14
    assert find_pulses(t, v, rise=0.5, fall=0.2, min_duration=0.5) == []


def test_a_pulse_open_at_the_end_of_the_video_is_closed():
    t = [i * 0.1 for i in range(10)]
    v = [0.0] * 3 + [1.0] * 7
    p = find_pulses(t, v, rise=0.5, fall=0.2)
    assert len(p) == 1


def test_hsv_ratio_counts_only_the_requested_range():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5, :, 2] = 255  # half pure red
    ranges = [{"lo": [0, 120, 90], "hi": [10, 255, 255]}]
    assert hsv_ratio(img, ranges) == pytest.approx(0.5)
    assert hsv_ratio(img, []) == 0.0


def test_border_mask_covers_only_the_frame():
    m = border_mask((100, 100), 0.1)
    assert m[0, 0] and m[-1, -1]
    assert not m[50, 50]


def test_an_empty_template_bank_does_not_break(tmp_path):
    bank = TemplateBank.from_dir(tmp_path / "does_not_exist")
    assert not bank
    assert bank.best_match(np.zeros((10, 10, 3), np.uint8)) == (None, 0.0)


def test_a_template_finds_itself(tmp_path):
    import cv2

    img = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(img, (30, 30), 15, (255, 255, 255), -1)
    tdir = tmp_path / "t"
    tdir.mkdir()
    cv2.imwrite(str(tdir / "target.png"), img[15:45, 15:45])
    name, score = TemplateBank.from_dir(tdir).best_match(img)
    assert name == "target"
    assert score > 0.9


# ── glyphs: an icon's mark, without position, scale or polarity ────────────


def _arrow(side: int) -> np.ndarray:
    """An asymmetric mark, so rotations and mirrors do not pass for it."""
    import cv2

    m = np.zeros((side, side), np.uint8)
    pts = np.array([[side // 2, 0], [side - 1, side // 2], [int(side * 0.68), side // 2],
                    [int(side * 0.68), side - 1], [int(side * 0.32), side - 1],
                    [int(side * 0.32), side // 2], [0, side // 2]], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def test_the_normalised_glyph_ignores_size_and_position():
    """The same drawing, small in a corner and large in the middle, must come
    out the same -- that is what makes matching at several scales unnecessary."""
    small = np.zeros((80, 80), np.uint8)
    small[5:25, 5:25] = _arrow(20)
    large = np.zeros((80, 80), np.uint8)
    large[20:76, 12:68] = _arrow(56)

    a, b = normalized_glyph(small), normalized_glyph(large)
    assert a is not None and b is not None
    assert a.shape == b.shape == (GLYPH_SIDE, GLYPH_SIDE)
    assert float(np.mean((a > 127) == (b > 127))) > 0.93


def test_a_mark_too_small_has_no_glyph():
    assert normalized_glyph(np.zeros((40, 40), np.uint8)) is None


def _write_icon(folder, hero: str, name: str, side: int = 128) -> None:
    import cv2

    (folder / hero).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / hero / f"{name}.png"), 255 - _arrow(side))


def test_an_empty_icon_bank_does_not_break(tmp_path):
    bank = IconBank.from_dir(tmp_path / "does_not_exist")
    assert not bank
    assert bank.best_match(np.zeros((GLYPH_SIDE, GLYPH_SIDE), np.uint8)) == (None, 0.0)


def test_the_bank_recognises_the_mark_in_both_polarities(tmp_path):
    """On the HUD the same icon shows up black on a white disc (ultimate) and
    white on a dark box (regular ability). One template serves both -- it is
    the mark that is compared, not the pixels."""
    import cv2

    _write_icon(tmp_path, "sample", "arrow")
    bank = IconBank.from_dir(tmp_path)
    assert len(bank) == 1

    # ultimate: white disc, black mark
    disc = np.full((60, 60, 3), 20, np.uint8)
    cv2.circle(disc, (30, 30), 26, (250, 250, 250), -1)
    mark = _arrow(30)
    disc[15:45, 15:45][mark > 0] = (15, 15, 15)
    key, score = bank.best_match(glyph_in_disc(disc))
    assert key == "sample/arrow" and score > 0.85

    # regular ability: dark box, light mark
    box = np.full((40, 40, 3), 50, np.uint8)
    mark = _arrow(30)
    box[5:35, 5:35][mark > 0] = (240, 240, 240)
    key, score = bank.best_match(glyph_on_dark(box))
    assert key == "sample/arrow" and score > 0.85


def test_the_icon_key_carries_hero_and_ability(tmp_path):
    _write_icon(tmp_path, "orisa", "energy_javelin")
    _write_icon(tmp_path, "domina", "panopticon")
    assert set(IconBank.from_dir(tmp_path).keys) == {
        "orisa/energy_javelin", "domina/panopticon",
    }


def test_the_crosshair_inside_the_roi_comes_from_the_roi_geometry():
    """The kills ROI is shifted upwards, so the crosshair -- which is the centre
    of the SCREEN -- does not fall at its centre. Deriving that from the
    geometry avoids a second number in the profile to forget to fix together."""
    from owcore.models import RoiSpec

    roi = RoiSpec(name="kills", x=0.42, y=0.40, w=0.16, h=0.18)
    x, y = roi.relative(0.5, 0.5)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5556, abs=1e-3)


def test_crops_come_out_with_canonical_colour(tmp_path):
    """Regression of a bug that only showed up between machines.

    The detectors decide by saturation, and the YUV->RGB matrix changes
    exactly the saturation. With the crop tagged as BT.709, the same file was
    read with saturation 231 on the host and 205 inside the container -- and the
    detector found 20 kills in one place and 10 in the other, with the same
    code. Tagging BT.601, which is what decoders assume for small frames, makes
    both read the same. This test pins the tagging.
    """
    import json
    import subprocess

    from owcore.config import get_settings
    from owcore.ffmpeg import extract_rois
    from owcore.models import RoiSpec

    src = tmp_path / "src.mp4"
    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x180:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    roi = RoiSpec(name="r", x=0.25, y=0.25, w=0.5, h=0.5, fps=5, width_px=160)
    out = extract_rois(src, [roi], tmp_path / "out")

    probe = json.loads(subprocess.run(
        [get_settings().ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-print_format", "json", str(out["r"])],
        capture_output=True, text=True, check=True,
    ).stdout)["streams"][0]

    # `color_space` (the matrix) and `color_range` are the two that change the
    # YUV->RGB conversion, and so the saturation. Primaries and transfer are
    # omitted by ffmpeg when they add nothing, so there is nothing to assert.
    assert probe.get("color_space") == "smpte170m"
    assert probe.get("color_range") == "tv"


def test_the_crop_reports_its_progress(tmp_path):
    """Cropping is ~3/4 of an analysis's time. Without it reporting how far it
    has got, the screen sits on the same number for minutes on a match
    recording -- and standing still is how the user reads frozen."""
    import subprocess

    from owcore.config import get_settings
    from owcore.ffmpeg import extract_rois
    from owcore.models import RoiSpec

    # heavy enough for ffmpeg to have something to count: it only speaks every
    # half second of wall clock, and a crop that ends before that would report
    # only once
    src = tmp_path / "src.mp4"
    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=1280x720:rate=30:duration=40",
         "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    rois = [
        RoiSpec(name=f"r{i}", x=0.0, y=0.0, w=1.0, h=1.0, fps=30, width_px=320)
        for i in range(5)
    ]

    seen: list[float] = []
    out = extract_rois(src, rois, tmp_path / "out", on_progress=seen.append)

    assert out["r0"].exists(), "the crop must come out the same, with or without a reporter"
    assert seen == sorted(seen), "the bar must not go backwards"
    assert all(0.0 <= v <= 1.0 for v in seen), f"fraction outside 0..1: {seen}"
    # what matters is not how many times it reported, but having reported
    # WHILE working: a reporter that only speaks at the end moves no bar
    assert any(v < 0.99 for v in seen), f"only reported at the end: {seen}"
    assert seen[-1] > 0.5, f"stopped too early at {seen[-1]:.2f}"


# ── the HUD icon versus "anything red" ─────────────────────────────────────

MAGENTA = (115, 23, 235)  # BGR of the skull's magenta
RANGE = [{"lo": [156, 185, 150], "hi": [178, 255, 255]}]


def _frame(draw) -> np.ndarray:
    img = np.full((64, 102, 3), (70, 60, 55), np.uint8)  # neutral scene
    draw(img)
    return img


def _find(img):
    from owcore.vision import find_icon

    return find_icon(img, RANGE, min_area_frac=0.04, max_offset=0.30,
                     aspect_range=(0.55, 1.9))


def test_a_small_red_splash_is_not_an_icon():
    """The complaint that prompted the size filter: a 20-pixel blob in the
    middle of the screen is not a kill."""
    import cv2

    img = _frame(lambda i: cv2.circle(i, (51, 32), 3, MAGENTA, -1))
    assert _find(img) is None


def test_an_icon_of_the_right_size_is_found():
    import cv2

    img = _frame(lambda i: cv2.circle(i, (51, 32), 14, MAGENTA, -1))
    blob = _find(img)
    assert blob is not None
    assert 0.04 < blob.area_frac < 0.30
    assert abs(blob.offset_x) < 0.1 and abs(blob.offset_y) < 0.1


def test_a_damage_arc_is_rejected_by_shape_and_position():
    """The directional damage indicator is wide, flat and sits above the
    crosshair."""
    import cv2

    img = _frame(lambda i: cv2.ellipse(i, (51, 8), (34, 5), 0, 0, 360, MAGENTA, -1))
    assert _find(img) is None


def test_inner_holes_tell_a_skull_from_a_blot():
    """The eye sockets are what separate a skull from a blob of the same
    size."""
    import cv2

    solid = _frame(lambda i: cv2.circle(i, (51, 32), 14, MAGENTA, -1))

    def with_sockets(i):
        cv2.circle(i, (51, 32), 14, MAGENTA, -1)
        cv2.circle(i, (46, 29), 3, (20, 20, 20), -1)
        cv2.circle(i, (56, 29), 3, (20, 20, 20), -1)

    skull = _frame(with_sockets)
    assert _find(solid).hole_ratio == 0.0
    assert _find(skull).hole_ratio > 0.05


# ── a schema that evolves without losing what was already there ────────────


def test_a_new_column_reaches_an_existing_database(isolated):
    """`create_all` ignores a table that already exists -- and that nearly cost
    a lot.

    Anyone who had already run the system had the `renders` table without the
    manual montages column. Without reconciling, the first montage would blow
    up with "column renders.timelines does not exist", and the way out would be
    deleting the database together with the matches already analysed.
    """
    from sqlalchemy import inspect, text

    from owcore.db import engine, init_db, session
    from owcore.models import Job, Render

    with session() as s:
        s.add(Job(id="j1", video_key="k", video_name="v.mp4"))
        s.add(Render(id="r1", job_id="j1", stage="queued"))

    # take the database back to its state before this feature
    eng = engine()
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE renders DROP COLUMN timelines"))
        conn.execute(text("DROP TABLE tracks"))
    assert "timelines" not in {
        c["name"] for c in inspect(eng).get_columns("renders")
    }

    init_db()

    columns = {c["name"] for c in inspect(eng).get_columns("renders")}
    assert "timelines" in columns, "the new column did not reach the old database"
    assert "tracks" in inspect(eng).get_table_names(), "the new table was not created"

    with session() as s:
        request = s.get(Render, "r1")
        # the old request is still whole, and the new column reads as empty
        assert request.stage == "queued"
        assert request.timelines == []


def test_a_new_column_does_not_leave_old_rows_NULL(isolated):
    """Backfill for the simple types too, not just JSON.

    A new column comes in nullable -- adding NOT NULL to a table with rows would
    require rewriting it. If the old rows stay NULL, the consumer breaks far
    from here: it was `round(job.fps, 3)` bringing the whole match listing down
    after `fps` entered the model.
    """
    from sqlalchemy import text

    from owcore.db import engine, init_db, session
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="j2", video_key="k", video_name="v.mp4"))

    eng = engine()
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE jobs DROP COLUMN fps"))
        conn.execute(text("ALTER TABLE jobs DROP COLUMN proxy_key"))

    init_db()

    with session() as s:
        job = s.get(Job, "j2")
        assert job.fps == 0.0, "the old row was left NULL in a Float"
        assert job.proxy_key == "", "the old row was left NULL in a String"


def test_repairs_the_NULL_a_previous_boot_left(isolated):
    """The column already exists, but with NULL where the model promises a
    value.

    It happens when it was created by a version of the reconciler that did not
    fill that type yet. Finding the column ready and leaving would keep the
    database with NULL forever.
    """
    from sqlalchemy import text

    from owcore.db import engine, init_db, session
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="j3", video_key="k", video_name="v.mp4"))

    eng = engine()
    with eng.begin() as conn:
        # the exact state an old boot would leave: the column exists, nullable
        # (adding NOT NULL to a table with rows would require rewriting it),
        # and nobody filled in the rows from before
        conn.execute(text("ALTER TABLE jobs DROP COLUMN fps"))
        conn.execute(text("ALTER TABLE jobs ADD COLUMN fps FLOAT"))

    init_db()

    with session() as s:
        assert s.get(Job, "j3").fps == 0.0


def test_reconciling_is_idempotent(isolated):
    """Every worker calls `init_db` on boot; running it again must touch
    nothing."""
    from owcore.db import _reconcile_columns, engine, init_db

    init_db()
    with engine().begin() as conn:
        assert _reconcile_columns(conn) == []
