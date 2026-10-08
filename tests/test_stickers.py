"""The stickers library, and pictures placed whole on the frame.

* the catalogue on its own (`owcore.stickers`): every sticker is a real
  transparent PNG, the same bytes on every call, drawn without numpy;
* the routes: listing, the shelf's small copies, and adding a sticker in a
  colour to a match as an ordinary image item of its library;
* `TimelineClip.fit`: a picture placed whole sits where the transform puts
  it, with the layer below showing all around -- checked on rendered pixels.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from owcore import stickers
from owcore.db import session
from owcore.models import Job, Media
from test_pipeline import api
from test_timeline import _render_with_library, raw_frame


def _decode(png: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, "not a PNG anything can read"
    return image  # BGRA


# ── the catalogue ───────────────────────────────────────────────────────────


def test_every_sticker_is_a_transparent_png():
    ids = [st.id for st in stickers.STICKERS]
    assert len(ids) == len(set(ids)), "an id names one sticker"
    assert len(ids) >= 10

    for st in stickers.STICKERS:
        assert st.category in stickers.CATEGORIES
        image = _decode(stickers.png_bytes(st.id, "red"))
        assert image.shape == (stickers.SIZE, stickers.SIZE, 4), st.id
        alpha = image[:, :, 3]
        # transparent around, and the picture's edge never cuts the outline
        assert alpha[0, 0] == 0 and alpha[-1, -1] == 0, st.id
        edge = max(alpha[0].max(), alpha[-1].max(), alpha[:, 0].max(), alpha[:, -1].max())
        assert edge == 0, f"{st.id} touches the edge"
        covered = (alpha > 0).mean()
        assert 0.08 < covered < 0.9, f"{st.id} covers {covered:.0%}"
        # the outline is dark, and the fill is the colour asked for
        solid = image[alpha == 255]
        reds = ((solid[:, 2] > 230) & (solid[:, 1] < 90) & (solid[:, 0] < 80)).mean()
        darks = (solid[:, :3].max(axis=1) < 40).mean()
        assert reds > 0.2 and darks > 0.03, st.id


def test_a_colour_changes_the_fill_and_not_the_shape():
    import numpy as np

    red = _decode(stickers.png_bytes("star", "red"))
    blue = _decode(stickers.png_bytes("star", "blue"))
    assert np.array_equal(red[:, :, 3], blue[:, :, 3])
    centre = stickers.SIZE // 2
    assert tuple(blue[centre, centre, :3]) == (255, 132, 10)  # BGR of #0a84ff


def test_stickers_are_drawn_the_same_on_every_call():
    stickers.png_bytes.cache_clear()
    stickers._masks.cache_clear()
    first = stickers.png_bytes("skull", "white")
    stickers.png_bytes.cache_clear()
    stickers._masks.cache_clear()
    assert stickers.png_bytes("skull", "white") == first


def test_the_details_are_drawn_inside_the_fill():
    """A skull's eyes are holes in the outline's colour, not transparent."""
    image = _decode(stickers.png_bytes("skull", "white"))
    half = stickers.SIZE / 2
    # the left eye, in the sticker's own coordinates, shrunk by FIT
    ex = int(half + (-0.27 * stickers.FIT) * half)
    ey = int(half + (-0.12 * stickers.FIT) * half)
    assert image[ey, ex, 3] == 255
    assert image[ey, ex, :3].max() < 40


def test_an_unknown_sticker_or_colour_is_a_key_error():
    with pytest.raises(KeyError):
        stickers.png_bytes("nope")
    with pytest.raises(KeyError):
        stickers.png_bytes("arrow", "beige")


def test_the_gateway_draws_stickers_without_numpy_or_pillow():
    gateway = Path(__file__).resolve().parents[1] / "services" / "gateway"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['numpy'] = None; sys.modules['PIL'] = None; "
         "sys.modules['cv2'] = None; import app; "
         "from owcore import stickers; stickers.png_bytes('crown', 'yellow')"],
        cwd=gateway, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# ── the routes ──────────────────────────────────────────────────────────────


def _job() -> str:
    with session() as s:
        job = Job(video_key="x/match.mp4", video_name="match.mp4")
        s.add(job)
        s.flush()
        return job.id


def test_the_catalogue_is_listed_with_its_colours(isolated):
    body = api().get("/api/stickers").json()
    assert body["categories"] == list(stickers.CATEGORIES)
    assert [st["id"] for st in body["stickers"]] == [st.id for st in stickers.STICKERS]
    assert [c["id"] for c in body["colors"]] == list(stickers.COLORS)
    assert body["colors"][0] == {"id": "red", "hex": "#ff3b30"}
    arrow = next(st for st in body["stickers"] if st["id"] == "arrow")
    assert arrow == {"id": "arrow", "name": "Arrow", "category": "point",
                     "preview_url": "/api/stickers/arrow.png"}


def test_the_shelf_gets_a_small_copy_in_any_colour(isolated):
    client = api()
    resp = client.get("/api/stickers/heart.png", params={"color": "pink"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert _decode(resp.content).shape == (stickers.PREVIEW_SIZE, stickers.PREVIEW_SIZE, 4)
    assert client.get("/api/stickers/heart.png", params={"color": "beige"}).status_code == 422
    assert client.get("/api/stickers/nope.png").status_code == 404


def test_adding_a_sticker_makes_a_ready_image_item(isolated):
    job_id = _job()
    client = api()

    resp = client.post(f"/api/jobs/{job_id}/stickers",
                       json={"sticker_id": "arrow", "color": "yellow"})
    assert resp.status_code == 201, resp.text
    item = resp.json()
    assert item["status"] == "ready"
    assert item["kind"] == "image"
    assert item["name"] == "Arrow (yellow)"
    assert item["sticker_id"] == "arrow:yellow"
    assert (item["width"], item["height"]) == (stickers.SIZE, stickers.SIZE)

    stored = client.get(item["file_url"])
    assert stored.content == stickers.png_bytes("arrow", "yellow")
    thumb = client.get(item["thumb_url"])
    assert thumb.headers["content-type"] == "image/png"
    assert _decode(thumb.content).shape[2] == 4, "the thumbnail keeps its transparency"

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert [m["id"] for m in detail["media"]] == [item["id"]]


def test_the_same_sticker_and_colour_reuses_the_item(isolated):
    job_id = _job()
    client = api()
    url = f"/api/jobs/{job_id}/stickers"
    first = client.post(url, json={"sticker_id": "ring", "color": "red"}).json()
    again = client.post(url, json={"sticker_id": "ring", "color": "red"}).json()
    other = client.post(url, json={"sticker_id": "ring", "color": "green"}).json()
    # no colour is the default one
    default = client.post(url, json={"sticker_id": "ring"}).json()

    assert again["id"] == first["id"] == default["id"]
    assert other["id"] != first["id"]
    with session() as s:
        assert s.query(Media).filter_by(job_id=job_id).count() == 2


def test_an_unknown_sticker_colour_or_match_is_refused(isolated):
    job_id = _job()
    client = api()
    url = f"/api/jobs/{job_id}/stickers"
    assert client.post(url, json={"sticker_id": "nope"}).status_code == 404
    assert client.post(url, json={}).status_code == 404
    assert client.post(url, json={"sticker_id": "star", "color": "beige"}).status_code == 422
    assert client.post("/api/jobs/nope/stickers", json={"sticker_id": "star"}).status_code == 404


def test_an_old_database_gains_the_sticker_column(isolated):
    from sqlalchemy import inspect, text

    from owcore import db

    with db.engine().begin() as conn:
        conn.execute(text("ALTER TABLE tracks DROP COLUMN sticker_id"))
    with db.engine().begin() as conn:
        added = db._reconcile_columns(conn)
    assert "tracks.sticker_id" in added
    columns = {c["name"] for c in inspect(db.engine()).get_columns("tracks")}
    assert "sticker_id" in columns


# ── placed whole ────────────────────────────────────────────────────────────


def test_a_clip_fit_is_optional_and_leaves_the_simple_path():
    from owcore.models import Fit, TimelineClip

    plain = TimelineClip(at_s=0, duration_s=2)
    assert plain.fit is None and plain.is_simple
    whole = TimelineClip(at_s=0, duration_s=2, fit="contain")
    assert whole.fit is Fit.CONTAIN
    assert not whole.is_simple


def test_a_picture_placed_whole_shows_the_layer_below_around_it(
    isolated, short_sample, tmp_path
):
    """A square blue picture, half the frame's height, on the right, over a
    red frame. Filling the frame (the export's `cover`) it would have been
    cropped to a wide strip covering the whole right half; placed whole it is
    a square, with red above, below and to its left."""
    import cv2
    import numpy as np

    from owcore.compose import LibraryFile
    from owcore.models import Layer, Timeline, TimelineClip

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.jpg"
    cv2.imwrite(str(red), np.full((360, 640, 3), (0, 0, 255), np.uint8))
    # a JPEG has no alpha of its own: what is around it must still be nothing
    cv2.imwrite(str(blue), np.full((200, 200, 3), (255, 0, 0), np.uint8))
    library = {"red": LibraryFile(red, "image"), "blue": LibraryFile(blue, "image")}

    def render(fit, name):
        timeline = Timeline(layers=[
            Layer(clips=[TimelineClip(at_s=0, duration_s=1, source="media",
                                      media_id="red")]),
            Layer(clips=[TimelineClip(at_s=0, duration_s=1, source="media",
                                      media_id="blue", fit=fit,
                                      transform={"scale": 0.5, "x": 0.5})]),
        ])
        video = _render_with_library(timeline, library, short_sample, tmp_path / name)
        return raw_frame(video, 0.5).reshape(45, 80, 3)

    whole = render("contain", "whole.mp4")
    # the square: 22.5 px tall on the 80x45 frame, centred at x=60, y=22.5
    r, g, b = whole[22, 60]
    assert b > 180 and r < 60, "the picture is where the transform put it"
    for y, x in ((3, 60), (41, 60), (22, 45)):
        r, g, b = whole[y, x]
        assert r > 180 and b < 60, f"red shows around the picture at ({x}, {y})"

    cover = render(None, "cover.mp4")
    r, g, b = cover[22, 45]
    assert b > 180 and r < 60, "filling the frame, the same picture is a wide strip"


def test_a_sticker_renders_with_its_transparency(isolated, short_sample, tmp_path):
    import numpy as np

    from owcore.compose import LibraryFile
    from owcore.models import Layer, Timeline, TimelineClip

    green = tmp_path / "green.png"
    import cv2
    cv2.imwrite(str(green), np.full((360, 640, 3), (0, 200, 0), np.uint8))
    ring = tmp_path / "ring.png"
    ring.write_bytes(stickers.png_bytes("ring", "yellow"))
    library = {"bg": LibraryFile(green, "image"), "ring": LibraryFile(ring, "image")}

    timeline = Timeline(layers=[
        Layer(clips=[TimelineClip(at_s=0, duration_s=1, source="media", media_id="bg")]),
        Layer(clips=[TimelineClip(at_s=0, duration_s=1, source="media",
                                  media_id="ring", fit="contain")]),
    ])
    frame = raw_frame(
        _render_with_library(timeline, library, short_sample, tmp_path / "ring.mp4"), 0.5
    ).reshape(45, 80, 3)
    # the ring's middle is see-through: the green shows
    r, g, b = frame[22, 40]
    assert g > 150 and r < 80
    # and on the ring itself, yellow (radius 0.78 * FIT of half the height)
    y = int(round(22.5 - 0.78 * stickers.FIT * 22.5))
    r, g, b = frame[y, 40]
    assert r > 180 and g > 150 and b < 80, (r, g, b)
