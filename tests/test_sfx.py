"""The sound effects library: the catalogue the server synthesises, and the
path from browsing an effect to hearing it in the rendered video.

* the catalogue on its own (`owcore.sfx`): every effect is a real, audible,
  well-formed WAV, the same bytes on every call;
* the routes: listing, listening before adding, and adding an effect to a
  match as an ordinary audio item of its library;
* a render with effects on audio layers, checked on the sound of the mp4.
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from conftest import service_module
from owcore import sfx
from owcore.db import session
from owcore.models import Job, Media
from test_pipeline import api, run_analysis
from test_timeline import _volume_between, run_render

#: the shortest block the app lets you make; an effect shorter than that could
#: not be placed whole
APP_MIN_CUT_S = 0.2


# ── the catalogue ───────────────────────────────────────────────────────────


def test_every_effect_is_a_well_formed_wav():
    ids = [e.id for e in sfx.EFFECTS]
    assert len(ids) == len(set(ids)), "an id names one sound"
    assert len(ids) >= 10

    for effect in sfx.EFFECTS:
        with wave.open(io.BytesIO(sfx.wav_bytes(effect.id))) as w:
            # what `owcore.audio.read_wav` and the render read
            assert w.getsampwidth() == 2
            assert w.getnchannels() == 1
            assert w.getframerate() == sfx.SR
            frames = w.getnframes()
            samples = struct.unpack(f"<{frames}h", w.readframes(frames))

        assert frames / sfx.SR == pytest.approx(sfx.duration_s(effect.id), abs=1e-3)
        assert sfx.duration_s(effect.id) >= APP_MIN_CUT_S, effect.id
        assert sfx.duration_s(effect.id) <= 3.0, "an effect, not a song"
        top = max(abs(v) for v in samples)
        # normalised: loud enough to be heard, never clipped
        assert 0.4 * 32767 < top <= 0.9 * 32767, effect.id
        # and it does not start or end on a click
        assert abs(samples[0]) < 0.05 * 32767, effect.id
        assert abs(samples[-1]) < 0.05 * 32767, effect.id


def test_the_sounds_are_synthesised_the_same_on_every_call():
    """The library item and the preview the app played must be the same
    sound -- and the same on any machine, since nothing is downloaded."""
    for effect in sfx.EFFECTS:
        assert sfx._finish(effect.make()) == list(sfx._signal(effect.id))


def test_every_effect_has_a_waveform_and_a_known_category():
    for effect in sfx.EFFECTS:
        assert effect.category in sfx.CATEGORIES
        peaks = sfx.peaks(effect.id)
        assert len(peaks) >= 64
        assert all(0.0 <= p <= 1.0 for p in peaks)
        assert max(peaks) == 1.0


def test_an_unknown_effect_is_a_key_error():
    with pytest.raises(KeyError):
        sfx.wav_bytes("nope")


def test_the_gateway_does_not_need_numpy():
    """The gateway's image has no numpy: a module it imports that needed it
    would put the API in a restart loop, as once happened to the
    preprocessor. So the gateway is imported, and an effect made, with numpy
    made impossible to import."""
    gateway = Path(__file__).resolve().parents[1] / "services" / "gateway"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['numpy'] = None; import app; "
         "from owcore import sfx; sfx.wav_bytes('whoosh')"],
        cwd=gateway, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# ── the routes ──────────────────────────────────────────────────────────────


def _job() -> str:
    """A match row, without analysing a recording: adding an effect does not
    look at the video."""
    with session() as s:
        job = Job(video_key="x/match.mp4", video_name="match.mp4")
        s.add(job)
        s.flush()
        return job.id


def test_the_catalogue_is_listed_by_category(isolated):
    body = api().get("/api/sfx").json()
    assert body["categories"] == list(sfx.CATEGORIES)
    assert [e["id"] for e in body["effects"]] == [e.id for e in sfx.EFFECTS]
    whoosh = next(e for e in body["effects"] if e["id"] == "whoosh")
    assert whoosh["name"] == "Whoosh"
    assert whoosh["category"] == "transition"
    assert whoosh["duration_s"] == sfx.duration_s("whoosh")
    assert len(whoosh["peaks"]) >= 64
    assert whoosh["audio_url"] == "/api/sfx/whoosh/audio"


def test_an_effect_can_be_heard_before_it_is_added(isolated):
    client = api()
    whole = client.get("/api/sfx/hit/audio")
    assert whole.status_code == 200
    assert whole.headers["content-type"] == "audio/wav"
    assert whole.content == sfx.wav_bytes("hit")

    # the player seeks with Range
    part = client.get("/api/sfx/hit/audio", headers={"range": "bytes=0-43"})
    assert part.status_code == 206
    assert part.content == sfx.wav_bytes("hit")[:44]
    assert part.headers["content-range"] == f"bytes 0-43/{len(whole.content)}"

    assert client.get("/api/sfx/nope/audio").status_code == 404


def test_adding_an_effect_makes_a_ready_library_item(isolated):
    job_id = _job()
    client = api()

    resp = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "boom"})
    assert resp.status_code == 201, resp.text
    item = resp.json()
    # no analysis to wait for: the server made the sound and knows it
    assert item["status"] == "ready"
    assert item["kind"] == "audio"
    assert item["name"] == "Boom"
    assert item["sfx_id"] == "boom"
    assert item["duration_s"] == sfx.duration_s("boom")
    assert item["peaks"] == list(sfx.peaks("boom"))
    assert item["bpm"] == 0 and item["beats"] == []

    # the stored file is the effect
    stored = client.get(item["file_url"])
    assert stored.status_code == 200
    assert stored.content == sfx.wav_bytes("boom")

    # it is part of the library, but not of the songs to build a montage on
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert [m["id"] for m in detail["media"]] == [item["id"]]
    assert detail["tracks"] == []


def test_adding_the_same_effect_twice_reuses_the_item(isolated):
    job_id = _job()
    client = api()
    first = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "hit"}).json()
    again = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "hit"}).json()
    other = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "ding"}).json()

    assert again["id"] == first["id"]
    assert other["id"] != first["id"]
    with session() as s:
        assert s.query(Media).filter_by(job_id=job_id).count() == 2


def test_removing_the_item_lets_the_effect_be_added_again(isolated):
    job_id = _job()
    client = api()
    first = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "zap"}).json()
    assert client.delete(f"/api/media/{first['id']}").status_code == 204
    again = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "zap"})
    assert again.status_code == 201
    assert again.json()["id"] != first["id"]


def test_an_unknown_effect_or_match_is_refused(isolated):
    job_id = _job()
    client = api()
    assert client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "nope"}).status_code == 404
    assert client.post(f"/api/jobs/{job_id}/sfx", json={}).status_code == 404
    assert client.post("/api/jobs/nope/sfx", json={"sfx_id": "hit"}).status_code == 404


def test_an_old_database_gains_the_column(isolated):
    """Whoever ran the system before has a `tracks` table without `sfx_id`;
    the reconciler adds it, and the old rows read as uploads."""
    from sqlalchemy import inspect, text

    from owcore import db

    with db.engine().begin() as conn:
        conn.execute(text("ALTER TABLE tracks DROP COLUMN sfx_id"))
    with db.engine().begin() as conn:
        added = db._reconcile_columns(conn)
    assert "tracks.sfx_id" in added
    columns = {c["name"] for c in inspect(db.engine()).get_columns("tracks")}
    assert "sfx_id" in columns


# ── on the ruler, and in the video ──────────────────────────────────────────


def _graph(*audio_blocks: dict, **kw):
    from owcore.compose import LibraryFile, compose_graph
    from owcore.models import Timeline

    t = Timeline(
        layers=[
            {"clips": [{"at_s": 0.0, "duration_s": 4.0, "start_s": 1.0}]},
            {"kind": "audio", "clips": list(audio_blocks)},
        ],
        **kw,
    )
    graph = compose_graph(
        t, source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600,
        library={"song": LibraryFile(Path("song.mp3"), "audio"),
                 "fx": LibraryFile(Path("fx.wav"), "audio")},
    )
    return t, graph.filter_complex


def _block(media_id: str, at: float, duration: float, **kw) -> dict:
    return {"at_s": at, "duration_s": duration, "source": "media",
            "media_id": media_id, **kw}


def test_an_effect_is_not_music():
    """One whoosh on a montage without a song must not take the game sound
    away -- which is what music does with `game_volume` at 0."""
    t, graph = _graph(_block("fx", 1.0, 0.7, kind="sfx"), game_volume=0.0)
    assert not t.has_music
    mix = graph.split(";")[-1]
    # the recording's sound and the effect, side by side, the game untouched
    assert mix.startswith("[a1][a2]amix=inputs=2"), mix
    assert "[game]" not in graph, "with no music the game is not turned down"

    # a song, by contrast, replaces the game sound at `game_volume` 0
    with_song, graph = _graph(_block("song", 0.0, 4.0), game_volume=0.0)
    assert with_song.has_music
    assert graph.split(";")[-1] == "[a2]anull[aout]", "the song alone"


def test_the_music_volume_does_not_govern_the_effects():
    _, graph = _graph(
        _block("song", 0.0, 4.0),
        _block("fx", 4.0, 0.7, kind="sfx"),
        music_volume=0.5, game_volume=0.3,
    )
    music = next(f for f in graph.split(";") if f.endswith("[music]"))
    # one music input, turned down -- the effect is not among them
    assert music.count("[a") == 1
    assert "volume=0.5000" in music
    # the effect goes into the final mix by itself, at its own level
    assert graph.split(";")[-1].startswith("[music][game][a3]amix=inputs=3")



def test_effects_on_audio_layers_are_heard_where_they_were_put(
    isolated, short_sample
):
    """End to end: two effects at the same instant, on two audio layers, over
    a video with the game muted. The second they play in is loud; the rest is
    silence."""
    job_id = run_analysis(short_sample)
    client = api()
    boom = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "boom"}).json()
    hit = client.post(f"/api/jobs/{job_id}/sfx", json={"sfx_id": "hit"}).json()

    timeline = {
        "title": "With effects",
        "game_volume": 0.0,
        "layers": [
            {"clips": [{"at_s": 0.0, "duration_s": 4.0, "start_s": 1.0,
                        "audio": {"mute": True}}]},
            {"kind": "audio", "name": "Effects", "clips": [
                {"at_s": 1.0, "duration_s": boom["duration_s"], "kind": "sfx",
                 "source": "media", "media_id": boom["id"]},
            ]},
            {"kind": "audio", "name": "Effects 2", "clips": [
                {"at_s": 1.0, "duration_s": hit["duration_s"], "kind": "sfx",
                 "source": "media", "media_id": hit["id"]},
            ]},
        ],
    }
    resp = client.post(
        f"/api/jobs/{job_id}/renders", data={"timelines": json.dumps([timeline])}
    )
    assert resp.status_code == 201, resp.text
    run_render()

    request = client.get(f"/api/renders/{resp.json()['id']}").json()
    assert request["status"] == "done", request["error"]

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    local = local_copy(key, Path(isolated.work_dir) / "check")
    info = ffmpeg.probe(local)
    assert info.duration_s == pytest.approx(4.0, abs=0.35)
    assert info.has_audio

    during = _volume_between(local, 1.0, 1.5)
    before = _volume_between(local, 0.0, 0.9)
    after = _volume_between(local, 3.0, 3.9)
    assert during > before + 20
    assert during > after + 20
