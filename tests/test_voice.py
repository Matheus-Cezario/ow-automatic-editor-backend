"""Voice-overs recorded in the editor.

* a voice block is not music: it does not take the game away and plays at
  its own volume, like a sound effect;
* while it speaks, the music and the game step back to the duck level --
  checked in the graph and on the sound of a render;
* the upload: a voice-over is audio, stays out of the track picker, and the
  browser's `.weba` is read as audio.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from owcore.db import session
from owcore.models import Job, Media
from test_pipeline import api
from test_timeline import _volume_between


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
                 "vo": LibraryFile(Path("vo.weba"), "audio")},
    )
    return t, graph.filter_complex


def _block(media_id: str, at: float, duration: float, **kw) -> dict:
    return {"at_s": at, "duration_s": duration, "source": "media",
            "media_id": media_id, **kw}


# ── the model and the graph ─────────────────────────────────────────────────


def test_a_voice_over_is_not_music():
    t, graph = _graph(_block("vo", 1.0, 2.0, kind="voice"), game_volume=0.0)
    assert not t.has_music, "a voice alone must not silence the game"
    assert t.voice_spans() == [(1.0, 3.0)]
    # the game steps back while it speaks, and the voice goes in on its own
    game = next(f for f in graph.split(";") if f.endswith("[game]"))
    assert "volume='1-0.7000*(" in game
    assert graph.split(";")[-1].startswith("[game][a2]amix=inputs=2")


def test_the_music_and_the_game_step_back_under_a_voice():
    _, graph = _graph(
        _block("song", 0.0, 4.0),
        _block("vo", 4.0, 1.5, kind="voice"),
        music_volume=0.8, game_volume=0.5, duck_level=0.25,
    )
    music = next(f for f in graph.split(";") if f.endswith("[music]"))
    assert music.count("[a") == 1, "the voice is not mixed as music"
    assert "volume='0.8000*(1-0.7500*(" in music
    # 4.0 - attack .. 5.5 + release, on the output's clock
    assert "(t-3.880)" in music and "(5.850-t)" in music
    game = next(f for f in graph.split(";") if f.endswith("[game]"))
    assert "volume='(0.5000)*(1-0.7500*(" in game


def test_a_muted_or_hidden_voice_does_not_duck():
    from owcore.models import Timeline

    t = Timeline(layers=[
        {"clips": [{"at_s": 0.0, "duration_s": 4.0, "start_s": 1.0}]},
        {"kind": "audio", "muted": True,
         "clips": [_block("vo", 1.0, 1.0, kind="voice")]},
        {"kind": "audio",
         "clips": [_block("vo", 2.0, 1.0, kind="voice", audio={"mute": True})]},
    ])
    assert t.voice_spans() == []


def test_without_a_voice_the_mix_is_as_before():
    _, graph = _graph(_block("song", 0.0, 4.0), game_volume=0.3)
    assert "(t-" not in graph, "no curve on the music or the game"


# ── on the sound of a render ────────────────────────────────────────────────


def _tone(dest: Path, seconds: float, freq: int = 440, volume: float = 0.5) -> Path:
    from owcore.config import get_settings

    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency={freq}:duration={seconds}",
         "-af", f"volume={volume}", str(dest)],
        check=True,
    )
    return dest


def _silence(dest: Path, seconds: float) -> Path:
    from owcore.config import get_settings

    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"anullsrc=r=44100:cl=mono:d={seconds}", str(dest)],
        check=True,
    )
    return dest


def test_the_music_is_quieter_while_the_voice_speaks(isolated, short_sample, tmp_path):
    """A loud song for four seconds, the game muted, and a silent voice-over
    from 1.5 to 2.5 s: whatever is heard is the song, so its dip is the
    voice's doing. At a duck level of 0.25 it comes out about 12 dB down."""
    from owcore import ffmpeg
    from owcore.compose import LibraryFile, compose_graph
    from owcore.models import Timeline

    library = {
        "song": LibraryFile(_tone(tmp_path / "song.wav", 6), "audio"),
        "vo": LibraryFile(_silence(tmp_path / "vo.wav", 2), "audio"),
    }
    timeline = Timeline(
        duck_level=0.25,
        layers=[
            {"clips": [{"at_s": 0.0, "duration_s": 4.0, "start_s": 1.0,
                        "audio": {"mute": True}}]},
            {"kind": "audio", "clips": [_block("song", 0.0, 4.0)]},
            {"kind": "audio", "name": "Voice",
             "clips": [_block("vo", 1.5, 1.0, kind="voice")]},
        ],
    )
    c = compose_graph(timeline, source=short_sample, width=640, height=360,
                      fps=30, library=library)
    out = tmp_path / "voice.mp4"
    ffmpeg.compose(c, out)

    under = _volume_between(out, 1.7, 2.3)
    before = _volume_between(out, 0.5, 1.2)
    after = _volume_between(out, 3.2, 3.8)
    assert before - under == pytest.approx(12, abs=2.5), (before, under)
    assert after - under == pytest.approx(12, abs=2.5), (after, under)


# ── the upload ──────────────────────────────────────────────────────────────


def _job() -> str:
    with session() as s:
        job = Job(video_key="x/match.mp4", video_name="match.mp4")
        s.add(job)
        s.flush()
        return job.id


def test_a_voice_over_upload_is_audio_and_not_a_song(isolated, tmp_path):
    job_id = _job()
    client = api()
    data = _tone(tmp_path / "vo.ogg", 1.0).read_bytes()

    resp = client.post(
        f"/api/jobs/{job_id}/media",
        files={"file": ("voice-over.weba", data, "audio/webm")},
        data={"voice": "true"},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["kind"] == "audio", ".weba is audio, not video"
    media_id = resp.json()["id"]

    with session() as s:
        assert s.get(Media, media_id).voice is True
    item = client.get(f"/api/media/{media_id}").json()
    assert item["voice"] is True
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert [m["id"] for m in detail["media"]] == [media_id]
    assert detail["tracks"] == [], "a voice-over is not a song to build on"


def test_only_audio_can_be_a_voice_over(isolated):
    job_id = _job()
    resp = api().post(
        f"/api/jobs/{job_id}/media",
        files={"file": ("x.png", b"not really", "image/png")},
        data={"voice": "true"},
    )
    assert resp.status_code == 422


def test_an_old_database_gains_the_voice_column(isolated):
    from sqlalchemy import inspect, text

    from owcore import db

    with db.engine().begin() as conn:
        conn.execute(text("ALTER TABLE tracks DROP COLUMN voice"))
    with db.engine().begin() as conn:
        added = db._reconcile_columns(conn)
    assert "tracks.voice" in added
    columns = {c["name"] for c in inspect(db.engine()).get_columns("tracks")}
    assert "voice" in columns


def test_a_browser_recording_is_read_as_audio(isolated, tmp_path):
    """What Chrome's MediaRecorder sends: Opus in WebM, named `.weba`. The
    analyser has to find its length, or the block would have none."""
    from conftest import service_module as module
    from owcore.config import get_settings
    from owcore.models import STREAM_MEDIA
    from test_pipeline import drain

    weba = tmp_path / "voice-over.weba"
    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=1.5", "-c:a", "libopus",
         "-f", "webm", str(weba)],
        check=True,
    )
    job_id = _job()
    resp = api().post(
        f"/api/jobs/{job_id}/media",
        files={"file": (weba.name, weba.read_bytes(), "audio/webm")},
        data={"voice": "true"},
    )
    assert resp.status_code == 201, resp.text
    worker = module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
        worker.handle(payload)

    item = api().get(f"/api/media/{resp.json()['id']}").json()
    assert item["status"] == "ready", item["error"]
    assert item["voice"] is True
    assert item["duration_s"] == pytest.approx(1.5, abs=0.1)
    file = api().get(item["file_url"])
    assert file.headers["content-type"] == "audio/webm"
