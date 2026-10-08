"""Several matches in one montage.

A moment brought from another match is a recording clip carrying that
match's `job_id`: it is cut from that recording instead of this one's, and
otherwise it is a moment like any other -- it ducks the music where its play
happens and goes through every transformation.

* the model: who it cuts from, and why it leaves the simple path;
* the graph: the other file, trimmed to its own length;
* the render and the preview, on the colour of their frames;
* the gateway: an unknown match is refused, and a match whose moments another
  montage uses cannot be deleted from under it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from owcore.db import session
from owcore.models import Job, JobStatus, RenderStatus
from test_pipeline import api
from test_timeline import _rgb, run_render


def _solid(dest: Path, colour: str, seconds: float = 4.0) -> Path:
    """A flat-colour 640x360 video with a tone, so each frame says which
    recording it came from."""
    from owcore.config import get_settings

    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c={colour}:s=640x360:r=30:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(dest)],
        check=True,
    )
    return dest


def _match(path: Path, name: str) -> str:
    """A match already analysed, with its recording in storage."""
    from owcore.storage import get_storage

    with session() as s:
        job = Job(video_key="", video_name=name, status=JobStatus.READY,
                  duration_s=4.0, fps=30, width=640, height=360)
        s.add(job)
        s.flush()
        job.video_key = get_storage().put_file(f"{job.id}/source.mp4", path)
        return job.id


def _two_matches(tmp_path) -> tuple[str, str]:
    red = _match(_solid(tmp_path / "red.mp4", "red"), "red.mp4")
    blue = _match(_solid(tmp_path / "blue.mp4", "blue"), "blue.mp4")
    return red, blue


def _red_then_blue(blue: str) -> dict:
    """Two seconds of this match, then two of the other one."""
    return {
        "title": "Both",
        "layers": [{"clips": [
            {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0, "source_t": 1.5},
            {"at_s": 2.0, "duration_s": 2.0, "start_s": 1.0, "source_t": 1.5,
             "job_id": blue},
        ]}],
    }


def _is_red(rgb) -> bool:
    return rgb[0] > 150 and rgb[2] < 80


def _is_blue(rgb) -> bool:
    return rgb[2] > 150 and rgb[0] < 80


# ── the model and the graph ─────────────────────────────────────────────────


def test_a_moment_from_another_match_says_where_it_is_cut_from():
    from owcore.models import Timeline

    t = Timeline(**_red_then_blue("other"))
    own, brought = t.layers[0].clips
    assert own.job_id is None and brought.job_id == "other"
    assert t.recording_jobs() == {"other"}
    # one source per cut is all the cut-and-splice path knows
    assert own.is_simple and not brought.is_simple
    assert not t.single_layer
    # its play still ducks the music, where it happens in the video
    assert t.play_times() == [0.5, 2.5]


def test_the_graph_cuts_it_from_the_other_recording_trimmed_to_its_length():
    from owcore.compose import Recording, compose_graph
    from owcore.models import Timeline

    t = Timeline(layers=[{"clips": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0},
        # asks for 3 s from 2.5 of a 4 s recording: 1.5 s are there
        {"at_s": 2.0, "duration_s": 3.0, "start_s": 2.5, "job_id": "other"},
    ]}])
    c = compose_graph(
        t, source=Path("own.mp4"), width=640, height=360, fps=30,
        source_duration_s=600,
        recordings={"other": Recording(Path("other.mp4"), 4.0)},
    )
    own, other = [i for i in c.inputs if not i.lavfi]
    assert (own.path, own.seek, own.duration) == ("own.mp4", 1.0, 2.0)
    assert (other.path, other.seek) == ("other.mp4", 2.5)
    assert other.duration == pytest.approx(1.5)


def test_a_match_that_is_not_there_is_an_error_not_a_silent_gap():
    from owcore.compose import compose_graph
    from owcore.models import Timeline

    t = Timeline(**_red_then_blue("gone"))
    with pytest.raises(ValueError, match="gone"):
        compose_graph(t, source=Path("own.mp4"), width=640, height=360, fps=30)


# ── on the frames of a render and a preview ─────────────────────────────────


def test_the_render_shows_both_matches_in_order(isolated, tmp_path):
    red, blue = _two_matches(tmp_path)
    resp = api().post(
        f"/api/jobs/{red}/renders",
        data={"timelines": json.dumps([_red_then_blue(blue)])},
    )
    assert resp.status_code == 201, resp.text
    run_render()

    detail = api().get(f"/api/jobs/{red}").json()
    render = next(r for r in detail["renders"] if r["id"] == resp.json()["id"])
    assert render["status"] == RenderStatus.DONE, render["error"]
    clip = next(c for c in detail["clips"] if c["render_id"] == render["id"])
    video = tmp_path / "out.mp4"
    video.write_bytes(api().get(clip["video_url"]).content)

    assert _is_red(_rgb(video, 1.0)), _rgb(video, 1.0)
    assert _is_blue(_rgb(video, 3.0)), _rgb(video, 3.0)


def test_the_preview_shows_both_matches_in_order(isolated, tmp_path):
    from test_pipeline import request_preview, run_previews

    red, blue = _two_matches(tmp_path)
    preview = request_preview(red, _red_then_blue(blue), from_s=0.0, to_s=4.0)
    run_previews()

    done = api().get(f"/api/previews/{preview['id']}").json()
    assert done["status"] == RenderStatus.DONE, done["error"]
    video = tmp_path / "preview.mp4"
    video.write_bytes(api().get(done["video_url"]).content)
    assert _is_red(_rgb(video, 1.0)), _rgb(video, 1.0)
    assert _is_blue(_rgb(video, 3.0)), _rgb(video, 3.0)


# ── the gateway ─────────────────────────────────────────────────────────────


def test_a_montage_naming_an_unknown_match_is_refused(isolated, tmp_path):
    red, _ = _two_matches(tmp_path)
    resp = api().post(
        f"/api/jobs/{red}/renders",
        data={"timelines": json.dumps([_red_then_blue("nope")])},
    )
    assert resp.status_code == 422
    assert "nope" in resp.text


def test_a_match_used_by_another_montage_is_not_deleted(isolated, tmp_path):
    red, blue = _two_matches(tmp_path)
    client = api()
    resp = client.post(
        f"/api/jobs/{red}/montages", json={"data": _red_then_blue(blue)}
    )
    assert resp.status_code == 201, resp.text
    montage_id = resp.json()["id"]

    refused = client.delete(f"/api/jobs/{blue}")
    assert refused.status_code == 409
    assert "red.mp4" in refused.text, "says which match uses it"
    assert client.get(f"/api/jobs/{blue}").status_code == 200

    # taken out of the montage, it can go
    only_red = {"title": "Red", "layers": [{"clips": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0}]}]}
    saved = client.put(
        f"/api/jobs/{red}/montages/{montage_id}", json={"data": only_red}
    )
    assert saved.status_code == 200, saved.text
    assert client.delete(f"/api/jobs/{blue}").status_code == 204
    # and the match that used it is untouched
    assert client.delete(f"/api/jobs/{red}").status_code == 204
