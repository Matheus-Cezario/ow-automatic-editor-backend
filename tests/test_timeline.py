"""The montage the **user** makes: blocks placed by hand on the music.

Two layers, as in the rest of the project:

* the timeline maths (`owcore.timeline`), with no ffmpeg and no database --
  this is where the screen's central promise is checked: a block comes out
  exactly where it was placed, whatever it costs its neighbours;
* the whole path through the microservices, from the music upload to the mp4
  -- this is where gateway, rhythm and editor are checked to agree on the
  format.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import MUSIC, service_module
from owcore.db import session
from owcore.models import (
    MIN_CUT_S,
    STREAM_THUMBS,
    STREAM_RENDER_READY,
    STREAM_MEDIA,
    Job,
    Timeline,
    TimelineCut,
)
from owcore.timeline import plan, snap, total_duration_s
from test_pipeline import api, drain, run_analysis

# ── the maths, on its own ───────────────────────────────────────────────────


def cut(at: float, dur: float, start: float = 10.0, **kw) -> TimelineCut:
    return TimelineCut(at_s=at, duration_s=dur, start_s=start, **kw)


def test_adjacent_blocks_become_cuts_only():
    pieces = plan([cut(0, 2, start=5), cut(2, 1.5, start=30)])

    assert [p.is_cut for p in pieces] == [True, True]
    assert [(p.start_s, p.end_s) for p in pieces] == [(5.0, 7.0), (30.0, 31.5)]
    assert total_duration_s(pieces) == pytest.approx(3.5)


def test_a_gap_between_two_blocks_becomes_black():
    """The gap does NOT shorten the video.

    It is the screen's promise: the second block was placed at 5s of the music
    and must come out at 5s of the video. Splicing the blocks would save an
    encode and move the cut away from the beat the user fitted it to.
    """
    pieces = plan([cut(0, 2), cut(5, 1)])

    assert [p.black for p in pieces] == [False, True, False]
    assert pieces[1].duration_s == pytest.approx(3.0)
    assert total_duration_s(pieces) == pytest.approx(6.0)


def test_space_before_the_first_block_also_becomes_black():
    """Starting the video with the music alone is a legitimate choice."""
    pieces = plan([cut(4, 2)])

    assert pieces[0].black and pieces[0].duration_s == pytest.approx(4.0)
    assert total_duration_s(pieces) == pytest.approx(6.0)


def test_black_at_the_end_is_left_out():
    """The video ends at the last cut: nobody wants 8s of black at the end."""
    pieces = plan([cut(0, 2)])

    assert len(pieces) == 1 and pieces[0].is_cut


def test_a_cut_past_the_end_of_the_recording_is_trimmed_without_moving_the_others():
    pieces = plan(
        [cut(0, 3, start=59), cut(4, 1, start=1)], source_duration_s=60
    )

    assert pieces[0].is_cut and pieces[0].duration_s == pytest.approx(1.0)
    # the 2s trimmed become black, not an advance of the next block
    assert pieces[1].black and pieces[1].duration_s == pytest.approx(3.0)
    assert total_duration_s(pieces[:2]) == pytest.approx(4.0)


def test_a_gap_shorter_than_a_frame_is_not_a_piece():
    """Splicing 20ms would cost a whole encode for nobody to see anything."""
    pieces = plan([cut(0, 2), cut(2.02, 1)])

    assert [p.is_cut for p in pieces] == [True, True]


def test_consecutive_blacks_become_one():
    """Each piece costs an encode; two blacks in a row are a waste."""
    pieces = plan(
        [cut(0, 3, start=59), cut(5, 1, start=1)], source_duration_s=60
    )

    assert [p.black for p in pieces] == [False, True, False]


def test_the_magnet_snaps_to_a_near_beat_and_ignores_a_far_one():
    beats = [0.0, 0.5, 1.0, 1.5]

    assert snap(0.52, beats) == 0.5
    assert snap(0.75, beats) == 0.75  # equidistant from both, too far


def test_the_timeline_sorts_and_refuses_overlap():
    spec = Timeline(cuts=[cut(4, 1), cut(0, 2)])
    assert [c.at_s for c in spec.cuts] == [0.0, 4.0]
    assert spec.duration_s == pytest.approx(5.0)

    with pytest.raises(ValueError, match="overlap"):
        Timeline(cuts=[cut(0, 2), cut(1, 1)])

    with pytest.raises(ValueError):
        Timeline(cuts=[cut(0, MIN_CUT_S / 2)])


# ── the whole path ──────────────────────────────────────────────────────────


def upload_music(job_id: str, music: Path = MUSIC) -> str:
    """Sends the music and runs the worker that listens to it, as happens in
    production."""
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("music.wav", music.read_bytes(), "audio/wav")},
    )
    assert resp.status_code == 201, resp.text
    track_id = resp.json()["id"]
    assert resp.json()["status"] == "pending"

    analyzer = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
        analyzer.handle(payload)
    return track_id


def render_montage(job_id: str, timelines: list[dict]) -> str:
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps(timelines)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def run_render() -> None:
    editor = service_module("editor", "main").Editor()
    for payload in drain(STREAM_RENDER_READY, "editor"):
        editor.handle(payload)


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_music_is_uploaded_before_any_video_and_comes_back_ready_to_draw(
    isolated, short_sample
):
    """The app needs the *analysed* music to draw the montage screen."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    track = api().get(f"/api/tracks/{track_id}").json()
    assert track["status"] == "ready", track["error"]
    assert track["duration_s"] > 5
    assert track["bpm"] > 0
    assert len(track["beats"]) > 4, "without beats no cut can snap anywhere"
    assert len(track["peaks"]) > 100, "without a waveform the chorus cannot be found"
    assert all(0.0 <= v <= 1.0 for v in track["peaks"])
    # the canonical URL is now the library's; the music is one of its items
    assert track["audio_url"].endswith(f"/api/media/{track_id}/file")
    assert track["kind"] == "audio"
    # and the old route still answers, because the app still uses it
    assert api().get(f"/api/tracks/{track_id}/audio",
                     headers={"range": "bytes=0-31"}).status_code == 206

    # and it shows up in the job, so the app does not have to keep any id
    detail = api().get(f"/api/jobs/{job_id}").json()
    assert [t["id"] for t in detail["tracks"]] == [track_id]


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_the_music_audio_is_served_with_range_for_the_player(isolated, short_sample):
    """Without Range the player cannot jump to the chorus."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    resp = api().get(f"/api/tracks/{track_id}/audio", headers={"range": "bytes=0-99"})
    assert resp.status_code == 206
    assert len(resp.content) == 100
    assert resp.headers["content-range"].startswith("bytes 0-99/")


def test_the_recording_is_served_with_range_for_the_preview(isolated, short_sample):
    """The montage screen's monitor seeks inside the original recording.

    Without `Range` it would have to download the whole match to show a frame
    at 3 minutes -- and really rendering on every adjustment would cost a trip
    through ffmpeg per drag.
    """
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert detail["video_url"] == f"/api/jobs/{job_id}/video"

    resp = api().get(f"/api/jobs/{job_id}/video", headers={"range": "bytes=0-511"})
    assert resp.status_code == 206
    assert len(resp.content) == 512
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers["accept-ranges"] == "bytes"

    whole = api().get(f"/api/jobs/{job_id}/video")
    assert whole.status_code == 200
    assert int(whole.headers["content-length"]) == short_sample.stat().st_size


def test_the_preview_of_a_missing_job_is_404(isolated):
    assert api().get("/api/jobs/doesnotexist/video").status_code == 404


# ── moment thumbnails ───────────────────────────────────────────────────────


def run_thumbs() -> int:
    worker = service_module("thumbs", "main").Thumbs()
    count = 0
    for payload in drain(STREAM_THUMBS, "thumbs"):
        worker.handle(payload)
        count += 1
    return count


def test_the_analysis_already_asks_for_the_moment_thumbnails(isolated, short_sample):
    """The editor's sidebar needs a picture to choose among thirty kills;
    without it, they are thirty identical clocks."""
    job_id = run_analysis(short_sample)
    assert run_thumbs() >= 1, "the planner did not ask for the thumbnails"

    detail = api().get(f"/api/jobs/{job_id}").json()
    moments = [
        e["t"] for e in detail["events"]
        if e["kind"] in {"kill", "sleep", "stun", "ult_negated", "escape"}
    ]
    assert moments, "the analysis found no moment"

    resp = api().get(f"/api/jobs/{job_id}/frame", params={"t": moments[0]})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 500, "the thumbnail came out empty"
    # an instant's frame never changes
    assert "max-age" in resp.headers.get("cache-control", "")


def test_a_moment_without_a_thumbnail_answers_404_instead_of_breaking(
    isolated, short_sample
):
    """404 here means 'not extracted yet' -- the app shows its placeholder."""
    job_id = run_analysis(short_sample)
    assert api().get(f"/api/jobs/{job_id}/frame", params={"t": 999}).status_code == 404


def test_asking_again_does_not_re_extract_what_already_exists(isolated, short_sample):
    """The app asks when opening the editor; the service must skip what is
    already there."""
    job_id = run_analysis(short_sample)
    run_thumbs()

    resp = api().post(f"/api/jobs/{job_id}/frames")
    assert resp.status_code == 202

    worker = service_module("thumbs", "main").Thumbs()
    # the second request finds nothing to extract, and says so without blowing up
    for payload in drain(STREAM_THUMBS, "thumbs"):
        worker.handle(payload)

    detail = api().get(f"/api/jobs/{job_id}").json()
    t = next(e["t"] for e in detail["events"] if e["kind"] == "kill")
    assert api().get(f"/api/jobs/{job_id}/frame", params={"t": t}).status_code == 200


def test_asking_thumbnails_for_a_missing_job_is_404(isolated):
    assert api().post("/api/jobs/doesnotexist/frames").status_code == 404


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_a_manual_montage_becomes_a_video_with_blocks_where_the_user_put_them(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    detail = api().get(f"/api/jobs/{job_id}").json()
    moments = [e["t"] for e in detail["events"] if e["kind"] == "kill"][:2]
    assert moments, "the analysis found no kill to assemble"

    # two blocks with a 1s gap between them: the video must last
    # 1.5 + 1.0 + 1.5 = 4s, not the 3s of the cuts
    cuts = [
        {"source_t": moments[0], "start_s": max(0.0, moments[0] - 1.0),
         "duration_s": 1.5, "at_s": 0.0, "kind": "kill"},
        {"source_t": moments[-1], "start_s": max(0.0, moments[-1] - 1.0),
         "duration_s": 1.5, "at_s": 2.5, "kind": "kill"},
    ]
    render_id = render_montage(
        job_id,
        [{"title": "My montage", "cuts": cuts}],
    )
    run_render()

    request = api().get(f"/api/renders/{render_id}").json()
    assert request["status"] == "done", request["error"]
    assert len(request["clips"]) == 1
    clip = request["clips"][0]

    assert clip["kind"] == "custom"
    assert clip["title"] == "My montage"
    assert clip["meta"]["hand_made"] is True
    assert clip["meta"]["segments"] == 2
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.0, abs=0.05)
    assert clip["video_url"], "the video did not come out"
    assert clip["segments_zip_url"], "the loose cuts did not come out"

    # the file itself: the gap is there, with the music playing over it
    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    local = local_copy(key, Path(isolated.work_dir) / "check")
    info = ffmpeg.probe(local)
    assert info.duration_s == pytest.approx(4.0, abs=0.35)
    assert info.has_audio


def test_without_music_a_manual_montage_keeps_the_match_audio(
    isolated, short_sample
):
    """And the gap's black must come out with *compatible* silence.

    Without a track the cuts keep the match audio, and `concat` refuses to
    join pieces whose audio does not match -- a stretch with sound and a mute
    black do not concatenate. The black comes out with silence at the same
    rate, and that is what this test guards: a gap in the middle of a montage
    without music.
    """
    job_id = run_analysis(short_sample)
    render_id = render_montage(
        job_id,
        [{"cuts": [
            {"start_s": 1.0, "duration_s": 1.5, "at_s": 0.0},
            {"start_s": 6.0, "duration_s": 1.5, "at_s": 3.0},
        ]}],
    )
    run_render()

    clip = api().get(f"/api/renders/{render_id}").json()["clips"][0]
    assert clip["meta"]["original_audio"] is True
    assert clip["meta"]["music_name"] is None
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.5, abs=0.05)
    assert clip["video_url"], "the montage with a gap and no music did not come out"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    info = ffmpeg.probe(local_copy(key, Path(isolated.work_dir) / "no_music"))
    assert info.duration_s == pytest.approx(4.5, abs=0.35)
    assert info.has_audio, "the match audio got lost in the splice with the black"


def test_an_empty_request_is_refused(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(f"/api/jobs/{job_id}/renders", data={"timelines": "[]"})
    assert resp.status_code == 422
    assert "timeline" in resp.json()["detail"]


def test_music_from_another_job_is_refused(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([{"layers": [
            {"clips": [{"start_s": 1, "duration_s": 1, "at_s": 0}]},
            {"kind": "audio", "clips": [
                {"at_s": 0, "duration_s": 1, "start_s": 0,
                 "source": "media", "media_id": "doesnotexist"},
            ]},
        ]}])},
    )
    assert resp.status_code == 422
    assert "unknown media" in resp.json()["detail"]


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_music_not_analysed_yet_blocks_the_request(isolated, short_sample):
    """Without beats or duration the screen could not have placed anything --
    a request like that can only be the app's mistake."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("music.wav", MUSIC.read_bytes(), "audio/wav")},
    )
    track_id = resp.json()["id"]  # on purpose: without running the analyzer

    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"track_id": track_id,
             "cuts": [{"start_s": 1, "duration_s": 1, "at_s": 0}]}
        ])},
    )
    assert resp.status_code == 409
    assert "has not been analysed yet" in resp.json()["detail"]


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_unreadable_music_fails_alone_without_bringing_the_job_down(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/tracks",
        files={"audio": ("broken.mp3", b"this is not audio", "audio/mpeg")},
    )
    track_id = resp.json()["id"]

    analyzer = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
        try:
            analyzer.handle(payload)
        except Exception as exc:  # the real worker turns this into on_error
            analyzer.on_error(payload, exc)

    track = api().get(f"/api/tracks/{track_id}").json()
    assert track["status"] == "failed"
    assert track["error"]
    # and the match analysis still stands
    assert api().get(f"/api/jobs/{job_id}").json()["status"] == "ready"


# ── the draft: the montage survives an F5 ───────────────────────────────────


def test_the_montage_in_progress_comes_back_with_the_job(isolated, short_sample):
    """Reloading the page used to cost the whole montage; now it belongs to the
    job."""
    job_id = run_analysis(short_sample)
    assert api().get(f"/api/jobs/{job_id}").json()["draft"] == {}

    draft = {
        "title": "Work in progress",
        "music_start_s": 12.5,
        "cuts": [
            {"source_t": 30.0, "start_s": 29.0, "duration_s": 1.5, "at_s": 0.0,
             "kind": "kill"},
            {"source_t": 75.0, "start_s": 74.0, "duration_s": 1.0, "at_s": 2.0,
             "kind": "sleep"},
        ],
    }
    resp = api().put(f"/api/jobs/{job_id}/draft", json=draft)
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_cuts"] == 2

    back = api().get(f"/api/jobs/{job_id}").json()["draft"]
    assert back["title"] == "Work in progress"
    assert back["music_start_s"] == 12.5
    assert [c["at_s"] for c in back["cuts"]] == [0.0, 2.0]
    assert back["cuts"][0]["kind"] == "kill"


def test_the_draft_remembers_the_beat_grid_corrections(
    isolated, short_sample
):
    """They do not change the video -- the cut stores absolute instants -- but
    they change where the magnet snaps. Fixing the grid twice is annoying."""
    job_id = run_analysis(short_sample)
    api().put(
        f"/api/jobs/{job_id}/draft",
        json={"cuts": [], "beat_offset_s": 0.12, "beat_multiplier": 2.0,
              "beat_bar": 4},
    )

    draft = api().get(f"/api/jobs/{job_id}").json()["draft"]
    assert draft["beat_offset_s"] == pytest.approx(0.12)
    assert draft["beat_multiplier"] == pytest.approx(2.0)
    assert draft["beat_bar"] == 4


def test_the_draft_remembers_the_mix_and_the_output_format(
    isolated, short_sample
):
    """Whoever lowered the game volume and picked 9:16 does not want to redo
    both after an F5. That is work like any other."""
    job_id = run_analysis(short_sample)
    api().put(
        f"/api/jobs/{job_id}/draft",
        json={
            "layers": [{"clips": [{"at_s": 0, "duration_s": 1, "start_s": 2}]}],
            "music_volume": 0.4,
            "game_volume": 0.8,
            "export": {"width": 1080, "height": 1920, "crf": 26,
                       "fit": "contain", "from_s": 1.0},
        },
    )

    draft = api().get(f"/api/jobs/{job_id}").json()["draft"]
    assert draft["music_volume"] == pytest.approx(0.4)
    assert draft["game_volume"] == pytest.approx(0.8)
    assert draft["export"]["width"] == 1080
    assert draft["export"]["height"] == 1920
    assert draft["export"]["crf"] == 26
    assert draft["export"]["fit"] == "contain"
    assert draft["export"]["from_s"] == pytest.approx(1.0)


def test_a_draft_with_an_impossible_output_is_refused(isolated, short_sample):
    """Storing garbage now means handing garbage back later."""
    job_id = run_analysis(short_sample)
    resp = api().put(
        f"/api/jobs/{job_id}/draft",
        json={"cuts": [], "export": {"width": 1080}},
    )
    assert resp.status_code == 422


def test_a_draft_without_any_cut_is_valid(isolated, short_sample):
    """A draft exists before the first block goes in -- saving just the chosen
    music must work."""
    job_id = run_analysis(short_sample)
    resp = api().put(
        f"/api/jobs/{job_id}/draft",
        json={"title": "just the music for now", "cuts": []},
    )
    assert resp.status_code == 200
    assert resp.json()["n_cuts"] == 0


def test_a_draft_with_an_impossible_cut_is_refused(isolated, short_sample):
    """Storing garbage now would mean handing garbage back on the next
    opening."""
    job_id = run_analysis(short_sample)
    resp = api().put(
        f"/api/jobs/{job_id}/draft",
        json={"cuts": [{"start_s": -5, "duration_s": 1, "at_s": 0}]},
    )
    assert resp.status_code == 422
    # the name changed in Phase 8: a draft is now one montage among several
    assert "invalid montage" in resp.json()["detail"]


def test_saving_again_replaces_the_previous_one(isolated, short_sample):
    job_id = run_analysis(short_sample)
    for n in (1, 2, 3):
        api().put(
            f"/api/jobs/{job_id}/draft",
            json={"cuts": [
                {"start_s": 1, "duration_s": 1, "at_s": float(i)}
                for i in range(n)
            ]},
        )
    assert len(api().get(f"/api/jobs/{job_id}").json()["draft"]["cuts"]) == 3


def test_discarding_the_draft(isolated, short_sample):
    job_id = run_analysis(short_sample)
    api().put(f"/api/jobs/{job_id}/draft",
              json={"cuts": [{"start_s": 1, "duration_s": 1, "at_s": 0}]})

    assert api().delete(f"/api/jobs/{job_id}/draft").status_code == 204
    assert api().get(f"/api/jobs/{job_id}").json()["draft"] == {}


def test_the_draft_of_a_missing_job_is_404(isolated):
    assert api().put("/api/jobs/doesnotexist/draft", json={"cuts": []}).status_code == 404
    assert api().delete("/api/jobs/doesnotexist/draft").status_code == 404


def test_rendering_does_not_delete_the_draft(isolated, short_sample):
    """After rendering, the normal thing is to want to adjust and render again
    -- losing the montage at that point would be the same damage as the F5."""
    job_id = run_analysis(short_sample)
    cuts = [{"source_t": 3.0, "start_s": 1.0, "duration_s": 1.5, "at_s": 0.0}]
    api().put(f"/api/jobs/{job_id}/draft", json={"title": "v1", "cuts": cuts})

    render_montage(job_id, [{"title": "v1", "cuts": cuts}])
    run_render()

    assert api().get(f"/api/jobs/{job_id}").json()["draft"]["title"] == "v1"


# ── the proxy and the match waveform (Phase 2) ──────────────────────────────


def test_the_proxy_comes_out_of_the_same_decode_and_is_much_smaller(
    isolated, short_sample
):
    """The reduced copy exists for the editor's monitor.

    Seeking inside the original recording dozens of times a second used to
    bring the browser's video element down. The proxy comes out as one more
    output of the decode that already happens, so its cost is close to zero --
    and that is what this test guards along with the size.
    """
    from owcore import ffmpeg
    from owcore.models import Job
    from owcore.storage import get_storage, local_copy

    job_id = run_analysis(short_sample)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.proxy_key, "the preprocessor did not generate the proxy"
        key = job.proxy_key
    storage = get_storage()
    assert storage.exists(key)

    proxy = local_copy(key, Path(isolated.work_dir) / "check")
    info = ffmpeg.probe(proxy)
    original = ffmpeg.probe(short_sample)

    # same match: the duration must match
    assert info.duration_s == pytest.approx(original.duration_s, abs=0.5)
    # and the whole screen, not a crop
    assert info.width / info.height == pytest.approx(
        original.width / original.height, abs=0.02
    )
    assert info.width <= 640
    assert proxy.stat().st_size < short_sample.stat().st_size


def test_the_job_says_the_recording_size(isolated, short_sample):
    """It is the export default -- and what lets the editor warn that the
    requested output will crop the frame."""
    from owcore import ffmpeg

    job_id = run_analysis(short_sample)
    expected = ffmpeg.probe(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert detail["width"] == expected.width
    assert detail["height"] == expected.height


def test_the_proxy_is_served_with_range(isolated, short_sample):
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert detail["proxy_url"] == f"/api/jobs/{job_id}/proxy"

    resp = api().get(f"/api/jobs/{job_id}/proxy", headers={"range": "bytes=0-511"})
    assert resp.status_code == 206
    assert len(resp.content) == 512
    assert resp.headers["content-type"] == "video/mp4"


def test_a_match_analysed_before_proxies_says_so_instead_of_breaking(isolated):
    """The app falls back to the original recording when `proxy_url` is null."""
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="old0000000000001", video_key="k", video_name="v.mp4"))

    assert api().get("/api/jobs/old0000000000001").json()["proxy_url"] is None
    assert api().get("/api/jobs/old0000000000001/proxy").status_code == 404


def test_an_old_match_is_measured_when_the_editor_opens(isolated, short_sample):
    """The new column is born empty on a match analysed before it existed, and
    the schema reconciler has no way of knowing what it should hold -- only the
    file knows. Without this the editor would open unable to say whether a 9:16
    crops its frame."""
    from owcore import ffmpeg
    from owcore.models import Job, JobStatus
    from owcore.storage import get_storage

    key = get_storage().put_file("old/video.mp4", short_sample)
    with session() as s:
        # `ready` because that is what an old match is: it was already
        # analysed, just by a version that did not have this column
        s.add(Job(id="old0000000000002", video_key=key, video_name="v.mp4",
                  status=JobStatus.READY))

    expected = ffmpeg.probe(short_sample)
    detail = api().get("/api/jobs/old0000000000002").json()
    assert detail["width"] == expected.width
    assert detail["height"] == expected.height

    # and the measurement is stored: the next GET does not pay another ffprobe
    with session() as s:
        assert s.get(Job, "old0000000000002").width == expected.width


def test_a_match_being_analysed_does_not_pay_an_ffprobe(isolated, short_sample):
    """While the analysis runs, the preprocessor will store the real size in
    seconds -- there is nothing to patch. And reading the header over the
    network right while it downloads the same file competes for the same
    bandwidth: measured, a 0.5s query went past 30s in that window, and the
    screen, which polls every two seconds, shows that as the server being
    down."""
    from owcore.models import Job, JobStatus
    from owcore.storage import get_storage

    key = get_storage().put_file("running/video.mp4", short_sample)
    with session() as s:
        s.add(Job(id="running000000001", video_key=key, video_name="v.mp4",
                  status=JobStatus.PREPROCESSING))

    detail = api().get("/api/jobs/running000000001").json()
    assert detail["width"] == 0, "measured a recording that is still being read"
    with session() as s:
        assert s.get(Job, "running000000001").width in (0, None)


def test_measuring_a_vanished_recording_does_not_bring_the_screen_down(isolated):
    """Better to open the editor without the size than not to open it."""
    from owcore.models import Job

    with session() as s:
        s.add(Job(id="old0000000000003", video_key="vanished.mp4", video_name="v"))

    detail = api().get("/api/jobs/old0000000000003").json()
    assert detail["width"] == 0


def test_the_match_waveform_comes_back_with_the_job(isolated, short_sample):
    """It is what shows the shot and the explosion on the editor's ruler."""
    job_id = run_analysis(short_sample)

    detail = api().get(f"/api/jobs/{job_id}").json()
    wave_ = detail["waveform"]

    assert len(wave_) > 100, "without a waveform the cut cannot be matched to the sound"
    assert all(0.0 <= v <= 1.0 for v in wave_)
    assert max(wave_) == pytest.approx(1.0), "the waveform is normalised by its peak"


def test_the_waveform_is_not_in_the_listing(isolated, short_sample):
    """It is a few thousand numbers per match, and the list does not use them."""
    run_analysis(short_sample)
    jobs = api().get("/api/jobs").json()["jobs"]

    assert jobs, "no match in the listing"
    assert "waveform" not in jobs[0]
    # the proxy, on the other hand, goes: the list is how the app decides what
    # to open
    assert "proxy_url" in jobs[0]


# ── layered montage (Phase 3) ───────────────────────────────────────────────


def test_the_v1_format_still_comes_in_and_goes_out_as_layers(isolated):
    """No migration runs on the database: the old format is valid input.

    A draft saved before this version, or a request stored in an old render,
    arrives with `cuts` and is converted on read -- a single layer of recording
    clips.
    """
    from owcore.models import ClipSource, Timeline

    old = Timeline(
        cuts=[
            {"start_s": 10, "duration_s": 2, "at_s": 0, "kind": "kill"},
            {"start_s": 30, "duration_s": 1, "at_s": 3},
        ]
    )

    assert len(old.layers) == 1
    assert [c.at_s for c in old.clips] == [0.0, 3.0]
    assert old.clips[0].source is ClipSource.RECORDING
    assert old.clips[0].kind == "kill"
    assert old.duration_s == pytest.approx(4.0)
    # and it still knows how to present itself as V1, for the old path
    assert [c.duration_s for c in old.cuts] == [2.0, 1.0]
    assert old.single_layer


def test_a_layer_or_transform_takes_the_montage_off_the_old_path(isolated):
    """The choice of path is what protects the render.

    Cut-and-splice is more resilient -- a bad cut costs only itself -- so it
    keeps the common case. The graph only comes in when needed.
    """
    from owcore.models import Layer, Timeline, TimelineClip

    simple = Timeline(layers=[Layer(clips=[TimelineClip(at_s=0, duration_s=1)])])
    assert simple.single_layer

    two = Timeline(
        layers=[
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
        ]
    )
    assert not two.single_layer

    scaled = Timeline(
        layers=[
            Layer(clips=[
                TimelineClip(at_s=0, duration_s=1, transform={"scale": 1.5})
            ])
        ]
    )
    assert not scaled.single_layer

    # a hidden layer does not count: one is left, and it is simple
    with_hidden = Timeline(
        layers=[
            Layer(clips=[TimelineClip(at_s=0, duration_s=1)]),
            Layer(hidden=True, clips=[TimelineClip(at_s=0, duration_s=1)]),
        ]
    )
    assert with_hidden.single_layer


def test_two_layers_become_one_video_with_the_upper_on_top(
    isolated, short_sample
):
    """The new path, from the request to the mp4."""
    job_id = run_analysis(short_sample)

    layers = [
        {"clips": [
            {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0, "kind": "kill"},
            {"at_s": 3.0, "duration_s": 1.5, "start_s": 6.0},
        ]},
        {"name": "corner", "clips": [
            {"at_s": 0.5, "duration_s": 1.5, "start_s": 9.0,
             "transform": {"scale": 0.35, "x": 0.6, "y": -0.6, "opacity": 0.9}},
        ]},
    ]
    render_id = render_montage(job_id, [{"title": "Layered", "layers": layers}])
    run_render()

    request = api().get(f"/api/renders/{render_id}").json()
    assert request["status"] == "done", request["error"]
    clip = request["clips"][0]

    assert clip["meta"]["composed"] is True
    assert clip["meta"]["layers"] == 2
    assert clip["meta"]["segments"] == 3
    assert clip["video_url"], "the video did not come out"
    # the request knows how to count itself
    assert request["timelines"][0]["n_layers"] == 2
    assert request["timelines"][0]["n_cuts"] == 3

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    output = local_copy(key, Path(isolated.work_dir) / "layers")
    info = ffmpeg.probe(output)
    original = ffmpeg.probe(short_sample)

    # 0 -> 4.5s: the last clip ends at 4.5
    assert info.duration_s == pytest.approx(4.5, abs=0.35)
    # the frame is the recording's: overlaying does not change it
    assert (info.width, info.height) == (original.width, original.height)


def test_a_non_default_output_takes_the_montage_off_the_old_path(isolated):
    """Cut-and-splice cannot change the aspect ratio or add a watermark: those
    only exist in the filter graph. That is how a 9:16 request came out 16:9
    without complaining about anything."""
    from owcore.models import Layer, Timeline, TimelineClip

    def montage(**export):
        return Timeline(
            export=export,
            layers=[Layer(clips=[TimelineClip(at_s=0, duration_s=2, start_s=1)])],
        )

    assert montage().single_layer, "with nothing requested, the old path"
    assert not montage(width=1080, height=1920).single_layer
    assert not montage(from_s=1.0).single_layer
    assert not montage(watermark_id="m1").single_layer
    assert not montage(crf=30).single_layer
    assert not montage(fps=24).single_layer


def test_a_single_layer_montage_still_goes_the_old_path(
    isolated, short_sample
):
    """It is the path that survives a bad cut, so it keeps the common case."""
    job_id = run_analysis(short_sample)
    render_id = render_montage(
        job_id,
        [{"layers": [{"clips": [
            {"at_s": 0.0, "duration_s": 1.5, "start_s": 1.0},
            {"at_s": 3.0, "duration_s": 1.5, "start_s": 6.0},
        ]}]}],
    )
    run_render()

    clip = api().get(f"/api/renders/{render_id}").json()["clips"][0]
    assert "composed" not in clip["meta"]
    # and the cuts zip, which only the old path produces, still comes
    assert clip["segments_zip_url"]
    assert clip["meta"]["blackfill_s"] == pytest.approx(1.5, abs=0.05)


def test_a_clip_outside_the_recording_becomes_background_without_moving_the_others(isolated):
    """The same promise as V1, now in the graph."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=59),   # only 1s exists
        TimelineClip(at_s=4, duration_s=1, start_s=1),
    ])])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
               source_duration_s=60)

    # the first comes in trimmed to 1s, and the second still comes in at 4s
    assert "trim=duration=1.000" in c.filter_complex
    assert "between(t,4.000,5.000)" in c.filter_complex
    assert c.duration_s == pytest.approx(5.0)


def test_a_source_that_cannot_be_rendered_yet_is_refused(isolated):
    """Ignoring it silently would be worse than not accepting it."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=1, source="color", fill="black"),
    ])])
    with pytest.raises(ValueError, match="cannot be rendered yet"):
        compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30)


def test_a_muted_layer_comes_in_without_sound(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[
        Layer(clips=[TimelineClip(at_s=0, duration_s=1, start_s=1)]),
        Layer(muted=True, clips=[TimelineClip(at_s=0, duration_s=1, start_s=5)]),
    ])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30)

    # two videos, a single audio
    assert c.filter_complex.count("overlay=") == 2
    assert "amix" not in c.filter_complex
    assert c.audio_map == "[aout]"


# ── the media library (Phase 4) ─────────────────────────────────────────────


def upload_media(job_id: str, name: str, data: bytes) -> str:
    """Brings a file in and runs the worker that analyses it."""
    resp = api().post(
        f"/api/jobs/{job_id}/media", files={"file": (name, data, "application/octet-stream")}
    )
    assert resp.status_code == 201, resp.text
    media_id = resp.json()["id"]

    worker = service_module("beats", "main").MediaAnalyzer()
    for payload in drain(STREAM_MEDIA, "media"):
        worker.handle(payload)
    return media_id


def make_png(dest: Path, colour: str = "red") -> Path:
    """Any image, made by ffmpeg itself."""
    from owcore.config import get_settings

    subprocess.run(
        [get_settings().ffmpeg, "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={colour}:s=320x180", "-frames:v", "1", str(dest)],
        check=True,
    )
    return dest



def test_music_became_a_library_item(isolated, short_sample):
    """Generalising `Track` cost one column and avoided a second upload system
    living next to the first."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert [m["id"] for m in detail["media"]] == [track_id]
    assert detail["media"][0]["kind"] == "audio"
    # and it still shows up as music, which is what the track picker uses
    assert [t["id"] for t in detail["tracks"]] == [track_id]


def test_an_imported_video_gets_dimensions_thumbnail_and_proxy(
    isolated, short_sample
):
    job_id = run_analysis(short_sample)
    media_id = upload_media(job_id, "clip.mp4", short_sample.read_bytes())

    item = api().get(f"/api/media/{media_id}").json()
    assert item["status"] == "ready", item["error"]
    assert item["kind"] == "video"
    assert item["width"] > 0 and item["height"] > 0
    assert item["fps"] > 0
    assert item["duration_s"] > 5
    assert item["thumb_url"], "without a thumbnail it cannot be picked in the list"
    assert item["proxy_url"], "without a proxy the monitor would drag the full file"

    # and both are served
    assert api().get(item["thumb_url"]).status_code == 200
    assert api().get(item["proxy_url"]).status_code == 200


def test_an_image_gets_dimensions_and_thumbnail_but_no_duration(
    isolated, short_sample, tmp_path
):
    """How long an image stays on screen is the montage's choice, not a
    property of the file."""
    job_id = run_analysis(short_sample)
    png = make_png(tmp_path / "logo.png")
    media_id = upload_media(job_id, "logo.png", png.read_bytes())

    item = api().get(f"/api/media/{media_id}").json()
    assert item["status"] == "ready", item["error"]
    assert item["kind"] == "image"
    assert (item["width"], item["height"]) == (320, 180)
    assert item["duration_s"] == 0
    assert item["thumb_url"]
    assert item["proxy_url"] is None, "an image needs no proxy"


def test_a_file_of_unknown_kind_is_refused(isolated, short_sample):
    """Accepting it and failing later would be worse than saying no now."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/media",
        files={"file": ("sheet.xlsx", b"not media", "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert "don't know what to do" in resp.json()["detail"]


def test_an_image_goes_into_the_montage_like_any_clip(
    isolated, short_sample, tmp_path
):
    """The whole path: import, assemble on top and check the mp4."""
    job_id = run_analysis(short_sample)
    png = make_png(tmp_path / "badge.png", colour="blue")
    media_id = upload_media(job_id, "badge.png", png.read_bytes())

    layers = [
        {"clips": [{"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0}]},
        {"name": "badge", "clips": [
            {"at_s": 0.5, "duration_s": 1.0, "source": "media",
             "media_id": media_id,
             "transform": {"scale": 0.5, "x": 0.5, "y": -0.5}},
        ]},
    ]
    render_id = render_montage(job_id, [{"title": "With badge", "layers": layers}])
    run_render()

    request = api().get(f"/api/renders/{render_id}").json()
    assert request["status"] == "done", request["error"]
    clip = request["clips"][0]
    assert clip["meta"]["media"] == 1
    assert clip["video_url"], "the video did not come out"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    output = local_copy(key, Path(isolated.work_dir) / "with_badge")
    assert ffmpeg.probe(output).duration_s == pytest.approx(2.0, abs=0.35)


def test_media_from_another_job_is_refused_in_the_request(isolated, short_sample):
    """The montage would come out without it, and with no warning."""
    job_id = run_analysis(short_sample)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"layers": [{"clips": [
                {"at_s": 0, "duration_s": 1, "source": "media",
                 "media_id": "doesnotexist"},
            ]}]}
        ])},
    )
    assert resp.status_code == 422
    assert "unknown media" in resp.json()["detail"]


def test_removing_from_the_library(isolated, short_sample, tmp_path):
    job_id = run_analysis(short_sample)
    png = make_png(tmp_path / "x.png")
    media_id = upload_media(job_id, "x.png", png.read_bytes())

    assert api().delete(f"/api/media/{media_id}").status_code == 204
    assert api().get(f"/api/media/{media_id}").status_code == 404
    assert api().get(f"/api/jobs/{job_id}").json()["media"] == []


# ── effects (Phase 5) ───────────────────────────────────────────────────────


def test_speed_changes_how_much_source_the_clip_consumes(isolated):
    """Not its duration in the video -- that is what the user drags."""
    from owcore.models import TimelineClip

    slow = TimelineClip(at_s=0, duration_s=2, start_s=10, speed=0.5)
    fast = TimelineClip(at_s=0, duration_s=2, start_s=10, speed=2.0)

    assert slow.source_consumed_s == pytest.approx(1.0)
    assert fast.source_consumed_s == pytest.approx(4.0)
    # and where it ends in the recording changes along
    assert slow.end_s == pytest.approx(11.0)
    assert fast.end_s == pytest.approx(14.0)
    # but both take the same 2s of the video
    assert slow.until_s == fast.until_s == pytest.approx(2.0)


def test_the_graph_speeds_picture_and_sound_up_together(isolated):
    """Picture and sound out of step is worse than having no sound."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=10, speed=0.4),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    # 2s of video at 0.4x consume 0.8s of recording
    assert "trim=duration=0.800" in g
    assert "setpts=PTS/0.4000" in g
    # `atempo` only accepts 0.5 and up, so 0.4 becomes 0.5 x 0.8
    assert "atempo=0.5" in g and "atempo=0.8000" in g


def test_the_filter_order_puts_the_fade_on_the_videos_clock(isolated):
    """A half-second fade lasts half a second in the video, not in the source."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, speed=2.0,
                     fade={"in_s": 0.5, "out_s": 0.5}),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    # speed comes before the fade: it changes the clip's clock
    assert g.index("setpts=PTS/2.0000") < g.index("fade=t=in")
    # and the fade out starts counting the duration in the *video*
    assert "fade=t=out:st=1.500:d=0.500" in g


def test_colour_is_applied_and_neutral_does_not_pollute_the_graph(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    def graph(**kw):
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1, start_s=1, **kw),
        ])])
        return compose_graph(t, source=Path("x.mp4"), width=640, height=360,
                      fps=30).filter_complex

    assert "eq=" not in graph()
    assert "saturation=1.4000" in graph(color={"saturation": 1.4})


def test_the_music_lets_the_game_show_through_underneath(isolated):
    """With `game_volume` at 0 it replaces, as in V1; above that, it mixes."""
    from owcore.compose import compose_graph

    def graph(**kw):
        return compose_graph(
            _with_music_on_the_ruler(**kw),
            source=Path("x.mp4"), width=640, height=360, fps=30,
            source_duration_s=600, library=_audio_library(Path("m.mp3")),
        ).filter_complex

    # the default is still V1's: the music rules alone
    assert "[game]" not in graph()
    mixed = graph(game_volume=0.5, music_volume=0.8)
    assert "volume=0.8000[music]" in mixed
    assert "volume=0.5000[game]" in mixed
    assert "[music][game]amix" in mixed


def test_with_no_music_at_all_the_cuts_sound_stands_on_its_own(isolated):
    """Neither `music_volume` nor `game_volume` has anything to do here: there
    are not two things to balance."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(game_volume=0.5, layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=1, start_s=1),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360,
               fps=30).filter_complex

    assert "[game]" not in g
    assert "volume=0.5000" not in g
    assert "[a1]anull[aout]" in g


def test_an_effect_takes_the_montage_off_the_cut_and_splice_path(isolated):
    """Cut-and-splice cannot do any of this."""
    from owcore.models import Layer, Timeline, TimelineClip

    def single_layer(**kw):
        return Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1, start_s=1, **kw),
        ])]).single_layer

    assert single_layer()
    assert not single_layer(speed=2.0)
    assert not single_layer(fade={"in_s": 0.2})
    assert not single_layer(color={"contrast": 1.2})


def test_absurd_effects_are_refused(isolated):
    """Storing garbage now would be a broken render later."""
    from owcore.models import TimelineClip

    with pytest.raises(ValueError, match="speed"):
        TimelineClip(at_s=0, duration_s=1, start_s=0, speed=50)
    with pytest.raises(ValueError, match="fades together"):
        TimelineClip(at_s=0, duration_s=1, start_s=0,
                     fade={"in_s": 0.7, "out_s": 0.7})
    with pytest.raises(ValueError, match="saturation"):
        TimelineClip(at_s=0, duration_s=1, start_s=0,
                     color={"saturation": 9})


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_slow_motion_and_fades_become_a_real_video(isolated, short_sample):
    """From the request to the mp4, with the clock checked."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    layers = [{"clips": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0, "speed": 0.5,
         "fade": {"in_s": 0.4}, "color": {"saturation": 1.3}},
        {"at_s": 2.0, "duration_s": 1.5, "start_s": 6.0, "speed": 2.0,
         "fade": {"out_s": 0.5}},
    ]}]
    render_id = render_montage(job_id, [{
        "title": "With effects", "track_id": track_id,
        "music_volume": 0.9, "game_volume": 0.3, "layers": layers,
    }])
    run_render()

    request = api().get(f"/api/renders/{render_id}").json()
    assert request["status"] == "done", request["error"]
    clip = request["clips"][0]
    assert clip["meta"]["composed"] is True
    assert clip["video_url"], "the video did not come out"

    from owcore import ffmpeg
    from owcore.storage import local_copy

    with session() as s:
        key = next(c.key for c in s.get(Job, job_id).clips)
    output = local_copy(key, Path(isolated.work_dir) / "effects")
    info = ffmpeg.probe(output)

    # 2s + 1.5s: speed changes what is consumed from the source, not what is
    # seen
    assert info.duration_s == pytest.approx(3.5, abs=0.35)
    assert info.has_audio


# ── keyframes, freeze, reverse (Phase 5, second half) ───────────────────────


def raw_frame(video: Path, t: float) -> "np.ndarray":
    """A small RGB frame, straight from ffmpeg."""
    import numpy as np

    from owcore.config import get_settings

    out = subprocess.run(
        [get_settings().ffmpeg, "-v", "error", "-ss", f"{t:.3f}",
         "-i", str(video), "-frames:v", "1", "-vf", "scale=80:45",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).astype(float)


def compose_and_render(timeline, source: Path, dest: Path) -> Path:
    from owcore import ffmpeg
    from owcore.compose import compose_graph

    info = ffmpeg.probe(source)
    c = compose_graph(timeline, source=source, width=info.width, height=info.height,
               fps=info.fps, source_duration_s=info.duration_s)
    ffmpeg.compose(c, dest)
    return dest


def test_the_fade_reveals_the_layer_below_instead_of_painting_black(
    isolated, short_sample, tmp_path
):
    """ffmpeg's `fade` paints black; on an upper layer that is a dark smear
    over what should show. With `alpha=1` it reveals.

    Over the black background both come out the same -- which is why the bug
    went unnoticed in the first half of the phase.
    """
    from owcore.models import Layer, Timeline, TimelineClip

    lower = TimelineClip(at_s=0, duration_s=2, start_s=1)
    upper = TimelineClip(at_s=0, duration_s=2, start_s=6, fade={"out_s": 1.0})

    together = compose_and_render(
        Timeline(layers=[Layer(clips=[lower]), Layer(clips=[upper])]),
        short_sample, tmp_path / "together.mp4",
    )
    lower_only = compose_and_render(
        Timeline(layers=[Layer(clips=[lower])]),
        short_sample, tmp_path / "lower.mp4",
    )

    # at the end of the fade, the composite must be the lower layer
    a = raw_frame(together, 1.95)
    b = raw_frame(lower_only, 1.95)
    assert a.size > 0 and a.size == b.size
    assert abs(a - b).mean() < 12, "the fade painted black instead of revealing"


def test_the_graph_fades_the_alpha(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, fade={"in_s": 0.5}),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    assert "fade=t=in:st=0:d=0.500:alpha=1" in g
    # without rgba the alpha does not exist, and the filter would have nothing
    # to work on
    assert g.index("format=rgba") < g.index("fade=t=in")


def test_zoom_interpolates_between_keyframes(isolated):
    """`zoompan` animates it, with expressions in the frame's time (`it`).

    It used to be `crop`, which computes width and height only once -- and so
    the lens never moved."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1,
                     zoom=[{"t": 0, "scale": 1}, {"t": 0.5, "scale": 2}]),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex

    assert "zoompan=z=" in g
    assert "crop=w=" not in g
    # the 0.5 fraction of a 2s clip is second 1
    assert "lt(it,1.0000)" in g
    # and it comes out at the canvas size and frame rate
    assert ":s=640x360:fps=30.000" in g


def test_keyframes_are_fractions_and_follow_the_block(isolated):
    """A zoom that closes at the end still closes at the end after stretching."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    def graph(duration: float) -> str:
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=duration, start_s=1,
                         zoom=[{"t": 0, "scale": 1}, {"t": 1.0, "scale": 2}]),
        ])])
        return compose_graph(t, source=Path("x.mp4"), width=640, height=360,
                      fps=30).filter_complex

    assert "lt(it,2.0000)" in graph(2.0)
    assert "lt(it,5.0000)" in graph(5.0)


def test_freezing_consumes_a_single_frame_of_the_recording(isolated):
    from owcore.models import TimelineClip

    c = TimelineClip(at_s=0, duration_s=3, start_s=10, freeze=True)

    assert c.source_consumed_s < 0.2, "a still frame does not consume three seconds"
    assert c.until_s == pytest.approx(3.0), "but it takes all three in the video"


def test_freezing_and_reversing_become_video(isolated, short_sample, tmp_path):
    from owcore import ffmpeg
    from owcore.models import Layer, Timeline, TimelineClip

    for name, kw in [("frozen", {"freeze": True}),
                     ("reversed", {"reverse": True})]:
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=1.5, start_s=3, **kw),
        ])])
        output = compose_and_render(t, short_sample, tmp_path / f"{name}.mp4")
        assert ffmpeg.probe(output).duration_s == pytest.approx(1.5, abs=0.35)


def test_a_frozen_frame_has_no_running_sound(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, freeze=True),
    ])])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30)

    assert c.audio_map is None


def test_an_animated_zoom_really_zooms_in(isolated, short_sample, tmp_path):
    """It is not enough for the graph to be right: the picture must really
    zoom in.

    The clip is **frozen** on purpose: with the content still, the only thing
    that changes between one instant and another is the lens. Comparing two
    videos whose content also runs in time would say nothing.
    """
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=3, freeze=True,
                     zoom=[{"t": 0, "scale": 1}, {"t": 1, "scale": 3}]),
    ])])
    video = compose_and_render(t, short_sample, tmp_path / "zoom.mp4")

    start = raw_frame(video, 0.1)
    end = raw_frame(video, 1.8)

    assert start.size > 0 and start.size == end.size
    # same picture, different lenses: the frames must be clearly different
    assert abs(start - end).mean() > 10, "the lens did not move"


def test_a_frozen_clip_shows_its_picture_from_start_to_end(
    isolated, short_sample, tmp_path
):
    """In ffmpeg 7, `tpad` read the frame rate that `setpts` had cleared, and a
    frozen clip came out as two frames followed by the black background."""
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=3, freeze=True),
    ])])
    video = compose_and_render(t, short_sample, tmp_path / "frozen.mp4")

    start, middle, end = (raw_frame(video, s) for s in (0.1, 1.0, 1.8))
    assert start.mean() > 20, "the frozen clip came out black"
    # and still: the same frame the whole time
    assert abs(start - middle).mean() < 2
    assert abs(start - end).mean() < 2


def test_zoom_also_animates_on_a_running_clip(
    isolated, short_sample, tmp_path
):
    """The same stretch with and without a lens: equal at the start, different
    at the end."""
    from owcore.models import Layer, Timeline, TimelineClip

    def render(name, **extra):
        t = Timeline(layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=2, start_s=3, **extra),
        ])])
        return compose_and_render(t, short_sample, tmp_path / f"{name}.mp4")

    plain = render("plain")
    zoomed = render("zoomed", zoom=[{"t": 0, "scale": 1}, {"t": 1, "scale": 3}])

    assert abs(raw_frame(zoomed, 0.05) - raw_frame(plain, 0.05)).mean() < 8
    assert abs(raw_frame(zoomed, 1.8) - raw_frame(plain, 1.8)).mean() > 10


# ── text (Phase 6) ──────────────────────────────────────────────────────────


def test_text_escapes_what_would_break_the_graph(isolated):
    """Colons and quotes show up in real text -- and each of them, loose,
    splits the filtergraph in two."""
    from owcore.textfx import escape

    assert escape("TRIPLE KILL: 50") == r"TRIPLE KILL\: 50"
    assert escape("a 'x'") == r"a \'x\'"
    assert escape("a\\b") == r"a\\b"
    # a raw line break would split the graph: the filtergraph is one line
    assert "\n" not in escape("two\nlines")


def test_the_percent_goes_through_whole_and_expansion_stays_off(isolated):
    """Escaping `%` with a backslash makes drawtext complain "Stray %" at
    warning level and **draw nothing** -- the text vanished from the video with
    no error at all.

    `expansion=none` is what solves it: without expansion, `%` is just a
    character.
    """
    from owcore.models import TimelineClip
    from owcore.textfx import escape, filter_chain

    assert escape("50%") == "50%"
    c = filter_chain(
        TimelineClip(at_s=0, duration_s=1, source="text", text="50% health"),
        height=720,
    )
    assert "expansion=none" in c
    assert "50% health" in c


def test_the_text_size_is_a_fraction_of_the_height(isolated):
    """The same montage must come out the same in 720p and in 4K."""
    from owcore.models import TimelineClip
    from owcore.textfx import filter_chain

    clip = TimelineClip(at_s=0, duration_s=1, source="text", text="hi",
                        text_style={"size": 0.1})

    assert "fontsize=72" in filter_chain(clip, 720)
    assert "fontsize=216" in filter_chain(clip, 2160)


def test_a_text_clip_needs_text(isolated):
    from owcore.models import TimelineClip

    with pytest.raises(ValueError, match="needs text"):
        TimelineClip(at_s=0, duration_s=1, source="text", text="   ")


def test_text_goes_onto_a_transparent_canvas(isolated):
    """If the canvas were black, the text would come inside a box."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=1, source="text", text="hi"),
    ])])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30)

    # the alpha must come from the **source**: requested later, in the chain,
    # `color` has already negotiated yuv420p with `drawtext` and drawn opaque
    # black -- and the alpha added there is born at 1, covering the layer below
    canvas = next(e for e in c.inputs if "color=c=black@0.0" in e.path)
    assert canvas.path.endswith(",format=rgba")
    # and a text has no sound running along
    assert c.audio_map is None


def test_text_shows_up_in_the_video_and_leaves_without_a_box(
    isolated, short_sample, tmp_path
):
    """What matters is not the graph: it is the frame."""
    from owcore.models import Layer, Timeline, TimelineClip

    base = TimelineClip(at_s=0, duration_s=2, start_s=1)
    text = TimelineClip(
        at_s=0.2, duration_s=1.2, source="text", text="TRIPLE KILL: 50%",
        text_style={"size": 0.14, "color": "yellow"}, transform={"y": -0.5},
    )

    with_text = compose_and_render(
        Timeline(layers=[Layer(clips=[base]), Layer(clips=[text])]),
        short_sample, tmp_path / "with.mp4",
    )
    without = compose_and_render(
        Timeline(layers=[Layer(clips=[base])]), short_sample, tmp_path / "without.mp4"
    )

    import numpy as np

    def halves(video, t):
        q = raw_frame(video, t).reshape(45, 80, 3)
        return q[:20, :, :], q[25:, :, :]

    top_with, bottom_with = halves(with_text, 0.8)
    top_without, bottom_without = halves(without, 0.8)

    # the text takes the top half (`y=-0.5`): there the frames change
    assert np.abs(top_with - top_without).mean() > 5
    # and the bottom stays **the same** -- the text canvas is transparent, and
    # the video keeps showing underneath it. Without this half, a black canvas
    # over everything would pass the test: it also "changes the frame"
    assert np.abs(bottom_with - bottom_without).mean() < 1

    # after the text, identical: it leaves no box behind
    assert abs(raw_frame(with_text, 1.9) - raw_frame(without, 1.9)).mean() < 5


def test_text_is_rendered_by_the_graph_and_not_the_old_path(isolated):
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=1, source="text", text="hi"),
    ])])
    assert not t.single_layer


def test_without_a_font_the_error_shows_up_at_the_right_time(isolated, monkeypatch):
    """Finding out there is no font in the middle of a render would be worse."""
    from owcore import fonts

    fonts.default_font.cache_clear()
    monkeypatch.setattr(fonts, "CANDIDATES", ())
    monkeypatch.setenv("OW_FONT", "")

    import owcore.config as config

    config.get_settings.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="OW_FONT"):
            fonts.default_font()
    finally:
        fonts.default_font.cache_clear()
        config.get_settings.cache_clear()


# ── export (Phase 7) ────────────────────────────────────────────────────────


def _simple_timeline(**export):
    from owcore.models import Layer, Timeline, TimelineClip

    return Timeline(
        export=export,
        layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=2, start_s=1),
            TimelineClip(at_s=2, duration_s=2, start_s=8),
        ])],
    )


def test_the_same_montage_comes_out_in_any_aspect_ratio(
    isolated, short_sample, tmp_path
):
    """What changes between 16:9 and 9:16 is not the montage: it is the window
    one looks through. Nothing about the clips needs to change."""
    from owcore import ffmpeg

    for name, exp, expected in [
        ("default", {}, (1280, 720)),
        ("vertical", {"width": 1080, "height": 1920}, (1080, 1920)),
        ("square", {"width": 720, "height": 720}, (720, 720)),
    ]:
        output = compose_and_render(
            _simple_timeline(**exp), short_sample, tmp_path / f"{name}.mp4"
        )
        info = ffmpeg.probe(output)
        assert (info.width, info.height) == expected, name
        assert info.duration_s == pytest.approx(4.0, abs=0.35), name


def test_cover_fills_and_contain_leaves_bars(
    isolated, short_sample, tmp_path
):
    """Both answers are legitimate, and give quite different pictures."""
    cover = compose_and_render(
        _simple_timeline(width=720, height=1280),
        short_sample, tmp_path / "cover.mp4",
    )
    contain = compose_and_render(
        _simple_timeline(width=720, height=1280, fit="contain"),
        short_sample, tmp_path / "contain.mp4",
    )

    a, b = raw_frame(cover, 1.0), raw_frame(contain, 1.0)
    assert abs(a - b).mean() > 15, "both framings came out the same"
    # `contain` has black bars: it is visibly darker overall
    assert b.mean() < a.mean()


def test_exporting_a_range_repositions_the_clips(isolated):
    """It is not cutting the finished video: the clips are repositioned as if
    the window were the beginning."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _simple_timeline(from_s=1.0, to_s=3.0),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600,
    )

    assert c.duration_s == pytest.approx(2.0)
    # of the two clips, both come in -- but each halfway
    assert c.filter_complex.count("overlay=") == 2
    assert "between(t,0.000,1.000)" in c.filter_complex
    assert "between(t,1.000,2.000)" in c.filter_complex


def test_a_clip_that_starts_before_the_window_comes_in_halfway(isolated):
    """And the entry point into the source moves along, or the picture would
    jump."""
    from owcore.compose import _within_window
    from owcore.models import TimelineClip

    clip = TimelineClip(at_s=0, duration_s=4, start_s=10)
    seen = _within_window(clip, 1.0, 3.0)

    assert seen is not None
    assert seen.at_s == 0.0, "it now starts at the first frame"
    assert seen.duration_s == pytest.approx(2.0)
    assert seen.start_s == pytest.approx(11.0), "it skipped 1s of the recording too"


def test_speed_counts_in_the_window_skip(isolated):
    from owcore.compose import _within_window
    from owcore.models import TimelineClip

    # at 2x, one skipped second of video costs two of recording
    clip = TimelineClip(at_s=0, duration_s=4, start_s=10, speed=2.0)
    seen = _within_window(clip, 1.0, 3.0)

    assert seen.start_s == pytest.approx(12.0)


def test_a_clip_outside_the_window_does_not_come_in(isolated):
    from owcore.compose import _within_window
    from owcore.models import TimelineClip

    clip = TimelineClip(at_s=10, duration_s=2, start_s=1)
    assert _within_window(clip, 0.0, 5.0) is None
    assert _within_window(TimelineClip(at_s=0, duration_s=1, start_s=1), 5.0, 9.0) is None


def test_an_empty_range_is_refused(isolated):
    from owcore.compose import compose_graph

    with pytest.raises(ValueError, match="empty"):
        compose_graph(_simple_timeline(from_s=50, to_s=60), source=Path("x.mp4"),
               width=640, height=360, fps=30, source_duration_s=600)


def test_the_watermark_goes_on_top_of_everything(isolated, short_sample, tmp_path):
    """A mark some layer covers is not a watermark."""
    from owcore.compose import LibraryFile, compose_graph
    from owcore import ffmpeg

    png = make_png(tmp_path / "mark.png", colour="white")
    t = _simple_timeline(watermark_id="m1", watermark_scale=0.3)
    info = ffmpeg.probe(short_sample)
    c = compose_graph(t, source=short_sample, width=info.width, height=info.height,
               fps=info.fps, source_duration_s=info.duration_s,
               library={"m1": LibraryFile(png, "image")})

    # the mark is the last overlay before the output: what comes out of it goes
    # straight to the final trim, with no layer on top
    filters = c.filter_complex
    assert "[mark]overlay" in filters
    assert "[watermarked]trim=" in filters

    with_mark = tmp_path / "with_mark.mp4"
    ffmpeg.compose(c, with_mark)
    without = compose_and_render(_simple_timeline(), short_sample, tmp_path / "without.mp4")
    assert abs(raw_frame(with_mark, 1.0) - raw_frame(without, 1.0)).mean() > 3


def test_a_watermark_not_in_the_library_is_refused(isolated):
    from owcore.compose import compose_graph

    with pytest.raises(ValueError, match="watermark"):
        compose_graph(_simple_timeline(watermark_id="gone"), source=Path("x.mp4"),
               width=640, height=360, fps=30, source_duration_s=600)


def test_the_requested_quality_reaches_the_file(isolated, short_sample, tmp_path):
    """A high CRF and a low resolution must give a visibly smaller file."""
    from owcore import ffmpeg

    full = compose_and_render(
        _simple_timeline(), short_sample, tmp_path / "full.mp4"
    )
    light = compose_and_render(
        _simple_timeline(width=854, height=480, fps=24, crf=32),
        short_sample, tmp_path / "light.mp4",
    )

    assert light.stat().st_size < full.stat().st_size / 3
    assert ffmpeg.probe(light).fps == pytest.approx(24, abs=1)


# ── reuse ───────────────────────────────────────────────────────────────────


def _montage(**kw):
    return {"layers": [{"clips": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 10.0},
    ]}], **kw}


def test_a_match_keeps_several_montages(isolated, short_sample):
    """The 30s cut for Shorts and the long montage are different jobs over the
    same material. Until Phase 8 one had to be chosen."""
    job_id = run_analysis(short_sample)

    short = api().post(f"/api/jobs/{job_id}/montages",
                       json={"name": "short vertical", "data": _montage()}).json()
    long_ = api().post(f"/api/jobs/{job_id}/montages",
                       json={"name": "the long one"}).json()

    items = api().get(f"/api/jobs/{job_id}/montages").json()["items"]
    assert {m["name"] for m in items} == {"short vertical", "the long one"}
    assert short["n_clips"] == 1
    assert short["duration_s"] == pytest.approx(2.0)
    assert long_["n_clips"] == 0, "a new montage starts empty"


def test_the_list_goes_from_most_recent_to_oldest(isolated, short_sample):
    """The one being edited is the one wanted back."""
    job_id = run_analysis(short_sample)
    first = api().post(f"/api/jobs/{job_id}/montages",
                       json={"name": "first"}).json()
    api().post(f"/api/jobs/{job_id}/montages", json={"name": "second"})
    api().put(f"/api/jobs/{job_id}/montages/{first['id']}",
              json={"data": _montage()})

    items = api().get(f"/api/jobs/{job_id}/montages").json()["items"]
    assert items[0]["name"] == "first"


def test_repeated_names_get_a_number(isolated, short_sample):
    """Two "Montage"s in a list to pick from are as good as no names."""
    job_id = run_analysis(short_sample)
    a = api().post(f"/api/jobs/{job_id}/montages", json={"name": "test"}).json()
    b = api().post(f"/api/jobs/{job_id}/montages", json={"name": "test"}).json()

    assert a["name"] == "test"
    assert b["name"] == "test 2"


def test_a_montage_without_a_name_gets_one(isolated, short_sample):
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages", json={}).json()
    assert m["name"] == "Montage 1"


def test_duplicating_to_experiment_without_risk(isolated, short_sample):
    job_id = run_analysis(short_sample)
    original = api().post(f"/api/jobs/{job_id}/montages",
                          json={"name": "good", "data": _montage()}).json()

    copy = api().post(
        f"/api/jobs/{job_id}/montages/{original['id']}/duplicate"
    ).json()

    assert copy["id"] != original["id"]
    assert copy["name"] == "good (copy)"
    assert copy["data"] == original["data"]

    # touching the copy does not touch the original
    api().put(f"/api/jobs/{job_id}/montages/{copy['id']}",
              json={"data": {"layers": []}})
    back = api().get(f"/api/jobs/{job_id}/montages").json()["items"]
    by_id = {m["id"]: m for m in back}
    assert by_id[original["id"]]["n_clips"] == 1
    assert by_id[copy["id"]]["n_clips"] == 0


def test_renaming_and_deleting(isolated, short_sample):
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages", json={"name": "old"}).json()

    api().put(f"/api/jobs/{job_id}/montages/{m['id']}", json={"name": "new"})
    assert api().get(f"/api/jobs/{job_id}/montages").json()["items"][0]["name"] == "new"

    assert api().delete(f"/api/jobs/{job_id}/montages/{m['id']}").status_code == 204
    assert api().get(f"/api/jobs/{job_id}/montages").json()["items"] == []


def test_a_montage_from_another_match_is_404(isolated, short_sample):
    """The id alone is not enough: the montage must belong to this match."""
    a = run_analysis(short_sample)
    b = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{a}/montages", json={"name": "x"}).json()

    assert api().get(f"/api/jobs/{b}/montages/{m['id']}/versions").status_code == 404
    assert api().delete(f"/api/jobs/{b}/montages/{m['id']}").status_code == 404


def test_an_invalid_montage_is_refused(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(f"/api/jobs/{job_id}/montages",
                      json={"data": {"layers": [{"clips": [
                          {"at_s": -5, "duration_s": 2, "start_s": 1}]}]}})
    assert resp.status_code == 422


def test_deleting_the_match_takes_the_montages(isolated, short_sample):
    job_id = run_analysis(short_sample)
    api().post(f"/api/jobs/{job_id}/montages", json={"data": _montage()})

    api().delete(f"/api/jobs/{job_id}")
    with session() as s:
        from owcore.models import Montage as MontageModel
        assert s.query(MontageModel).filter_by(job_id=job_id).count() == 0


# ── migrating the single draft ──────────────────────────────────────────────


def test_the_old_draft_becomes_the_first_montage(isolated, short_sample):
    """The code that reads is what knows how to convert the old format -- and
    that is why a match untouched for months still opens."""
    job_id = run_analysis(short_sample)
    api().put(f"/api/jobs/{job_id}/draft",
              json={"title": "what I was doing", "cuts": [
                  {"at_s": 0.0, "duration_s": 2.0, "start_s": 10.0}]})

    # simulates the state before Phase 8: everything in the job's column
    with session() as s:
        from owcore.models import Job, Montage as MontageModel
        job = s.get(Job, job_id)
        stored = job.montages[0].data
        for m in list(job.montages):
            s.delete(m)
        job.draft = stored

    items = api().get(f"/api/jobs/{job_id}/montages").json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "what I was doing"
    assert items[0]["n_clips"] == 1

    # and the column goes away, so there are not two truths about one montage
    with session() as s:
        from owcore.models import Job
        assert not s.get(Job, job_id).draft


def test_the_migration_does_not_repeat_the_montage(isolated, short_sample):
    """Reading twice must not create two."""
    job_id = run_analysis(short_sample)
    api().put(f"/api/jobs/{job_id}/draft", json={"cuts": [
        {"at_s": 0.0, "duration_s": 2.0, "start_s": 10.0}]})

    api().get(f"/api/jobs/{job_id}")
    api().get(f"/api/jobs/{job_id}")
    assert len(api().get(f"/api/jobs/{job_id}/montages").json()["items"]) == 1


def test_the_old_app_keeps_saving(isolated, short_sample):
    """`PUT /draft` writes to the most recent montage instead of silently
    losing the work."""
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages", json={"name": "current"}).json()

    api().put(f"/api/jobs/{job_id}/draft", json=_montage())

    items = api().get(f"/api/jobs/{job_id}/montages").json()["items"]
    assert len(items) == 1, "it created a second one"
    assert items[0]["id"] == m["id"]
    assert items[0]["n_clips"] == 1


def test_the_job_detail_brings_the_montages(isolated, short_sample):
    job_id = run_analysis(short_sample)
    api().post(f"/api/jobs/{job_id}/montages",
               json={"name": "one", "data": _montage()})

    detail = api().get(f"/api/jobs/{job_id}").json()
    assert [m["name"] for m in detail["montages"]] == ["one"]
    # and `draft` still answers with the most recent, for an older app
    assert detail["draft"]["layers"][0]["clips"][0]["at_s"] == 0.0


# ── version history ─────────────────────────────────────────────────────────


def test_marking_and_going_back_to_a_version(isolated, short_sample):
    """The "it was good yesterday" -- which is not undo: that one dies with the
    tab."""
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()
    base = f"/api/jobs/{job_id}/montages/{m['id']}"

    snapshot = api().post(f"{base}/versions", json={"label": "it was good"}).json()
    assert snapshot["n_clips"] == 1

    api().put(base, json={"data": {"layers": []}})
    assert api().get(f"/api/jobs/{job_id}/montages").json()["items"][0]["n_clips"] == 0

    restored = api().post(f"{base}/versions/{snapshot['id']}/restore").json()
    assert restored["n_clips"] == 1


def test_restoring_does_not_delete_what_was_in_front(isolated, short_sample):
    """Restoring swaps what is in front; it does not throw work away."""
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()
    base = f"/api/jobs/{job_id}/montages/{m['id']}"
    snapshot = api().post(f"{base}/versions", json={"label": "first"}).json()

    two = _montage()
    two["layers"][0]["clips"].append(
        {"at_s": 5.0, "duration_s": 2.0, "start_s": 20.0})
    api().put(base, json={"data": two})
    api().post(f"{base}/versions/{snapshot['id']}/restore")

    snapshots = api().get(f"{base}/versions").json()["items"]
    assert "before restoring" in [f["label"] for f in snapshots]
    kept = [f for f in snapshots if f["label"] == "before restoring"][0]
    assert kept["n_clips"] == 2, "the previous state was kept whole"


def test_marking_the_same_thing_twice_creates_no_version(isolated, short_sample):
    """A list of identical states helps nobody find yesterday's."""
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()
    base = f"/api/jobs/{job_id}/montages/{m['id']}"

    assert api().post(f"{base}/versions", json={}).status_code == 201
    assert api().post(f"{base}/versions", json={}).status_code == 409
    assert len(api().get(f"{base}/versions").json()["items"]) == 1


def test_the_history_stops_growing(isolated, short_sample):
    """Twenty markers is already more history than anyone scrolls through in a
    list."""
    from owcore.models import Montage as MontageModel, MontageVersion

    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()

    with session() as s:
        target = s.get(MontageModel, m["id"])
        for i in range(30):
            data = _montage()
            data["music_start_s"] = float(i)
            target.data = data
            target.versions.append(MontageVersion(label=f"n{i}", data=data))
            s.flush()

    snapshots = api().get(
        f"/api/jobs/{job_id}/montages/{m['id']}/versions"
    ).json()["items"]
    assert len(snapshots) == 30, "storing straight in the database skips pruning"

    # whereas the normal path prunes
    api().put(f"/api/jobs/{job_id}/montages/{m['id']}",
              json={"data": {"layers": [], "music_start_s": 99.0}})
    api().post(f"/api/jobs/{job_id}/montages/{m['id']}/versions", json={})
    snapshots = api().get(
        f"/api/jobs/{job_id}/montages/{m['id']}/versions"
    ).json()["items"]
    assert len(snapshots) == 20


def test_a_copy_does_not_take_the_history(isolated, short_sample):
    """The snapshots say where *that* montage has been; the copy has not been
    anywhere yet."""
    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()
    api().post(f"/api/jobs/{job_id}/montages/{m['id']}/versions", json={})

    copy = api().post(
        f"/api/jobs/{job_id}/montages/{m['id']}/duplicate"
    ).json()
    assert copy["n_versions"] == 0
    snapshots = api().get(
        f"/api/jobs/{job_id}/montages/{copy['id']}/versions"
    ).json()["items"]
    assert snapshots == []


def test_deleting_the_montage_takes_the_versions(isolated, short_sample):
    from owcore.models import MontageVersion

    job_id = run_analysis(short_sample)
    m = api().post(f"/api/jobs/{job_id}/montages",
                   json={"name": "x", "data": _montage()}).json()
    api().post(f"/api/jobs/{job_id}/montages/{m['id']}/versions", json={})

    api().delete(f"/api/jobs/{job_id}/montages/{m['id']}")
    with session() as s:
        assert s.query(MontageVersion).filter_by(montage_id=m["id"]).count() == 0


# ── presets ─────────────────────────────────────────────────────────────────


def test_a_preset_crosses_matches(isolated, short_sample):
    """It is what makes the second match cost one click instead of half an hour
    of fitting -- which is why it belongs to no job."""
    a = run_analysis(short_sample)
    recipe = {"kinds": ["kill"], "duration_s": 1.8, "beats_per_cut": 2.0,
              "zoom": True, "export": {"width": 1080, "height": 1920}}
    api().post("/api/presets", json={"name": "shorts", "data": recipe})

    items = api().get("/api/presets").json()["items"]
    assert [p["name"] for p in items] == ["shorts"]
    assert items[0]["data"]["beats_per_cut"] == 2.0
    assert items[0]["data"]["export"]["width"] == 1080
    # the list is the same seen from any match
    assert api().get("/api/presets").json() == api().get("/api/presets").json()
    assert a  # the match does not come into it


def test_a_preset_keeps_the_way_of_cutting_and_not_the_cuts(isolated):
    """A list of cuts is only good for that match; a way of cutting is good for
    any of them."""
    from owcore.models import Recipe

    r = Recipe(**{"kinds": ["kill", "sleep"], "lead_s": 1.2, "duration_s": 2.0})
    assert not hasattr(r, "clips")
    assert not hasattr(r, "layers")
    assert r.kinds == ["kill", "sleep"]


def test_an_impossible_recipe_is_refused(isolated):
    for bad in ({"duration_s": 0.0}, {"lead_s": -1}, {"speed": 0},
                {"gap_s": -0.5}, {"music_volume": 5}):
        resp = api().post("/api/presets", json={"name": "x", "data": bad})
        assert resp.status_code == 422, bad


def test_a_preset_without_a_name_is_refused(isolated):
    assert api().post("/api/presets", json={"data": {}}).status_code == 422
    assert api().post("/api/presets", json={"name": "  "}).status_code == 422


def test_editing_and_deleting_a_preset(isolated):
    p = api().post("/api/presets", json={"name": "one", "data": {}}).json()

    api().put(f"/api/presets/{p['id']}",
              json={"name": "another", "data": {"duration_s": 3.0}})
    back = api().get("/api/presets").json()["items"][0]
    assert back["name"] == "another"
    assert back["data"]["duration_s"] == 3.0

    assert api().delete(f"/api/presets/{p['id']}").status_code == 204
    assert api().get("/api/presets").json()["items"] == []


# ── music on the ruler (Phase 9) ────────────────────────────────────────────


def _with_music_on_the_ruler(**kw):
    from owcore.models import Timeline

    blocks = kw.pop("blocks", [
        {"at_s": 0.0, "duration_s": 1.5, "start_s": 10.0,
         "source": "media", "media_id": "m1"},
    ])
    return Timeline(
        layers=[
            {"clips": [{"at_s": 0.0, "duration_s": 4.0, "start_s": 1.0}]},
            {"kind": "audio", "clips": blocks},
        ],
        **kw,
    )


def _audio_library(path):
    from owcore.compose import LibraryFile

    return {"m1": LibraryFile(path, "audio")}


def test_an_audio_layer_draws_nothing(isolated):
    """It plays. If it entered the stacking, the next video clip would show up
    over an `overlay` that does not exist."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    assert c.filter_complex.count("overlay=") == 1, "only the video clip"
    assert "[2:a]atrim" in c.filter_complex, "but its sound comes in"


def test_a_music_block_is_trimmed_and_positioned(isolated):
    """It is what the continuous track never knew how to do: come in mid-video,
    with a chosen piece of the music."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(blocks=[
            {"at_s": 2.5, "duration_s": 1.5, "start_s": 30.0,
             "source": "media", "media_id": "m1"},
        ]),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    # the piece comes from 30s into the music...
    assert any(
        e.seek == pytest.approx(30.0) and "m.mp3" in e.path
        for e in c.inputs
    )
    # ...lasts 1.5s and comes in at 2.5s of the video
    assert "atrim=duration=1.500" in c.filter_complex
    assert "adelay=2500|2500" in c.filter_complex


def test_two_music_blocks_mix(isolated):
    """Switching tracks mid-video was the request; it is two blocks."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(blocks=[
            {"at_s": 0.0, "duration_s": 2.0, "start_s": 0.0,
             "source": "media", "media_id": "m1"},
            {"at_s": 2.0, "duration_s": 2.0, "start_s": 60.0,
             "source": "media", "media_id": "m1", "audio": {"volume": 0.4}},
        ]),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    # the two blocks mix with each other; the game sound stays out because
    # `game_volume` is 0 -- with music playing, the default is music alone
    assert "amix=inputs=2" in c.filter_complex
    assert "[music]" in c.filter_complex
    assert "volume=0.4000" in c.filter_complex


def test_silence_is_the_absence_of_a_block(isolated):
    """There is no "silence block": where there is no music, there is no music.
    It is the same the gap between clips already does with the picture."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(blocks=[
            {"at_s": 0.0, "duration_s": 1.0, "start_s": 0.0,
             "source": "media", "media_id": "m1"},
            {"at_s": 3.0, "duration_s": 1.0, "start_s": 0.0,
             "source": "media", "media_id": "m1"},
        ]),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    # nothing covers the gap from 1s to 3s, and no filter tries to fill it
    assert "adelay=3000|3000" in c.filter_complex
    assert c.duration_s == pytest.approx(4.0)


def test_a_muted_audio_layer_is_ignored(isolated):
    from owcore.compose import compose_graph

    t = _with_music_on_the_ruler()
    t.layers[1].muted = True
    c = compose_graph(
        t, source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    assert "[2:a]" not in c.filter_complex


def test_with_video_only_the_audio_layer_is_not_even_opened(isolated):
    """Building its input would mean paying for a file nobody would hear."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
        video_only=True,
    )

    assert not any("m.mp3" in e.path for e in c.inputs)
    assert c.audio_map is None


def test_a_music_block_enters_the_export_window(isolated):
    """Exporting a range repositions the sound along with the picture --
    otherwise the music would come out shifted from the video."""
    from owcore.compose import compose_graph

    c = compose_graph(
        _with_music_on_the_ruler(
            export={"from_s": 1.0, "to_s": 3.0},
            blocks=[{"at_s": 0.0, "duration_s": 4.0, "start_s": 10.0,
                     "source": "media", "media_id": "m1"}],
        ),
        source=Path("x.mp4"), width=640, height=360, fps=30,
        source_duration_s=600, library=_audio_library(Path("m.mp3")),
    )

    # the block started at 0s and went until 4s; seen through the window it
    # starts at the first frame and takes the music from 11s on
    assert any(e.seek == pytest.approx(11.0) for e in c.inputs)
    assert "atrim=duration=2.000" in c.filter_complex


def test_music_on_the_ruler_takes_the_montage_off_the_short_path(isolated):
    """Cut-and-splice does not mix sound that runs outside the cuts, and reusing
    the picture assumes the sound comes afterwards, from outside."""
    t = _with_music_on_the_ruler()

    assert t.has_music
    assert not t.single_layer


def test_an_empty_audio_layer_is_not_music_yet(isolated):
    """Creating the layer is just making room; nothing changed yet in the video
    that comes out."""
    from owcore.models import Timeline

    t = Timeline(layers=[
        {"clips": [{"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0}]},
        {"kind": "audio", "clips": []},
    ])
    assert not t.has_music


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_the_video_comes_out_with_the_music_trimmed_and_positioned(
    isolated, short_sample, tmp_path
):
    """End to end: the output file has sound, lasts what was asked, and the
    stretch without a music block is quieter than the rest."""
    from owcore import ffmpeg
    from owcore.compose import compose_graph

    info = ffmpeg.probe(short_sample)
    t = _with_music_on_the_ruler(
        game_volume=0.0,
        blocks=[{"at_s": 0.0, "duration_s": 2.0, "start_s": 5.0,
                 "source": "media", "media_id": "m1"}],
    )
    # without the game sound, what is left after 2s is real silence
    for layer in t.layers:
        for clip in layer.clips:
            if clip.source == "recording":
                clip.audio.mute = True

    c = compose_graph(
        t, source=short_sample, width=info.width, height=info.height,
        fps=info.fps, source_duration_s=info.duration_s,
        library=_audio_library(MUSIC),
    )
    output = tmp_path / "with_music.mp4"
    ffmpeg.compose(c, output)

    result = ffmpeg.probe(output)
    assert result.duration_s == pytest.approx(4.0, abs=0.35)
    assert result.has_audio

    assert _volume_between(output, 0.0, 1.8) > _volume_between(output, 2.2, 3.8) + 10


def _volume_between(video: Path, start: float, end: float) -> float:
    """The mean volume of a stretch, in dB. The closer to zero, the louder."""
    from owcore.config import get_settings

    out = subprocess.run(
        # `-v info` on purpose: `volumedetect` writes its result as info, and
        # with `-v error` it would say nothing
        [get_settings().ffmpeg, "-v", "info", "-ss", f"{start:.3f}",
         "-t", f"{end - start:.3f}", "-i", str(video),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    for line in out.splitlines():
        if "mean_volume" in line:
            return float(line.split(":")[1].strip().split()[0])
    return -91.0


# ── what the server refuses ─────────────────────────────────────────────────


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_music_on_a_video_layer_is_refused(isolated, short_sample):
    """It would make ffmpeg try to resize an audio stream, and the whole render
    would die with a message that explains nothing."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([{
            "layers": [{"clips": [
                {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0},
                {"at_s": 3.0, "duration_s": 2.0, "start_s": 0.0,
                 "source": "media", "media_id": track_id},
            ]}],
        }])},
    )
    assert resp.status_code == 422
    assert "audio layer" in resp.json()["detail"]


def test_an_image_on_an_audio_layer_is_refused(isolated, short_sample, tmp_path):
    """Worse than an error: it would come out silent, with no error at all, and
    the user would look for the problem in the mix."""
    job_id = run_analysis(short_sample)
    png = make_png(tmp_path / "badge.png")
    media_id = upload_media(job_id, "badge.png", png.read_bytes())

    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([{
            "layers": [
                {"clips": [{"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0}]},
                {"kind": "audio", "clips": [
                    {"at_s": 0.0, "duration_s": 2.0, "start_s": 0.0,
                     "source": "media", "media_id": media_id},
                ]},
            ],
        }])},
    )
    assert resp.status_code == 422
    assert "only accepts music" in resp.json()["detail"]


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_the_continuous_track_becomes_a_block_on_read(isolated):
    """There were two ways of having music and one is left. The code that reads
    is what converts the old format -- there is no migration to run on the
    database."""
    from owcore.models import MontageDraft, Timeline

    t = Timeline(
        track_id="m1", music_start_s=12.0,
        layers=[{"clips": [
            {"at_s": 0.0, "duration_s": 2.0, "start_s": 1.0},
            {"at_s": 2.0, "duration_s": 3.0, "start_s": 6.0},
        ]}],
    )

    assert t.track_id is None, "the continuous track does not survive reading"
    assert t.music_start_s == 0.0
    sound = t.layers[-1]
    assert sound.is_audio and len(sound.clips) == 1
    block = sound.clips[0]
    assert block.media_id == "m1"
    assert block.at_s == 0.0, "the music came in with the video"
    assert block.duration_s == pytest.approx(5.0), "and covered the whole video"
    assert block.start_s == pytest.approx(12.0), "from the same point in the music"
    assert t.has_music

    # the draft saved in V1 (cuts, no layers) ends up in the same place
    d = MontageDraft(
        track_id="m1", music_start_s=3.0,
        cuts=[{"start_s": 10.0, "duration_s": 2.0, "at_s": 0.0}],
    )
    assert [l.is_audio for l in d.layers] == [False, True]
    assert d.layers[0].clips[0].start_s == pytest.approx(10.0)
    assert d.layers[1].clips[0].start_s == pytest.approx(3.0)


@pytest.mark.skipif(not MUSIC.exists(), reason="needs data/sample/music.wav")
def test_a_request_in_the_old_format_still_becomes_a_video(isolated, short_sample):
    """The conversion is not only the model's: a request from an app older than
    this phase must come out the other side as a video with music."""
    job_id = run_analysis(short_sample)
    track_id = upload_music(job_id)

    render_id = render_montage(job_id, [{
        "title": "the usual track", "track_id": track_id,
        "music_start_s": 2.0,
        "layers": [{"clips": [
            {"at_s": 0.0, "duration_s": 1.5, "start_s": 1.0},
            {"at_s": 2.0, "duration_s": 1.5, "start_s": 6.0},
        ]}],
    }])
    run_render()

    clip = api().get(f"/api/renders/{render_id}").json()["clips"][0]
    assert clip["video_url"], clip.get("meta")
    assert clip["meta"]["composed"] is True, "music only exists in the graph"
    assert clip["meta"]["original_audio"] is False
    assert clip["meta"]["music_name"], "the list says which music it came out with"


# ── transitions ─────────────────────────────────────────────────────────────


def _solid_colours(tmp_path):
    """Two flat-colour frames, so every pixel can be traced to its clip."""
    import cv2
    import numpy as np

    from owcore.compose import LibraryFile

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    # OpenCV writes BGR
    cv2.imwrite(str(red), np.full((360, 640, 3), (0, 0, 255), np.uint8))
    cv2.imwrite(str(blue), np.full((360, 640, 3), (255, 0, 0), np.uint8))
    return {
        "red": LibraryFile(path=red, kind="image"),
        "blue": LibraryFile(path=blue, kind="image"),
    }


def _red_to_blue(transition):
    from owcore.models import Layer, Timeline, TimelineClip

    return Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, source="media", media_id="red"),
        TimelineClip(at_s=2, duration_s=2, source="media", media_id="blue",
                     transition=transition),
    ])])


def _render_with_library(timeline, library, short_sample, dest):
    from owcore import ffmpeg
    from owcore.compose import compose_graph

    c = compose_graph(timeline, source=short_sample, width=640, height=360,
                      fps=30, library=library)
    ffmpeg.compose(c, dest)
    return dest


def _rgb(video, t):
    """The frame's mean colour (R, G, B)."""
    return raw_frame(video, t).reshape(-1, 3).mean(axis=0)


def test_transition_is_in_the_model_and_leaves_the_simple_path(isolated):
    from owcore.models import TimelineClip

    c = TimelineClip(at_s=0, duration_s=2,
                     transition={"kind": "dissolve", "duration_s": 0.5})
    assert c.transition.kind == "dissolve"
    assert not c.is_simple

    with pytest.raises(ValueError, match="transition"):
        TimelineClip(at_s=0, duration_s=1,
                     transition={"kind": "dissolve", "duration_s": 2})
    with pytest.raises(ValueError):
        TimelineClip(at_s=0, duration_s=2,
                     transition={"kind": "spinning", "duration_s": 0.5})


def test_dissolve_mixes_both_and_ends_on_the_new_clip(
    isolated, short_sample, tmp_path
):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _red_to_blue({"kind": "dissolve", "duration_s": 1.0}),
        lib, short_sample, tmp_path / "dissolve.mp4",
    )

    r, _, b = _rgb(video, 1.5)
    assert r > 180 and b < 60, "before the cut, only red"
    # halfway through, both show at the same time: that is what sets a
    # dissolve apart from a fade over the black background
    r, _, b = _rgb(video, 2.5)
    assert r > 60 and b > 60, f"halfway, a mix (r={r:.0f}, b={b:.0f})"
    r, _, b = _rgb(video, 3.5)
    assert b > 180 and r < 60, "afterwards, only blue"


def test_dip_to_black_goes_dark_at_the_cut(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _red_to_blue({"kind": "fade_black", "duration_s": 1.0}),
        lib, short_sample, tmp_path / "black.mp4",
    )

    assert _rgb(video, 1.0)[0] > 180
    assert _rgb(video, 2.0).max() < 40, "black at the cut"
    assert _rgb(video, 3.0)[2] > 180


def test_dip_to_white_goes_bright_at_the_cut(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _red_to_blue({"kind": "fade_white", "duration_s": 1.0}),
        lib, short_sample, tmp_path / "white.mp4",
    )

    assert _rgb(video, 2.0).min() > 200, "white at the cut"


def test_slide_pushes_the_new_clip_over_the_old_one(
    isolated, short_sample, tmp_path
):
    import numpy as np

    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _red_to_blue({"kind": "slide_left", "duration_s": 1.0}),
        lib, short_sample, tmp_path / "slide.mp4",
    )

    q = raw_frame(video, 2.5).reshape(45, 80, 3)
    left = q[:, :30, :].mean(axis=(0, 1))
    right = q[:, 50:, :].mean(axis=(0, 1))
    # halfway: blue came in from the right, red is still on the left -- and
    # underneath it, not black
    assert left[0] > 180 and left[2] < 60
    assert right[2] > 180 and right[0] < 60
    assert np.abs(_rgb(video, 3.5) - (0, 0, 255)).max() < 40


def test_the_previous_clip_overrun_is_picture_only(isolated):
    """A dissolve stretches the previous clip's picture under the new one; its
    sound stops where the clip stops on the ruler."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1),
        TimelineClip(at_s=2, duration_s=2, start_s=6,
                     transition={"kind": "dissolve", "duration_s": 0.5}),
    ])])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60)

    graph = c.filter_complex
    assert "[1:v]trim=duration=2.500" in graph, "the picture runs past the cut"
    assert "[1:a]atrim=duration=2.000" in graph, "the sound does not"
    # and the previous clip stays visible until the transition ends
    assert "between(t,0.000,2.500)" in graph


def test_a_transition_cut_off_by_the_export_window_is_dropped(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(
        export={"from_s": 2.2},
        layers=[Layer(clips=[
            TimelineClip(at_s=0, duration_s=2, start_s=1),
            TimelineClip(at_s=2, duration_s=2, start_s=6,
                         transition={"kind": "slide_left", "duration_s": 1}),
        ])],
    )
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60)

    # the entrance happened before the window: the clip is already in place
    assert "*W*" not in c.filter_complex


# ── the exact preview's frame and window ────────────────────────────────────


def test_the_preview_keeps_the_export_shape_on_a_small_frame():
    from owcore.preview import preview_size

    assert preview_size(0, 0, 2560, 1440) == (640, 360)
    # a vertical export stays vertical
    assert preview_size(1080, 1920, 2560, 1440) == (360, 640)
    assert preview_size(1080, 1080, 1920, 1080) == (640, 640)


def test_the_preview_window_is_clamped_to_the_montage():
    import pytest
    from owcore.preview import PREVIEW_MAX_S, preview_window

    assert preview_window(10, None, None) == (0, 10)
    assert preview_window(10, 4, 30) == (4, 10)
    assert preview_window(500, 0, None) == (0, PREVIEW_MAX_S)
    with pytest.raises(ValueError):
        preview_window(10, 12, 20)


def test_the_preview_timeline_only_changes_the_export():
    from owcore.preview import PREVIEW_CRF, preview_timeline

    spec = Timeline(
        cuts=[{"start_s": 1, "duration_s": 2, "at_s": 0}],
        export={"fit": "contain", "fps": 60},
    )
    small = preview_timeline(
        spec, from_s=0.5, to_s=1.5,
        source_width=640, source_height=360, source_fps=24,
    )
    assert small.layers == spec.layers
    assert (small.export.width, small.export.height) == (640, 360)
    assert small.export.fps == 30
    assert small.export.crf == PREVIEW_CRF
    assert small.export.fit == spec.export.fit
    assert (small.export.from_s, small.export.to_s) == (0.5, 1.5)


# ── keyframes and easing ────────────────────────────────────────────────────


def test_keyframes_are_validated_and_leave_the_simple_path(isolated):
    from owcore.models import TimelineClip

    c = TimelineClip(at_s=0, duration_s=2, keys=[
        {"prop": "opacity", "t": 1, "value": 0},
        {"prop": "opacity", "t": 0, "value": 1, "ease": "in_out"},
    ])
    assert not c.is_simple
    assert [k.t for k in c.keys_for("opacity")] == [0, 1], "in time order"

    with pytest.raises(ValueError, match="opacity"):
        TimelineClip(at_s=0, duration_s=2,
                     keys=[{"prop": "opacity", "t": 0, "value": 2}])
    with pytest.raises(ValueError, match="same instant"):
        TimelineClip(at_s=0, duration_s=2, keys=[
            {"prop": "x", "t": 0.5, "value": 0},
            {"prop": "x", "t": 0.5, "value": 1},
        ])
    with pytest.raises(ValueError):
        TimelineClip(at_s=0, duration_s=2,
                     keys=[{"prop": "spin", "t": 0, "value": 1}])


def test_easing_shapes_the_curve(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1,
                     zoom=[{"t": 0, "scale": 1, "ease": "in_out"},
                           {"t": 1, "scale": 2}]),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30).filter_complex
    # smoothstep: u*u*(3-2u)
    assert "*(3-2*(" in g


def test_zoom_keyframes_follow_the_clip_as_placed_not_as_drawn(isolated):
    """Under a dissolve the clip's picture runs longer, and through an export
    window it starts partway: the keyframes still sit where the user put them."""
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    zoom = [{"t": 0, "scale": 1}, {"t": 1, "scale": 2}]
    tail = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, zoom=zoom),
        TimelineClip(at_s=2, duration_s=2, start_s=6,
                     transition={"kind": "dissolve", "duration_s": 0.5}),
    ])])
    g = compose_graph(tail, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60).filter_complex
    assert "lt(it,2.0000)" in g, "the zoom ends at the cut, not under the dissolve"

    windowed = Timeline(export={"from_s": 1}, layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, zoom=zoom),
    ])])
    g = compose_graph(windowed, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60).filter_complex
    # the window starts halfway through the zoom: its first point is 1s gone
    assert "lt(it,-1.0000)" in g


def test_animated_volume_runs_on_the_clips_clock(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=3, duration_s=2, start_s=1, keys=[
            {"prop": "volume", "t": 0, "value": 0},
            {"prop": "volume", "t": 1, "value": 1},
        ]),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60).filter_complex
    assert ":eval=frame" in g
    # the volume comes before the delay that places the clip at 3s
    assert g.index("volume='") < g.index("adelay=3000")


def _blue_over_red(*keys, scale=1.0):
    from owcore.models import Layer, Timeline, TimelineClip

    return Timeline(layers=[
        Layer(clips=[TimelineClip(at_s=0, duration_s=2, source="media",
                                  media_id="red")]),
        Layer(clips=[TimelineClip(at_s=0, duration_s=2, source="media",
                                  media_id="blue", transform={"scale": scale},
                                  keys=list(keys))]),
    ])


def test_scale_and_opacity_animate_together(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _blue_over_red({"prop": "scale", "t": 0, "value": 0.5},
                       {"prop": "scale", "t": 1, "value": 0.5},
                       {"prop": "opacity", "t": 0, "value": 0},
                       {"prop": "opacity", "t": 1, "value": 1}),
        lib, short_sample, tmp_path / "both.mp4",
    )
    q = raw_frame(video, 1.95).reshape(45, 80, 3)
    assert q[22, 40, 2] > 180, "the centre is blue at the end"
    assert q[2, 2, 0] > 180, "the corner shows the red underneath"


def test_opacity_keyframes_fade_the_upper_clip_in(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _blue_over_red({"prop": "opacity", "t": 0, "value": 0},
                       {"prop": "opacity", "t": 1, "value": 1}),
        lib, short_sample, tmp_path / "opacity.mp4",
    )
    early, mid, late = (_rgb(video, s) for s in (0.05, 1.0, 1.95))
    assert early[0] > 200 and early[2] < 40, "starts on the red underneath"
    assert 90 < mid[2] < 170 and 90 < mid[0] < 170, "half way, half and half"
    assert late[2] > 200 and late[0] < 40


def test_an_ease_in_holds_back_the_first_half(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _blue_over_red({"prop": "opacity", "t": 0, "value": 0, "ease": "in"},
                       {"prop": "opacity", "t": 1, "value": 1}),
        lib, short_sample, tmp_path / "ease.mp4",
    )
    # u² at the half: a quarter of the blue, where linear gives half
    assert 30 < _rgb(video, 1.0)[2] < 100


def test_position_keyframes_move_the_clip_across(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _blue_over_red({"prop": "x", "t": 0, "value": -0.5},
                       {"prop": "x", "t": 1, "value": 0.5}, scale=0.5),
        lib, short_sample, tmp_path / "move.mp4",
    )

    def blue_side(t):
        q = raw_frame(video, t).reshape(45, 80, 3)
        return q[:, :40, 2].mean(), q[:, 40:, 2].mean()

    left, right = blue_side(0.05)
    assert left > right + 60, "starts on the left"
    left, right = blue_side(1.95)
    assert right > left + 60, "ends on the right"


def test_scale_keyframes_grow_the_clip(isolated, short_sample, tmp_path):
    lib = _solid_colours(tmp_path)
    video = _render_with_library(
        _blue_over_red({"prop": "scale", "t": 0, "value": 0.2},
                       {"prop": "scale", "t": 1, "value": 1}),
        lib, short_sample, tmp_path / "grow.mp4",
    )
    import numpy as np

    def blue_width(t):
        q = raw_frame(video, t).reshape(45, 80, 3)
        blue = (q[:, :, 2] > 180) & (q[:, :, 0] < 80)
        cols = np.where(blue.any(axis=0))[0]
        return (cols.max() - cols.min() + 1) / 80 if len(cols) else 0.0

    # it really grows -- `scale` with `eval=frame` once kept the first size
    assert blue_width(0.1) < 0.35
    assert 0.5 < blue_width(1.0) < 0.7, "half way: 0.6 of the frame"
    assert blue_width(1.95) > 0.9


# ── speed ramps ─────────────────────────────────────────────────────────────

_RAMP = [  # 0.5x → 2x across a 2s clip, linear: speed(t) = 0.5 + 0.75 t
    {"prop": "speed", "t": 0, "value": 0.5},
    {"prop": "speed", "t": 1, "value": 2},
]


def _ramp_source_offset(t: float) -> float:
    """∫ (0.5 + 0.75 τ) dτ from 0 to t."""
    return 0.5 * t + 0.375 * t * t


def test_a_ramp_integrates_the_speed(isolated):
    from owcore.models import TimelineClip

    c = TimelineClip(at_s=0, duration_s=2, start_s=1, keys=_RAMP)
    assert c.is_ramped
    assert c.source_offset(1.0) == pytest.approx(_ramp_source_offset(1.0), abs=1e-4)
    assert c.source_consumed_s == pytest.approx(_ramp_source_offset(2.0), abs=1e-4)
    assert c.local_for_source(_ramp_source_offset(1.3)) == pytest.approx(1.3, abs=1e-3)
    # without keys it is the old constant speed
    plain = TimelineClip(at_s=0, duration_s=2, start_s=1, speed=2)
    assert plain.source_consumed_s == 4
    assert plain.local_for_source(3) == 1.5


def test_a_ramp_cannot_be_frozen_or_reversed(isolated):
    from owcore.models import TimelineClip

    for flag in ("freeze", "reverse"):
        with pytest.raises(ValueError, match="ramp"):
            TimelineClip(at_s=0, duration_s=2, keys=_RAMP, **{flag: True})


def test_a_ramp_is_retimed_and_has_no_clip_sound(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, keys=_RAMP),
    ])])
    g = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60).filter_complex
    assert f"trim=duration={_ramp_source_offset(2.0):.3f}" in g
    assert "setpts='(if(lt(T," in g
    assert "atrim" not in g, "atempo takes one rate, not a curve"


def test_an_export_window_starts_a_ramp_where_its_source_is(isolated):
    from owcore.compose import compose_graph
    from owcore.models import Layer, Timeline, TimelineClip

    t = Timeline(export={"from_s": 1}, layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=10, keys=_RAMP),
    ])])
    c = compose_graph(t, source=Path("x.mp4"), width=640, height=360, fps=30,
                      source_duration_s=60)
    seek = c.inputs[1].seek
    assert seek == pytest.approx(10 + _ramp_source_offset(1.0), abs=1e-3)


def _time_coded_source(tmp_path) -> Path:
    """A video whose red channel says what second it is: 20 levels a second."""
    from owcore.config import get_settings

    out = tmp_path / "clock.mp4"
    subprocess.run(
        [get_settings().ffmpeg, "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=black:s=160x90:r=30:d=10",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-vf", "format=rgb24,geq=r='min(255,20*T)':g=0:b=0,format=yuv420p",
         "-c:v", "libx264", "-qp", "0", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )
    return out


def test_a_ramp_shows_the_source_where_the_integral_says(isolated, tmp_path):
    from owcore.models import Layer, Timeline, TimelineClip

    source = _time_coded_source(tmp_path)
    t = Timeline(layers=[Layer(clips=[
        TimelineClip(at_s=0, duration_s=2, start_s=1, keys=_RAMP),
    ])])
    video = compose_and_render(t, source, tmp_path / "ramp.mp4")

    def shown(at: float) -> float:
        return raw_frame(video, at).reshape(-1, 3)[:, 0].mean() / 20

    def rate(a: float, b: float) -> float:
        # a difference cancels the reading's own lag (~0.1s, measured at 1x
        # and 2x), which an absolute comparison would trip on
        return (shown(b) - shown(a)) / (b - a)

    expected = lambda a, b: (_ramp_source_offset(b) - _ramp_source_offset(a)) / (b - a)
    assert rate(0.1, 0.5) == pytest.approx(expected(0.1, 0.5), abs=0.3), "slow start"
    assert rate(1.5, 1.9) == pytest.approx(expected(1.5, 1.9), abs=0.3), "fast end"
    assert rate(1.5, 1.9) > 2 * rate(0.1, 0.5), "it really ramps"
    assert shown(1.0) == pytest.approx(1 + _ramp_source_offset(1.0), abs=0.2)
