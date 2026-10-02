"""Integration: the job goes through every microservice.

It starts no processes -- it instantiates each worker and hands it the messages
the bus produced. That is what checks the *contracts*: if the preprocessor
changes the `RoiReady` format, or a detector stops notifying the planner, this
test breaks.

The system has two phases, and the tests follow that split:

* **analysis** -- runs once per recording and ends at `ready`, with the
  match's moments. No music goes through here.
* **rendering** -- the user assembles what they want in the editor, with each
  video's music, and can ask as many times as they like over the same
  analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import service_module
from owcore.bus import get_bus
from owcore.db import session
from owcore.models import (
    DETECTORS,
    STREAM_EDIT,
    STREAM_JOBS,
    STREAM_RENDER_READY,
    STREAM_ROI,
    Job,
    JobStage,
    JobStatus,
    Render,
    RenderStage,
    RenderStatus,
)


def drain(stream: str, group: str) -> list[dict]:
    """Takes everything waiting for that group off the bus."""
    bus = get_bus()
    out: list[dict] = []
    while True:
        msgs = list(bus.consume(stream, group, "test", block_ms=0))
        if not msgs:
            return out
        for m in msgs:
            out.append(m.payload)
            bus.ack(stream, group, m.id)


def api() -> TestClient:
    return TestClient(service_module("gateway", "app").app)


# ── phase 1: analysis ───────────────────────────────────────────────────────


def run_analysis(video_path, params: str = "{}", *, detectors=None) -> str:
    """Uploads the recording through the API and runs the whole analysis by
    hand."""
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("match.mp4", video_path.read_bytes(), "video/mp4")},
        data={"params": params},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["id"]

    preprocessor = service_module("preprocessor", "main").Preprocessor()
    for payload in drain(STREAM_JOBS, "preprocessor"):
        preprocessor.handle(payload)

    workers = detectors if detectors is not None else all_detectors()
    for w in workers:
        for payload in drain(STREAM_ROI, w.group):
            if w.accepts(payload):
                w.handle(payload)

    planner = service_module("planner", "main").Planner()
    for payload in drain(STREAM_EDIT, "planner"):
        planner.handle(payload)

    return job_id


def all_detectors() -> list:
    return [
        service_module("detector_kills", "main").KillsDetector(),
        service_module("detector_survival", "main").SurvivalDetector(),
        service_module("detector_ults", "main").UltsDetector(),
        service_module("detector_banner", "main").BannerDetector(),
        service_module("detector_killfeed", "main").KillfeedDetector(),
    ]


# ── phase 2: rendering ──────────────────────────────────────────────────────

#: the events the editor puts on the shelf -- and where the cuts here come from
MOMENTS = {"kill", "headshot", "ability_kill", "sleep", "stun", "ult_negated",
           "escape"}


def moments_of(job_id: str) -> list[float]:
    detail = api().get(f"/api/jobs/{job_id}").json()
    return [e["t"] for e in detail["events"] if e["kind"] in MOMENTS]


def montage(
    job_id: str,
    *,
    count: int = 3,
    duration: float = 1.0,
    gap: float = 0.0,
    title: str = "Montage",
) -> dict:
    """A timeline like the one that comes out of the editor: N moments in a
    row.

    It is the only way to render video in the system. Until Phase 11 there was
    another -- the app picked ready-made proposals and the server decided the
    cuts -- and that is what these tests used.
    """
    instants = sorted(set(moments_of(job_id)))[:count]
    assert instants, "the analysis found no moment to assemble"
    return {
        "title": title,
        "cuts": [
            {
                "source_t": t,
                # half a second of run-up, without going before the recording
                "start_s": max(0.0, t - 0.5),
                "duration_s": duration,
                "at_s": i * (duration + gap),
            }
            for i, t in enumerate(instants)
        ],
    }


def request_render(job_id: str, timelines: list[dict] | None = None) -> str:
    """Asks for a render, as the app does."""
    if timelines is None:
        timelines = [montage(job_id)]
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps(timelines)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def run_render() -> None:
    """Runs the editor for every request waiting in the queue."""
    editor = service_module("editor", "main").Editor()
    for payload in drain(STREAM_RENDER_READY, "editor"):
        editor.handle(payload)


def run_pipeline(
    video_path, params: str = "{}", *, timelines: list[dict] | None = None
) -> tuple[str, str]:
    """Analysis + one render. Returns (job_id, render_id)."""
    job_id = run_analysis(video_path, params)
    render_id = request_render(job_id, timelines)
    run_render()
    return job_id, render_id


# ── the analysis, on its own ────────────────────────────────────────────────


def test_analysis_stops_at_ready_with_the_matchs_moments(isolated, short_sample):
    job_id = run_analysis(short_sample)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.READY, job.error
        assert job.duration_s > 10
        assert {r.detector for r in job.reports} == set(DETECTORS)
        assert all(r.ok for r in job.reports)
        assert any(e.kind == "kill" for e in job.events)
        # the count is a number, and the stage a code: the sentence is the app's
        assert job.stage == JobStage.READY
        assert job.n_moments == len(job.events)
        # the analysis renders no video: that is the second phase
        assert job.clips == []
        assert job.renders == []


def test_the_analysis_delivers_the_moments_the_editor_shows(isolated, short_sample):
    """The contract between the analysis and the editor: what the detector
    found shows up on the shelf. A detected kind that does not get this far is
    wasted work -- which is what happened to headshots and ability kills."""
    job_id = run_analysis(short_sample)
    detail = api().get(f"/api/jobs/{job_id}").json()

    assert detail["status"] == "ready"
    kinds = {e["kind"] for e in detail["events"]}
    assert kinds & MOMENTS, "no moment to assemble"
    # the `meta` comes along: it is where *which* ability killed lives
    assert all("meta" in e for e in detail["events"])


def test_the_analysis_does_not_end_while_a_detector_is_missing(isolated, short_sample):
    kills = service_module("detector_kills", "main").KillsDetector()
    job_id = run_analysis(short_sample, detectors=[kills])

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.DETECTING


def test_one_detector_failing_does_not_bring_the_job_down(isolated, short_sample):
    """An important contract: if the ults detector breaks, the user still gets
    the moments the others found."""
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    )
    job_id = resp.json()["id"]

    preprocessor = service_module("preprocessor", "main").Preprocessor()
    for payload in drain(STREAM_JOBS, "preprocessor"):
        preprocessor.handle(payload)

    for w in all_detectors():
        for payload in drain(STREAM_ROI, w.group):
            if not w.accepts(payload):
                continue
            if w.detector == "ults":
                w._dispatch(_FakeBus(), _FakeMsg(payload))  # blows up on purpose
            else:
                w.handle(payload)

    planner = service_module("planner", "main").Planner()
    for payload in drain(STREAM_EDIT, "planner"):
        planner.handle(payload)

    with session() as s:
        job = s.get(Job, job_id)
        assert job.status == JobStatus.READY, job.error
        ults = next(r for r in job.reports if r.detector == "ults")
        assert not ults.ok and ults.error
        assert job.events, "it should deliver what the others found"


class _FakeMsg:
    def __init__(self, payload: dict):
        self.id = "x"
        self.payload = dict(payload)
        # a missing artifact: the detector blows up trying to download it
        self.payload["artifacts"] = [
            {"key": "does/not/exist.mp4", "kind": "roi", "meta": {"roi": "killfeed"}}
        ]


class _FakeBus:
    def ack(self, *_a): ...


def test_a_negated_ultimate_is_stored_as_an_event(isolated):
    """Regression: it only existed in the proposal generator's head.

    A negated ultimate is an enemy ultimate followed by a kill -- no detector
    alone sees both halves, so whoever closes the analysis crosses the two and
    stores the result. While there were proposals, that crossing happened
    inside the proposal generator and died there: the kind showed up in the
    editor's list and in thumbnail extraction, but no event of that kind ever
    existed in the database.
    """
    from owcore.jobs import load_events, save_events
    from owcore.models import DetectionEvent, EventKind, Job, JobStatus

    with session() as s:
        s.add(Job(
            id="j1", video_key="k", video_name="v.mp4",
            status=JobStatus.DETECTING, duration_s=60.0,
        ))
    save_events("j1", "ults", [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)])
    save_events("j1", "kills", [DetectionEvent(kind=EventKind.KILL, t=41.5)])

    service_module("planner", "main").Planner()._close_analysis("j1")

    negated = [e for e in load_events("j1") if e.kind == EventKind.ULT_NEGATED]
    assert len(negated) == 1, "the crossing did not become an event"
    assert negated[0].t == 41.5
    assert negated[0].meta["delay_s"] == 1.5

    with session() as s:
        job = s.get(Job, "j1")
        assert job.status == JobStatus.READY
        assert job.n_moments == 3, "the crossed event counts as a moment too"


def test_an_old_match_gets_its_moments_counted(isolated):
    """Before `n_moments`, the count only lived inside a sentence in
    `stage`. The gateway counts the events of such a match once, and swaps
    the sentence for the plain code."""
    from owcore.jobs import save_events
    from owcore.models import DetectionEvent, EventKind, Job

    with session() as s:
        s.add(Job(
            id="old", video_key="k", video_name="v.mp4", status=JobStatus.READY,
            stage="68 momento(s) encontrados — abra o editor",
        ))
    save_events("old", "kills", [
        DetectionEvent(kind=EventKind.KILL, t=10.0),
        DetectionEvent(kind=EventKind.KILL, t=20.0),
    ])

    listed = api().get("/api/jobs").json()["jobs"][0]
    assert listed["n_moments"] == 2
    assert listed["stage"] == "ready"
    with session() as s:
        assert s.get(Job, "old").n_moments == 2, "the count was not stored"


def test_closing_twice_does_not_duplicate_what_was_crossed(isolated):
    """All detectors report almost together; only one may close the analysis."""
    from owcore.jobs import load_events, save_events
    from owcore.models import DetectionEvent, EventKind, Job, JobStatus

    with session() as s:
        s.add(Job(
            id="j1", video_key="k", video_name="v.mp4",
            status=JobStatus.DETECTING, duration_s=60.0,
        ))
    save_events("j1", "ults", [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)])
    save_events("j1", "kills", [DetectionEvent(kind=EventKind.KILL, t=41.5)])

    planner = service_module("planner", "main").Planner()
    planner._close_analysis("j1")
    planner._close_analysis("j1")

    negated = [e for e in load_events("j1") if e.kind == EventKind.ULT_NEGATED]
    assert len(negated) == 1


# ── rendering ───────────────────────────────────────────────────────────────


def test_rendering_produces_the_assembled_clips(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)

    with session() as s:
        request = s.get(Render, render_id)
        assert request.status == RenderStatus.DONE, request.error
        assert request.clips, "no clip was rendered"
        for c in request.clips:
            assert c.key, "clip without a blob in storage"
            assert c.kind == "custom", "every video comes out of the editor"
            assert c.meta["hand_made"] is True


def test_the_same_moment_serves_several_videos(isolated, short_sample):
    """Using a cut in one video does not use it up: the same instants can be
    assembled again, and the previous request stays intact."""
    job_id = run_analysis(short_sample)
    request_render(job_id, [montage(job_id, title="Short", duration=0.8)])
    run_render()
    request_render(job_id, [montage(job_id, title="Long", duration=1.5)])
    run_render()

    with session() as s:
        job = s.get(Job, job_id)
        assert len(job.renders) == 2
        assert all(r.status == RenderStatus.DONE for r in job.renders)
        assert all(r.clips for r in job.renders)


def test_two_videos_in_the_same_request(isolated, short_sample):
    """A request can carry more than one montage -- that is what lets the
    Shorts cut and the long version come out together, from the same match."""
    job_id = run_analysis(short_sample)
    render_id = request_render(job_id, [
        montage(job_id, title="Shorts", count=2, duration=0.8),
        montage(job_id, title="Full", count=3, duration=1.5),
    ])
    run_render()

    with session() as s:
        request = s.get(Render, render_id)
        assert request.status == RenderStatus.DONE, request.error
        assert {c.title for c in request.clips} == {"Shorts", "Full"}


def test_a_request_before_the_analysis_ends_is_refused(isolated, short_sample):
    client = api()
    resp = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    )
    job_id = resp.json()["id"]
    resp = client.post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([
            {"cuts": [{"start_s": 0, "duration_s": 1, "at_s": 0}]}
        ])},
    )
    assert resp.status_code == 409


def test_a_request_without_a_montage_is_refused(isolated, short_sample):
    job_id = run_analysis(short_sample)
    resp = api().post(f"/api/jobs/{job_id}/renders", data={"timelines": "[]"})
    assert resp.status_code == 422


def test_a_montage_with_another_jobs_media_is_refused(isolated, short_sample):
    job_id = run_analysis(short_sample)
    spec = montage(job_id)
    resp = api().post(
        f"/api/jobs/{job_id}/renders",
        data={"timelines": json.dumps([{
            "layers": [{"clips": [{
                "at_s": 0, "duration_s": 1, "start_s": 0,
                "source": "media", "media_id": "made_up",
            }]}],
        }])},
    )
    assert resp.status_code == 422, spec


def test_deleting_a_request_does_not_delete_the_match(isolated, short_sample):
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    assert client.delete(f"/api/renders/{render_id}").status_code == 204

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["events"], "the moments must not disappear with the request"
    assert detail["renders"] == []
    assert detail["clips"] == []


# ── no music, the original audio ────────────────────────────────────────────


def _has_audio(client, video_url: str, tmp_path: Path) -> bool:
    from owcore import ffmpeg

    tmp_path.mkdir(parents=True, exist_ok=True)
    dest = tmp_path / "downloaded.mp4"
    dest.write_bytes(client.get(video_url).content)
    return ffmpeg.probe(dest).has_audio


def test_without_music_the_video_keeps_the_original_audio(
    isolated, short_sample, tmp_path
):
    """Explicit requirement: whoever puts no music on the ruler keeps the
    match's sound."""
    job_id, render_id = run_pipeline(short_sample)
    client = api()
    request = client.get(f"/api/renders/{render_id}").json()

    clips = [c for c in request["clips"] if c["video_url"]]
    assert clips
    for c in clips:
        assert c["meta"]["original_audio"] is True
        assert c["meta"].get("music_name") is None
        assert _has_audio(client, c["video_url"], tmp_path / c["id"])


# ── API input ──────────────────────────────────────────────────────────────


def test_an_upload_without_a_video_is_rejected(isolated):
    assert api().post("/api/jobs", data={"params": "{}"}).status_code == 422


def test_invalid_parameters_are_rejected(isolated, short_sample):
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", b"x", "video/mp4")},
        data={"params": "this is not json"},
    )
    assert resp.status_code == 422


def test_a_truncated_upload_is_refused_at_the_door(isolated, short_sample):
    """A half upload looks like no error at all.

    The multipart closes properly, the `Content-Length` matches what actually
    arrived, and what is left is half a recording stored as if it were whole --
    the damage only showed up stages later, in the preprocessor, as an `ffprobe
    exited with 1`. Comparing with the size the client says it sent hands the
    problem back to the upload screen, which is where something can be done.
    """
    video_bytes = short_sample.read_bytes()
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", video_bytes[: len(video_bytes) // 2],
                         "video/mp4")},
        data={"params": "{}", "size": str(len(video_bytes))},
    )
    assert resp.status_code == 400
    assert "incomplete" in resp.json()["detail"]
    # and nothing was left behind: no job in the queue, no half blob
    assert api().get("/api/jobs").json()["jobs"] == []
    assert not list(isolated.blob_dir.rglob("*.mp4"))


def test_a_whole_upload_passes_with_the_size_checked(isolated, short_sample):
    video_bytes = short_sample.read_bytes()
    resp = api().post(
        "/api/jobs",
        files={"video": ("m.mp4", video_bytes, "video/mp4")},
        data={"params": "{}", "size": str(len(video_bytes))},
    )
    assert resp.status_code == 201, resp.text


def test_a_missing_job_returns_404(isolated):
    assert api().get("/api/jobs/doesnotexist").status_code == 404


def test_times_come_out_with_a_timezone(isolated, short_sample):
    """A date without a timezone is read as *local* time by whoever receives it
    -- Dart does that. Since what leaves here is UTC, without the suffix the
    app showed every time shifted, and the "time left" came out negative."""
    from datetime import datetime, timezone

    client = api()
    job_id = client.post(
        "/api/jobs",
        files={"video": ("m.mp4", short_sample.read_bytes(), "video/mp4")},
        data={"params": "{}"},
    ).json()["id"]
    d = client.get(f"/api/jobs/{job_id}").json()
    for field in ("created_at", "updated_at"):
        read = datetime.fromisoformat(d[field])
        assert read.tzinfo is not None, f"{field} came without a timezone: {d[field]!r}"
        # and the time must be now, not three hours from now
        drift = abs((datetime.now(timezone.utc) - read).total_seconds())
        assert drift < 300, f"{field} is off from the clock by {drift:.0f}s"


def test_a_missing_request_returns_404(isolated):
    assert api().get("/api/renders/doesnotexist").status_code == 404


def test_clips_show_up_in_the_api(isolated, short_sample):
    job_id, _render = run_pipeline(short_sample)
    client = api()

    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["clips"]

    clip = detail["clips"][0]
    video = client.get(clip["video_url"])
    assert video.status_code == 200
    assert video.content[4:8] == b"ftyp"

    partial = client.get(clip["video_url"], headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206
    assert len(partial.content) == 100


# ── zip with the cuts ──────────────────────────────────────────────────────
#
# The zip only exists on the cut-and-splice path -- one layer, no music --
# because only there does each cut become a file. A layered montage goes
# through a filter graph, where the pieces never exist separately.


def _open_zip(resp):
    import io
    import zipfile

    return zipfile.ZipFile(io.BytesIO(resp.content))


def test_a_montage_offers_its_cuts_as_a_zip(isolated, short_sample):
    job_id, _render = run_pipeline(short_sample)
    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    assembled = detail["clips"][0]

    assert assembled["segments_zip_url"]
    resp = client.get(assembled["segments_zip_url"])
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]

    with _open_zip(resp) as zf:
        names = zf.namelist()
        assert names == sorted(names), "the cuts come in chronological order"
        assert len(names) == assembled["meta"]["segments"]
        assert all(n.endswith(".mp4") for n in names)
        assert zf.testzip() is None
        # the name says where the cut came from in the recording
        assert all("m" in n and "s.mp4" in n for n in names)


def test_the_same_stretch_used_twice_goes_into_the_zip_once(
    isolated, short_sample
):
    """The zip is material for re-editing: keeping the same cut twice would add
    nothing for whoever re-edits, even if the video repeats it."""
    job_id = run_analysis(short_sample)
    instant = sorted(set(moments_of(job_id)))[0]
    cut = {"source_t": instant, "start_s": max(0.0, instant - 0.5),
           "duration_s": 1.0}
    render_id = request_render(job_id, [{
        "title": "Repeated",
        "cuts": [{**cut, "at_s": 0.0}, {**cut, "at_s": 1.0}],
    }])
    run_render()

    detail = api().get(f"/api/jobs/{job_id}").json()
    assembled = detail["clips"][0]
    assert assembled["meta"]["segments"] == 2

    with _open_zip(api().get(assembled["segments_zip_url"])) as zf:
        names = zf.namelist()
    assert len(names) == 1, "the same stretch is not stored twice"
    assert render_id


def test_a_clip_without_separate_cuts_returns_404(isolated, short_sample):
    """A layered montage produces no zip: the pieces never become files."""
    job_id = run_analysis(short_sample)
    instant = sorted(set(moments_of(job_id)))[0]
    request_render(job_id, [{
        "title": "Layered",
        "layers": [
            {"clips": [{"at_s": 0, "duration_s": 1.0,
                        "start_s": max(0.0, instant - 0.5),
                        "source_t": instant}]},
            {"name": "text", "clips": [{"at_s": 0, "duration_s": 1.0,
                                        "start_s": 0, "source": "text",
                                        "text": "HI"}]},
        ],
    }])
    run_render()

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    only = detail["clips"][0]
    assert only["meta"]["composed"] is True
    assert only["segments_zip_url"] is None
    assert client.get(f"/api/clips/{only['id']}/cuts.zip").status_code == 404


# ── the whole match's package ──────────────────────────────────────────────


def test_the_match_zip_brings_videos_and_cuts(isolated, short_sample):
    """The overall package must be reachable through the job, without going
    through any clip -- that is what it exists for."""
    job_id, _render = run_pipeline(short_sample)
    client = api()

    # the URL already comes in the listing: no need to open the job or video
    listing = client.get("/api/jobs").json()["jobs"]
    summary = next(j for j in listing if j["id"] == job_id)
    assert summary["zip_url"] == f"/api/jobs/{job_id}/cuts.zip"
    assert summary["has_cuts"] is True

    resp = client.get(summary["zip_url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]

    with _open_zip(resp) as zf:
        names = zf.namelist()
        assert zf.testzip() is None
        assert [n for n in names if "/videos/" in n], "the final videos are missing"
        assert [n for n in names if "/cuts/" in n], "the loose cuts are missing"
        assert all(n.endswith(".mp4") for n in names)
        # each request has its own folder: two requests do not overwrite
        assert all(n.startswith("request_") for n in names)


def test_the_match_zip_separates_the_requests(isolated, short_sample):
    job_id = run_analysis(short_sample)
    request_render(job_id, [montage(job_id, title="A", duration=0.8)])
    run_render()
    request_render(job_id, [montage(job_id, title="B", duration=1.2)])
    run_render()

    with _open_zip(api().get(f"/api/jobs/{job_id}/cuts.zip")) as zf:
        folders = {n.split("/", 1)[0] for n in zf.namelist()}
    assert folders == {"request_01", "request_02"}


def test_a_missing_matchs_zip_returns_404(isolated):
    assert api().get("/api/jobs/doesnotexist/cuts.zip").status_code == 404


# ── the cuts are delivered even when the video does not come out ───────────


def test_a_failing_montage_still_delivers_the_cuts(
    isolated, short_sample, monkeypatch
):
    """Material already cut is not thrown away because the next step broke."""
    from owcore import ffmpeg

    render = service_module("editor", "render")
    original = ffmpeg.concat

    def broken_concat(*a, **k):
        raise ffmpeg.FFmpegError("simulated failure joining the stretches")

    monkeypatch.setattr(render.ffmpeg, "concat", broken_concat)
    try:
        job_id, render_id = run_pipeline(short_sample)
    finally:
        monkeypatch.setattr(render.ffmpeg, "concat", original)

    with session() as s:
        request = s.get(Render, render_id)
        assert request.status == RenderStatus.DONE, request.error
        assembled = request.clips[0]
        assert assembled.key == "", "there should be no final video"
        assert assembled.meta["segments_zip_key"], "the cuts were lost"
        assert assembled.meta["render_error"]
        assert request.stage == RenderStage.DONE

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clip = detail["clips"][0]
    assert clip["video_url"] is None
    assert clip["segments_zip_url"]
    assert client.get(clip["segments_zip_url"]).status_code == 200
    # and the match package still works, just without a video inside
    assert detail["clips_only_cuts"] >= 1
    assert client.get(detail["zip_url"]).status_code == 200


def test_the_video_of_a_clip_without_a_montage_returns_404(
    isolated, short_sample, monkeypatch
):
    from owcore import ffmpeg

    render = service_module("editor", "render")
    monkeypatch.setattr(
        render.ffmpeg, "concat",
        lambda *a, **k: (_ for _ in ()).throw(ffmpeg.FFmpegError("failure")),
    )
    job_id, _render = run_pipeline(short_sample)

    client = api()
    detail = client.get(f"/api/jobs/{job_id}").json()
    clip = detail["clips"][0]
    assert client.get(f"/api/clips/{clip['id']}/video").status_code == 404
