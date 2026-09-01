"""Preprocessing microservice.

The only service that touches the heavy video. It does *one* decode, and out of
it comes everything the detectors need: a small crop per detector, at low FPS,
plus the audio track. After that, no other service opens the original file --
only the editor, and only for the seconds that matter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore.audio import waveform_of
from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import session
from owcore.ffmpeg import extract_audio, extract_rois, probe
from owcore.jobs import get_params, set_status
from owcore.models import (
    PROXY_ROI,
    Artifact,
    Job,
    JobStatus,
    RoiReady,
    STREAM_JOBS,
    STREAM_ROI,
    proxy_roi,
)
from owcore.profiles import load_profile
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

#: which ROIs each detector receives. This is where it is decided how many
#: pixels each microservice gets to see.
#:
#: One crop can go to more than one detector -- `killfeed` goes to two -- and
#: that costs no extra decoding: the crop is made once and the same blob is
#: addressed to both. It is the questions that differ: `ults` looks for an enemy
#: ultimate icon across the whole strip, `killfeed` reads the line's anatomy to
#: say which ability each kill was made with.
DETECTOR_ROIS: dict[str, list[str]] = {
    "kills": ["kills"],
    "survival": ["health"],
    "ults": ["killfeed", "ult"],
    "banner": ["banner"],
    # the player's card comes along: it is where the killfeed detector reads
    # whose kill it was
    "killfeed": ["killfeed", "player"],
}
#: detectors that also want the match audio
DETECTOR_AUDIO = {"ults"}

# Which slice of the bar belongs to each phase. The numbers come from timing a
# real 11-minute match (1080p60, 502 MB) end to end: 82s downloading, 356s
# cropping, 14s on audio and uploads, 29s in the detectors -- 482s in all. The
# old bar gave cropping the range 0.15 to 0.25, that is, three quarters of the
# time squeezed into a tenth of the bar; it sat on the same number for six
# minutes, which reads as frozen, not as working.
#
# The slices are approximate on purpose -- the proportion changes with the
# recording's resolution and with the machine -- but the *order of magnitude* is
# stable: cropping always dominates. And a bar moving at the wrong rate is still
# far better than one that does not move.
BAND_DOWNLOAD = (0.0, 0.17)
BAND_CROP = (0.17, 0.90)
BAND_AUDIO = (0.90, 0.94)
DETECTION_START = 0.94

#: how far the bar has to move to be worth a database write. ffmpeg reports
#: twice a second and the download on every chunk: storing all of it would be
#: hundreds of UPDATEs per job, to move a number the screen reads every two
#: seconds.
BAR_STEP = 0.01


def _progress_reporter(job_id: str):
    """Returns the function that carries a phase's progress to the job's bar."""
    last = 0.0

    def report(stage: str, band: tuple[float, float], fraction: float) -> None:
        nonlocal last
        lo, hi = band
        value = lo + (hi - lo) * max(0.0, min(1.0, fraction))
        if value - last < BAR_STEP:
            return
        last = value
        set_status(job_id, stage=stage, progress=value)

    return report


class Preprocessor(Worker):
    name = "preprocessor"
    stream = STREAM_JOBS
    group = "preprocessor"

    def handle(self, payload: dict[str, Any]) -> None:
        job_id = payload["job_id"]
        settings = get_settings()
        storage = get_storage()

        with session() as s:
            job = s.get(Job, job_id)
            if job is None:
                self.log.warning("job %s sumiu; ignorando", job_id)
                return
            video_key = job.video_key

        set_status(job_id, JobStatus.PREPROCESSING, stage="baixando o video", progress=0.0)

        work = Path(settings.work_dir) / job_id
        work.mkdir(parents=True, exist_ok=True)
        report = _progress_reporter(job_id)
        source = local_copy(
            video_key, work,
            on_progress=lambda f: report("baixando o video", BAND_DOWNLOAD, f),
        )

        info = probe(source)
        self.log.info(
            "%s: %.1fs %dx%d @%.1ffps",
            job_id, info.duration_s, info.width, info.height, info.fps,
        )
        with session() as s:
            job = s.get(Job, job_id)
            if job is not None:
                job.duration_s = info.duration_s
                job.fps = info.fps
                job.width = info.width
                job.height = info.height

        params = get_params(job_id)
        profile = load_profile(params.profile or settings.profile)

        set_status(job_id, stage="recortando as regioes da HUD",
                   progress=BAND_CROP[0])
        wanted = sorted({r for rois in DETECTOR_ROIS.values() for r in rois})
        # the editor's proxy comes along: one more output in the same decode,
        # instead of a second pass over the heavy video
        crops = extract_rois(
            source, [*profile.rois(wanted), proxy_roi()], work / "rois",
            on_progress=lambda f: report(
                "recortando as regioes da HUD", BAND_CROP, f
            ),
        )

        roi_keys: dict[str, str] = {}
        proxy_key = ""
        for name, path in crops.items():
            if name == PROXY_ROI:
                proxy_key = storage.put_file(f"{job_id}/proxy.mp4", path)
                self.log.info(
                    "  proxy    %6.1f MB (original: %.1f MB)",
                    path.stat().st_size / 1024 / 1024,
                    source.stat().st_size / 1024 / 1024,
                )
                continue
            key = f"{job_id}/rois/{name}.mp4"
            storage.put_file(key, path)
            roi_keys[name] = key
            self.log.info("  roi %-9s %6.1f KB", name, path.stat().st_size / 1024)

        audio_key: str | None = None
        waveform_points: list[float] = []
        if info.has_audio:
            set_status(job_id, stage="extraindo o audio",
                       progress=BAND_AUDIO[0])
            wav = extract_audio(source, work / "audio.wav")
            if wav is not None:
                audio_key = f"{job_id}/audio.wav"
                storage.put_file(audio_key, wav)
                # the match's waveform goes to the editor to draw: it is where
                # the shot and the explosion show, so a cut can be matched to
                # the game's sound
                waveform_points, _dur = waveform_of(wav)

        with session() as s:
            job = s.get(Job, job_id)
            if job is not None:
                job.proxy_key = proxy_key
                job.waveform = waveform_points

        set_status(job_id, JobStatus.DETECTING, stage="detectando eventos",
                   progress=DETECTION_START)

        bus = get_bus()
        for detector, roi_names in DETECTOR_ROIS.items():
            artifacts = [
                Artifact(key=roi_keys[n], kind="roi", meta={"roi": n})
                for n in roi_names
                if n in roi_keys
            ]
            if detector in DETECTOR_AUDIO and audio_key:
                artifacts.append(Artifact(key=audio_key, kind="audio"))
            bus.publish(
                STREAM_ROI,
                RoiReady(
                    job_id=job_id,
                    detector=detector,
                    artifacts=artifacts,
                    duration_s=info.duration_s,
                    params=params,
                ).model_dump(),
            )

        self.log.info("job %s distribuido para os detectores", job_id)


if __name__ == "__main__":
    sys.exit(run_worker(Preprocessor))
