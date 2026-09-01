"""Microservice of the crosshair region: kills and critical hits.

One crop, two readings. The magenta skull says somebody died; the red X marker
says the shot landed on the head. They are the same handful of pixels at the
same instant, so asking the preprocessor for two identical crops would be
paying twice for the same work.
"""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.config import get_settings
from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_headshots, detect_kills


class KillsDetector(DetectorWorker):
    name = "detector-kills"
    group = "detector-kills"
    detector = "kills"

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        templates = Path(get_settings().templates_dir) / "kills"
        roi = artifacts["kills"]
        events = detect_kills(roi, profile, templates)
        events += detect_headshots(roi, profile)
        events.sort(key=lambda e: e.t)
        return events


if __name__ == "__main__":
    sys.exit(run_worker(KillsDetector))
