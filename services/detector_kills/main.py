"""Microsservico detector de eliminacoes."""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.config import get_settings
from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_kills


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
        return detect_kills(artifacts["kills"], profile, templates)


if __name__ == "__main__":
    sys.exit(run_worker(KillsDetector))
