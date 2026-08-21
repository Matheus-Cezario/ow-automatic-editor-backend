"""Microsservico detector de sobrevivencia (vida baixa, morte, fuga)."""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_survival


class SurvivalDetector(DetectorWorker):
    name = "detector-survival"
    group = "detector-survival"
    detector = "survival"

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        return detect_survival(artifacts["health"], profile)


if __name__ == "__main__":
    sys.exit(run_worker(SurvivalDetector))
