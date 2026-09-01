"""Killfeed microservice: which ability each kill was made with.

One detector per *screen region* -- not per ability. The killfeed line is a
single one, and what changes from one kill to the next is the icon in the
middle of it. Adding a new ability to the repertoire means adding a file to
`templates/abilities/`, not a microservice.
"""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.config import get_settings
from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_ability_kills


class KillfeedDetector(DetectorWorker):
    name = "detector-killfeed"
    group = "detector-killfeed"
    detector = "killfeed"

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        icons = Path(get_settings().templates_dir) / "abilities"
        return detect_ability_kills(
            artifacts["killfeed"], artifacts.get("player"), profile, icons
        )


if __name__ == "__main__":
    sys.exit(run_worker(KillfeedDetector))
