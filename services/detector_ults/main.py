"""Ultimate detector microservice -- the player's own and everyone else's.

Two screen regions, because the question is the same and the answer comes from
different places: the **footer button** reports the player's own ultimate (and
says which one it was, by its icon), and the **killfeed** reports the others.
Splitting that into two microservices would only make the end of the analysis
wait for one more.
"""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.config import get_settings
from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_self_ults, detect_ults


class UltsDetector(DetectorWorker):
    name = "detector-ults"
    group = "detector-ults"
    detector = "ults"

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        root = Path(get_settings().templates_dir)
        events = detect_ults(
            artifacts.get("killfeed"), artifacts.get("audio"), profile, root / "ults"
        )
        if "ult" in artifacts:
            events += detect_self_ults(artifacts["ult"], profile, root / "abilities")
        events.sort(key=lambda e: e.t)
        return events


if __name__ == "__main__":
    sys.exit(run_worker(UltsDetector))
