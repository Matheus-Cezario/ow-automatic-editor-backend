"""Microsservico detector dos avisos do rodape.

Um detector por *regiao da tela*, nao por habilidade: a faixa do rodape e uma
so, e cada habilidade se distingue pelo icone. Acrescentar uma habilidade nova
e acrescentar um molde ao profile -- nao um microsservico.
"""

from __future__ import annotations

import sys
from pathlib import Path

from owcore.config import REPO_ROOT
from owcore.detector import DetectorWorker
from owcore.models import DetectionEvent, JobParams
from owcore.profiles import Profile
from owcore.worker import run_worker

from detect import detect_abilities


class BannerDetector(DetectorWorker):
    name = "detector-banner"
    group = "detector-banner"
    detector = "banner"

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        return detect_abilities(
            artifacts["banner"], profile, REPO_ROOT / "config" / "shapes"
        )


if __name__ == "__main__":
    sys.exit(run_worker(BannerDetector))
