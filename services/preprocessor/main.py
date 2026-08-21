"""Microsservico de pre-processamento.

O unico servico que toca o video pesado. Faz *uma* decodificacao e dela sai
tudo que os detectores precisam: um recorte pequeno por detector, em baixo
FPS, mais a faixa de audio. Depois disso, nenhum outro servico abre o arquivo
original -- so o editor, e so nos segundos que interessam.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import session
from owcore.ffmpeg import extract_audio, extract_rois, probe
from owcore.jobs import get_params, set_status
from owcore.models import (
    Artifact,
    Job,
    JobStatus,
    RoiReady,
    STREAM_JOBS,
    STREAM_ROI,
)
from owcore.profiles import load_profile
from owcore.storage import get_storage, local_copy
from owcore.worker import Worker, run_worker

#: quais ROIs cada detector recebe. E aqui que se decide quantos pixels cada
#: microsservico enxerga.
DETECTOR_ROIS: dict[str, list[str]] = {
    "kills": ["kills"],
    "survival": ["health"],
    "ults": ["killfeed"],
    "banner": ["banner"],
}
#: detectores que tambem querem o audio da partida
DETECTOR_AUDIO = {"ults"}


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

        set_status(job_id, JobStatus.PREPROCESSING, stage="lendo o video", progress=0.05)

        work = Path(settings.work_dir) / job_id
        work.mkdir(parents=True, exist_ok=True)
        source = local_copy(video_key, work)

        info = probe(source)
        self.log.info(
            "%s: %.1fs %dx%d @%.1ffps",
            job_id, info.duration_s, info.width, info.height, info.fps,
        )
        with session() as s:
            job = s.get(Job, job_id)
            if job is not None:
                job.duration_s = info.duration_s

        params = get_params(job_id)
        profile = load_profile(params.profile or settings.profile)

        set_status(job_id, stage="recortando as regioes da HUD", progress=0.15)
        wanted = sorted({r for rois in DETECTOR_ROIS.values() for r in rois})
        crops = extract_rois(source, profile.rois(wanted), work / "rois")

        roi_keys: dict[str, str] = {}
        for name, path in crops.items():
            key = f"{job_id}/rois/{name}.mp4"
            storage.put_file(key, path)
            roi_keys[name] = key
            self.log.info("  roi %-9s %6.1f KB", name, path.stat().st_size / 1024)

        audio_key: str | None = None
        if info.has_audio:
            set_status(job_id, stage="extraindo o audio", progress=0.25)
            wav = extract_audio(source, work / "audio.wav")
            if wav is not None:
                audio_key = f"{job_id}/audio.wav"
                storage.put_file(audio_key, wav)

        set_status(job_id, JobStatus.DETECTING, stage="detectando eventos", progress=0.3)

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
