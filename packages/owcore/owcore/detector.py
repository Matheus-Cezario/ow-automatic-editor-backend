"""Base dos microsservicos detectores.

Todos eles fazem a mesma coreografia -- pegar so as suas ROIs, rodar a
deteccao, gravar os eventos e avisar o editor -- entao ela mora aqui e cada
detector implementa apenas o `detect`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bus import get_bus
from .config import get_settings
from .jobs import record_report, save_events, set_status
from .models import (
    Artifact,
    DetectionEvent,
    EditRequested,
    JobParams,
    RoiReady,
    STREAM_EDIT,
    STREAM_ROI,
)
from .profiles import Profile, load_profile
from .storage import local_copy
from .worker import Worker


class DetectorWorker(Worker):
    stream = STREAM_ROI
    #: nome com que o preprocessor enderecou este detector
    detector: str

    def accepts(self, payload: dict[str, Any]) -> bool:
        return payload.get("detector") == self.detector

    # ── contrato do detector concreto ───────────────────────────────────────

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        raise NotImplementedError

    # ── coreografia comum ───────────────────────────────────────────────────

    def handle(self, payload: dict[str, Any]) -> None:
        msg = RoiReady(**payload)
        settings = get_settings()
        work = Path(settings.work_dir) / msg.job_id / self.detector
        work.mkdir(parents=True, exist_ok=True)

        files = self._fetch(msg.artifacts, work)
        profile = load_profile(msg.params.profile or settings.profile)

        events = self.detect(
            msg.job_id, files, profile, msg.params, msg.duration_s
        )
        save_events(msg.job_id, self.detector, events)
        record_report(msg.job_id, self.detector, len(events))
        set_status(msg.job_id, stage=f"{self.detector}: {len(events)} evento(s)")

        get_bus().publish(
            STREAM_EDIT, EditRequested(job_id=msg.job_id).model_dump()
        )

    def _fetch(self, artifacts: list[Artifact], work: Path) -> dict[str, Path]:
        """Baixa os artefatos e indexa por nome de ROI (ou pelo tipo, no audio)."""
        out: dict[str, Path] = {}
        for a in artifacts:
            label = a.meta.get("roi") or a.kind
            out[label] = local_copy(a.key, work)
        return out

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """Um detector que falha nao pode travar o job: ele registra o erro e
        libera o editor, que monta com os eventos que os outros acharam."""
        job_id = payload.get("job_id")
        if not job_id:
            return
        record_report(job_id, self.detector, 0, error=str(exc)[:1000])
        get_bus().publish(STREAM_EDIT, EditRequested(job_id=job_id).model_dump())
