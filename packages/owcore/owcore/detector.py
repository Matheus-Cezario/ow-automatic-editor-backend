"""Base for the detector microservices.

They all perform the same choreography -- take only their own ROIs, run the
detection, store the events and notify the editor -- so it lives here and each
detector implements just `detect`.
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
    #: the name the preprocessor addressed this detector by
    detector: str

    def accepts(self, payload: dict[str, Any]) -> bool:
        return payload.get("detector") == self.detector

    # -- contract of the concrete detector ----------------------------------

    def detect(
        self,
        job_id: str,
        artifacts: dict[str, Path],
        profile: Profile,
        params: JobParams,
        duration_s: float,
    ) -> list[DetectionEvent]:
        raise NotImplementedError

    # -- shared choreography ------------------------------------------------

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
        """Downloads the artifacts, indexed by ROI name (or by kind, for audio)."""
        out: dict[str, Path] = {}
        for a in artifacts:
            label = a.meta.get("roi") or a.kind
            out[label] = local_copy(a.key, work)
        return out

    def on_error(self, payload: dict[str, Any], exc: Exception) -> None:
        """A detector that fails must not stall the job: it records the error
        and releases the editor, which builds with what the others found."""
        job_id = payload.get("job_id")
        if not job_id:
            return
        record_report(job_id, self.detector, 0, error=str(exc)[:1000])
        get_bus().publish(STREAM_EDIT, EditRequested(job_id=job_id).model_dump())
