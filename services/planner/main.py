"""The microservice that closes the analysis.

It is where the detectors meet. Each works on a region of the screen and
finishes when it finishes; this service waits for all of them, crosses what can
only be seen by looking at two kinds of event at once, and calls the match
analysed.

It used to be the "planner": on top of that, it wrote a list of videos ready to
generate, and the app offered that list. It does not write it any more. The
system is a video editor with automatic event detection -- what the analysis
delivers is the match's timeline, and who decides what becomes a video is
whoever edits.

Once the analysis is over the job sits at `ready` and the thumbnails are
requested: the editor's shelf of moments lives off them.
"""

from __future__ import annotations

import sys
from typing import Any

from owcore.bus import get_bus
from owcore.config import get_settings
from owcore.db import session
from owcore.jobs import (
    all_detectors_done,
    claim_for_planning,
    expected_detectors,
    load_events,
    reported_detectors,
    save_events,
    set_status,
    stale_detecting_jobs,
)
from owcore.models import (
    STREAM_EDIT,
    STREAM_THUMBS,
    EventKind,
    Job,
    JobParams,
    JobStatus,
    ThumbsRequested,
)
from owcore.rules import derive_negated_ults
from owcore.worker import Worker, run_worker


class Planner(Worker):
    name = "planner"
    stream = STREAM_EDIT
    group = "planner"

    # -- normal entry: a detector finished ----------------------------------

    def handle(self, payload: dict[str, Any]) -> None:
        job_id = payload["job_id"]
        expected = expected_detectors(job_id)
        if not all_detectors_done(job_id, expected):
            missing = set(expected) - reported_detectors(job_id)
            self.log.info("job %s ainda espera: %s", job_id, ", ".join(sorted(missing)))
            return
        self._close_analysis(job_id)

    # -- rescue: some detector died and never reported ----------------------

    def idle(self) -> None:
        timeout = get_settings().detector_timeout_s
        for job_id in stale_detecting_jobs(timeout):
            missing = set(expected_detectors(job_id)) - reported_detectors(job_id)
            self.log.warning(
                "job %s parado ha mais de %.0fs sem %s; fecho com o que tenho",
                job_id, timeout, ", ".join(sorted(missing)) or "ninguem",
            )
            try:
                self._close_analysis(job_id)
            except Exception:
                self.log.exception("resgate do job %s falhou", job_id)

    # -- end of the analysis ------------------------------------------------

    def _close_analysis(self, job_id: str) -> None:
        # the detectors finish almost together and all of them notify; the
        # atomic claim guarantees the crossing runs exactly once
        if not claim_for_planning(job_id):
            self.log.debug("job %s ja foi fechado", job_id)
            return

        with session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            params = JobParams(**(job.params or {}))

        events = load_events(job_id)
        derived = self._cross_detectors(job_id, events, params)

        total = len(events) + len(derived)
        self.log.info(
            "job %s: %d evento(s), %d deles cruzados aqui",
            job_id, total, len(derived),
        )
        set_status(
            job_id, JobStatus.READY,
            stage=(
                f"{total} momento(s) encontrados — abra o editor"
                if total else "nenhum momento encontrado"
            ),
            progress=1.0,
        )
        self._request_thumbnails(job_id)

    def _cross_detectors(self, job_id: str, events, params: JobParams) -> list:
        """The events that only exist by crossing two detectors.

        A negated ultimate is an enemy ultimate followed by a kill: the ults
        detector sees the first half, the kills detector the second, and neither
        sees the play. Only here, with everything in one list, does it appear.

        They are **stored as events**, and not recomputed on every read: that is
        what puts them on the editor's shelf alongside the others. While
        proposals existed they were born and died inside the proposal generator,
        and the editor never got to see them.
        """
        if any(e.kind == EventKind.ULT_NEGATED for e in events):
            return []  # reprocessing: they were already stored
        derived = derive_negated_ults(events, params.ult_negate_window_s)
        if derived:
            save_events(job_id, "planner", derived)
        return derived

    def _request_thumbnails(self, job_id: str) -> None:
        """Notifies whoever extracts the moments' thumbnails.

        It goes from here because this is where the analysis is known to be
        over -- and it is no more than publishing a message: this service still
        opens no video at all.
        """
        get_bus().publish(
            STREAM_THUMBS, ThumbsRequested(job_id=job_id).model_dump()
        )


if __name__ == "__main__":
    sys.exit(run_worker(Planner))
