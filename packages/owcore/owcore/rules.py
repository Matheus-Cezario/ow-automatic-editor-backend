"""Rules that cross what more than one detector saw.

A pure function (events -> events), with no I/O, so it can be tested on its own.

This was once the engine that turned events into video *proposals* -- the
"here is what we can generate" list the app offered ready-made. That phase no
longer exists: the system is an editor, and whoever edits decides what becomes
a video. What is left here is the work no detector can do alone, because it
depends on looking at two kinds of event at the same time.
"""

from __future__ import annotations

from typing import Sequence

from .models import DetectionEvent, EventKind


def _times(events: Sequence[DetectionEvent], kind: EventKind) -> list[float]:
    return sorted(e.t for e in events if e.kind == kind)


def derive_negated_ults(
    events: Sequence[DetectionEvent], window_s: float
) -> list[DetectionEvent]:
    """Crosses `ULT_USED` with `KILL` to produce `ULT_NEGATED`.

    It lives here, and not in the ults detector, on purpose: no detector alone
    sees both kinds of event. Correlation across microservices is the job of
    whoever aggregates them.
    """
    kills = _times(events, EventKind.KILL)
    out: list[DetectionEvent] = []
    for ult in (e for e in events if e.kind == EventKind.ULT_USED):
        after = [k for k in kills if ult.t <= k <= ult.t + window_s]
        if not after:
            continue
        out.append(
            DetectionEvent(
                kind=EventKind.ULT_NEGATED,
                t=round(after[0], 3),
                confidence=round(min(1.0, ult.confidence * 0.9), 3),
                meta={
                    "ult_at": ult.t,
                    "delay_s": round(after[0] - ult.t, 2),
                    "ult": ult.meta.get("ult"),
                    "source": ult.meta.get("source"),
                },
            )
        )
    return out
