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


def unconfirmed_kills(
    events: Sequence[DetectionEvent], before_s: float, after_s: float
) -> list[DetectionEvent]:
    """The crosshair skulls that no killfeed line backs up.

    The skull also shows when the player destroys a deployable -- Symmetra's
    turrets, a teleporter, a Torbjorn turret -- and those put nothing in the
    killfeed. So a skull counts as a kill only if a line the player could have
    made appears near it: one with the player's name on the killer's plate, or
    one whose name could not be read (a real kill lost to a bad crop is worse
    than a turret let through).

    Any line in the window confirms, without pairing lines to skulls one to
    one: the tracker can merge two lines that look alike, and pairing would
    then throw a real kill away.

    Returns nothing when there is no killfeed evidence at all. A match with
    skulls and not a single line means the killfeed was not read (a region off
    target, a recording without it) -- not that every kill was a turret.
    """
    lines = [e for e in events if e.kind == EventKind.KILLFEED_LINE]
    if not lines:
        return []
    mine = sorted(
        e.t for e in lines if e.meta.get("killer", "unknown") != "other"
    )
    return [
        k
        for k in events
        if k.kind == EventKind.KILL
        and not any(k.t - before_s <= t <= k.t + after_s for t in mine)
    ]
