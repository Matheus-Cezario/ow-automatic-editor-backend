"""Rules that cross what more than one detector saw. A pure function, so it
can be covered well without touching video.

This file was once three times larger: it covered the engine that turned
events into video proposals -- kill streaks, "solo wipe", montages per ability.
That phase no longer exists. What is left is the only rule that still fits in
no detector, because it depends on looking at two kinds of event at once.
"""

from __future__ import annotations

from owcore.models import DetectionEvent, EventKind
from owcore.rules import derive_negated_ults


def kills(*ts: float) -> list[DetectionEvent]:
    return [DetectionEvent(kind=EventKind.KILL, t=t) for t in ts]


def test_ult_followed_by_a_kill_becomes_a_negated_ult():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(41.5)
    out = derive_negated_ults(ev, 6.0)
    assert len(out) == 1
    assert out[0].t == 41.5
    assert out[0].meta["delay_s"] == 1.5


def test_ult_without_a_kill_in_the_window_does_not_count():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(50.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_a_kill_before_the_ult_does_not_count():
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=40.0)] + kills(39.0)
    assert derive_negated_ults(ev, 6.0) == []


def test_one_ult_at_a_time_and_the_nearest_kill_closes_the_play():
    """Two negated ultimates in the same match are two moments, and each one
    closes on the first kill that follows it -- not on the last."""
    ev = [DetectionEvent(kind=EventKind.ULT_USED, t=t) for t in (10.0, 30.0)]
    ev += kills(11.0, 12.0, 31.0)
    out = derive_negated_ults(ev, 6.0)
    assert [e.t for e in out] == [11.0, 31.0]
