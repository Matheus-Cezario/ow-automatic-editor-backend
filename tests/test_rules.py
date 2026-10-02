"""Rules that cross what more than one detector saw. A pure function, so it
can be covered well without touching video.

This file was once three times larger: it covered the engine that turned
events into video proposals -- kill streaks, "solo wipe", montages per ability.
That phase no longer exists. What is left is the only rule that still fits in
no detector, because it depends on looking at two kinds of event at once.
"""

from __future__ import annotations

from owcore.models import DetectionEvent, EventKind
from owcore.rules import derive_negated_ults, unconfirmed_kills


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


# ── a skull needs a killfeed line ──────────────────────────────────────────


def line(t: float, killer: str = "player") -> DetectionEvent:
    return DetectionEvent(kind=EventKind.KILLFEED_LINE, t=t, meta={"killer": killer})


def test_a_skull_with_no_killfeed_line_is_a_deployable():
    """A Symmetra turret draws the skull and puts nothing in the killfeed."""
    ev = kills(10.0, 20.0) + [line(10.4)]
    assert [k.t for k in unconfirmed_kills(ev, 1.0, 2.0)] == [20.0]


def test_a_teammates_line_does_not_confirm_the_players_skull():
    ev = kills(10.0) + [line(10.3, killer="other")]
    assert [k.t for k in unconfirmed_kills(ev, 1.0, 2.0)] == [10.0]


def test_a_line_whose_name_could_not_be_read_confirms():
    """Losing a real kill to a bad crop of the name is worse than letting a
    turret through."""
    ev = kills(10.0) + [line(10.3, killer="unknown")]
    assert unconfirmed_kills(ev, 1.0, 2.0) == []


def test_the_line_may_come_a_little_before_or_after_the_skull():
    ev = kills(10.0, 30.0) + [line(9.2), line(31.9)]
    assert unconfirmed_kills(ev, 1.0, 2.0) == []
    ev = kills(10.0) + [line(12.5)]
    assert len(unconfirmed_kills(ev, 1.0, 2.0)) == 1


def test_without_any_killfeed_line_nothing_is_dropped():
    """No line at all means the killfeed was not read -- not that every kill
    in the match was a turret."""
    assert unconfirmed_kills(kills(10.0, 20.0), 1.0, 2.0) == []


def test_one_line_can_back_up_skulls_close_together():
    """Lines are not paired one to one with skulls: the tracker can merge two
    lines that look alike, and pairing would then throw a real kill away."""
    ev = kills(10.0, 10.9) + [line(10.2)]
    assert unconfirmed_kills(ev, 1.0, 2.0) == []
