"""Detection of ability kills, read from the killfeed.

Input: the strip in the top-right corner where OW2 stacks its kill lines, plus
the player's own card in the bottom-left corner. Every line always has the same
anatomy:

    [ killer's plate ] [ ability icon ] > [ victim's plate ]

The plates do **not** say which team is which. That was the assumption here for
a long time -- cyan on our side, red on theirs -- and measuring it on a real
match knocked it down: the same player shows up on the blue plate of one line
and on the red plate of another. Blue is whoever killed and red is whoever
died, on both sides of the match. So "a blue plate to the left of a red one"
selects *every* kill in the match, and the detector was reporting a teammate's
ability kills as if they were the player's.

What answers it is the name written on the blue plate, compared against the
name on the player's own card. Not by reading the letters -- see
`owcore.nameplate`: what is asked is only whether the two writings are the same
name.

What says **what** the kill was made with is the icon between the two plates. It
appears in two forms, and the detector reads both:

* a normal ability -- a bright drawing in a dark box;
* an **ultimate** -- a white disc with the drawing in black, the same way as on
  the footer button (and with a blue glow around it).

The icon is compared against `templates/abilities/`, which
`tools/fetch_ability_icons.py` downloads. Without those files there are no
ability kills: a kill without knowing which ability it was is already what the
kill detector reports, and repeating it here would only duplicate the event.

What it still reports without them is every line that appears
(`KILLFEED_LINE`, with whose it was). That is not a moment: it is what tells a
real kill from the crosshair skull of a destroyed deployable -- a Symmetra
turret draws the same skull and puts nothing in the killfeed. The planner
crosses the two (`rules.unconfirmed_kills`).

Measured on the reference recordings (Orisa and Domina, 2558x1438): the right
icon scores between 0.65 and 0.93 and the runner-up between 0.27 and 0.55 --
Orisa's javelin, the spinning javelin, the shove and Domina's ultimate, all
correct.

Then checked over 16 minutes of a full match, which is another problem: there
the killfeed stacks several lines, they slide when a new one arrives, and the
icon crop fails for 3 to 5 seconds at a stretch in the middle of a line's life.
So each line is **tracked** from one frame to the next (see `_Line`) rather than
counted: counting cannot tell "the same line vanished and came back" from "a new
line appeared with the same ability", and those two are exactly the cases that
show up.

That measurement -- 7 of a match's 11 ability kills -- was taken before the
team-colour assumption fell, so its 11 were the kills of *everyone*, and its 7
were mostly teammates'. It is kept here only as the reason the tracking exists.
What the killfeed answers now is a narrower question, and one worth answering:
which of these were the player's.

Two things a single frame cannot be trusted with are now decided over the
line's whole life:

* **which icon it was.** Every frame of the line casts a vote for its best
  match, and the line takes the ability that won most of them (see
  `_Line.ability`). A single frame above the threshold used to be enough to
  name it -- and a line whose icon never got a good frame was lost even when it
  matched the same ability in twenty frames in a row. Voting is also what keeps
  the gun kills out: a gap with no icon matches whatever is closest in each
  frame, and noise never agrees with itself for long;
* **whose it was.** The killer's name is read on every frame where the plate is
  at its widest, and one good reading is enough (see `_Line.killer_is`).
  Keeping only the last reading meant that one frame where two letters touched
  -- a different letter count, and so a score of 0 -- gave the player's own kill
  to "someone else".

The icon itself is compared in shades of grey and at the size it was seen
(`owcore.vision.soft_glyph_on_dark`, `IconBank.rank`) rather than as a
black-and-white cut. The cut is where small icons went: at low resolution the
thin strokes of a drawing come out grey and fall under it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from owcore.models import DetectionEvent, EventKind
from owcore.nameplate import read_name, read_player_name, same_name
from owcore.profiles import Profile
from owcore.vision import (
    Glyph,
    IconBank,
    iter_frames,
    soft_glyph_in_disc,
    soft_glyph_on_dark,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Plate:
    """One of the two coloured boxes of a killfeed line."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def _plates(bgr: np.ndarray, ranges: Sequence[dict], cfg: dict) -> list[Plate]:
    """The boxes of one colour, top to bottom.

    The player's name and the hero portrait break the box into pieces of
    different colours, so the mask is closed horizontally before measuring: what
    matters is the whole rectangle, not the pieces.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    for r in ranges:
        mask |= cv2.inRange(
            hsv, np.array(r["lo"], np.uint8), np.array(r["hi"], np.uint8)
        )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    )
    lo_h, hi_h = cfg.get("row_height_range", [0.04, 0.25])
    min_aspect = float(cfg.get("min_aspect", 1.5))
    min_fill = float(cfg.get("min_fill", 0.5))

    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    found: list[Plate] = []
    for k in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[k])
        if not (lo_h * h < bh < hi_h * h):
            continue
        if bw / max(1, bh) < min_aspect or area / max(1, bw * bh) < min_fill:
            continue
        found.append(Plate(x=x, y=y, w=bw, h=bh))
    return sorted(found, key=lambda p: p.y)


def _line_icon(
    bgr: np.ndarray, killer: Plate, victim: Plate, cfg: dict
) -> tuple[Glyph | None, str]:
    """Crops the icon's glyph from between the two plates.

    The window is a fraction of the **gap** between them, and not an offset in
    pixels: the killfeed shrinks and grows with the recording's resolution, but
    the line's internal proportions do not change. The window runs to the end
    of the gap on purpose, `>` included: the glyph step erases the `>` as the
    rightmost piece of the mark (see `owcore.vision._drop_chevron`), and that
    only works if the whole `>` is always inside.
    """
    lo, hi = cfg.get("icon_span", [0.03, 1.0])
    gap = victim.x - killer.right
    x0 = int(killer.right + lo * gap)
    x1 = int(killer.right + hi * gap)
    half_h = int(float(cfg.get("icon_height", 1.24)) * killer.h / 2)
    cy = int(killer.cy)
    h, w = bgr.shape[:2]
    crop = bgr[max(0, cy - half_h): min(h, cy + half_h), max(0, x0): min(w, x1)]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None, ""
    # ultimate first: it is a white disc, and the disc test is the more
    # specific of the two -- the bright-on-dark glyph would match the whole disc
    # and return a circle instead of the drawing
    glyph = soft_glyph_in_disc(crop)
    if glyph is not None:
        return glyph, "ult"
    return soft_glyph_on_dark(crop, drop_chevron=True), "ability"


def _read_killer(bgr: np.ndarray, killer: Plate, line: "_Line") -> None:
    """Stores the name written on the killer's plate.

    It only reads when the plate is about as wide as the widest seen so far on
    this line. The line slides in and the plate opens up: in the first frames
    the name is still half shown, and comparing half a word with the player's
    full name only ever answers "no". The widest plate is the one with the full
    name.

    Every reading at that width is kept, not just the last: one frame where an
    explosion covers the plate, or where compression glues two letters together,
    reads as a name of another length -- and a single bad reading was enough to
    give the player's own kill to someone else.
    """
    tol = max(2, int(0.1 * killer.h))
    if killer.w < line.plate_w - tol:
        return
    pad = max(1, int(0.12 * killer.h))
    h, w = bgr.shape[:2]
    crop = bgr[max(0, killer.y - pad): min(h, killer.y + killer.h + pad),
               max(0, killer.x): min(w, killer.right)]
    if crop.size == 0:
        return
    line.plate_w = max(line.plate_w, killer.w)
    name = read_name(crop)
    if name:
        line.readings.append((killer.w, name))


def detect_ability_kills(
    roi_video: Path, player_video: Path | None, profile: Profile, icons_dir: Path
) -> list[DetectionEvent]:
    """Only the player's ability kills -- see `read_killfeed`."""
    return [
        e
        for e in read_killfeed(roi_video, player_video, profile, icons_dir)
        if e.kind == EventKind.ABILITY_KILL
    ]


def read_killfeed(
    roi_video: Path, player_video: Path | None, profile: Profile, icons_dir: Path
) -> list[DetectionEvent]:
    """Two readings of the same killfeed, in one pass over its frames.

    * `ABILITY_KILL` -- the player's kills whose icon is a known ability;
    * `KILLFEED_LINE` -- every line that appears, icon or not, with whose it
      was. It is what confirms the crosshair skull: destroying a deployable (a
      Symmetra turret) draws the skull and puts nothing here.

    Both come from the same tracked lines: every line is a `KILLFEED_LINE`,
    and the ones whose icon the frames agree on are also an `ABILITY_KILL`.
    The plain lines need no icon, and so they also work without
    `templates/abilities/` and without the player's name -- then they come out
    with killer "unknown".
    """
    cfg = profile.section("killfeed")
    roi = profile.roi("killfeed")

    bank = IconBank.from_dir(icons_dir)
    if bank:
        log.info("%d ability icon(s) loaded", len(bank))
    else:
        log.warning(
            "no icons in %s -- ability kills are off; the killfeed lines are "
            "still read. Run tools/fetch_ability_icons.py to download them.",
            icons_dir,
        )

    # Without knowing who the player is there is no ability kill to report: the
    # killfeed announces all ten players', and picking the player's needs their
    # name. Returning everything would go back to what was wrong -- a
    # teammate's kill entering the montage as if it were the user's.
    player = read_player_name(player_video, profile.roi("player").fps) if player_video else None
    if player is None:
        log.warning(
            "could not read the player's name on the footer plate -- without it "
            "there is no telling whose each kill was: ability kills are off, and "
            "the killfeed lines come out with an unknown killer"
        )
    else:
        log.info(
            "player name read: %d letter(s), in %.0f%% of frames",
            len(player.letters), 100 * player.agreement,
        )

    name_threshold = float(cfg.get("name_threshold", 0.40))
    threshold = float(cfg.get("icon_threshold", 0.80))
    min_votes = int(cfg.get("icon_min_votes", 2))
    vote_share = float(cfg.get("icon_vote_share", 0.5))
    vote_floor = float(cfg.get("icon_vote_floor", 0.6))
    min_side = int(cfg.get("icon_min_side_px", 8))
    gap_lo, gap_hi = cfg.get("gap_range", [0.4, 4.0])
    hold = float(cfg.get("hold_s", 7.0))
    slide = float(cfg.get("slide_s", 0.6))

    #: each kill is a LINE appearing, and not a frame above the threshold: the
    #: line stays on screen for seconds, so its presence marks no instant at
    #: all. Tracking the line, rather than counting how many there are per
    #: ability, is what separates the two cases that look alike in a count: the
    #: same line disappearing and coming back (the icon crop fails for seconds
    #: at a stretch on a real recording), and a new line appearing with the
    #: ability that was already on screen.
    #:
    #: There used to be a second set of tracks just for ability lines, opened
    #: only on a frame whose icon passed the threshold -- because back then
    #: every plate was crossed with every other and a meaningless pair could
    #: land on a track and split it. `_assign` gives each track one pair per
    #: frame, so the icon now rides on the same tracks as everything else and
    #: every frame of the line gets to vote.
    feed: list[_Line] = []

    def _follow(ln: _Line, killer: Plate, victim: Plate, t: float) -> None:
        ln.last_seen = t
        # the line slides as it enters and the edge wobbles with the
        # compression: the track follows rather than demanding the same pixel
        # every time
        ln.inner_left, ln.inner_right = killer.right, victim.x
        ln.outer_left, ln.outer_right = killer.x, victim.right
        ln.h = killer.h

    for frame in iter_frames(roi_video, fps_hint=roi.fps):
        victims = _plates(frame.bgr, cfg.get("hsv_victim", []), cfg)
        pairs: list[tuple[Plate, Plate]] = []
        for killer in _plates(frame.bgr, cfg.get("hsv_killer", []), cfg):
            for victim in victims:
                if abs(killer.cy - victim.cy) > 0.4 * killer.h:
                    continue
                if victim.x <= killer.right - 2:
                    # red on the left is not a killfeed line at all: the
                    # order is always killer then victim
                    continue
                gap = victim.x - killer.right
                if not (gap_lo * killer.h < gap < gap_hi * killer.h):
                    continue
                pairs.append((killer, victim))

        for (killer, victim), line in _assign(feed, pairs, frame.t, hold, slide):
            if line is None:
                line = _Line(killer.right, victim.x, killer.x, victim.right,
                             killer.h, frame.t, frame.t)
                feed.append(line)
            else:
                _follow(line, killer, victim, frame.t)
            if player is not None:
                _read_killer(frame.bgr, killer, line)
            if bank and player is not None:
                glyph, style = _line_icon(frame.bgr, killer, victim, cfg)
                # a drawing a few pixels tall has lost what tells one icon
                # from its neighbours, and it does not say "I don't know": it
                # agrees, frame after frame, on the wrong one
                if glyph is not None and glyph.side >= min_side:
                    line.vote(*bank.rank(glyph)[:2], style)

    events: list[DetectionEvent] = []
    per_ability: dict[str, int] = {}
    by_others = 0
    whose: dict[str, int] = {}
    for ln in feed:
        mine = ln.killer_is(player, name_threshold)
        killer = {None: "unknown", True: "player", False: "other"}[mine]
        whose[killer] = whose.get(killer, 0) + 1
        events.append(
            DetectionEvent(
                kind=EventKind.KILLFEED_LINE,
                t=round(ln.start, 3),
                meta={"killer": killer},
            )
        )

        named = ln.ability(threshold, min_votes, vote_share, vote_floor)
        if named is None:
            # no icon, or no icon that the frames agree on: almost always a
            # kill with a normal weapon, which draws none
            continue
        if not mine:
            # the line exists and the ability was recognised, but the killer
            # was someone else: not the user's material
            by_others += 1
            continue
        key, score, style = named
        hero, _, ability = key.partition("/")
        per_ability[key] = per_ability.get(key, 0) + 1
        events.append(
            DetectionEvent(
                kind=EventKind.ABILITY_KILL,
                t=round(ln.start, 3),
                confidence=round(min(1.0, 0.5 + 0.5 * score), 3),
                meta={
                    "ability": key,
                    "hero": hero,
                    "name": ability,
                    "icon_score": round(float(score), 3),
                    "icon_votes": len(ln.votes.get(key, [])),
                    "ultimate": style == "ult",
                },
            )
        )
    for key, n in sorted(per_ability.items()):
        log.info("%s: %d kill(s)", key, n)
    if by_others:
        log.info("%d line(s) discarded: the killer was not the player",
                 by_others)
    log.info("killfeed lines: %s", whose or "none")

    events.sort(key=lambda e: e.t)
    return events


def _assign(
    tracks: list["_Line"],
    pairs: list[tuple[Plate, Plate]],
    t: float,
    hold: float,
    slide: float,
) -> list[tuple[tuple[Plate, Plate], "_Line | None"]]:
    """Which track each pair of plates in this frame continues -- or `None`
    for a new line.

    One pair per track and one track per pair, the closest first. Taking, for
    each pair, the first track it could belong to is not enough here: while a
    line slides in only its inner edges count (`_Line.same_as`), and lines with
    no icon all have the same inner edges -- so a new line's track grabbed the
    plates of the lines above it, and one kill came out as four.
    """
    candidates = []
    for ti, ln in enumerate(tracks):
        if t - ln.last_seen > hold:
            continue
        for pi, (killer, victim) in enumerate(pairs):
            if ln.same_as(killer, victim, t, slide):
                cost = (
                    abs(killer.right - ln.inner_left)
                    + abs(victim.x - ln.inner_right)
                    + abs(killer.x - ln.outer_left)
                    + abs(victim.right - ln.outer_right)
                )
                candidates.append((cost, ti, pi))
    taken_tracks: set[int] = set()
    chosen: dict[int, _Line] = {}
    for _cost, ti, pi in sorted(candidates):
        if ti in taken_tracks or pi in chosen:
            continue
        taken_tracks.add(ti)
        chosen[pi] = tracks[ti]
    return [(pair, chosen.get(pi)) for pi, pair in enumerate(pairs)]


@dataclass(slots=True)
class _Line:
    """A killfeed line tracked across frames."""

    #: the line's four horizontal edges. The height (`cy`) is not among them:
    #: when a new kill arrives, the whole stack slides downwards, and an
    #: identity tied to `cy` would switch lines at exactly that moment.
    #:
    #: The two inner ones -- end of the killer's plate, start of the victim's --
    #: are the ones that stay put from the first frame: they bracket the icon,
    #: which has a fixed width. The outer ones depend on the length of the names
    #: and are what tells one line from another, but they are still growing
    #: while the line comes in.
    inner_left: int
    inner_right: int
    outer_left: int
    outer_right: int
    h: int
    start: float
    last_seen: float
    #: for each ability that was a frame's best match, that frame's score
    votes: dict[str, list[float]] = field(default_factory=dict)
    #: how the icon was drawn when it won (an ultimate's disc, or a box)
    styles: dict[str, str] = field(default_factory=dict)
    #: (plate width, letters) of each reading of the name on the killer's
    #: plate. It is what separates the player's kill from a teammate's.
    readings: list[tuple[int, list[np.ndarray]]] = field(default_factory=list)
    #: the widest the killer's plate has been seen -- see `_read_killer`
    plate_w: int = 0

    def vote(self, key: str | None, score: float, style: str) -> None:
        if not key:
            return
        self.votes.setdefault(key, []).append(score)
        if score >= max(self.votes[key]):
            self.styles[key] = style

    def ability(
        self, threshold: float, min_votes: int, share: float, floor: float = 0.0
    ) -> tuple[str, float, str] | None:
        """(key, best score, style) of the ability this line was made with, or
        None when the frames do not agree on one.

        Only frames whose best match reaches `floor` vote at all. The winner
        must have taken at least `min_votes` of those frames and `share` of
        them, and its best frame must reach `threshold`. The vote is what tells
        an icon from no icon: a gun kill's empty gap still matches *something*
        in each frame, but a different something each time and poorly, while a
        real icon wins frame after frame. The floor keeps those poor frames from
        diluting a real icon's share: on a real match a track can run for
        seconds over frames where the crop landed on a hero portrait, and a
        blade that won 13 frames at 0.97 lost the line to 20 frames of noise
        at 0.4-0.5.
        """
        votes = {k: [s for s in v if s >= floor] for k, v in self.votes.items()}
        votes = {k: v for k, v in votes.items() if v}
        if not votes:
            return None
        total = sum(len(v) for v in votes.values())
        key = max(votes, key=lambda k: (len(votes[k]), max(votes[k])))
        wins = votes[key]
        if len(wins) < min_votes or len(wins) < share * total:
            return None
        best = max(wins)
        if best < threshold:
            return None
        return key, best, self.styles.get(key, "ability")

    def killer_is(self, player, threshold: float) -> bool | None:
        """Whether the killer's plate carries the player's name; None when
        that cannot be known (no player name, or the plate never read).

        Only the readings taken at the plate's full width count -- the ones
        before it finished opening show half a name -- and among those, one
        reading that matches is enough. The names of other players stay far
        below the threshold in every frame (0.21 at most against 0.40,
        measured), so a second chance costs nothing in precision; what it
        saves is the player's kill whose best frame happened to be a bad one.
        """
        if player is None or not self.readings:
            return None
        # measured against the widest plate that gave a READING, not the
        # widest seen: that one may be the frame an explosion covered
        tol = max(2, int(0.1 * self.h))
        widest = max(w for w, _name in self.readings)
        full = [name for w, name in self.readings if w >= widest - tol]
        return max(same_name(player.letters, name) for name in full) >= threshold

    def same_as(self, killer: Plate, victim: Plate, t: float, slide: float) -> bool:
        """Whether this line is the same as this pair of plates.

        The tolerance comes from the plate's height, not from a pixel count: the
        killfeed grows with the recording's resolution.

        For `slide` seconds after the line appears, only the inner edges count.
        That is because the line comes in sliding and the plates are still
        opening up -- measured on a real recording, one of the outer edges moves
        16 to 20 pixels between two consecutive frames. Once the entrance is
        over, the outer ones count again: without them, the same player killing
        twice would give a single line, because the inner half of the line is
        identical in both.
        """
        tol = max(4.0, 0.5 * self.h)
        if abs(killer.right - self.inner_left) > tol:
            return False
        if abs(victim.x - self.inner_right) > tol:
            return False
        if t - self.start <= slide:
            return True
        return (
            abs(killer.x - self.outer_left) <= tol
            and abs(victim.right - self.outer_right) <= tol
        )
