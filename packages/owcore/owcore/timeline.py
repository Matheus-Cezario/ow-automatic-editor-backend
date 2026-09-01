"""From the timeline the user built to the list of pieces to cut.

Pure rules, no ffmpeg and no database -- which is why the whole montage can be
tested without opening a single video.

The difference from `rules.py` is who decides. There the system reads events;
here it decides nothing: it receives blocks already positioned and only has to
say what goes to ffmpeg, in what order, and what to do with the empty space
between one block and the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import MIN_CUT_S, TimelineCut

#: A gap smaller than this does not become black: it is less than one frame at
#: 24 fps, and filling it would cost a whole re-encode for nobody to see the
#: difference. The next block simply starts that much earlier.
MIN_GAP_S = 0.04


@dataclass(slots=True)
class Piece:
    """A piece of the final video, in the order it will be concatenated.

    It is either a cut from the recording (`black=False`, with `start_s`/`end_s`
    in the recording) or the black filling a gap the user left.
    """

    duration_s: float
    start_s: float = 0.0
    end_s: float = 0.0
    black: bool = False
    #: instant of the moment the cut came from -- only used to name the file
    source_t: float = 0.0
    kind: str = ""

    @property
    def is_cut(self) -> bool:
        return not self.black


def plan(
    cuts: Sequence[TimelineCut],
    *,
    source_duration_s: float = 0.0,
    min_gap_s: float = MIN_GAP_S,
) -> list[Piece]:
    """The pieces, in order, covering the whole video with no hole.

    What the user left empty becomes black with the music playing over it --
    which is what any editor does, and what keeps the screen's promise: every
    block lands exactly on the point of the music where it was placed. Closing
    the gap by joining the blocks would move every one that follows.

    A cut running past the end of the recording is trimmed, and whatever is
    left of its slot also becomes black -- again, so as not to push the ones
    behind it.
    """
    ordered = sorted(cuts, key=lambda c: c.at_s)
    pieces: list[Piece] = []
    cursor = 0.0

    for cut in ordered:
        gap = cut.at_s - cursor
        if gap >= min_gap_s:
            pieces.append(Piece(duration_s=gap, black=True))
            cursor += gap

        duration = cut.duration_s
        if source_duration_s > 0:
            duration = min(duration, max(0.0, source_duration_s - cut.start_s))

        if duration >= MIN_CUT_S:
            pieces.append(
                Piece(
                    duration_s=duration,
                    start_s=cut.start_s,
                    end_s=cut.start_s + duration,
                    source_t=cut.source_t,
                    kind=cut.kind,
                )
            )

        # whatever the trim ate (or the whole block, if it fell outside the
        # recording) becomes black: the next block stays where it was marked
        leftover = cut.duration_s - max(0.0, duration)
        if leftover >= min_gap_s:
            pieces.append(Piece(duration_s=leftover, black=True))

        cursor = cut.until_s

    # trailing black adds nothing: the video ends on the last cut
    while pieces and pieces[-1].black:
        pieces.pop()

    # two blacks in a row are one black -- every piece costs an encode
    merged: list[Piece] = []
    for piece in pieces:
        if piece.black and merged and merged[-1].black:
            merged[-1].duration_s += piece.duration_s
            continue
        merged.append(piece)
    return merged


def total_duration_s(pieces: Sequence[Piece]) -> float:
    return sum(p.duration_s for p in pieces)


def snap(value: float, beats: Sequence[float], tolerance_s: float = 0.12) -> float:
    """Snaps an instant to the nearest beat, if there is one close by.

    It lives here, and not only in the app, because the server needs the same
    answer: the app snaps while the user drags, and whoever checks afterwards
    has to arrive at the same number.
    """
    if not beats:
        return value
    best = min(beats, key=lambda b: abs(b - value))
    return best if abs(best - value) <= tolerance_s else value
