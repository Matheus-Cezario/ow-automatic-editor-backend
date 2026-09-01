"""Reading the names the HUD writes -- without reading them.

The killfeed does not say whose kill it was in any way a colour can answer.
Measured on a real match: the same player appears on the blue side of one line
and on the red side of another, so the plates are painted **killer** and
**victim**, not by team. The only thing that separates the player's kill from a
teammate's is the name written on the killer's plate.

Which does not mean OCR. Recognising the letters would be answering a much
harder question than the one being asked: it is not *what* the name says that
matters, it is whether two names written on screen are **the same name**. So
each name is reduced to the sequence of drawings of its letters, and two names
are compared letter by letter, in order. No alphabet, no language, no model --
and it survives the accents (`TONKA` with the umlaut) and the odd characters
that a battle tag is full of.

The two writings do not come out the same size: the name on the player's card
is bigger than the one in the killfeed, and the fonts are not tracked the same
way -- the same name comes out 50% wider in one than in the other, normalised
by height. That is why the comparison is per letter, with each letter
normalised on its own: what is stable between the two is the *shape* of each
letter, not the spacing between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: side of the box each letter is normalised into, in pixels. Big enough for
#: `O` and `Q` not to become the same drawing, small enough that a letter read
#: from the killfeed -- which is around 16 pixels tall on a 1440p recording --
#: is not being invented by the interpolation.
LETTER_BOX = 16

#: the HUD writes names in near-white, on plates of every colour there is.
#: Asking for *little colour and much light* selects the letters and leaves the
#: plate, the hero portraits and the background out of it, without having to
#: know which colour the plate is.
_TEXT_LO = np.array([0, 0, 185], np.uint8)
_TEXT_HI = np.array([179, 75, 255], np.uint8)


def text_mask(bgr: np.ndarray) -> np.ndarray:
    """What is written, in white on black."""
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), _TEXT_LO, _TEXT_HI)


def _letter_like(mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """The blobs that could be a letter, left to right.

    The filters are all proportions, never pixel counts: the HUD grows with the
    recording's resolution. What they throw out is the glare of an explosion
    (fills the whole strip), the badge beside the name (taller than a capital),
    and the light parts of the hero portrait (a blob with holes, far from
    filling its own bounding box).
    """
    h = mask.shape[0]
    # A letter does not always come out in one piece: the crossbar of an `A`
    # separates from its apex when the strokes are thin, and at the size the
    # killfeed writes at the compression is enough to do it. A vertical close
    # puts them back together -- the kernel is one pixel WIDE on purpose, so it
    # cannot join two letters standing side by side, only the parts of one
    # standing on top of each other.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    )
    count, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for k in range(1, count):
        x, y, w, bh, area = (int(v) for v in stats[k])
        if not (0.18 * h <= bh <= 0.85 * h):
            continue
        if w > 2.2 * bh or w < 0.08 * bh:
            continue
        if area < 0.14 * w * bh:
            continue
        found.append((x, y, w, bh, (labels[y:y + bh, x:x + w] == k).view(np.uint8)))
    return sorted(found, key=lambda blob: blob[0])


def read_name(bgr: np.ndarray) -> list[np.ndarray] | None:
    """The name written in this crop, as the drawing of each of its letters.

    The crop can carry more than the name -- the rank number and the emblem on
    the player's card, the hero portraits on the killfeed plate. What is taken
    is the **longest word**: letters lined up at the same height, with the gaps
    between them that a word has. Everything else on those plates is one or two
    blobs, never nine in a row.

    Returns None when there is nothing there that looks like writing, which is
    the honest answer for a HUD covered by an explosion.
    """
    letters = _letter_like(text_mask(bgr))
    if len(letters) < 3:
        return None
    heights = sorted(blob[3] for blob in letters)
    cap = heights[len(heights) // 2]

    words: list[list] = []
    current = [letters[0]]
    for prev, nxt in zip(letters, letters[1:]):
        together = (
            nxt[0] - (prev[0] + prev[2]) <= 0.9 * cap
            and abs(nxt[3] - cap) <= 0.45 * cap
            and abs((nxt[1] + nxt[3] / 2) - (prev[1] + prev[3] / 2)) <= 0.4 * cap
        )
        if together:
            current.append(nxt)
        else:
            words.append(current)
            current = [nxt]
    words.append(current)

    word = max(words, key=len)
    if len(word) < 3:
        return None
    return [
        cv2.resize(blob[4] * 255, (LETTER_BOX, LETTER_BOX),
                   interpolation=cv2.INTER_AREA)
        for blob in word
    ]


def same_name(a: list[np.ndarray] | None, b: list[np.ndarray] | None) -> float:
    """0..1: how much two readings are the same name.

    Different letter counts settle it on their own -- and they settle most of
    the cases, since two battle tags rarely have the same length. When they do
    (`CENOURAET` and `MATEUSKCZ`, both nine, in the same match), what separates
    them is the shape of each letter.

    The worst letter pulls the score down on purpose: a name of the same length
    with one letter different is still another name, and averaging alone would
    let it through on the strength of the eight that do match.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    scores = []
    for one, other in zip(a, b):
        x = one.astype(np.float32) / 255
        y = other.astype(np.float32) / 255
        union = float(np.maximum(x, y).sum())
        scores.append(float(np.minimum(x, y).sum() / union) if union else 0.0)
    return float(np.mean(scores) * min(scores) ** 0.5)


@dataclass(slots=True)
class PlayerName:
    """The name on the player's own card, read off the recording."""

    #: the letters, ready to compare with `same_name`
    letters: list[np.ndarray]
    #: in how many of the frames read this same name came out. It is the
    #: measure of trust: a HUD that is there says the same thing hundreds of
    #: times, and a region pointed at the wrong place says something different
    #: every frame.
    agreement: float

    def matches(self, other: list[np.ndarray] | None, threshold: float) -> bool:
        return same_name(self.letters, other) >= threshold


def read_player_name(roi_video: Path, fps_hint: float | None = None) -> PlayerName | None:
    """Learns the player's name from the card in the bottom-left corner.

    It reads it in every frame and keeps the **most repeated** reading, rather
    than the first: the card is covered now and then -- a chat message, the
    scoreboard, the HUD blinking out during a killcam -- and one bad frame
    would poison every comparison afterwards. What repeats over a whole match
    is the name.

    The reading with the most common letter count wins, and among those the one
    that agrees most with its peers is taken: even with the count right, a
    frame with the plate half-covered gives deformed letters.
    """
    from .vision import iter_frames  # here to keep the import graph acyclic

    readings = [
        name for name in (read_name(f.bgr) for f in iter_frames(roi_video, fps_hint))
        if name
    ]
    if not readings:
        return None

    counts: dict[int, int] = {}
    for name in readings:
        counts[len(name)] = counts.get(len(name), 0) + 1
    size = max(counts, key=lambda n: counts[n])
    peers = [name for name in readings if len(name) == size]

    best, best_score = peers[0], -1.0
    for candidate in peers:
        score = float(np.mean([same_name(candidate, other) for other in peers]))
        if score > best_score:
            best, best_score = candidate, score
    return PlayerName(letters=best, agreement=counts[size] / len(readings))
