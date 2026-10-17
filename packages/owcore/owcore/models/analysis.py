"""What the analysis works with: screen crops, artifacts, detected events, job
parameters and the per-moment thumbnails."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import EventKind

# ---------------------------- message models -------------------------------


class RoiSpec(BaseModel):
    """A normalised (0..1) crop of the screen, plus the downscale applied."""

    name: str
    x: float
    y: float
    w: float
    h: float
    fps: float = 10.0
    width_px: int = 320
    #: if True, the crop is the whole screen scaled down (used to detect death)
    fullscreen: bool = False

    def relative(self, sx: float, sy: float) -> tuple[float, float]:
        """Where a point on the **screen** falls inside this crop, in 0..1.

        This is for elements anchored to a fixed screen position rather than to
        the middle of the ROI -- the crosshair, for instance, sits at (0.5, 0.5)
        of the screen, and the kills ROI is shifted upwards, so inside it the
        crosshair is not centred. Deriving that from the geometry itself avoids
        a second number in the profile that would have to be corrected in step
        every time the ROI moved.
        """
        return ((sx - self.x) / self.w, (sy - self.y) / self.h)


#: Name of the output that becomes the editor's proxy. It is no detector's
#: ROI: it is the whole screen scaled down, hung off the same decode because
#: decoding the heavy video is already paid for -- asking for a second pass just
#: for this would spend again exactly what the system saves most.
PROXY_ROI = "proxy"


def proxy_roi() -> RoiSpec:
    """The whole screen, small and at low FPS: enough to edit with.

    The final cut still comes out of the original recording, at full quality --
    this exists only so the monitor can seek to an instant without dragging half
    a gigabyte over HTTP on every scrub.
    """
    return RoiSpec(
        name=PROXY_ROI,
        x=0.0,
        y=0.0,
        w=1.0,
        h=1.0,
        fps=24.0,
        width_px=640,
        fullscreen=True,
    )


class Artifact(BaseModel):
    """A reference to a blob in storage."""

    key: str
    kind: str
    meta: dict[str, Any] = Field(default_factory=dict)


class DetectionEvent(BaseModel):
    kind: EventKind
    t: float  # seconds since the start of the video
    confidence: float = 1.0
    meta: dict[str, Any] = Field(default_factory=dict)


class BeatGrid(BaseModel):
    bpm: float
    beats: list[float] = Field(default_factory=list)


class JobParams(BaseModel):
    """Parameters of the **analysis**: how to read the match.

    They apply to the whole job. There used to be many more -- how many kills
    made a streak, how many made a "solo wipe" -- because the analysis ended by
    proposing ready-made videos. It does not propose any more: it delivers the
    moments, and grouping them is the job of whoever edits. What is left here is
    what still changes **which events exist**.

    An unknown field is ignored (not `extra="forbid"`): a match recorded when
    those parameters existed still opens.
    """

    #: an enemy ultimate followed by a kill within this window counts as a
    #: negated ultimate
    ult_negate_window_s: float = 6.0
    profile: str | None = None

#: Events worth a thumbnail in the editor's sidebar: the ones that become
#: blocks. Low health and interruption are the context of the play, not the
#: play.
#:
#: This has to match the list the editor shows (`_usefulMoments`, in the app):
#: a kind that appears there and is missing here becomes a card with no frame
#: forever -- nobody extracts the thumbnail, and the app keeps asking for it
#: until it gives up.
THUMB_KINDS = (
    EventKind.KILL,
    EventKind.HEADSHOT,
    EventKind.ABILITY_KILL,
    EventKind.SLEEP,
    EventKind.STUN,
    EventKind.ULT_NEGATED,
    EventKind.ESCAPE,
)


def frame_key(job_id: str, t: float) -> str:
    """Where the thumbnail for instant `t` of that match lives.

    It derives from the instant rather than becoming a database column: writer
    and reader arrive at the same key on their own, and one more thumbnail is
    not one more migration. The rounding to hundredths is the same the API uses
    to talk about time, so the app asks for exactly what the worker wrote.
    """
    return f"{job_id}/frames/{t:.2f}.jpg"
