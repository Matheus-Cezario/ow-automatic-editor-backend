"""Domain models (pydantic) and tables (SQLAlchemy)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------- enums -----------------------------------


class JobStatus(StrEnum):
    """Lifecycle of the *analysis*. It runs once, on its own, and ends at
    `READY`: from there the job waits for the user to open the editor."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    DETECTING = "detecting"
    READY = "ready"
    FAILED = "failed"


class RenderStatus(StrEnum):
    """Lifecycle of *one render request*. A job has as many as it likes."""

    PENDING = "pending"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


class JobStage(StrEnum):
    """Where the analysis is, finer than [JobStatus].

    A code, never a sentence: the database stores what happened and the app
    decides how to say it. Numbers do not go in here either -- how many
    moments were found is `Job.n_moments`, how far along it is `progress`.
    """

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    CROPPING = "cropping"
    EXTRACTING_AUDIO = "extracting_audio"
    DETECTING = "detecting"
    #: the detectors are done and the planner is crossing their events
    PLANNING = "planning"
    READY = "ready"
    ERROR = "error"


class RenderStage(StrEnum):
    """Where a render request is. A code, like [JobStage]: how far it got is
    `progress`, and how many videos came out is read from its clips."""

    QUEUED = "queued"
    PREPARING = "preparing"
    RENDERING = "rendering"
    DONE = "done"
    #: the request brought no timeline
    NOTHING_CHOSEN = "nothing_chosen"
    #: none of the timelines has anything that can be cut
    NOTHING_TO_CUT = "nothing_to_cut"
    ERROR = "error"


class TrackStatus(StrEnum):
    """Lifecycle of *one media item* uploaded to the job.

    Music arrives before any video exists: you have to hear it in the app and
    see the beats before you can place cuts on top of it. With the media
    library, the same holds for an imported clip or image -- the app needs the
    thumbnail and the dimensions before it will let you build with them.
    """

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class MediaKind(StrEnum):
    """What the imported file is.

    Decides what the analysis does with it: audio gets beats and a waveform,
    video gets a thumbnail and a proxy, an image gets a thumbnail.
    """

    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"


class EventKind(StrEnum):
    KILL = "kill"
    DEATH = "death"
    LOW_HP = "low_hp"
    ESCAPE = "escape"
    #: an ultimate was used. `meta["side"]` says whose: "self" when the footer
    #: button discharged (the player's own), "enemy" when the icon appeared in
    #: the killfeed or the audio spiked.
    ULT_USED = "ult_used"
    ULT_NEGATED = "ult_negated"
    #: critical hit -- the red X marker on the crosshair
    HEADSHOT = "headshot"
    #: someone on our team killed with an ability named in the killfeed;
    #: `meta["ability"]` carries "hero/ability"
    ABILITY_KILL = "ability_kill"
    #: Ana's sleep dart landing on someone
    SLEEP = "sleep"
    #: Sigma's Accretion rock stunning someone
    STUN = "stun"
    #: a line appearing in the killfeed, whatever it was made with.
    #: `meta["killer"]` is "player", "other" or "unknown" (the name on the plate
    #: could not be read). Not a moment: it is the evidence the planner uses to
    #: confirm the crosshair's skulls, and it is consumed there -- see
    #: `rules.unconfirmed_kills`.
    KILLFEED_LINE = "killfeed_line"


#: What a generated video is. There used to be a kind per rule -- "kill
#: streak", "solo wipe", "beat montage" -- because the system proposed
#: ready-made videos from the events. It does not any more: every video comes
#: out of the editor, and what it is, is whatever title the user gave it.
#:
#: It is still a text column in the database, so clips already generated keep
#: the old kind they had -- the app only needs to know how to draw them.
CLIP_KIND_CUSTOM = "custom"


#: Detectors the analysis waits for before considering itself finished.
#: One per *screen region*, not per ability: `banner` reads the footer strip and
#: tells the abilities apart by their icon.
DETECTORS = ("kills", "survival", "ults", "banner", "killfeed")


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


#: nothing below this is worth a cut: it is less than a frame on any recording
MIN_CUT_S = 0.05

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


class TimelineCut(BaseModel):
    """A block on the timeline: a piece of the recording placed at a point of
    the video.

    It is the unit of the montage. The user says *which* stretch (`start_s` +
    `duration_s`, in the recording) and *where* it comes in (`at_s`, in the
    video that will come out). The two are independent: the same moment can
    appear twice, at different points of the music, with different durations.
    """

    #: instant of the moment the block came from. Does not affect the cut --
    #: it lets the app know which event this block came from, and names the file
    source_t: float = 0.0
    #: where the cut starts in the recording
    start_s: float
    #: how long it lasts
    duration_s: float
    #: where it comes in, in the final video; 0 is the first frame
    at_s: float
    #: kill/sleep/stun -- a label only
    kind: str = ""

    @model_validator(mode="after")
    def _check_coherent(self) -> "TimelineCut":
        if self.start_s < 0:
            raise ValueError("start_s cannot be negative")
        if self.at_s < 0:
            raise ValueError("at_s cannot be negative")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"a cut must last at least {MIN_CUT_S}s")
        return self

    @property
    def end_s(self) -> float:
        """Where the cut ends *in the recording*."""
        return self.start_s + self.duration_s

    @property
    def until_s(self) -> float:
        """Where the cut ends *in the video*."""
        return self.at_s + self.duration_s


class ClipSource(StrEnum):
    """Where a clip's picture comes from.

    It is born discriminated so the media library could arrive without touching
    the model: the editor produces `RECORDING` and `COLOR`, with `MEDIA` and
    `TEXT` arriving in the phases after.
    """

    #: a stretch of the match recording
    RECORDING = "recording"
    #: solid colour. The black of the gaps stops being a special case of the
    #: render and becomes a clip like any other
    COLOR = "color"
    #: a file imported by the user (Phase 4)
    MEDIA = "media"
    #: text (Phase 6)
    TEXT = "text"


class Transform(BaseModel):
    """Where and at what size the clip appears in the frame.

    `x` and `y` are offsets from the centre, normalised by half the frame: -1
    touches the left/top edge, +1 the right/bottom, 0 is the centre. That way
    the same montage holds at any resolution -- what matters in a montage is the
    proportion, not the pixel.
    """

    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0
    opacity: float = 1.0

    #: What is cut off each edge, as a fraction of the frame. The picture
    #: keeps its size and place: the cut edges become transparent, so a
    #: cropped killfeed can sit over another clip.
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0
    #: degrees, clockwise, around the frame's centre; the corners that turn
    #: out of the frame are kept whole, the frame edge cuts them
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False

    @model_validator(mode="after")
    def _check_coherent(self) -> "Transform":
        if self.scale <= 0:
            raise ValueError("scale must be greater than zero")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1")
        crops = (self.crop_left, self.crop_top, self.crop_right, self.crop_bottom)
        if any(not 0.0 <= c <= 0.9 for c in crops):
            raise ValueError("a crop is between 0 and 0.9 of the frame")
        if (
            self.crop_left + self.crop_right > 0.95
            or self.crop_top + self.crop_bottom > 0.95
        ):
            raise ValueError("the crop leaves nothing of the picture")
        if not -360.0 <= self.rotation <= 360.0:
            raise ValueError("rotation is between -360 and 360 degrees")
        return self

    @property
    def has_crop(self) -> bool:
        return any(
            (self.crop_left, self.crop_top, self.crop_right, self.crop_bottom)
        )

    @property
    def has_turn(self) -> bool:
        """Rotated or mirrored."""
        return self.rotation % 360 != 0 or self.flip_h or self.flip_v

    @property
    def is_neutral(self) -> bool:
        """True when the clip comes in as it was, with nothing on top."""
        return (
            self.scale == 1.0
            and self.x == 0.0
            and self.y == 0.0
            and self.opacity == 1.0
            and not self.has_crop
            and not self.has_turn
        )


class ClipAudio(BaseModel):
    """The clip's own sound -- which is not the video's soundtrack."""

    volume: float = 1.0
    mute: bool = False
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipAudio":
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        if self.fade_in_s < 0 or self.fade_out_s < 0:
            raise ValueError("fade cannot be negative")
        return self

    @property
    def is_neutral(self) -> bool:
        return (
            self.volume == 1.0
            and not self.mute
            and self.fade_in_s == 0.0
            and self.fade_out_s == 0.0
        )


class ClipColor(BaseModel):
    """The clip's colour adjustment.

    The three that solve almost everything in a gameplay montage: a dark
    recording, a washed-out recording, a colourless recording. The rest (curves,
    temperature) arrives when it is missed.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipColor":
        if not -1.0 <= self.brightness <= 1.0:
            raise ValueError("brightness must be between -1 and 1")
        if not 0.0 <= self.contrast <= 3.0:
            raise ValueError("contrast must be between 0 and 3")
        if not 0.0 <= self.saturation <= 3.0:
            raise ValueError("saturation must be between 0 and 3")
        return self

    @property
    def is_neutral(self) -> bool:
        return (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
        )


class Look(StrEnum):
    """A colour grade in one click -- what a LUT does, built from ffmpeg's own
    filters so no file has to travel with the montage."""

    NONE = "none"
    NOIR = "noir"
    TEAL_ORANGE = "teal_orange"
    WARM = "warm"
    COLD = "cold"
    VIVID = "vivid"
    FADED = "faded"


class ClipFx(BaseModel):
    """The clip's visual effects; each is 0 (off) to 1 (full).

    `impact` is the punch of a play: a white flash and a burst of shake at
    the clip's play -- or at its first frame, when it has none.
    """

    look: Look = Look.NONE
    blur: float = 0.0
    sharpen: float = 0.0
    vignette: float = 0.0
    shake: float = 0.0
    impact: float = 0.0

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipFx":
        for name in ("blur", "sharpen", "vignette", "shake", "impact"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self

    @property
    def is_neutral(self) -> bool:
        return self.look is Look.NONE and not any(
            (self.blur, self.sharpen, self.vignette, self.shake, self.impact)
        )


class BlendMode(StrEnum):
    """How a clip mixes with the layers under it. NORMAL covers them."""

    NORMAL = "normal"
    SCREEN = "screen"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"
    ADD = "add"
    LIGHTEN = "lighten"
    DARKEN = "darken"
    DIFFERENCE = "difference"


class ChromaKey(BaseModel):
    """A colour made transparent -- a green screen.

    `similarity` is how far from the colour still counts as it; `softness`
    how gradually the edge goes from transparent to solid.
    """

    color: str = "#00ff00"
    similarity: float = 0.3
    softness: float = 0.1

    @model_validator(mode="after")
    def _check_coherent(self) -> "ChromaKey":
        c = self.color.lstrip("#")
        if len(c) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in c):
            raise ValueError("the key colour is #rrggbb")
        if not 0.01 <= self.similarity <= 1.0:
            raise ValueError("similarity must be between 0.01 and 1")
        if not 0.0 <= self.softness <= 1.0:
            raise ValueError("softness must be between 0 and 1")
        return self

    @property
    def hex(self) -> str:
        return self.color.lstrip("#").lower()


class ClipFade(BaseModel):
    """The clip's fade in and out, in seconds.

    These are transitions to and from the **background** -- which in a layered
    montage is black. A transition *between two clips* (a crossfade) is another
    thing entirely and does not fit the overlay format: it requires both
    existing at the same time with shifting weights.
    """

    in_s: float = 0.0
    out_s: float = 0.0

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipFade":
        if self.in_s < 0 or self.out_s < 0:
            raise ValueError("fade cannot be negative")
        return self

    @property
    def is_neutral(self) -> bool:
        return self.in_s == 0.0 and self.out_s == 0.0


class TransitionKind(StrEnum):
    """How a clip enters over the one before it on the same layer."""

    #: the new clip appears over the old one, which keeps running underneath
    DISSOLVE = "dissolve"
    #: the old one goes dark, the new one comes out of the dark
    FADE_BLACK = "fade_black"
    #: same, through white -- the flash of an impact
    FADE_WHITE = "fade_white"
    #: the new clip slides in over the old one, from the side it is named after
    #: the *movement*: `slide_left` comes in from the right moving left
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    #: a moving edge uncovers the new clip; named after the edge's movement
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    #: the new clip arrives enlarged and settles while it appears
    ZOOM = "zoom"
    #: the new clip turns and grows into place while it appears
    SPIN = "spin"
    #: a hard cut, torn: colour channels split and bands of the picture jump
    GLITCH = "glitch"

    @property
    def overlaps(self) -> bool:
        """Does it need the previous clip on screen while this one comes in?

        A dissolve or a slide mixes two pictures, and the overlay graph only has
        two pictures at the same instant if the previous clip runs **past** the
        cut. A dip to black or white does not mix anything: one goes out, the
        other comes in.
        """
        return self not in (
            TransitionKind.FADE_BLACK,
            TransitionKind.FADE_WHITE,
            TransitionKind.GLITCH,
        )


class ClipTransition(BaseModel):
    """How the clip enters, at the cut with the previous one on its layer.

    It belongs to the clip that **enters**, and not to the cut: moving the clip
    takes its entrance with it, and a cut with nothing before it (the first
    clip, or one after a gap) still has an entrance -- out of the background.
    """

    kind: TransitionKind
    duration_s: float = 0.5

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipTransition":
        if not 0.1 <= self.duration_s <= 3.0:
            raise ValueError("a transition lasts between 0.1 and 3 seconds")
        return self


class Fit(StrEnum):
    """What to do when the clip's aspect is not the output's.

    It comes up for real when somebody exports 9:16 from a 16:9 recording -- and
    both answers are legitimate, depending on what you want.
    """

    #: fills the screen and crops the overflow. The default: in a gameplay
    #: montage the action is in the middle, and black bars top and bottom are
    #: wasted screen on a phone
    COVER = "cover"
    #: shows the whole frame and accepts bars. For whoever needs what is in
    #: the corners -- the HUD, the scoreboard
    CONTAIN = "contain"
    #: the whole frame, over a blurred, darkened copy of itself filling the
    #: rest -- the usual way to put a landscape clip in a vertical video
    BLUR = "blur"


#: The formats a montage can also be rendered in, by aspect.
EXTRA_FORMATS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}


class ExportSpec(BaseModel):
    """How the final video is written.

    Kept apart from the montage on purpose: the same montage becomes a 16:9 for
    YouTube and a 9:16 for Shorts without anything in it changing. What changes
    is the window you look through.
    """

    #: `0` in either of them = the recording's size
    width: int = 0
    height: int = 0
    #: `0` = the recording's fps
    fps: float = 0.0
    #: H.264 quality: lower is better. 20 is the system default
    crf: int = 20
    fit: Fit = Fit.COVER

    #: time window, in seconds of the assembled video. `None` = everything
    from_s: float = 0.0
    to_s: float | None = None

    #: library item drawn over everything
    watermark_id: str | None = None
    #: mark size, as a fraction of the frame width
    watermark_scale: float = 0.12
    #: the corner it sits in, from the centre: (1, -1) is the top right
    watermark_x: float = 0.82
    watermark_y: float = -0.82
    watermark_opacity: float = 0.65

    #: In a portrait output, the recording's killfeed (top right of the
    #: frame, which a centre crop cuts away) brought back at the top.
    killfeed_inset: bool = False

    #: Other aspects rendered together with this one (see `EXTRA_FORMATS`),
    #: and how they are framed. The app turns them into separate outputs;
    #: they live here so the choice is saved with the montage.
    extra_formats: list[str] = Field(default_factory=list)
    extra_fit: Fit = Fit.COVER
    extra_killfeed: bool = False

    @model_validator(mode="after")
    def _check_coherent(self) -> "ExportSpec":
        unknown = set(self.extra_formats) - set(EXTRA_FORMATS)
        if unknown:
            raise ValueError(f"unknown formats: {sorted(unknown)}")
        if self.width < 0 or self.height < 0:
            raise ValueError("dimensions cannot be negative")
        if (self.width > 0) != (self.height > 0):
            raise ValueError("give both dimensions or neither")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf vai de 0 a 51")
        if self.from_s < 0:
            raise ValueError("from_s cannot be negative")
        if self.to_s is not None and self.to_s <= self.from_s:
            raise ValueError("to_s must be greater than from_s")
        if not 0.0 <= self.watermark_opacity <= 1.0:
            raise ValueError("watermark_opacity vai de 0 a 1")
        if not 0.01 <= self.watermark_scale <= 1.0:
            raise ValueError("watermark_scale vai de 0.01 a 1")
        return self

    @property
    def is_default(self) -> bool:
        """Is this the export the system would produce on its own?"""
        return (
            self.width == 0
            and self.fps == 0
            and self.crf == 20
            and self.fit is Fit.COVER
            and self.from_s == 0
            and self.to_s is None
            and self.watermark_id is None
            and not self.killfeed_inset
        )

    def dimensions(self, source_width: int, source_height: int) -> tuple[int, int]:
        """The size of the final canvas.

        Rounded down to even: H.264 in `yuv420p` stores colour in 2x2 blocks,
        and an odd dimension simply does not encode.
        """
        w = self.width or source_width
        h = self.height or source_height
        return (int(w) // 2 * 2, int(h) // 2 * 2)


class TextAnim(StrEnum):
    """How a text comes in or goes out. The same maths runs in the editor's
    monitor (`text_anim.dart`) and in `owcore.textfx`."""

    NONE = "none"
    FADE = "fade"
    #: grows from half size
    POP = "pop"
    #: comes up from below (in) / goes down (out)
    SLIDE = "slide"
    #: types itself out, letter by letter -- in only
    TYPEWRITER = "typewriter"


class TextStyle(BaseModel):
    """How the text looks.

    Size and outline are **fractions of the frame height**, not pixels: the
    same montage has to come out identical at 720p and at 4K, and a 48px body
    that looks right in one would be tiny in the other.
    """

    #: letter height, 0 to 1 of the frame height
    size: float = 0.08
    color: str = "white"
    #: outline thickness, as a fraction of the letter size. The outline is not
    #: decoration: without it, white text disappears in a bright scene
    outline: float = 0.12
    outline_color: str = "black"
    #: a font id from the catalogue (`owcore.fonts`); empty is the default.
    #: Montages saved before the catalogue stored a path, still accepted
    font: str = ""
    #: how it comes in and goes out, and how long each takes
    anim_in: TextAnim = TextAnim.NONE
    anim_out: TextAnim = TextAnim.NONE
    anim_s: float = 0.35

    @model_validator(mode="after")
    def _check_coherent(self) -> "TextStyle":
        if not 0.01 <= self.size <= 0.5:
            raise ValueError("text size goes from 0.01 to 0.5 of the height")
        if not 0.0 <= self.outline <= 1.0:
            raise ValueError("outline goes from 0 to 1 of the letter size")
        if not 0.05 <= self.anim_s <= 3.0:
            raise ValueError("a text animation lasts from 0.05 to 3 seconds")
        if self.anim_out is TextAnim.TYPEWRITER:
            raise ValueError("typewriter is an entrance only")
        return self


class Ease(StrEnum):
    """How a value travels from one keyframe to the next.

    It belongs to the keyframe the segment **leaves**: an `in_out` key starts
    slow, speeds up and arrives slow at the next one. Linear is the default
    because it is what the zoom always did.
    """

    LINEAR = "linear"
    #: starts slow, arrives fast -- a punch into the kill
    IN = "in"
    #: starts fast, arrives slow -- settling after a move
    OUT = "out"
    #: slow at both ends -- the smooth camera move
    IN_OUT = "in_out"


class KeyProp(StrEnum):
    """What a [ClipKey] animates. Each one replaces its static value while the
    clip has at least one keyframe for it."""

    #: `transform.x` / `transform.y`: offset from the centre, in half frames
    X = "x"
    Y = "y"
    #: `transform.scale`: the clip's size on the frame
    SCALE = "scale"
    #: `transform.opacity`
    OPACITY = "opacity"
    #: `audio.volume`
    VOLUME = "volume"
    #: `speed`: how fast the source runs. Keyframed, it is a speed ramp
    SPEED = "speed"


#: the range each animated property accepts -- the same one its static field
#: accepts, give or take what makes no sense to animate to
KEY_RANGES: dict[KeyProp, tuple[float, float]] = {
    KeyProp.X: (-4.0, 4.0),
    KeyProp.Y: (-4.0, 4.0),
    KeyProp.SCALE: (0.05, 8.0),
    KeyProp.OPACITY: (0.0, 1.0),
    KeyProp.VOLUME: (0.0, 4.0),
    KeyProp.SPEED: (0.1, 10.0),
}

#: each easing as a function of the progress `u` (0 to 1) between two keys --
#: the same shapes `owcore.compose` writes as ffmpeg expressions
EASE_SHAPES = {
    "linear": lambda u: u,
    "in": lambda u: u * u,
    "out": lambda u: u * (2 - u),
    "in_out": lambda u: u * u * (3 - 2 * u),
}


def curve_at(points: list[tuple[float, float, str]], at: float) -> float:
    """A value along (time, value, ease) points, in time order: the endpoint's
    outside them, the ease of the point it leaves in between."""
    if at < points[0][0]:
        return points[0][1]
    for (t0, v0, ease), (t1, v1, _) in zip(points, points[1:]):
        if at < t1:
            u = min(1.0, max(0.0, (at - t0) / max(1e-6, t1 - t0)))
            return v0 + (v1 - v0) * EASE_SHAPES[str(ease)](u)
    return points[-1][1]


class ClipKey(BaseModel):
    """One keyframe of one property, inside the clip.

    `t` is a fraction of the clip, like the zoom's: the animation survives
    stretching or trimming the block.
    """

    prop: KeyProp
    t: float
    value: float
    ease: Ease = Ease.LINEAR

    @model_validator(mode="after")
    def _check_coherent(self) -> "ClipKey":
        if not 0.0 <= self.t <= 1.0:
            raise ValueError("a keyframe's t goes from 0 to 1")
        low, high = KEY_RANGES[self.prop]
        if not low <= self.value <= high:
            raise ValueError(f"{self.prop} goes from {low} to {high}")
        return self


class ZoomKey(BaseModel):
    """One point of the zoom animation, inside the clip.

    `t` runs from 0 to 1 -- it is a fraction of the clip, not seconds. That way
    the animation survives stretching or trimming the block: a zoom that closes
    at the end goes on closing at the end.
    """

    t: float
    #: 1 = full size; 2 = double, i.e. half the frame filling the screen
    scale: float = 1.0
    #: where the lens points, from the centre, -1 to 1
    x: float = 0.0
    y: float = 0.0
    #: how the zoom travels to the next point
    ease: Ease = Ease.LINEAR

    @model_validator(mode="after")
    def _check_coherent(self) -> "ZoomKey":
        if not 0.0 <= self.t <= 1.0:
            raise ValueError("a keyframe's t goes from 0 to 1")
        if not 1.0 <= self.scale <= 8.0:
            raise ValueError(
                "zoom scale goes from 1 to 8: below 1 would show outside the frame"
            )
        return self


class TimelineClip(BaseModel):
    """A piece of the final video: what appears, where, and for how long.

    It is V1's `TimelineCut` with room for the rest: which source it comes from,
    how it is positioned in the frame and what happens to its sound. A V1 cut is
    exactly a `RECORDING` clip with a neutral transform and neutral audio, and
    that is how the migration reads it.
    """

    source: ClipSource = ClipSource.RECORDING
    #: where it comes in, in the video; 0 is the first frame
    at_s: float
    duration_s: float
    #: where it starts in the source (recording or media). Ignored by colour and text
    start_s: float = 0.0
    #: instant of the moment the clip came from -- a label, does not affect the cut
    source_t: float = 0.0
    #: kill/sleep/stun, used to name and colour it
    kind: str = ""
    #: solid colour when `source` is COLOR. It is called `fill` because
    #: `color` is already the colour correction -- they are the *colour of the
    #: content* and the *adjustment of the content*, two things meeting in the
    #: same clip
    fill: str = "black"
    #: id of the library item when `source` is MEDIA
    media_id: str | None = None
    #: the name the user gave the clip -- shown on the editor's ruler, not
    #: drawn in the video
    label: str = ""
    #: what is written, when `source` is TEXT
    text: str = ""
    text_style: TextStyle = Field(default_factory=TextStyle)
    transform: Transform = Field(default_factory=Transform)
    audio: ClipAudio = Field(default_factory=ClipAudio)
    color: ClipColor = Field(default_factory=ClipColor)
    fx: ClipFx = Field(default_factory=ClipFx)
    #: how the clip mixes with the layers below; NORMAL covers them
    blend: BlendMode = BlendMode.NORMAL
    #: a colour made transparent (a green screen); None keys nothing
    chroma: ChromaKey | None = None
    fade: ClipFade = Field(default_factory=ClipFade)
    #: How the clip enters over the previous one on its layer. None = a cut.
    transition: ClipTransition | None = None

    #: Zoom animation inside the clip. Empty = no animation.
    #:
    #: It is the *punch* on the beat: two keyframes are enough. Zoom is about
    #: the content -- looking more closely at what is there -- and is not to be
    #: confused with `transform.scale`, which is the clip's size within the
    #: frame (the PiP).
    zoom: list[ZoomKey] = Field(default_factory=list)

    #: Keyframes of position, scale, opacity and volume. A property with at
    #: least one key is animated, and its static value (`transform`, `audio`)
    #: stops counting; one with none keeps the static value.
    keys: list[ClipKey] = Field(default_factory=list)

    #: Freezes on the last frame instead of running. The duration is still the
    #: clip's; what changes is that the picture stops.
    freeze: bool = False
    #: Plays backwards.
    reverse: bool = False

    #: How much faster the clip runs. 2 = double, 0.5 = slow motion.
    #:
    #: It changes how much source it consumes: a 2s clip at 2x eats 4s of
    #: recording. It does not change how much it occupies in the video -- that
    #: is `duration_s`, and that is what the user drags.
    speed: float = 1.0

    @model_validator(mode="after")
    def _check_coherent(self) -> "TimelineClip":
        if self.start_s < 0:
            raise ValueError("start_s cannot be negative")
        if self.at_s < 0:
            raise ValueError("at_s cannot be negative")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"a clip must last at least {MIN_CUT_S}s")
        if not 0.1 <= self.speed <= 10.0:
            raise ValueError("speed must be between 0.1 and 10")
        if self.fade.in_s + self.fade.out_s > self.duration_s + 1e-6:
            raise ValueError("the fades together exceed the clip duration")
        if self.zoom:
            if len(self.zoom) < 2:
                raise ValueError("a zoom animation needs two points")
            ts = [k.t for k in self.zoom]
            if ts != sorted(ts):
                raise ValueError("keyframes must be in order")
        for prop in KeyProp:
            ts = [k.t for k in self.keys_for(prop)]
            if len(set(ts)) != len(ts):
                raise ValueError(f"two {prop} keyframes at the same instant")
        if self.freeze and self.reverse:
            raise ValueError("freezing and reversing at the same time makes no sense")
        if self.is_ramped and (self.freeze or self.reverse):
            raise ValueError("a speed ramp cannot be frozen or reversed")
        if len(self.label) > 80:
            raise ValueError("a clip name has at most 80 characters")
        if self.source is ClipSource.TEXT and not self.text.strip():
            raise ValueError("a text clip needs text")
        if self.transition and self.transition.duration_s > self.duration_s + 1e-6:
            raise ValueError("the transition is longer than the clip")
        return self

    @property
    def source_consumed_s(self) -> float:
        """How much source this clip eats.

        It is not its duration in the video: at 2x, two seconds of video eat
        four of recording. `duration_s` is what the user drags on the ruler;
        this is a consequence.
        """
        # a frozen clip eats one frame: the rest is that same frame, still
        if self.freeze:
            return MIN_CUT_S
        return self.source_offset(self.duration_s)

    # -- speed, constant or ramped -----------------------------------------
    #
    # With a ramp the source no longer runs at one rate: how much of it has
    # gone by at an instant is the integral of the speed. Every place that
    # turns clip time into source time -- what the clip consumes, where a
    # split cuts, where an export window starts -- goes through here.

    @property
    def is_ramped(self) -> bool:
        return bool(self.keys_for(KeyProp.SPEED))

    def speed_at(self, local: float, span: float | None = None) -> float:
        """The speed `local` seconds into the clip. `span` is the clip's length
        as placed, which keyframe fractions refer to (the drawn or windowed
        copy of a clip is longer or shorter)."""
        keys = self.keys_for(KeyProp.SPEED)
        if not keys:
            return self.speed
        span = self.duration_s if span is None else span
        return curve_at([(k.t * span, k.value, k.ease) for k in keys], local)

    def source_offset(self, local: float, span: float | None = None) -> float:
        """How much source has gone by `local` seconds into the clip."""
        if not self.is_ramped:
            return local * self.speed
        if local <= 0:
            return local * self.speed_at(0.0, span)
        # Simpson over a fine grid: the curve is smooth between keys, and a
        # few hundred samples put the error far below a frame
        n = max(2, int(local * 240) // 2 * 2)
        h = local / n
        total = self.speed_at(0.0, span) + self.speed_at(local, span)
        for i in range(1, n):
            total += (4 if i % 2 else 2) * self.speed_at(i * h, span)
        return total * h / 3

    def local_for_source(self, offset: float, span: float | None = None) -> float:
        """The clip instant at which `offset` seconds of source have gone by --
        the inverse of `source_offset`; the speed is always positive, so it is
        monotonic and bisection finds it."""
        if not self.is_ramped:
            return offset / self.speed
        lo, hi = 0.0, max(1.0, offset / 0.1)
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.source_offset(mid, span) < offset:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @property
    def end_s(self) -> float:
        """Where it ends *in the source* -- speed already accounted for."""
        return self.start_s + self.source_consumed_s

    @property
    def until_s(self) -> float:
        """Where it ends *in the video*."""
        return self.at_s + self.duration_s

    @property
    def is_simple(self) -> bool:
        """A clip that V1's cut-and-splice path can handle."""
        return (
            self.source is ClipSource.RECORDING
            and self.transform.is_neutral
            and self.audio.is_neutral
            and self.color.is_neutral
            and self.fx.is_neutral
            and self.blend is BlendMode.NORMAL
            and self.chroma is None
            and self.fade.is_neutral
            and self.transition is None
            and self.speed == 1.0
            and not self.zoom
            and not self.keys
            and not self.freeze
            and not self.reverse
        )

    @property
    def play_local_s(self) -> float | None:
        """Where in the clip its play happens, in seconds of the video from
        the clip's first frame -- None for a clip with no play in it."""
        if self.source is not ClipSource.RECORDING or self.source_t <= 0:
            return None
        into = self.source_t - self.start_s
        if into < 0 or self.freeze or self.reverse:
            return None
        local = self.local_for_source(into)
        return local if local <= self.duration_s else None

    def keys_for(self, prop: KeyProp | str) -> list[ClipKey]:
        """This property's keyframes, in time order."""
        prop = KeyProp(prop)
        return sorted((k for k in self.keys if k.prop == prop), key=lambda k: k.t)

    def as_cut(self) -> TimelineCut:
        """The V1 view of this clip, for the cut-and-splice path."""
        return TimelineCut(
            source_t=self.source_t,
            start_s=self.start_s,
            duration_s=self.duration_s,
            at_s=self.at_s,
            kind=self.kind,
        )

    @classmethod
    def from_cut(cls, cut: TimelineCut) -> "TimelineClip":
        return cls(
            source=ClipSource.RECORDING,
            at_s=cut.at_s,
            duration_s=cut.duration_s,
            start_s=cut.start_s,
            source_t=cut.source_t,
            kind=cut.kind,
        )


class LayerKind(StrEnum):
    """A layer either draws or plays -- never both."""

    VIDEO = "video"
    #: sound only. Its clips point at library audio, and nothing in it appears
    #: on screen
    AUDIO = "audio"


class Layer(BaseModel):
    """A layer of the timeline.

    It is called a layer, and not a track, because `Track` in this system is
    already the music the user uploaded -- two `Track`s in the same model would
    be a trap.

    The order in the list is the stacking order: the first is the background,
    the last sits on top. An audio layer takes no part in that stacking: it
    draws nothing, it only plays.
    """

    kind: LayerKind = LayerKind.VIDEO
    name: str = ""
    muted: bool = False
    hidden: bool = False
    #: locked changes nothing in the render -- it is the app that refuses edits
    locked: bool = False
    #: drawn as a thin strip in the editor; nothing to do with the render
    collapsed: bool = False
    clips: list[TimelineClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlap(self) -> "Layer":
        ordered = sorted(self.clips, key=lambda c: c.at_s)
        for previous, following in zip(ordered, ordered[1:]):
            if following.at_s < previous.until_s - 1e-6:
                raise ValueError(
                    f"two clips overlap at {following.at_s:.2f}s of the layer"
                )
        self.clips = ordered
        return self

    @property
    def is_audio(self) -> bool:
        return self.kind is LayerKind.AUDIO

    @property
    def duration_s(self) -> float:
        return max((c.until_s for c in self.clips), default=0.0)


class Recipe(BaseModel):
    """How to build a video out of what happened in a match.

    A preset does not store cuts -- it stores the **way** of cutting. "Two
    seconds per kill, snapped to the beat, with zoom" holds for any match, while
    a list of cuts only holds for that one.

    It is what makes the second match cost one click instead of half an hour of
    fitting.
    """

    #: which events become cuts
    kinds: list[str] = Field(default_factory=lambda: ["kill", "sleep", "stun"])
    #: how long before the event the cut starts -- the moment needs a run-up,
    #: or the kill lands on the very first frame
    lead_s: float = 1.0
    #: length of each cut. Ignored when `beats_per_cut` rules
    duration_s: float = 2.0
    #: with a soundtrack, each cut lasts N beats instead of `duration_s`
    beats_per_cut: float = 0.0
    #: gap between one cut and the next
    gap_s: float = 0.0
    #: at most this many cuts. `0` = all there are
    max_cuts: int = 0

    #: effects applied to each cut
    zoom: bool = False
    #: the zoom eases in and out instead of punching
    zoom_smooth: bool = False
    fade_s: float = 0.0
    speed: float = 1.0

    #: how each cut enters over the one before -- empty is a plain cut
    transition: str = ""
    transition_s: float = 0.5
    #: slow motion through each play, full speed around it
    ramp: bool = False
    ramp_slow: float = 0.35

    #: ducking at the plays (see `Timeline.duck_plays`)
    duck_plays: bool = False
    duck_level: float = 0.3

    #: how the labels the system writes look -- `None` keeps the default
    label_style: TextStyle | None = None

    #: text the system writes by itself
    counter: bool = False
    streaks: bool = False

    music_volume: float = 1.0
    game_volume: float = 0.0
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="after")
    def _check_coherent(self) -> "Recipe":
        if self.lead_s < 0:
            raise ValueError("lead_s cannot be negative")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"each cut must last at least {MIN_CUT_S}s")
        if self.beats_per_cut < 0:
            raise ValueError("beats_per_cut cannot be negative")
        if self.gap_s < 0:
            raise ValueError("gap_s cannot be negative")
        if self.max_cuts < 0:
            raise ValueError("max_cuts cannot be negative")
        if self.transition and self.transition not in {k.value for k in TransitionKind}:
            raise ValueError(f"unknown transition {self.transition!r}")
        if not 0.05 <= self.transition_s <= 3.0:
            raise ValueError("transition_s goes from 0.05 to 3")
        if not 0.1 <= self.ramp_slow <= 1.0:
            raise ValueError("ramp_slow goes from 0.1 to 1")
        if not 0.0 <= self.duck_level <= 1.0:
            raise ValueError("duck_level goes from 0 to 1")
        if not 0.1 <= self.speed <= 8.0:
            raise ValueError("speed vai de 0.1 a 8")
        if self.fade_s < 0:
            raise ValueError("fade_s cannot be negative")
        for nome, v in (("music_volume", self.music_volume),
                        ("game_volume", self.game_volume)):
            if not 0.0 <= v <= 2.0:
                raise ValueError(f"{nome} fica entre 0 e 2")
        return self


def _track_as_block(
    track_id: str, music_start_s: float, duration_s: float
) -> "Layer":
    """The continuous track becomes a music block covering the video.

    There were two ways of having music: a continuous track under everything,
    which could not be cut, and blocks placed on the ruler. The second one
    survived -- and since the first is exactly a block starting where the music
    came in and covering the whole video, old montages need no database
    migration: **the code that reads converts the old format**, as it always has
    here.
    """
    return Layer(
        kind=LayerKind.AUDIO,
        name="Musica",
        clips=[
            TimelineClip(
                source=ClipSource.MEDIA,
                media_id=track_id,
                at_s=0.0,
                duration_s=duration_s,
                start_s=max(0.0, music_start_s),
            )
        ],
    )


class Marker(BaseModel):
    """A note pinned to an instant of the montage -- "the drop starts here".

    It lives only in the editor: the ruler shows it and the magnet snaps to it.
    The video never sees it.
    """

    t_s: float = Field(ge=0)
    label: str = Field(default="", max_length=40)


class MontageDraft(BaseModel):
    """The montage **in progress**, exactly as it was left on screen.

    It is the `Timeline` without the requirement of being finished: it accepts
    zero cuts, because a draft exists from before the first block comes in. Each
    block, on the other hand, is validated -- storing rubbish now would mean
    handing rubbish back later.

    It exists because reloading the page cost the whole montage: half an hour of
    fitting to the beat vanished on an F5.
    """

    title: str = ""
    track_id: str | None = None
    music_start_s: float = 0.0
    #: V1 format, still accepted while the app does not send layers
    cuts: list[TimelineCut] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)

    #: The user's corrections to the beat grid. They do not affect the video:
    #: a cut stores absolute instants, and the grid is only the screen's magnet.
    #: They travel in the draft so they are not lost on an F5 -- fixing the grid
    #: twice is more annoying than fixing it once.
    beat_offset_s: float = 0.0
    beat_multiplier: float = 1.0
    beat_bar: int = 1

    #: the mix and the output format are work too: whoever lowered the game
    #: volume and chose 9:16 does not want to redo both after an F5
    music_volume: float = 1.0
    game_volume: float = 0.0
    duck_plays: bool = False
    duck_level: float = 0.3
    export: ExportSpec = Field(default_factory=ExportSpec)

    #: the editor's notes on the ruler; work like the rest, so they survive
    #: an F5
    markers: list[Marker] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _track_becomes_block(self) -> "MontageDraft":
        if not self.track_id:
            return self
        # a V1 draft stores `cuts` instead of layers; materialising them first
        # is what stops the audio layer from becoming the only layer
        if not self.layers and self.cuts:
            self.layers = [
                Layer(clips=[TimelineClip.from_cut(c) for c in self.cuts])
            ]
            self.cuts = []
        end = max((c.until_s for c in self.clips), default=0.0)
        if end >= MIN_CUT_S:
            self.layers = [
                *self.layers,
                _track_as_block(self.track_id, self.music_start_s, end),
            ]
        self.track_id = None
        self.music_start_s = 0.0
        return self

    @property
    def clips(self) -> list[TimelineClip]:
        """Every clip, from every layer -- including those of a draft saved
        before layers existed."""
        if self.layers:
            return [c for l in self.layers for c in l.clips]
        return [TimelineClip.from_cut(c) for c in self.cuts]

    @property
    def duration_s(self) -> float:
        return max((c.until_s for c in self.clips), default=0.0)


class Timeline(BaseModel):
    """A hand-built video: the layers, and the music beneath them.

    **It reads the V1 format.** An old request (or a draft saved before this
    version) arrives with `cuts` instead of `layers`, and is converted here on
    read -- a single layer of recording clips. There is no migration to run on
    the database: the old format is still valid input, and comes out the other
    side in the new one.
    """

    title: str = ""
    #: **old format**: the continuous track under everything. Still valid
    #: input, and turned into a block on the audio layer on read -- nobody
    #: builds like this any more
    track_id: str | None = None
    #: which point of the music the continuous track came in at
    music_start_s: float = 0.0
    layers: list[Layer] = Field(default_factory=list)

    #: Volume of the music and of the game sound, 0 to 2.
    #:
    #: With `game_volume` at 0 the music **replaces** the audio, which is what
    #: V1 did. Above that the two mix -- which is what lets the shot show
    #: through underneath the music. With no music block at all, the cuts' audio
    #: stands on its own and neither volume has anything to do.
    music_volume: float = 1.0
    game_volume: float = 0.0

    #: Ducking: at each play the music drops to `duck_level` of its volume
    #: and the game sound comes up to full, so the shot is heard over the
    #: song. A play is a moment clip's instant -- the kill, the dart.
    duck_plays: bool = False
    duck_level: float = 0.3

    #: how the final video is written. The same montage becomes 16:9 and 9:16
    #: without anything in it changing -- what changes is the window you look
    #: through
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="before")
    @classmethod
    def _accept_v1_format(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("layers") or "cuts" not in data:
            return data
        cuts = data.pop("cuts") or []
        data["layers"] = [{"clips": [
            TimelineClip.from_cut(
                c if isinstance(c, TimelineCut) else TimelineCut(**c)
            ).model_dump()
            for c in cuts
        ]}]
        return data

    @model_validator(mode="after")
    def _has_something_to_build(self) -> "Timeline":
        if self.music_start_s < 0:
            raise ValueError("music_start_s cannot be negative")
        for nome, v in (("music_volume", self.music_volume),
                        ("game_volume", self.game_volume)):
            if not 0.0 <= v <= 2.0:
                raise ValueError(f"{nome} fica entre 0 e 2")
        if not 0.0 <= self.duck_level <= 1.0:
            raise ValueError("duck_level goes from 0 to 1")
        if not any(l.clips for l in self.layers):
            raise ValueError("an empty timeline does not make a video")
        return self

    @model_validator(mode="after")
    def _track_becomes_block(self) -> "Timeline":
        if not self.track_id:
            return self
        end = max((l.duration_s for l in self.layers), default=0.0)
        if end >= MIN_CUT_S:
            self.layers = [
                *self.layers,
                _track_as_block(self.track_id, self.music_start_s, end),
            ]
        self.track_id = None
        self.music_start_s = 0.0
        return self

    @property
    def duration_s(self) -> float:
        """How long the video will last -- gaps between clips included."""
        return max((l.duration_s for l in self.layers), default=0.0)

    @property
    def clips(self) -> list[TimelineClip]:
        """Every clip, from every layer, bottom to top."""
        return [c for l in self.layers for c in l.clips]

    @property
    def single_layer(self) -> bool:
        """Can this be built through V1's cut-and-splice path?

        One layer, no clip with a transform, adjusted sound or a source other
        than the recording, and the output in the recording's format. That is
        the case for most montages, and there the old path is more resilient: a
        cut that fails costs only itself, while an error in the filter graph
        brings the whole render down.

        A non-default output -- another aspect, a time window, a watermark --
        only exists in the filter graph, so that alone takes the montage off
        this path. The same goes for an audio layer: splicing cuts cannot mix
        sound that runs outside them.
        """
        if not self.export.is_default:
            return False
        if any(l.is_audio for l in self.layers):
            return False
        visible = [l for l in self.layers if not l.hidden]
        if len(visible) != 1:
            return False
        return all(c.is_simple for c in visible[0].clips)

    def play_times(self) -> list[float]:
        """When each play happens in the video: the instant of every moment
        clip on a visible picture layer, where its source reaches the event
        (the speed, ramps included, taken into account)."""
        out: list[float] = []
        for layer in self.layers:
            if layer.hidden or layer.is_audio:
                continue
            for c in layer.clips:
                local = c.play_local_s
                if local is not None:
                    out.append(c.at_s + local)
        return sorted(out)

    @property
    def has_music(self) -> bool:
        """The montage has music, that is: any block on an audio layer.

        It is what decides what the two volumes mean. With no music the cuts'
        audio comes out as it is; with music, `game_volume` says how much of the
        game shows through underneath it.
        """
        return any(l.is_audio and l.clips for l in self.layers)

    @property
    def cuts(self) -> list[TimelineCut]:
        """The V1 view of this timeline. Only meaningful with one layer."""
        return [c.as_cut() for c in self.clips]


# --------------------------- bus messages ----------------------------------

STREAM_JOBS = "ow.jobs"
STREAM_ROI = "ow.roi"
STREAM_EDIT = "ow.edit"
#: a render request, ready for the editor to cut.
#:
#: There used to be a stage between creating the request and this queue: the
#: rhythm service analysed the music of each chosen proposal and only then
#: released the editor. There are no proposals any more, and the editor's music
#: arrives already analysed through the library -- the request is born ready,
#: and the gateway publishes straight here.
STREAM_RENDER_READY = "ow.render.ready"
#: a freshly uploaded file, waiting to be analysed (beats and waveform for
#: audio; thumbnail, dimensions and proxy for video and image)
STREAM_MEDIA = "ow.media"
#: analysis finished: someone to extract a thumbnail for each moment
STREAM_THUMBS = "ow.thumbs"
#: an exact preview of a montage, waiting for the previewer. A stream of its
#: own so a preview never queues behind a full render
STREAM_PREVIEW = "ow.preview"


class JobCreated(BaseModel):
    job_id: str


class RoiReady(BaseModel):
    job_id: str
    detector: str
    artifacts: list[Artifact]
    duration_s: float
    params: JobParams


class EventsDetected(BaseModel):
    job_id: str
    detector: str
    events: list[DetectionEvent]
    error: str | None = None


class EditRequested(BaseModel):
    job_id: str


class RenderRequested(BaseModel):
    render_id: str


class MediaUploaded(BaseModel):
    media_id: str


class PreviewRequested(BaseModel):
    preview_id: str


class ThumbsRequested(BaseModel):
    job_id: str


# ──────────────────────────────── tabelas ───────────────────────────────────


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(24), default=JobStatus.PENDING)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: how many moments the analysis found, crossed events included. `None`
    #: until the analysis ends -- and on matches analysed before this column
    #: existed, until the gateway counts them
    n_moments: Mapped[int | None] = mapped_column(Integer, nullable=True)

    video_key: Mapped[str] = mapped_column(String(255))
    video_name: Mapped[str] = mapped_column(String(255), default="")

    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    #: frames per second of the recording -- the editor needs it for a
    #: one-frame step to make sense
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    #: size of the recording. It is the export default -- and what lets the
    #: editor say whether the requested output crops the frame or leaves bars
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    #: reduced copy of the recording, for the editor's monitor. It comes out
    #: of the same decode as the crops, so it costs almost nothing
    proxy_key: Mapped[str] = mapped_column(String(255), default="")
    #: waveform of the match audio, already reduced -- it is what shows the
    #: shot and the explosion on the ruler
    waveform: Mapped[list] = mapped_column(JSON, default=list)
    #: montage in progress, saved automatically while the user edits
    draft: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    events: Mapped[list["Event"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    renders: Mapped[list["Render"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    previews: Mapped[list["Preview"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    reports: Mapped[list["DetectorReport"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    montages: Mapped[list["Montage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    media: Mapped[list["Media"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Render(Base):
    """A render request: the montages the user sent to be rendered."""

    __tablename__ = "renders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), default=RenderStatus.PENDING)
    stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: serialised list of `Timeline` -- the videos the user built
    timelines: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="renders")
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="render", cascade="all, delete-orphan"
    )


class Preview(Base):
    """An exact preview: a stretch of one montage rendered small and fast.

    The editor's monitor composes the layers in the browser, which is instant
    but approximate. This is the server's own graph -- the same
    `compose_graph` the final video goes through -- on a reduced frame, so what
    it shows is what will come out, only smaller.

    It is not a [Render]: it never shows up among the generated videos, in the
    match's zip or in its counts, and only the latest one of a match is kept.
    """

    __tablename__ = "previews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(24), default=RenderStatus.PENDING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: one serialised `Timeline`, its export window already set
    timeline: Mapped[dict] = mapped_column(JSON, default=dict)
    #: where the stretch sits in the montage, so the app can line it up with
    #: the ruler
    from_s: Mapped[float] = mapped_column(Float, default=0.0)
    to_s: Mapped[float] = mapped_column(Float, default=0.0)
    video_key: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="previews")


class Media(Base):
    """A file the user brought in: music, clip or image.

    It began as "the job's music" and became the match's media library --
    because they were the same thing. Music is uploaded, a worker analyses it
    and the gateway serves it with `Range`: exactly the path an imported clip
    walks. Generalising cost one column (`kind`) and avoided a second upload
    system living beside the first.

    It belongs to the **job**, not to a request: the same file serves as many
    montages as the user likes, with no re-upload.

    > The table is still called `tracks`, for historical reasons. Renaming it
    > would mean migrating data for an aesthetic gain; the model, which is what
    > you read in the code, says what it holds.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default=TrackStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: audio, video or image. The default covers the rows from when there was
    #: only music: they were all audio
    kind: Mapped[str] = mapped_column(String(16), default=MediaKind.AUDIO)
    name: Mapped[str] = mapped_column(String(255), default="")
    key: Mapped[str] = mapped_column(String(255))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    # -- audio only ---------------------------------------------------------
    bpm: Mapped[float] = mapped_column(Float, default=0.0)
    #: beat instants, in seconds
    beats: Mapped[list] = mapped_column(JSON, default=list)
    #: waveform already reduced to a few thousand peaks (0..1), so the app can
    #: draw without downloading the whole audio
    peaks: Mapped[list] = mapped_column(JSON, default=list)

    # -- video and image ----------------------------------------------------
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    thumb_key: Mapped[str] = mapped_column(String(255), default="")
    #: reduced copy, for the same reason as the recording's proxy: the monitor
    #: cannot drag the full file on every seek
    proxy_key: Mapped[str] = mapped_column(String(255), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="media")

    @property
    def is_audio(self) -> bool:
        return self.kind == MediaKind.AUDIO


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(24))
    t: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped[Job] = relationship(back_populates="events")


class DetectorReport(Base):
    """One record per detector per job -- how the pipeline knows it can start."""

    __tablename__ = "detector_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    detector: Mapped[str] = mapped_column(String(32))
    ok: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    n_events: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="reports")


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    render_id: Mapped[str] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(24))
    title: Mapped[str] = mapped_column(String(160), default="")
    start_s: Mapped[float] = mapped_column(Float, default=0.0)
    end_s: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    key: Mapped[str] = mapped_column(String(255))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    job: Mapped[Job] = relationship(back_populates="clips")
    render: Mapped[Render] = relationship(back_populates="clips")

class Montage(Base):
    """A named montage of a match.

    Until Phase 8 there was **one** montage per job, kept in a column of the job
    itself. That forced a choice: either the 30-second cut for Shorts or the
    long montage, never both. They are different pieces of work over the same
    material, and now each has its own name.

    The content is still a `MontageDraft` -- the same format the app already
    sent. What changed is where it lives, and the fact that there are several.
    """

    __tablename__ = "montages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), default="")
    #: the montage itself, in `MontageDraft` format
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    job: Mapped[Job] = relationship(back_populates="montages")
    versions: Mapped[list["MontageVersion"]] = relationship(
        back_populates="montage", cascade="all, delete-orphan"
    )

    @property
    def summary(self) -> dict:
        """Enough for the montage list without downloading the whole montage.

        Whoever only wants to choose between "short vertical" and "the long one"
        does not need the clips of either.
        """
        try:
            m = MontageDraft(**(self.data or {}))
        except ValidationError:
            # a montage stored by an earlier version of the format must not
            # take down the list: it shows as empty and stays openable
            return {"n_clips": 0, "duration_s": 0.0, "has_music": False}
        clips = m.clips
        return {
            "n_clips": len(clips),
            "duration_s": round(max((c.until_s for c in clips), default=0.0), 2),
            "has_music": any(l.is_audio and l.clips for l in m.layers),
        }


class MontageVersion(Base):
    """A snapshot of a montage, kept so you can go back to it.

    It is not undo -- that lives in the app and dies with the tab. This is the
    "it was good yesterday": rare, deliberate markers, taken when a video is
    generated (what came out was *this*) or when the user asks.

    Keeping a snapshot on every autosave would fill the database with identical
    states and make the list useless from sheer length.
    """

    __tablename__ = "montage_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    montage_id: Mapped[str] = mapped_column(
        ForeignKey("montages.id", ondelete="CASCADE"), index=True
    )
    #: why this snapshot exists: "generated the video", "before restoring"...
    label: Mapped[str] = mapped_column(String(120), default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    montage: Mapped[Montage] = relationship(back_populates="versions")


class Preset(Base):
    """A preset: the way of building, saved for the next match.

    It belongs to no job on purpose -- crossing from one match to another is
    precisely why it exists.
    """

    __tablename__ = "presets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), default="")
    #: the recipe, in the `Recipe` format
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
