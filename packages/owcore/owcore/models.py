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
            raise ValueError("start_s nao pode ser negativo")
        if self.at_s < 0:
            raise ValueError("at_s nao pode ser negativo")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"um corte tem de durar ao menos {MIN_CUT_S}s")
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

    @model_validator(mode="after")
    def _check_coherent(self) -> "Transform":
        if self.scale <= 0:
            raise ValueError("scale tem de ser maior que zero")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity fica entre 0 e 1")
        return self

    @property
    def is_neutral(self) -> bool:
        """True when the clip comes in as it was, with nothing on top."""
        return (
            self.scale == 1.0
            and self.x == 0.0
            and self.y == 0.0
            and self.opacity == 1.0
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
            raise ValueError("volume nao pode ser negativo")
        if self.fade_in_s < 0 or self.fade_out_s < 0:
            raise ValueError("fade nao pode ser negativo")
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
            raise ValueError("brightness fica entre -1 e 1")
        if not 0.0 <= self.contrast <= 3.0:
            raise ValueError("contrast fica entre 0 e 3")
        if not 0.0 <= self.saturation <= 3.0:
            raise ValueError("saturation fica entre 0 e 3")
        return self

    @property
    def is_neutral(self) -> bool:
        return (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
        )


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
            raise ValueError("fade nao pode ser negativo")
        return self

    @property
    def is_neutral(self) -> bool:
        return self.in_s == 0.0 and self.out_s == 0.0


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

    @model_validator(mode="after")
    def _check_coherent(self) -> "ExportSpec":
        if self.width < 0 or self.height < 0:
            raise ValueError("as dimensoes nao podem ser negativas")
        if (self.width > 0) != (self.height > 0):
            raise ValueError("de as duas dimensoes ou nenhuma")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf vai de 0 a 51")
        if self.from_s < 0:
            raise ValueError("from_s nao pode ser negativo")
        if self.to_s is not None and self.to_s <= self.from_s:
            raise ValueError("to_s tem de ser maior que from_s")
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
        )

    def dimensions(self, source_width: int, source_height: int) -> tuple[int, int]:
        """The size of the final canvas.

        Rounded down to even: H.264 in `yuv420p` stores colour in 2x2 blocks,
        and an odd dimension simply does not encode.
        """
        w = self.width or source_width
        h = self.height or source_height
        return (int(w) // 2 * 2, int(h) // 2 * 2)


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
    #: font path; empty uses the system's
    font: str = ""

    @model_validator(mode="after")
    def _check_coherent(self) -> "TextStyle":
        if not 0.01 <= self.size <= 0.5:
            raise ValueError("size do texto vai de 0.01 a 0.5 da altura")
        if not 0.0 <= self.outline <= 1.0:
            raise ValueError("outline vai de 0 a 1 do tamanho da letra")
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

    @model_validator(mode="after")
    def _check_coherent(self) -> "ZoomKey":
        if not 0.0 <= self.t <= 1.0:
            raise ValueError("t de um quadro-chave vai de 0 a 1")
        if not 1.0 <= self.scale <= 8.0:
            raise ValueError(
                "scale de zoom vai de 1 a 8: menos que 1 mostraria fora do quadro"
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
    #: what is written, when `source` is TEXT
    text: str = ""
    text_style: TextStyle = Field(default_factory=TextStyle)
    transform: Transform = Field(default_factory=Transform)
    audio: ClipAudio = Field(default_factory=ClipAudio)
    color: ClipColor = Field(default_factory=ClipColor)
    fade: ClipFade = Field(default_factory=ClipFade)

    #: Zoom animation inside the clip. Empty = no animation.
    #:
    #: It is the *punch* on the beat: two keyframes are enough. Zoom is about
    #: the content -- looking more closely at what is there -- and is not to be
    #: confused with `transform.scale`, which is the clip's size within the
    #: frame (the PiP).
    zoom: list[ZoomKey] = Field(default_factory=list)

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
            raise ValueError("start_s nao pode ser negativo")
        if self.at_s < 0:
            raise ValueError("at_s nao pode ser negativo")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"um clipe tem de durar ao menos {MIN_CUT_S}s")
        if not 0.1 <= self.speed <= 10.0:
            raise ValueError("speed fica entre 0.1 e 10")
        if self.fade.in_s + self.fade.out_s > self.duration_s + 1e-6:
            raise ValueError("os fades somados passam da duracao do clipe")
        if self.zoom:
            if len(self.zoom) < 2:
                raise ValueError("uma animacao de zoom precisa de dois pontos")
            ts = [k.t for k in self.zoom]
            if ts != sorted(ts):
                raise ValueError("os quadros-chave tem de estar em ordem")
        if self.freeze and self.reverse:
            raise ValueError("congelar e inverter ao mesmo tempo nao faz sentido")
        if self.source is ClipSource.TEXT and not self.text.strip():
            raise ValueError("um clipe de texto precisa de texto")
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
        return self.duration_s * self.speed

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
            and self.fade.is_neutral
            and self.speed == 1.0
            and not self.zoom
            and not self.freeze
            and not self.reverse
        )

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
    clips: list[TimelineClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlap(self) -> "Layer":
        ordenados = sorted(self.clips, key=lambda c: c.at_s)
        for anterior, seguinte in zip(ordenados, ordenados[1:]):
            if seguinte.at_s < anterior.until_s - 1e-6:
                raise ValueError(
                    f"dois clipes se sobrepoem em {seguinte.at_s:.2f}s da camada"
                )
        self.clips = ordenados
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
    fade_s: float = 0.0
    speed: float = 1.0

    #: text the system writes by itself
    counter: bool = False
    streaks: bool = False

    music_volume: float = 1.0
    game_volume: float = 0.0
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="after")
    def _check_coherent(self) -> "Recipe":
        if self.lead_s < 0:
            raise ValueError("lead_s nao pode ser negativo")
        if self.duration_s < MIN_CUT_S:
            raise ValueError(f"cada corte tem de ter ao menos {MIN_CUT_S}s")
        if self.beats_per_cut < 0:
            raise ValueError("beats_per_cut nao pode ser negativo")
        if self.gap_s < 0:
            raise ValueError("gap_s nao pode ser negativo")
        if self.max_cuts < 0:
            raise ValueError("max_cuts nao pode ser negativo")
        if not 0.1 <= self.speed <= 8.0:
            raise ValueError("speed vai de 0.1 a 8")
        if self.fade_s < 0:
            raise ValueError("fade_s nao pode ser negativo")
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
    export: ExportSpec = Field(default_factory=ExportSpec)

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
        fim = max((c.until_s for c in self.clips), default=0.0)
        if fim >= MIN_CUT_S:
            self.layers = [
                *self.layers,
                _track_as_block(self.track_id, self.music_start_s, fim),
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

    #: how the final video is written. The same montage becomes 16:9 and 9:16
    #: without anything in it changing -- what changes is the window you look
    #: through
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="before")
    @classmethod
    def _accept_v1_format(cls, dados: Any) -> Any:
        if not isinstance(dados, dict):
            return dados
        if dados.get("layers") or "cuts" not in dados:
            return dados
        cortes = dados.pop("cuts") or []
        dados["layers"] = [{"clips": [
            TimelineClip.from_cut(
                c if isinstance(c, TimelineCut) else TimelineCut(**c)
            ).model_dump()
            for c in cortes
        ]}]
        return dados

    @model_validator(mode="after")
    def _has_something_to_build(self) -> "Timeline":
        if self.music_start_s < 0:
            raise ValueError("music_start_s nao pode ser negativo")
        for nome, v in (("music_volume", self.music_volume),
                        ("game_volume", self.game_volume)):
            if not 0.0 <= v <= 2.0:
                raise ValueError(f"{nome} fica entre 0 e 2")
        if not any(l.clips for l in self.layers):
            raise ValueError("uma linha do tempo vazia nao vira video")
        return self

    @model_validator(mode="after")
    def _track_becomes_block(self) -> "Timeline":
        if not self.track_id:
            return self
        fim = max((l.duration_s for l in self.layers), default=0.0)
        if fim >= MIN_CUT_S:
            self.layers = [
                *self.layers,
                _track_as_block(self.track_id, self.music_start_s, fim),
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
        visiveis = [l for l in self.layers if not l.hidden]
        if len(visiveis) != 1:
            return False
        return all(c.is_simple for c in visiveis[0].clips)

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

    video_key: Mapped[str] = mapped_column(String(255))
    video_name: Mapped[str] = mapped_column(String(255), default="")

    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    #: frames per second of the recording -- the editor needs it for the step
    #: um quadro fazer sentido
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
    #: montagem em andamento, salva sozinha enquanto o usuario edita
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
    #: a receita, no formato de `Recipe`
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
