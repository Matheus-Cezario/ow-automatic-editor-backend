"""The montage: cuts, clips, layers, effects, text, transitions, export and
the `Timeline` the editor renders. The app mirrors this contract."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

#: nothing below this is worth a cut: it is less than a frame on any recording
MIN_CUT_S = 0.05

#: The `kind` of a block made from the sound effects library (`owcore.sfx`).
#: It sits on an audio layer like the music but is not music: it does not
#: take the game sound away, and the music volume does not govern it.
SFX_KIND = "sfx"

#: The `kind` of a block recorded in the editor as a voice-over. Not music
#: either -- it plays at its own volume -- and the rest of the mix steps back
#: while it speaks (`Timeline.voice_spans`).
VOICE_KIND = "voice"


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

    #: Integrated loudness to bring the sound to, in LUFS (-14 is YouTube's
    #: and Spotify's); None leaves the mix as it is.
    loudness: float | None = None

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
        if self.loudness is not None and not -30.0 <= self.loudness <= -5.0:
            raise ValueError("loudness is between -30 and -5 LUFS")
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
            and self.loudness is None
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


class TextAlign(StrEnum):
    """How the lines of a text line up with each other."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


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
    #: how the lines line up; the block as a whole stays centred on the
    #: clip's position
    align: TextAlign = TextAlign.CENTER
    #: the box the text breaks inside, 0 to 1 of the frame **width**. 0 is no
    #: box: lines break only where they were typed
    width: float = 0.0
    #: a colour behind the text, or empty for none. One box behind all the
    #: lines, as wide as the widest (or as the `width`, when set)
    box: str = ""
    #: how much the box covers, 0 (see-through) to 1 (solid)
    box_opacity: float = 0.6
    #: a drop shadow's colour, or empty for none -- down and to the right
    shadow: str = ""

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
        if self.width and not 0.05 <= self.width <= 1.0:
            raise ValueError("a text box is 0.05 to 1 of the frame width, or 0")
        if not 0.0 <= self.box_opacity <= 1.0:
            raise ValueError("box opacity goes from 0 to 1")
        # they go into the filtergraph as they are
        for colour in (self.box, self.shadow):
            if colour and not re.fullmatch(r"[A-Za-z]+|#[0-9A-Fa-f]{6}", colour):
                raise ValueError(f"not a colour: {colour!r}")
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
    #: for a RECORDING clip, the job whose recording it cuts -- a moment
    #: brought from another match. None is this montage's own match
    job_id: str | None = None
    #: the name the user gave the clip -- shown on the editor's ruler, not
    #: drawn in the video
    label: str = ""
    #: how this clip fills the frame, when not the export's way. A sticker or
    #: a hero icon is placed whole (`CONTAIN`), transparent around, so that
    #: its size and place are the transform's; filling the frame would crop
    #: it to a slice. None follows the export
    fit: Fit | None = None
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

    #: Slow motion with in-between frames made by motion interpolation, instead
    #: of each source frame simply lasting longer (which stutters). Only
    #: matters while the clip runs slower than the recording.
    smooth: bool = False

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
            and self.job_id is None
            and self.transform.is_neutral
            and self.audio.is_neutral
            and self.color.is_neutral
            and self.fx.is_neutral
            and self.blend is BlendMode.NORMAL
            and self.chroma is None
            and self.fit is None
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

    @property
    def is_sound_effect(self) -> bool:
        return self.kind == SFX_KIND

    @property
    def is_voice(self) -> bool:
        return self.kind == VOICE_KIND

    @property
    def is_music(self) -> bool:
        """On an audio layer, a song: not an effect and not a voice-over."""
        return not (self.is_sound_effect or self.is_voice)

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

    def recording_jobs(self) -> set[str]:
        """The other matches this montage cuts from: the job of every moment
        brought from another recording."""
        return {
            c.job_id
            for layer in self.layers
            for c in layer.clips
            if c.source is ClipSource.RECORDING and c.job_id
        }

    def voice_spans(self) -> list[tuple[float, float]]:
        """Where a voice-over speaks in the video: (from, until) of every voice
        block on an audio layer that is heard."""
        return sorted(
            (c.at_s, c.until_s)
            for layer in self.layers
            if layer.is_audio and not layer.muted and not layer.hidden
            for c in layer.clips
            if c.is_voice and not c.audio.mute
        )

    @property
    def has_music(self) -> bool:
        """The montage has music, that is: any block on an audio layer that
        is not a sound effect.

        It is what decides what the two volumes mean. With no music the cuts'
        audio comes out as it is; with music, `game_volume` says how much of the
        game shows through underneath it. A whoosh is not music: one effect on
        a montage without a song must not silence the game -- and neither must a
        voice-over.
        """
        return any(
            c.is_music
            for l in self.layers
            if l.is_audio
            for c in l.clips
        )

    @property
    def cuts(self) -> list[TimelineCut]:
        """The V1 view of this timeline. Only meaningful with one layer."""
        return [c.as_cut() for c in self.clips]
