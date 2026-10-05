"""From the layered timeline to an ffmpeg filter graph.

This file **does not run ffmpeg**: it writes the `-filter_complex` and says
which inputs to pass. It is pure text assembly, and that is why the whole graph
can be tested without encoding a single frame -- which matters when an error in
the graph brings down the entire render, and not just one cut.

## Why a graph, and not cut-and-splice

V1 cut each stretch into a file and concatenated them. That works, it is
resilient (a bad cut costs only itself) and it **cannot hold layers**:
overlaying requires two pieces existing at the same time, and concatenation is
exactly the opposite of that.

So the old path stays alive for single-layer montages, and this one takes over
when there is a layer, a transform or adjusted sound. The choice is made by
`Timeline.single_layer`.

## The shape of the graph

A background canvas covers the whole video, and each clip is overlaid onto it
at the right moment:

    [bg][v0] overlay(enable=...) [t0]
    [t0][v1] overlay(enable=...) [t1]  ...

The background solves for free what was a special case in V1: a gap between
clips is where nothing was overlaid, and what you see there is the background
canvas itself.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from pathlib import Path

from . import textfx
from .models import (
    MIN_CUT_S,
    BlendMode,
    ClipSource,
    ClipTransition,
    Ease,
    Fit,
    KeyProp,
    Look,
    MediaKind,
    Timeline,
    TimelineClip,
    Transform,
    TransitionKind,
)


@dataclass(slots=True)
class LibraryFile:
    """A library item already downloaded, with what the graph needs to know.

    The kind matters because an image is not a video: it does not run in time,
    so it goes in on a loop and takes whatever duration the clip asks for.
    """

    path: Path
    kind: str = MediaKind.VIDEO

    @property
    def is_image(self) -> bool:
        return self.kind == MediaKind.IMAGE

    @property
    def is_audio(self) -> bool:
        return self.kind == MediaKind.AUDIO


@dataclass(slots=True)
class Input:
    """One ffmpeg `-i`, with whatever comes before it."""

    path: str
    #: `-ss` before the input: ffmpeg jumps to the nearest keyframe instead of
    #: decoding from the start, which makes each clip cost almost nothing
    seek: float | None = None
    duration: float | None = None
    #: synthetic inputs (colour, silence) come from `-f lavfi`
    lavfi: bool = False
    #: an image is a single frame; on a loop it becomes video for as long as asked
    loop: bool = False

    def args(self) -> list[str]:
        args: list[str] = []
        if self.lavfi:
            args += ["-f", "lavfi"]
        if self.loop:
            args += ["-loop", "1"]
        if self.seek is not None:
            args += ["-ss", f"{self.seek:.3f}"]
        if self.duration is not None:
            args += ["-t", f"{self.duration:.3f}"]
        args += ["-i", self.path]
        return args


@dataclass(slots=True)
class Composition:
    inputs: list[Input] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    video_map: str = ""
    audio_map: str | None = None
    duration_s: float = 0.0
    crf: int = 20

    @property
    def filter_complex(self) -> str:
        return ";".join(self.filters)

    def input_args(self) -> list[str]:
        return [a for e in self.inputs for a in e.args()]


def _position(clip: TimelineClip, clock: "KeyClock") -> tuple[str, str]:
    """Where the clip is overlaid, as expressions ffmpeg evaluates.

    The transform's `x` and `y` are offsets from the centre normalised by half
    the frame, so the same montage holds at any resolution: `W` and `H` are the
    canvas, `w` and `h` the already-scaled clip.

    **Text is the exception**: its canvas is already frame-sized and `drawtext`
    has already put the line in place inside it. Offsetting the canvas would
    move the text twice -- with `y=-0.5` it went over the edge and vanished.
    """
    if clip.source is ClipSource.TEXT:
        x, y = "0", "0"
    else:
        # keyframes in the overlay's own clock, which is the final video's
        local = f"(t-{clip.at_s:.3f})"
        cx = _animated(clip, KeyProp.X, local, clock) or f"{clip.transform.x:.4f}"
        cy = _animated(clip, KeyProp.Y, local, clock) or f"{clip.transform.y:.4f}"
        x = f"(W-w)/2+({cx})*(W/2)"
        y = f"(H-h)/2+({cy})*(H/2)"
    x, y = _slide(clip, x, y)
    # quoted when they carry commas: `if`/`min`/`max` would otherwise split
    # filters in the graph
    return tuple(f"'{e}'" if "," in e else e for e in (x, y))


#: Where a sliding clip starts, as a multiple of the frame: it comes in from
#: the side opposite to the movement.
_SLIDE_FROM = {
    TransitionKind.SLIDE_LEFT: (1, 0),
    TransitionKind.SLIDE_RIGHT: (-1, 0),
    TransitionKind.SLIDE_UP: (0, 1),
    TransitionKind.SLIDE_DOWN: (0, -1),
}


def _slide(clip: TimelineClip, x: str, y: str) -> tuple[str, str]:
    """The resting position, plus the stretch still to travel while it slides.

    The `t` of an overlay is the time of the final video, so the progress is
    counted from the clip's `at_s`; once it reaches 1 the extra term is zero and
    the clip sits where it was placed.
    """
    tr = clip.transition
    if tr is None or tr.kind not in _SLIDE_FROM:
        return x, y
    dx, dy = _SLIDE_FROM[tr.kind]
    remaining = f"(1-min(1,max(0,(t-{clip.at_s:.3f})/{tr.duration_s:.3f})))"
    if dx:
        x = f"{x}+({dx})*W*{remaining}"
    if dy:
        y = f"{y}+({dy})*H*{remaining}"
    return x, y


#: The colour each dip goes through.
_DIP_COLOR = {TransitionKind.FADE_BLACK: "black", TransitionKind.FADE_WHITE: "white"}


@dataclass(frozen=True)
class KeyClock:
    """Where a clip's keyframes fall on the clock its filters see.

    Keyframes are fractions of the clip **as the user placed it** (`span_s`).
    What reaches the filters is not always that clip: under a dissolve its
    picture runs longer, and through an export window it may start partway
    (`skipped_s` already gone). Converting with the drawn clip's duration put
    the zoom in the wrong place in both cases.
    """

    span_s: float
    skipped_s: float = 0.0

    def at(self, fraction: float) -> float:
        return fraction * self.span_s - self.skipped_s


@dataclass(frozen=True)
class Ramp:
    """A speed ramp seen from the filters: how much source has gone by at each
    instant of the drawn clip, and back.

    `clip` is the clip as placed (its keys are fractions of it); the clock
    says where the drawn copy starts inside it.
    """

    clip: TimelineClip
    clock: KeyClock

    def _base(self) -> float:
        return self.clip.source_offset(self.clock.skipped_s, self.clock.span_s)

    def src(self, local: float) -> float:
        """Source seconds gone by `local` seconds into the drawn clip."""
        at = self.clock.skipped_s + local
        return self.clip.source_offset(at, self.clock.span_s) - self._base()

    def local(self, src: float) -> float:
        """The drawn clip's instant at which `src` source seconds have gone by."""
        at = self.clip.local_for_source(self._base() + src, self.clock.span_s)
        return at - self.clock.skipped_s

    def setpts(self, length: float) -> str:
        """`setpts` placing each source frame at its instant in the clip.

        The map from source time to clip time is sampled every 1/30 s (at
        most 600 points) and joined with straight lines, in a balanced tree
        of `if`s so the expression stays shallow however long the clip.
        """
        n = max(1, min(600, int(length * 30)))
        outs = [length * i / n for i in range(n + 1)]
        pts = [(self.src(o), o) for o in outs]

        def seg(i: int) -> str:
            (s0, o0), (s1, o1) = pts[i], pts[i + 1]
            k = (o1 - o0) / max(1e-9, s1 - s0)
            return f"({o0:.5f}+(T-{s0:.5f})*{k:.6f})"

        def tree(i: int, j: int) -> str:
            if j - i == 1:
                return seg(i)
            mid = (i + j) // 2
            return f"if(lt(T,{pts[mid][0]:.5f}),{tree(i, mid)},{tree(mid, j)})"

        return f"setpts='({tree(0, n)})/TB'"


#: the shape of each easing, as an ffmpeg expression of the progress `u`
#: (0 to 1) between two keyframes
_EASE = {
    Ease.LINEAR: "{u}",
    Ease.IN: "({u})*({u})",
    Ease.OUT: "({u})*(2-({u}))",
    Ease.IN_OUT: "({u})*({u})*(3-2*({u}))",
}


def _curve(points: list[tuple[float, float, Ease]], var: str) -> str:
    """An ffmpeg expression running through (time, value, ease) points.

    What comes out is a ladder of `if`s, from the first point to the last;
    between each pair the value follows the ease of the point it leaves.
    Before the first point and after the last, the value is the endpoint's.
    """
    expr = f"{points[-1][1]:.4f}"
    for (t0, v0, ease), (t1, v1, _) in reversed(list(zip(points, points[1:]))):
        span = max(1e-6, t1 - t0)
        u = f"({var}-{t0:.4f})/{span:.4f}"
        line = f"({v0:.4f}+({v1 - v0:.4f})*{_EASE[ease].format(u=u)})"
        expr = f"if(lt({var},{t1:.4f}),{line},{expr})"
    return f"if(lt({var},{points[0][0]:.4f}),{points[0][1]:.4f},{expr})"


def _interpolate(keys: list, field_name: str, clock: KeyClock, var: str = "t") -> str:
    """The zoom's field between its keyframes. The `t` inside it is the
    clip's time with the speed already applied -- which is why keyframes are
    fractions: they follow the block when it stretches."""
    return _curve(
        [(clock.at(k.t), float(getattr(k, field_name)), k.ease) for k in keys],
        var,
    )


def _animated(
    clip: TimelineClip, prop: KeyProp, var: str, clock: KeyClock
) -> str | None:
    """The expression for an animated property, or `None` when it has no
    keyframes and its static value stands."""
    keys = clip.keys_for(prop)
    if not keys:
        return None
    return _curve([(clock.at(k.t), k.value, k.ease) for k in keys], var)


def _fit_chain(fit: Fit, width: int, height: int) -> list[str]:
    """Places the clip on the output canvas, which may have another aspect.

    This shows up for real when exporting 9:16 from a 16:9 recording, and both
    answers are legitimate: `cover` fills and crops the overflow -- in a
    gameplay montage the action is in the middle, and black bars on a phone are
    wasted screen; `contain` shows the whole frame and accepts the bars, for
    whoever needs what is in the corners.

    `force_original_aspect_ratio`'s `increase`/`decrease` does the maths on the
    longer side; the `crop` or the `pad` settles what is left over.
    """
    if fit is Fit.CONTAIN:
        return [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0.0",
        ]
    return [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
    ]


#: The looks, as ffmpeg filters -- a LUT's job without a LUT file.
_LOOKS = {
    Look.NOIR: ["hue=s=0", "eq=contrast=1.25:brightness=-0.02"],
    Look.TEAL_ORANGE: [
        "colorbalance=rs=-0.12:bs=0.14:rh=0.14:gh=0.03:bh=-0.12",
        "eq=saturation=1.15:contrast=1.05",
    ],
    Look.WARM: ["colorbalance=rm=0.10:bm=-0.10:rh=0.05", "eq=saturation=1.05"],
    Look.COLD: ["colorbalance=rm=-0.08:bm=0.12:bh=0.05", "eq=saturation=0.95"],
    Look.VIVID: ["eq=saturation=1.45:contrast=1.12"],
    Look.FADED: ["eq=contrast=0.82:brightness=0.06:saturation=0.75"],
}

#: How long an impact's flash lasts each side of the play, and how fast its
#: shake dies away (per second).
_FLASH_S = 0.18
_IMPACT_DECAY = 7.0


def _impact_at(clip: TimelineClip) -> float:
    """Where the impact hits, on the clip's clock: its play, or its start."""
    p = clip.play_local_s
    return 0.0 if p is None else p


def _shake_chain(clip: TimelineClip, width: int, height: int) -> list[str]:
    """Camera shake: the canvas-sized clip is enlarged by a margin and a
    canvas-sized window wanders inside it.

    `crop` evaluates `x`/`y` on every frame (only its size is fixed), and `t`
    is the clip's own clock here. The handheld part is constant; the impact's
    part starts at the play and dies away.
    """
    fx = clip.fx
    if not (fx.shake or fx.impact):
        return []
    mx = max(2, round(width * 0.03)) // 2 * 2
    my = max(2, round(height * 0.03)) // 2 * 2
    p = _impact_at(clip)
    amount = (
        f"min(1,{fx.shake * 0.5:.4f}"
        f"+{fx.impact:.4f}*gte(t,{p:.3f})*exp(-{_IMPACT_DECAY}*(t-{p:.3f})))"
    )
    wobble_x = "(0.6*sin(t*41.3)+0.4*sin(t*23.1+1.7))"
    wobble_y = "(0.6*sin(t*37.9+0.6)+0.4*sin(t*19.7+2.3))"
    return [
        f"scale={width + 2 * mx}:{height + 2 * my}",
        f"crop={width}:{height}"
        f":x='{mx}+{mx}*{amount}*{wobble_x}'"
        f":y='{my}+{my}*{amount}*{wobble_y}'",
    ]


def _fx_chain(clip: TimelineClip, height: int) -> list[str]:
    """Look, blur, sharpen, vignette and the impact's flash, in that order:
    the grade first, so the blur and the vignette work on the final colour."""
    fx = clip.fx
    steps: list[str] = list(_LOOKS.get(fx.look, []))
    if fx.blur:
        # up to a 12 px sigma on a 1080p frame, in proportion elsewhere
        steps.append(f"gblur=sigma={fx.blur * 12 * height / 1080:.2f}")
    if fx.sharpen:
        steps.append(f"unsharp=5:5:{0.3 + fx.sharpen * 1.7:.3f}:5:5:0")
    if fx.vignette:
        steps.append(f"vignette=angle={0.25 + fx.vignette * 0.9:.3f}")
    if fx.impact:
        p = _impact_at(clip)
        steps.append(
            f"eq=brightness='{fx.impact * 0.6:.3f}"
            f"*max(0,1-abs(t-{p:.3f})/{_FLASH_S})':eval=frame"
        )
    return steps


#: ffmpeg's names for the blend modes
_BLEND = {
    BlendMode.SCREEN: "screen",
    BlendMode.MULTIPLY: "multiply",
    BlendMode.OVERLAY: "overlay",
    BlendMode.ADD: "addition",
    BlendMode.LIGHTEN: "lighten",
    BlendMode.DARKEN: "darken",
    BlendMode.DIFFERENCE: "difference",
}


def _blend_chain(
    previous: str,
    n: int,
    output: str,
    mode: BlendMode,
    x: str,
    y: str,
    enable: str,
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
) -> list[str]:
    """A clip mixed with what is under it, instead of laid over it.

    `overlay` only covers; `blend` mixes two same-sized pictures. So the clip
    is first placed on a transparent canvas the size of the video, lasting
    all of it (so `blend` always has both pictures), and blended with
    everything below -- which is the *base*, the top input, as an image
    editor does for overlay. The result only counts where the clip is: its
    own alpha (crop, chroma key, fades, opacity) is the mask the blended
    picture is laid on with.
    """
    return [
        f"color=c=black@0.0:s={int(width)}x{int(height)}:r={fps:.3f}"
        f":d={duration:.3f},format=rgba[cv{n}]",
        f"[cv{n}][v{n}]overlay=x={x}:y={y}:{enable}:eof_action=pass"
        f":format=auto,format=gbrap,split[ca{n}][cb{n}]",
        f"[{previous}]format=gbrap,split[pa{n}][pb{n}]",
        f"[pa{n}][ca{n}]blend=all_mode={_BLEND[mode]}[bl{n}]",
        f"[cb{n}]alphaextract[m{n}]",
        f"[bl{n}][m{n}]alphamerge[bm{n}]",
        f"[pb{n}][bm{n}]overlay=format=auto[{output}]",
    ]


def _entrance_chain(tr: "ClipTransition") -> list[str]:
    """Wipes, zoom, spin and glitch: the entering clip's own pixels, remapped
    for the transition's length.

    One `geq` per kind, switched on only while the transition lasts
    (`enable`), so the rest of the clip pays nothing. `T` is the clip's own
    clock here and `p` the progress, 0 to 1. Zoom and spin sample the picture
    through an inverse scale and rotation around its centre, the way the
    keyframed scale does.
    """
    k = tr.kind
    d = f"{tr.duration_s:.3f}"
    p = f"min(1,T/{d})"
    q = f"(1-{p})"  # what is left
    rgb = "r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
    on = f"enable='lt(t,{d})'"
    if k in _WIPE_EDGE:
        visible = _WIPE_EDGE[k].format(p=p)
        return ["format=rgba", f"geq={rgb}:a='alpha(X,Y)*{visible}':{on}"]
    if k in (TransitionKind.ZOOM, TransitionKind.SPIN):
        if k is TransitionKind.ZOOM:
            z, th = f"(1+0.5*{q}*{q})", "0"
        else:
            z, th = f"(1-0.7*{q}*{q})", f"(-PI*{q}*{q})"
        dx, dy = "(X-W/2)", "(Y-H/2)"
        sx = f"(({dx}*cos({th})+{dy}*sin({th}))/{z}+W/2)"
        sy = f"((-{dx}*sin({th})+{dy}*cos({th}))/{z}+H/2)"
        inside = f"between({sx},0,W-1)*between({sy},0,H-1)"
        return [
            "format=rgba",
            f"geq=r='r({sx},{sy})':g='g({sx},{sy})':b='b({sx},{sy})'"
            f":a='if({inside},alpha({sx},{sy})*{p},0)':{on}",
        ]
    if k is TransitionKind.GLITCH:
        # bands a twelfth of the frame tall, each jumping its own way, the
        # red and blue channels pulled apart; it all calms down by the end
        o = f"(W*0.04*{q}*sin(T*91+floor(Y*12/H)*7.3))"
        return [
            "format=rgba",
            f"geq=r='r(X+{o},Y)':g='g(X+{o}/3,Y)':b='b(X-{o},Y)'"
            f":a='alpha(X,Y)':{on}",
        ]
    return []


#: The part of the frame a wipe has uncovered at progress `p`
_WIPE_EDGE = {
    TransitionKind.WIPE_LEFT: "gte(X,W*(1-{p}))",
    TransitionKind.WIPE_RIGHT: "lte(X,W*{p})",
    TransitionKind.WIPE_UP: "gte(Y,H*(1-{p}))",
    TransitionKind.WIPE_DOWN: "lte(Y,H*{p})",
}


def _crop_turn_chain(t: Transform, width: int, height: int) -> list[str]:
    """Crop, mirror and rotate the canvas-sized clip, in that order.

    The crop cuts the edges **away** -- the picture keeps its size and place
    and what was cut becomes transparent (`crop` then `pad` back), which is
    what lets a cropped killfeed sit over another clip. It works on the
    canvas-sized frame, so the fractions are the ones the editor drew.

    The rotation grows the frame to hold the turned picture whole
    (`rotw`/`roth`); the overlay centres it and the canvas cuts what sticks
    out, as the editor's monitor does.
    """
    if not (t.has_crop or t.has_turn):
        return []
    steps = ["format=rgba"]
    if t.has_crop:
        left = round(width * t.crop_left)
        top = round(height * t.crop_top)
        w = max(2, width - left - round(width * t.crop_right))
        h = max(2, height - top - round(height * t.crop_bottom))
        steps.append(f"crop={w}:{h}:{left}:{top}")
        steps.append(f"pad={width}:{height}:{left}:{top}:color=black@0.0")
    if t.flip_h:
        steps.append("hflip")
    if t.flip_v:
        steps.append("vflip")
    if t.rotation % 360:
        a = f"{math.radians(t.rotation):.6f}"
        steps.append(
            f"rotate=a={a}:ow='rotw({a})':oh='roth({a})':c=none"
        )
    return steps


def _zoom_chain(
    clip: TimelineClip, width: int, height: int, fps: float, clock: KeyClock
) -> list[str]:
    """The window that moves and tightens inside the clip.

    It used to be a `crop` with expressions in `t`, scaled back to the canvas.
    That never animated: `crop` evaluates its **width and height once**, when
    the filter is configured -- only `x` and `y` are per frame. The window kept
    a single size (the last keyframe's) and the lens stood still.

    `zoompan` is the filter made for this: zoom and position are evaluated on
    every frame. Its clock is `it`, the input frame's time, which here already
    starts at 0 and already counts the speed. It positions the window in whole
    pixels, and at canvas size a slow zoom would visibly step a pixel at a
    time; working on a frame twice as large halves the step.
    """
    z = _interpolate(clip.zoom, "scale", clock, "it")
    x = _interpolate(clip.zoom, "x", clock, "it")
    y = _interpolate(clip.zoom, "y", clock, "it")
    return [
        f"scale={2 * int(width)}:{2 * int(height)}",
        f"zoompan=z='{z}'"
        f":x='(iw-iw/zoom)*(0.5+({x})/2)'"
        f":y='(ih-ih/zoom)*(0.5+({y})/2)'"
        # one frame out per frame in: it is a video, not a still being panned
        f":d=1:s={int(width)}x{int(height)}:fps={fps:.3f}",
    ]


def _video_chain(
    clip: TimelineClip,
    input_index: int,
    output: str,
    width: int,
    height: int,
    fit: Fit,
    dip_out: tuple[str, float] | None = None,
    fps: float = 30.0,
    clock: KeyClock | None = None,
    ramp: Ramp | None = None,
) -> str:
    """What happens to a clip before it touches the canvas.

    `dip_out` is (colour, seconds): the clip after this one dips through that
    colour, and this one has to go into it on its way out.

    Order matters. Speed comes **before** everything, because it changes the
    clip's clock: a half-second fade has to last half a second in the final
    video, not half a second of the source.
    """
    clock = clock or KeyClock(clip.duration_s)
    # the trim is on the source, so it counts the speed
    consumed = ramp.src(clip.duration_s) if ramp else clip.source_consumed_s
    steps = [f"[{input_index}:v]trim=duration={consumed:.3f}"]
    steps.append("setpts=PTS-STARTPTS")

    if clip.reverse:
        # `reverse` needs the whole stretch in memory, and so only serves short
        # clips -- which is the case for a beat-synced montage
        steps.append("reverse")

    if clip.freeze:
        # A single frame, stretched over the block's duration.
        #
        # `tpad` turns that duration into frames using the link's frame rate,
        # and in ffmpeg 7 `setpts` leaves the rate **unknown**: the padding came
        # out as zero frames, and the frozen block as the black background.
        # The `fps` in front says the rate again.
        steps.append(f"fps={fps:.3f}")
        steps.append(f"tpad=stop_mode=clone:stop_duration={clip.duration_s:.3f}")
    elif ramp is not None:
        # each source frame goes to the instant the ramp brings it to
        steps.append(ramp.setpts(clip.duration_s))
    elif clip.speed != 1.0:
        # dividing the PTS speeds it up: at 2x, each frame is worth half the time
        steps.append(f"setpts=PTS/{clip.speed:.4f}")

    if clip.source is ClipSource.TEXT:
        # the canvas already arrives as rgba from the source itself (see
        # `_clip_input`): the text is drawn straight onto it
        steps.append(textfx.filter_chain(clip, height))

    # before anything that depends on size, the clip takes on the size of the
    # output canvas
    if clip.source is not ClipSource.TEXT:
        steps += _fit_chain(fit, width, height)

    # The lens comes after the framing: it zooms into what is on screen. Before
    # it, a 16:9 recording exported as 9:16 was zoomed in its own aspect and
    # then stretched into the other.
    if clip.zoom:
        steps += _zoom_chain(clip, width, height, fps, clock)

    if clip.source is not ClipSource.TEXT:
        steps += _shake_chain(clip, width, height)

    if not clip.color.is_neutral:
        steps.append(
            f"eq=brightness={clip.color.brightness:.4f}"
            f":contrast={clip.color.contrast:.4f}"
            f":saturation={clip.color.saturation:.4f}"
        )

    if clip.source is not ClipSource.TEXT:
        steps += _fx_chain(clip, height)

    if clip.chroma is not None and clip.source is not ClipSource.TEXT:
        # keyed on the graded picture, before the crop and the turn: the
        # transparent edges those make are not the screen's colour
        k = clip.chroma
        steps.append("format=rgba")
        steps.append(f"colorkey=0x{k.hex}:{k.similarity:.3f}:{k.softness:.3f}")

    if clip.source is not ClipSource.TEXT:
        steps += _crop_turn_chain(clip.transform, width, height)

    # Keyframes run on the clip's own clock: `T` below is the frame's time
    # with the speed applied, starting at the clip's first frame
    scale = _animated(clip, KeyProp.SCALE, "T", clock)
    opacity = _animated(clip, KeyProp.OPACITY, "T", clock)
    if scale is not None:
        # An animated size keeps the frame the canvas' size and resamples the
        # picture inside it, transparent around. `scale` with `eval=frame`
        # looked like the tool for this, but on ffmpeg 7.1 it keeps the first
        # frame's size -- the clip never grew. `geq` is slower (per pixel), so
        # only animated clips pay for it; it carries an animated opacity too.
        s_ = f"max(0.001,{scale})"
        sx = f"((X-W/2)/{s_}+W/2)"
        sy = f"((Y-H/2)/{s_}+H/2)"
        inside = f"between({sx},0,W-1)*between({sy},0,H-1)"
        alpha = f"alpha({sx},{sy})" + (f"*({opacity})" if opacity else "")
        steps.append("format=rgba")
        steps.append(
            f"geq=r='r({sx},{sy})':g='g({sx},{sy})':b='b({sx},{sy})'"
            f":a='if({inside},{alpha},0)'"
        )
        opacity = None  # already applied
    elif clip.transform.scale != 1.0:
        steps.append(
            f"scale=iw*{clip.transform.scale:.4f}:ih*{clip.transform.scale:.4f}"
        )

    tr = clip.transition
    dissolve = tr is not None and tr.kind is TransitionKind.DISSOLVE
    if tr is not None:
        steps += _entrance_chain(tr)

    # alpha only exists in rgba, and from here down everything touches it
    if (
        not clip.fade.is_neutral
        or (clip.transform.opacity < 1.0 and not clip.keys_for(KeyProp.OPACITY))
        or dissolve
        or opacity is not None
    ) and clip.source is not ClipSource.TEXT and scale is None:
        steps.append("format=rgba")

    # The entrance. A dissolve is a fade **of the alpha**: the previous clip
    # runs on underneath (see `compose_graph`) and shows through. A dip is a
    # fade of the colour, half the time on each side of the cut.
    if dissolve:
        steps.append(f"fade=t=in:st=0:d={tr.duration_s:.3f}:alpha=1")
    elif tr is not None and tr.kind in _DIP_COLOR:
        steps.append(
            f"fade=t=in:st=0:d={tr.duration_s / 2:.3f}:color={_DIP_COLOR[tr.kind]}"
        )
    if dip_out is not None:
        colour, d = dip_out
        start = max(0.0, clip.duration_s - d)
        steps.append(f"fade=t=out:st={start:.3f}:d={d:.3f}:color={colour}")

    if not clip.fade.is_neutral:
        # `alpha=1` is what makes the fade **reveal** what is underneath rather
        # than painting black over it. Over the black background it is the same
        # thing; over another layer it is the difference between a transition
        # and a dark smear.
        #
        # And it runs on the already-sped-up clock: the fade lasts what it
        # lasts in the video.
        if clip.fade.in_s > 0:
            steps.append(f"fade=t=in:st=0:d={clip.fade.in_s:.3f}:alpha=1")
        if clip.fade.out_s > 0:
            start = max(0.0, clip.duration_s - clip.fade.out_s)
            steps.append(
                f"fade=t=out:st={start:.3f}:d={clip.fade.out_s:.3f}:alpha=1"
            )

    if opacity is not None:
        # `colorchannelmixer` takes a number, not an expression: an animated
        # alpha is computed per pixel, which is slower, so only when asked
        steps.append(
            "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
            f":a='alpha(X,Y)*({opacity})'"
        )
    elif clip.transform.opacity < 1.0 and not clip.keys_for(KeyProp.OPACITY):
        steps.append(f"colorchannelmixer=aa={clip.transform.opacity:.4f}")

    # only now is the clip placed at its moment in the final video
    steps.append(f"setpts=PTS+{clip.at_s:.3f}/TB")
    return ",".join(steps) + f"[{output}]"


def _atempo_chain(factor: float) -> list[str]:
    """The `atempo` chain for an arbitrary factor.

    The filter only accepts 0.5 to 100 at a time; outside that, two are chained.
    Without this, slow motion below 0.5x would come out with the audio intact --
    and image and sound out of step is worse than having no sound.
    """
    steps: list[str] = []
    rest = factor
    while rest < 0.5:
        steps.append("atempo=0.5")
        rest /= 0.5
    while rest > 100.0:
        steps.append("atempo=100")
        rest /= 100.0
    if abs(rest - 1.0) > 1e-6:
        steps.append(f"atempo={rest:.4f}")
    return steps


#: Ducking's shape around a play: fully down from DUCK_BEFORE seconds before
#: it to DUCK_AFTER after, ramping down over DUCK_ATTACK and back up over
#: DUCK_RELEASE. The editor's monitor uses the same numbers.
DUCK_BEFORE = 0.15
DUCK_AFTER = 0.5
DUCK_ATTACK = 0.15
DUCK_RELEASE = 0.4


def _duck_curve(plays: list[float]) -> str:
    """How ducked the mix is at `t`, 0 to 1, as an ffmpeg expression: a
    trapezoid around each play, the highest one winning where they meet."""
    shapes = []
    for p in plays:
        start = p - DUCK_BEFORE - DUCK_ATTACK
        end = p + DUCK_AFTER + DUCK_RELEASE
        shapes.append(
            f"max(0,min(1,min((t-{start:.3f})/{DUCK_ATTACK},"
            f"({end:.3f}-t)/{DUCK_RELEASE})))"
        )
    expr = shapes[0]
    for shape in shapes[1:]:
        expr = f"max({expr},{shape})"
    return expr


def _audio_chain(
    clip: TimelineClip,
    input_index: int,
    output: str,
    clock: KeyClock | None = None,
) -> str | None:
    """The clip's sound, delayed until the moment it comes in."""
    clock = clock or KeyClock(clip.duration_s)
    volume = _animated(clip, KeyProp.VOLUME, "t", clock)
    if clip.audio.mute or (volume is None and clip.audio.volume <= 0):
        return None
    # a frozen frame has no sound running alongside it
    if clip.freeze:
        return None
    ms = int(round(clip.at_s * 1000))
    steps = [
        f"[{input_index}:a]atrim=duration={clip.source_consumed_s:.3f}",
        "asetpts=PTS-STARTPTS",
    ]
    if clip.reverse:
        steps.append("areverse")
    if clip.speed != 1.0:
        steps += _atempo_chain(clip.speed)
    if clip.audio.fade_in_s > 0:
        steps.append(f"afade=t=in:st=0:d={clip.audio.fade_in_s:.3f}")
    if clip.audio.fade_out_s > 0:
        start = max(0.0, clip.duration_s - clip.audio.fade_out_s)
        steps.append(
            f"afade=t=out:st={start:.3f}:d={clip.audio.fade_out_s:.3f}"
        )
    if volume is not None:
        # the clock here is the clip's, before the delay that places it
        steps.append(f"volume='{volume}':eval=frame")
    elif clip.audio.volume != 1.0:
        steps.append(f"volume={clip.audio.volume:.4f}")
    if ms > 0:
        steps.append(f"adelay={ms}|{ms}")
    return ",".join(steps) + f"[{output}]"


def _within_window(
    clip: TimelineClip, start: float, end: float, span: float | None = None
) -> TimelineClip | None:
    """The clip as seen through the export window, or `None` if it fell outside.

    Exporting a stretch is not trimming the video once it is finished: the clips
    are repositioned as if the window were the beginning. A clip that starts
    before it comes in partway -- and then its entry point **into the source**
    moves along with it, in proportion to the speed, or the image would jump.
    """
    if clip.until_s <= start + 1e-6 or clip.at_s >= end - 1e-6:
        return None

    eaten_before = max(0.0, start - clip.at_s)
    left_after = max(0.0, clip.until_s - end)
    new_duration = clip.duration_s - eaten_before - left_after
    if new_duration < MIN_CUT_S:
        return None

    return clip.model_copy(
        update={
            # an entrance that happened before the window is not seen in it
            "transition": clip.transition if eaten_before <= 0 else None,
            "at_s": max(0.0, clip.at_s - start),
            "duration_s": new_duration,
            # how much of the clip was skipped costs more source when it runs
            # sped up -- the integral, under a ramp; on a text or an image,
            # `start_s` means nothing
            "start_s": clip.start_s + clip.source_offset(eaten_before, span),
        }
    )


def _watermark(
    exp,
    input_index: int,
    previous: str,
    output: str,
    width: int,
) -> list[str]:
    """The mark over everything, in the chosen corner.

    It comes after every layer on purpose: a watermark some layer covers is not
    a watermark.
    """
    mark_width = max(1, int(round(exp.watermark_scale * width)))
    steps = [
        f"[{input_index}:v]scale={mark_width}:-1,format=rgba"
        f",colorchannelmixer=aa={exp.watermark_opacity:.4f}[mark]"
    ]
    x = f"(W-w)/2+({exp.watermark_x:.4f})*(W/2)"
    y = f"(H-h)/2+({exp.watermark_y:.4f})*(H/2)"
    steps.append(f"[{previous}][mark]overlay=x={x}:y={y}[{output}]")
    return steps


def _clip_input(
    clip: TimelineClip,
    *,
    source: Path,
    source_duration_s: float,
    library: dict[str, LibraryFile],
    width: int,
    height: int,
    fps: float,
    ramp: Ramp | None = None,
) -> tuple[Input | None, float, bool]:
    """This clip's ffmpeg input, its usable duration, and whether it has sound.

    Returns `None` when the clip falls outside the source -- its slot then shows
    the background canvas, and the clips after it do not move from where they
    were placed.
    """
    # what is asked of the source is what the speed consumes, not what the
    # clip occupies in the video -- under a ramp, the integral of the speed
    consumed = ramp.src(clip.duration_s) if ramp else clip.source_consumed_s

    if clip.source is ClipSource.RECORDING:
        wanted = consumed
        if source_duration_s > 0:
            wanted = min(wanted, max(0.0, source_duration_s - clip.start_s))
        if wanted <= 0:
            return None, 0.0, False
        # Trimmed at the source, it shrinks in the video by the same proportion
        # -- except when frozen, which consumes one frame and occupies the whole
        # block: there the duration in the video is not a consequence of what
        # was consumed.
        if clip.freeze:
            usable = clip.duration_s
        elif ramp:
            usable = ramp.local(wanted) if wanted < consumed - 1e-6 else clip.duration_s
        else:
            usable = wanted / clip.speed
        # a ramp has no sound: `atempo` takes one rate, not a curve
        return (
            Input(path=str(source), seek=clip.start_s, duration=wanted),
            usable,
            ramp is None,
        )

    if clip.source is ClipSource.TEXT:
        # A transparent canvas the size of the frame, where the text is drawn.
        # From then on it is a clip like any other -- it moves, it fades, it
        # travels through layers.
        #
        # The `format=rgba` goes **inside the source**, and not in the video
        # chain. The difference is not cosmetic: without it there, `color`
        # negotiates yuv420p with `drawtext`, draws opaque black, and the
        # following `format=rgba` only adds an alpha that was born at 1. The
        # result is a black canvas over everything -- the text appeared, and the
        # video vanished underneath it.
        return (
            Input(
                path=f"color=c=black@0.0:s={int(width)}x{int(height)}"
                f":r={fps:.3f},format=rgba",
                duration=clip.duration_s,
                lavfi=True,
            ),
            clip.duration_s,
            False,
        )

    if clip.source is ClipSource.MEDIA:
        item = library.get(clip.media_id or "")
        if item is None:
            raise ValueError(
                f"media {clip.media_id!r} is not in this job's library"
            )
        if item.is_image:
            # an image does not run in time: it goes in on a loop and lasts
            # whatever the clip asks for, with no `-ss` (there is nowhere to
            # seek in a single frame) and no speed (there is nothing to speed up
            # in a still frame)
            return (
                Input(
                    path=str(item.path),
                    duration=clip.duration_s,
                    loop=True,
                ),
                clip.duration_s,
                False,
            )
        return (
            Input(
                path=str(item.path),
                seek=clip.start_s,
                duration=consumed,
            ),
            clip.duration_s,
            ramp is None,
        )

    # solid colour arrives when it is missed; ignoring it silently would be
    # worse than refusing it
    raise ValueError(f"source '{clip.source}' cannot be rendered yet")


def compose_graph(
    timeline: Timeline,
    *,
    source: Path,
    width: int,
    height: int,
    fps: float,
    source_duration_s: float = 0.0,
    library: dict[str, LibraryFile] | None = None,
    video_only: bool = False,
) -> Composition:
    """The graph that builds this timeline.

    Layers come in bottom to top, and within each one the clips come in time
    order. A hidden layer does not come in; a muted layer comes in without sound.

    `library` maps each library item's id to its file on disk. A media clip is
    one more input in the graph, and from there on it goes through the same
    transformations as a stretch of the recording -- that is the point of having
    a single clip format.

    A clip running past the end of the recording is **trimmed**, as in V1 --
    whatever is left of its slot becomes the background canvas, and the clips
    after it do not move from where they were placed.
    """
    exp = timeline.export
    source_fps = fps if fps > 0 else 30.0
    fps = exp.fps or source_fps
    width, height = exp.dimensions(width, height)
    fit = exp.fit

    # the stretch asked for: everything, or a window of it
    start = exp.from_s
    end = min(exp.to_s, timeline.duration_s) if exp.to_s else timeline.duration_s
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError("the requested export range is empty")
    c = Composition(duration_s=duration, crf=exp.crf)

    # the background canvas: it is what shows at every instant nobody covered
    c.inputs.append(
        Input(
            path=f"color=c=black:s={int(width)}x{int(height)}:r={fps:.3f}",
            duration=duration,
            lavfi=True,
        )
    )
    c.filters.append("[0:v]setsar=1[bg]")

    # With music and `game_volume` at 0, it replaces the cuts' sound. Not
    # building their chain is not a saving: an audio chain with no output makes
    # the graph invalid, and ffmpeg refuses the whole set.
    has_music = timeline.has_music
    # the plays inside the stretch asked for, on its clock
    plays = (
        [p - start for p in timeline.play_times() if start - 2 <= p <= end + 2]
        if timeline.duck_plays and has_music
        else []
    )
    duck = _duck_curve(plays) if plays else None
    # ducking brings the game sound up at the plays, even when it is
    # otherwise left out of the mix
    game_comes_in = not has_music or timeline.game_volume > 0 or duck is not None

    previous = "bg"
    #: the sound coming from the cuts -- what `game_volume` governs
    cut_audio: list[str] = []
    #: the sound of the music blocks, which is not game sound and does not obey it
    music_audio: list[str] = []
    n = 0

    for layer in timeline.layers:
        if layer.hidden:
            continue
        # an audio layer draws nothing: with `video_only` it has nothing to do
        # here, and building its input would mean paying for a file nobody
        # would hear
        if layer.is_audio and video_only:
            continue
        clips = layer.clips
        for i, original in enumerate(clips):
            # How the **next** clip enters decides how this one leaves: a
            # dissolve or a slide needs this one still on screen underneath,
            # so its picture runs past the cut for the transition's length --
            # its sound does not, and the ruler does not move. A dip needs this
            # one to go dark on its way out.
            nxt = clips[i + 1] if i + 1 < len(clips) else None
            tail = 0.0
            dip_out: tuple[str, float] | None = None
            if (
                not layer.is_audio
                and nxt is not None
                and nxt.transition is not None
                and abs(nxt.at_s - original.until_s) < 1e-3
            ):
                tr = nxt.transition
                if tr.kind.overlaps:
                    tail = tr.duration_s
                elif tr.kind in _DIP_COLOR:
                    dip_out = (_DIP_COLOR[tr.kind], tr.duration_s / 2)
            drawn = (
                original.model_copy(
                    update={"duration_s": original.duration_s + tail}
                )
                if tail
                else original
            )
            clip = _within_window(drawn, start, end, original.duration_s)
            if clip is None:
                continue  # outside the stretch asked for
            heard = (
                _within_window(original, start, end, original.duration_s)
                if tail
                else clip
            )
            # a dip whose end fell outside the window would happen off screen
            if dip_out is not None and clip.until_s < original.until_s - start - 1e-6:
                dip_out = None

            # keyframes follow the clip as placed, not as drawn or windowed
            clock = KeyClock(
                span_s=original.duration_s,
                skipped_s=max(0.0, start - original.at_s),
            )
            ramp = Ramp(original, clock) if original.is_ramped else None

            clip_input, usable_duration, has_sound = _clip_input(
                clip,
                source=source,
                source_duration_s=source_duration_s,
                library=library or {},
                width=width,
                height=height,
                fps=fps,
                ramp=ramp,
            )
            if clip_input is None:
                continue  # falls outside the source; its slot stays background

            n += 1
            c.inputs.append(clip_input)
            trimmed = clip.model_copy(update={"duration_s": usable_duration})

            if not layer.is_audio:
                c.filters.append(
                    _video_chain(
                        trimmed,
                        n,
                        f"v{n}",
                        width,
                        height,
                        fit,
                        dip_out=dip_out,
                        fps=fps,
                        clock=clock,
                        ramp=ramp,
                    )
                )
                x, y = _position(trimmed, clock)
                output = f"t{n}"
                enable = (
                    f"enable='between(t,{trimmed.at_s:.3f},"
                    f"{trimmed.until_s:.3f})'"
                )
                if trimmed.blend is BlendMode.NORMAL:
                    c.filters.append(
                        f"[{previous}][v{n}]overlay=x={x}:y={y}:{enable}:"
                        f"eof_action=pass[{output}]"
                    )
                else:
                    c.filters += _blend_chain(
                        previous, n, output, trimmed.blend, x, y, enable,
                        width=width, height=height, fps=fps, duration=duration,
                    )
                previous = output

            # with `video_only` the clips' sound is not built either: an audio
            # chain with no output makes the graph invalid and ffmpeg refuses
            # the whole set
            keeps_sound = layer.is_audio or game_comes_in
            if (
                not video_only
                and not layer.muted
                and has_sound
                and keeps_sound
                and heard is not None
            ):
                # the extra picture past the cut is silent: the sound stops
                # where the clip stops on the ruler
                sound = trimmed.model_copy(
                    update={"duration_s": min(heard.duration_s, usable_duration)}
                )
                chain = _audio_chain(sound, n, f"a{n}", clock)
                if chain is not None:
                    c.filters.append(chain)
                    (music_audio if layer.is_audio else cut_audio).append(f"a{n}")

    if n == 0:
        raise ValueError("no clip falls inside the recording")

    if exp.watermark_id:
        mark = (library or {}).get(exp.watermark_id)
        if mark is None:
            raise ValueError(
                f"watermark {exp.watermark_id!r} is not in the library"
            )
        c.inputs.append(Input(path=str(mark.path), loop=mark.is_image,
                              duration=duration if mark.is_image else None))
        c.filters += _watermark(exp, len(c.inputs) - 1, previous, "watermarked",
                                width)
        previous = "watermarked"

    c.filters.append(
        f"[{previous}]trim=duration={duration:.3f},setpts=PTS-STARTPTS[vout]"
    )
    c.video_map = "[vout]"

    if video_only:
        # the picture alone, with no audio track at all: whoever asks like this
        # wants the video muted
        return c

    # The final mix. There are two sounds, and they do not mean the same thing:
    # the music blocks on one side, the cuts' sound on the other -- and
    # `game_volume` governs only the second.
    parts: list[str] = []

    if music_audio:
        volume = (
            f",volume={timeline.music_volume:.4f}"
            if timeline.music_volume != 1.0
            else ""
        )
        if duck is not None:
            # down to duck_level at each play, on the output's own clock
            drop = 1 - timeline.duck_level
            volume = (
                f",volume='{timeline.music_volume:.4f}*(1-{drop:.4f}*({duck}))'"
                ":eval=frame"
            )
        if len(music_audio) == 1 and not volume:
            parts.append(music_audio[0])
        else:
            entry = "".join(f"[{m}]" for m in music_audio)
            join = (
                f"amix=inputs={len(music_audio)}:dropout_transition=0:normalize=0"
                if len(music_audio) > 1
                else "anull"
            )
            c.filters.append(f"{entry}{join}{volume}[music]")
            parts.append("music")

    if cut_audio and game_comes_in:
        if has_music:
            # with music playing, the game sound comes in at the level asked
            # for -- which is what lets the shot show through underneath it
            game = "".join(f"[{a}]" for a in cut_audio)
            join = (
                f"amix=inputs={len(cut_audio)}:dropout_transition=0:normalize=0"
                if len(cut_audio) > 1
                else "anull"
            )
            volume = (
                f",volume={timeline.game_volume:.4f}"
                if timeline.game_volume != 1.0
                else ""
            )
            if duck is not None:
                # up to full at each play: the shot over the song
                low = timeline.game_volume
                boost = max(low, 1.0) - low
                volume = (
                    f",volume='{low:.4f}+{boost:.4f}*({duck})':eval=frame"
                )
            c.filters.append(f"{game}{join}{volume}[game]")
            parts.append("game")
        else:
            # with no music at all, the cuts' original audio stands on its own
            parts += cut_audio

    if len(parts) == 1:
        # mixing a single track is wasted work, and `amix` would also mess with
        # its volume for no reason
        c.filters.append(f"[{parts[0]}]anull[aout]")
        c.audio_map = "[aout]"
    elif parts:
        entry = "".join(f"[{p}]" for p in parts)
        c.filters.append(
            f"{entry}amix=inputs={len(parts)}:dropout_transition=0:"
            f"normalize=0[aout]"
        )
        c.audio_map = "[aout]"

    return c
