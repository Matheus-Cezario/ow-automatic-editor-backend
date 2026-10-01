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

from dataclasses import dataclass, field
from pathlib import Path

from . import textfx
from .models import (
    MIN_CUT_S,
    ClipSource,
    Fit,
    MediaKind,
    Timeline,
    TimelineClip,
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


def _position(clip: TimelineClip) -> tuple[str, str]:
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
        x = f"(W-w)/2+({clip.transform.x:.4f})*(W/2)"
        y = f"(H-h)/2+({clip.transform.y:.4f})*(H/2)"
    return _slide(clip, x, y)


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
    # quoted: the commas of `min`/`max` would otherwise split filters in the graph
    if dx:
        x = f"'{x}+({dx})*W*{remaining}'"
    if dy:
        y = f"'{y}+({dy})*H*{remaining}'"
    return x, y


#: The colour each dip goes through.
_DIP_COLOR = {TransitionKind.FADE_BLACK: "black", TransitionKind.FADE_WHITE: "white"}


def _interpolate(
    keys: list, field_name: str, duration_s: float, var: str = "t"
) -> str:
    """An ffmpeg expression interpolating the field between keyframes.

    What comes out is a ladder of `if`s, from the first point to the last, with
    a straight line between each pair. The `t` inside it is the clip's time with
    the speed already applied -- which is why the keyframes are fractions and
    not seconds: they follow the block when it stretches.
    """
    points = [
        (max(0.0, k.t * duration_s), float(getattr(k, field_name))) for k in keys
    ]

    # before the first point and after the last, the value is the endpoint's
    expr = f"{points[-1][1]:.4f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(points, points[1:]))):
        span = max(1e-6, t1 - t0)
        line = f"({v0:.4f}+({v1 - v0:.4f})*({var}-{t0:.4f})/{span:.4f})"
        expr = f"if(lt({var},{t1:.4f}),{line},{expr})"
    return f"if(lt({var},{points[0][0]:.4f}),{points[0][1]:.4f},{expr})"


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


def _zoom_chain(clip: TimelineClip, width: int, height: int, fps: float) -> list[str]:
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
    z = _interpolate(clip.zoom, "scale", clip.duration_s, "it")
    x = _interpolate(clip.zoom, "x", clip.duration_s, "it")
    y = _interpolate(clip.zoom, "y", clip.duration_s, "it")
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
) -> str:
    """What happens to a clip before it touches the canvas.

    `dip_out` is (colour, seconds): the clip after this one dips through that
    colour, and this one has to go into it on its way out.

    Order matters. Speed comes **before** everything, because it changes the
    clip's clock: a half-second fade has to last half a second in the final
    video, not half a second of the source.
    """
    # the trim is on the source, so it counts the speed
    steps = [f"[{input_index}:v]trim=duration={clip.source_consumed_s:.3f}"]
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
        steps += _zoom_chain(clip, width, height, fps)

    if not clip.color.is_neutral:
        steps.append(
            f"eq=brightness={clip.color.brightness:.4f}"
            f":contrast={clip.color.contrast:.4f}"
            f":saturation={clip.color.saturation:.4f}"
        )

    if clip.transform.scale != 1.0:
        steps.append(
            f"scale=iw*{clip.transform.scale:.4f}:ih*{clip.transform.scale:.4f}"
        )

    tr = clip.transition
    dissolve = tr is not None and tr.kind is TransitionKind.DISSOLVE

    # alpha only exists in rgba, and from here down everything touches it
    if (not clip.fade.is_neutral or clip.transform.opacity < 1.0 or dissolve) and (
        clip.source is not ClipSource.TEXT
    ):
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

    if clip.transform.opacity < 1.0:
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


def _audio_chain(clip: TimelineClip, input_index: int, output: str) -> str | None:
    """The clip's sound, delayed until the moment it comes in."""
    if clip.audio.mute or clip.audio.volume <= 0:
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
    if clip.audio.volume != 1.0:
        steps.append(f"volume={clip.audio.volume:.4f}")
    if ms > 0:
        steps.append(f"adelay={ms}|{ms}")
    return ",".join(steps) + f"[{output}]"


def _within_window(
    clip: TimelineClip, start: float, end: float
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
            # sped up; on a text or an image, `start_s` means nothing
            "start_s": clip.start_s + eaten_before * clip.speed,
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
) -> tuple[Input | None, float, bool]:
    """This clip's ffmpeg input, its usable duration, and whether it has sound.

    Returns `None` when the clip falls outside the source -- its slot then shows
    the background canvas, and the clips after it do not move from where they
    were placed.
    """
    if clip.source is ClipSource.RECORDING:
        # what is asked of the source is what the speed consumes, not what the
        # clip occupies in the video
        wanted = clip.source_consumed_s
        if source_duration_s > 0:
            wanted = min(wanted, max(0.0, source_duration_s - clip.start_s))
        if wanted <= 0:
            return None, 0.0, False
        # Trimmed at the source, it shrinks in the video by the same proportion
        # -- except when frozen, which consumes one frame and occupies the whole
        # block: there the duration in the video is not a consequence of what
        # was consumed.
        return (
            Input(path=str(source), seek=clip.start_s, duration=wanted),
            clip.duration_s if clip.freeze else wanted / clip.speed,
            True,
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
                f"a midia {clip.media_id!r} nao esta na biblioteca deste job"
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
                duration=clip.source_consumed_s,
            ),
            clip.duration_s,
            True,
        )

    # solid colour arrives when it is missed; ignoring it silently would be
    # worse than refusing it
    raise ValueError(f"fonte '{clip.source}' ainda nao e montavel")


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
        raise ValueError("o trecho pedido para exportar esta vazio")
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
    game_comes_in = not has_music or timeline.game_volume > 0

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
                else:
                    dip_out = (_DIP_COLOR[tr.kind], tr.duration_s / 2)
            drawn = (
                original.model_copy(
                    update={"duration_s": original.duration_s + tail}
                )
                if tail
                else original
            )
            clip = _within_window(drawn, start, end)
            if clip is None:
                continue  # outside the stretch asked for
            heard = _within_window(original, start, end) if tail else clip
            # a dip whose end fell outside the window would happen off screen
            if dip_out is not None and clip.until_s < original.until_s - start - 1e-6:
                dip_out = None

            clip_input, usable_duration, has_sound = _clip_input(
                clip,
                source=source,
                source_duration_s=source_duration_s,
                library=library or {},
                width=width,
                height=height,
                fps=fps,
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
                    )
                )
                x, y = _position(trimmed)
                output = f"t{n}"
                c.filters.append(
                    f"[{previous}][v{n}]overlay=x={x}:y={y}:"
                    f"enable='between(t,{trimmed.at_s:.3f},"
                    f"{trimmed.until_s:.3f})':"
                    f"eof_action=pass[{output}]"
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
                chain = _audio_chain(sound, n, f"a{n}")
                if chain is not None:
                    c.filters.append(chain)
                    (music_audio if layer.is_audio else cut_audio).append(f"a{n}")

    if n == 0:
        raise ValueError("nenhum clipe cai dentro da gravacao")

    if exp.watermark_id:
        mark = (library or {}).get(exp.watermark_id)
        if mark is None:
            raise ValueError(
                f"a marca d'agua {exp.watermark_id!r} nao esta na biblioteca"
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
