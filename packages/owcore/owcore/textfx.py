"""The editor's text turning into ffmpeg `drawtext`.

Kept apart from `compose.py` for a practical reason: escaping text for a
filtergraph is the kind of code where one backslash too many or too few slips
past review and only shows up when somebody types `50%` into a label. On its
own, it gets its own test.

The entrance and exit animations are expressions in `t`, the text canvas' own
clock (it starts at the clip's first frame). The editor's monitor runs the
same formulas (`lib/text_anim.dart`): change one, change the other.
"""

from __future__ import annotations

import math

from . import fonts, textlayout
from .models import TextAlign, TextAnim, TimelineClip

#: What needs a backslash in front of it, and why:
#:
#: * `\` -- the backslash itself, or it eats the next character;
#: * `:` -- separates one option from the next inside the filter;
#: * `'` -- delimits an option's value.
#:
#: `%` is **not** in here, and that lesson was expensive: escaped with a
#: backslash, `drawtext` warns "Stray %" and **draws nothing** -- the text
#: vanished from the whole video, with no error at all. What handles `%` is
#: `expansion=none` in the chain, which turns `%{...}` off for good.
#:
#: A line break becomes a space: `drawtext` accepts several lines, but the
#: filtergraph is a single line, and a raw break there splits the graph in two.
#: Typed breaks never reach here anyway: each line is its own `drawtext`.
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\\", "\\\\"),
    (":", "\\:"),
    ("'", "\\'"),
    ("\n", " "),
    ("\r", " "),
)

#: how far a slide travels, as a fraction of the frame height
SLIDE_DISTANCE = 0.08
#: the typewriter's pace, in characters a second
TYPE_RATE = 25.0
#: at most this many drawtext steps for the typewriter; a longer text types
#: several characters a step
TYPE_MAX_STEPS = 40
#: from one line's top to the next, in letter sizes -- the monitor's `height`
LINE_HEIGHT = 1.2
#: room between the text and the edge of its box, in letter sizes
BOX_PAD = 0.3
#: how far the shadow falls, down and right, in letter sizes
SHADOW_OFFSET = 0.06
#: how dark the shadow is
SHADOW_OPACITY = 0.8
#: a box follows the text's animation in steps this long
BOX_STEP_S = 1 / 30


def escape(text: str) -> str:
    """Lets the text through `drawtext` without becoming syntax."""
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _ease_out(p: str) -> str:
    return f"({p})*(2-({p}))"


def _motion(clip: TimelineClip) -> tuple[str, str, str]:
    """(alpha, size factor, downward offset in frame heights) as expressions
    in `t`. 1, 1 and 0 when the text does not animate."""
    style = clip.text_style
    d = min(style.anim_s, clip.duration_s / 2)
    p_in = f"min(1,max(0,t/{d:.4f}))"
    p_out = f"min(1,max(0,({clip.duration_s:.4f}-t)/{d:.4f}))"
    alpha, size, drop = ["1"], ["1"], ["0"]

    for anim, p in ((style.anim_in, p_in), (style.anim_out, p_out)):
        e = _ease_out(p)
        if anim is TextAnim.FADE:
            alpha.append(e)
        elif anim is TextAnim.POP:
            size.append(f"(0.5+0.5*{e})")
            alpha.append(f"min(1,3*{p})")
        elif anim is TextAnim.SLIDE:
            drop.append(f"{SLIDE_DISTANCE}*(1-{e})")
            alpha.append(e)
    return "*".join(alpha), "*".join(size), "+".join(drop)


def _motion_at(clip: TimelineClip, t: float) -> tuple[float, float, float]:
    """`_motion`, worked out at one instant: (alpha, size factor, drop)."""
    style = clip.text_style
    d = min(style.anim_s, clip.duration_s / 2)
    alpha, size, drop = 1.0, 1.0, 0.0
    for anim, p in (
        (style.anim_in, min(1.0, max(0.0, t / d))),
        (style.anim_out, min(1.0, max(0.0, (clip.duration_s - t) / d))),
    ):
        e = p * (2 - p)
        if anim is TextAnim.FADE:
            alpha *= e
        elif anim is TextAnim.POP:
            size *= 0.5 + 0.5 * e
            alpha *= min(1.0, 3 * p)
        elif anim is TextAnim.SLIDE:
            drop += SLIDE_DISTANCE * (1 - e)
            alpha *= e
    return alpha, size, drop


def _typing(clip: TimelineClip, lines: list[str]) -> list[tuple[list[str], float, float]]:
    """The typewriter's steps: (each line as shown, from, until). Empty when
    the text does not type itself.

    The letters are counted across the lines already broken, so a line never
    re-breaks as it fills: a word starts on the line it will end on.
    """
    if clip.text_style.anim_in is not TextAnim.TYPEWRITER:
        return []
    n = sum(len(line) for line in lines)
    if n == 0:
        return []
    d = min(clip.text_style.anim_s, clip.duration_s / 2)
    typing = min(max(d, n / TYPE_RATE), clip.duration_s)
    steps = min(n, TYPE_MAX_STEPS)
    out = []
    for k in range(1, steps + 1):
        left = math.ceil(n * k / steps)
        shown = []
        for line in lines:
            shown.append(line[:left])
            left = max(0, left - len(line))
        out.append((shown, typing * (k - 1) / steps, typing * k / steps))
    # the last step stays until the end
    shown, start, _ = out[-1]
    out[-1] = (shown, start, clip.duration_s + 1)
    return out


def filter_chain(clip: TimelineClip, height: int, width: int | None = None) -> str:
    """This clip's `drawtext`s (one per line), and its box, already positioned
    in the frame.

    Size and outline come as a **fraction of the height**: the same montage
    comes out identical at 720p and at 4K, and a 48px body that looks right in
    one would be tiny in the other. The outline is not decoration -- without
    it, white text disappears against a bright scene.

    The lines are a block centred on the clip's position, a line height apart;
    each line's *baseline* is placed, not its top, so a line of capitals and
    one with a `g` sit on the same rows the monitor draws.
    """
    width = width or int(round(height * 16 / 9))
    style = clip.text_style
    body_px = max(1, int(round(style.size * height)))
    outline_px = int(round(style.outline * body_px))
    font = fonts.resolve(style.font)
    alpha, size, drop = _motion(clip)
    animated = (alpha, size, drop) != ("1", "1", "0")

    box_w = style.width * width if style.width else None
    lines = textlayout.wrap(font, clip.text, body_px, box_w)
    n = len(lines)
    # the block: as wide as its box, or as its widest line
    block_w = box_w or max(textlayout.width(font, line, body_px) for line in lines)
    line_px = LINE_HEIGHT * body_px
    baseline = textlayout.ascent_share(font) * line_px

    # the clip's position: the centre is 0, the edges are -1 and 1
    cx = f"(w/2+({clip.transform.x:.4f})*(w/2))"
    cy = f"(h/2+({clip.transform.y:.4f})*(h/2))"
    k = f"({size})" if animated else "1"

    def x_of() -> str:
        if style.align is TextAlign.LEFT:
            return f"{cx}-{block_w:.2f}*{k}/2"
        if style.align is TextAlign.RIGHT:
            return f"{cx}+{block_w:.2f}*{k}/2-text_w"
        return f"{cx}-text_w/2"

    def y_of(i: int) -> str:
        top = f"{cy}-{n * line_px:.2f}*{k}/2"
        y = f"{top}+{i * line_px + baseline:.2f}*{k}-ascent"
        return f"{y}+({drop})*h" if animated else y

    shadow_px = max(1, int(round(SHADOW_OFFSET * body_px)))

    def drawtext(text: str, i: int, enable: str | None = None) -> str:
        parts = [
            f"fontfile='{escape(font)}'",
            f"text='{escape(text)}'",
            # no expansion whatsoever: the user's text is text, and a stray `%`
            # in it would make drawtext give up on drawing the whole line
            "expansion=none",
            f"fontsize='{body_px}*({size})'" if animated else f"fontsize={body_px}",
            f"fontcolor={style.color}",
            f"x='{x_of()}'",
            f"y='{y_of(i)}'",
        ]
        if animated:
            parts.append(f"alpha='{alpha}'")
        if outline_px > 0:
            parts += [f"borderw={outline_px}", f"bordercolor={style.outline_color}"]
        if style.shadow:
            parts += [
                f"shadowx={shadow_px}",
                f"shadowy={shadow_px}",
                f"shadowcolor={style.shadow}@{SHADOW_OPACITY}",
            ]
        if enable:
            parts.append(f"enable='{enable}'")
        return "drawtext=" + ":".join(parts)

    chain = _box_chain(clip, width, height, block_w, n * line_px, body_px)
    steps = _typing(clip, lines)
    if not steps:
        chain += [drawtext(line, i) for i, line in enumerate(lines) if line]
        return ",".join(chain)
    # each line gets a drawtext for each stretch it stays the same
    for i in range(n):
        runs: list[tuple[str, float, float]] = []
        for shown, start, until in steps:
            text = shown[i]
            if runs and runs[-1][0] == text:
                runs[-1] = (text, runs[-1][1], until)
            else:
                runs.append((text, start, until))
        chain += [
            drawtext(text, i, f"gte(t,{start:.4f})*lt(t,{until:.4f})")
            for text, start, until in runs
            if text
        ]
    return ",".join(chain)


def _box_chain(
    clip: TimelineClip, width: int, height: int,
    block_w: float, block_h: float, body_px: int,
) -> list[str]:
    """The box behind the text, as `drawbox`es.

    `drawbox` takes no expression in time, so a box under an animated text
    follows it in steps of `BOX_STEP_S`, each with its own size, place and
    opacity. It is drawn first, *replacing* the canvas' pixels: the canvas is
    transparent, and blended into nothing the box would keep alpha 0 and never
    show.
    """
    style = clip.text_style
    if not style.box or style.box_opacity <= 0:
        return []
    pad = BOX_PAD * body_px
    cx = width / 2 + clip.transform.x * width / 2
    cy = height / 2 + clip.transform.y * height / 2

    def drawbox(alpha: float, size: float, drop: float, enable: str | None) -> str:
        w = (block_w + 2 * pad) * size
        h = (block_h + 2 * pad) * size
        x = cx - w / 2
        y = cy + drop * height - h / 2
        part = (
            f"drawbox=x={x:.0f}:y={y:.0f}:w={w:.0f}:h={h:.0f}"
            f":color={style.box}@{style.box_opacity * alpha:.3f}:t=fill:replace=1"
        )
        return part + (f":enable='{enable}'" if enable else "")

    if (style.anim_in, style.anim_out) == (TextAnim.NONE, TextAnim.NONE) or (
        style.anim_out is TextAnim.NONE and style.anim_in is TextAnim.TYPEWRITER
    ):
        return [drawbox(1.0, 1.0, 0.0, None)]

    # sample the motion, and merge the samples that come out the same box
    runs: list[tuple[tuple[float, float, float], float, float]] = []
    t = 0.0
    while t < clip.duration_s:
        a, k, dr = _motion_at(clip, min(t + BOX_STEP_S / 2, clip.duration_s))
        key = (round(a, 2), round(k, 3), round(dr, 4))
        end = min(t + BOX_STEP_S, clip.duration_s)
        if runs and runs[-1][0] == key:
            runs[-1] = (key, runs[-1][1], end)
        else:
            runs.append((key, t, end))
        t = end
    # the last stretch holds to the end, whatever rounding left
    key, start, _ = runs[-1]
    runs[-1] = (key, start, clip.duration_s + 1)
    return [
        drawbox(a, k, dr, f"gte(t,{start:.4f})*lt(t,{until:.4f})")
        for (a, k, dr), start, until in runs
        if a > 0
    ]
