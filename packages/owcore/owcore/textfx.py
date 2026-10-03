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

from . import fonts
from .models import TextAnim, TimelineClip

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


def _typing(clip: TimelineClip) -> list[tuple[str, float, float]]:
    """The typewriter's steps: (text shown, from, until). Empty when the text
    does not type itself."""
    if clip.text_style.anim_in is not TextAnim.TYPEWRITER:
        return []
    text = clip.text
    n = len(text)
    if n == 0:
        return []
    d = min(clip.text_style.anim_s, clip.duration_s / 2)
    typing = min(max(d, n / TYPE_RATE), clip.duration_s)
    steps = min(n, TYPE_MAX_STEPS)
    out = []
    for k in range(1, steps + 1):
        shown = text[: math.ceil(n * k / steps)]
        out.append((shown, typing * (k - 1) / steps, typing * k / steps))
    # the last step stays until the end
    shown, start, _ = out[-1]
    out[-1] = (shown, start, clip.duration_s + 1)
    return out


def filter_chain(clip: TimelineClip, height: int) -> str:
    """This clip's `drawtext`, already positioned in the frame.

    Size and outline come as a **fraction of the height**: the same montage
    comes out identical at 720p and at 4K, and a 48px body that looks right in
    one would be tiny in the other. The outline is not decoration -- without
    it, white text disappears against a bright scene.
    """
    style = clip.text_style
    body_px = max(1, int(round(style.size * height)))
    outline_px = int(round(style.outline * body_px))
    font = fonts.resolve(style.font)
    alpha, size, drop = _motion(clip)
    animated = (alpha, size, drop) != ("1", "1", "0")

    def drawtext(text: str, enable: str | None = None) -> str:
        parts = [
            f"fontfile='{escape(font)}'",
            f"text='{escape(text)}'",
            # no expansion whatsoever: the user's text is text, and a stray `%`
            # in it would make drawtext give up on drawing the whole line
            "expansion=none",
            f"fontsize='{body_px}*({size})'" if animated else f"fontsize={body_px}",
            f"fontcolor={style.color}",
            # text moves through the frame with the same x/y as any clip: the
            # centre is 0, the edges are -1 and 1
            f"x=(w-text_w)/2+({clip.transform.x:.4f})*(w/2)",
            f"y='(h-text_h)/2+({clip.transform.y:.4f})*(h/2)+({drop})*h'"
            if animated
            else f"y=(h-text_h)/2+({clip.transform.y:.4f})*(h/2)",
        ]
        if animated:
            parts.append(f"alpha='{alpha}'")
        if outline_px > 0:
            parts += [f"borderw={outline_px}", f"bordercolor={style.outline_color}"]
        if enable:
            parts.append(f"enable='{enable}'")
        return "drawtext=" + ":".join(parts)

    steps = _typing(clip)
    if not steps:
        return drawtext(clip.text)
    return ",".join(
        drawtext(shown, f"gte(t,{start:.4f})*lt(t,{until:.4f})")
        for shown, start, until in steps
    )
