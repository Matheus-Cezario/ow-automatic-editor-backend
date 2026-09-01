"""The editor's text turning into ffmpeg `drawtext`.

Kept apart from `compose.py` for a practical reason: escaping text for a
filtergraph is the kind of code where one backslash too many or too few slips
past review and only shows up when somebody types `50%` into a label. On its
own, it gets its own test.
"""

from __future__ import annotations

from . import fonts
from .models import TimelineClip

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


def escape(text: str) -> str:
    """Lets the text through `drawtext` without becoming syntax."""
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


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
    font = style.font or fonts.default_font()

    parts = [
        f"fontfile='{escape(font)}'",
        f"text='{escape(clip.text)}'",
        # no expansion whatsoever: the user's text is text, and a stray `%` in
        # it would make drawtext give up on drawing the whole line
        "expansion=none",
        f"fontsize={body_px}",
        f"fontcolor={style.color}",
        # text moves through the frame with the same x/y as any clip: the
        # centre is 0, the edges are -1 and 1
        f"x=(w-text_w)/2+({clip.transform.x:.4f})*(w/2)",
        f"y=(h-text_h)/2+({clip.transform.y:.4f})*(h/2)",
    ]
    if outline_px > 0:
        parts += [f"borderw={outline_px}", f"bordercolor={style.outline_color}"]
    return "drawtext=" + ":".join(parts)
