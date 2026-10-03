"""The exact preview: a montage turned into its small, fast twin.

The editor's monitor composes layers in the browser -- instant, but its own
approximation of the filter graph. When the user wants to *see* what the server
will produce, the answer is the server's graph itself, run on a reduced frame,
over a short stretch, with the fastest encoder settings. Everything that
decides the picture (layers, effects, transitions, text, the fit to the export
shape) is untouched; only the size of the canvas and the encoder change.
"""

from __future__ import annotations

from owcore.models import Timeline

#: the longest side of the preview frame -- the proxy's own width, so the
#: preview never upscales what it decodes
PREVIEW_LONG_SIDE = 640
#: a preview is watched once: low quality and few frames are fine
PREVIEW_CRF = 30
PREVIEW_MAX_FPS = 30.0
#: the longest stretch one preview may cover, in seconds of the montage
PREVIEW_MAX_S = 60.0


def preview_size(
    export_width: int, export_height: int, source_width: int, source_height: int
) -> tuple[int, int]:
    """The preview frame: the export's shape (or the recording's), scaled so
    its longest side is [PREVIEW_LONG_SIDE], rounded to even numbers."""
    w = export_width or source_width
    h = export_height or source_height
    if w <= 0 or h <= 0:
        w, h = 16, 9
    k = PREVIEW_LONG_SIDE / max(w, h)
    return max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)


def preview_window(
    duration_s: float, from_s: float | None, to_s: float | None
) -> tuple[float, float]:
    """The stretch to render, clamped to the montage and to [PREVIEW_MAX_S]."""
    start = max(0.0, min(from_s or 0.0, duration_s))
    end = duration_s if to_s is None else min(to_s, duration_s)
    end = min(end, start + PREVIEW_MAX_S)
    if end - start <= 0:
        raise ValueError("the preview window is empty")
    return start, end


def preview_timeline(
    spec: Timeline,
    *,
    from_s: float,
    to_s: float,
    source_width: int,
    source_height: int,
    source_fps: float,
) -> Timeline:
    """The same montage, exported small, fast and only over [from_s, to_s]."""
    width, height = preview_size(
        spec.export.width, spec.export.height, source_width, source_height
    )
    fps = spec.export.fps or source_fps or PREVIEW_MAX_FPS
    export = spec.export.model_copy(
        update={
            "width": width,
            "height": height,
            "fps": min(fps, PREVIEW_MAX_FPS),
            "crf": PREVIEW_CRF,
            "from_s": from_s,
            "to_s": to_s,
        }
    )
    return spec.model_copy(update={"export": export})
