#!/usr/bin/env python
"""Calibrates the HUD profile for *your* recording.

The system only gets it right if it knows where the Overwatch HUD sits on your
screen and what colour it is. That changes with resolution, aspect ratio,
language, colour-blind mode and game patch -- so instead of guessing constants,
this tool shows what the detector is seeing.

Two modes:

    # 1. Where are the regions? Draws the rectangles over real frames.
    python tools/calibrate.py preview --video match.mp4 --at 30 90 150

    # 2. Which threshold? Measures the region frame by frame and suggests numbers.
    python tools/calibrate.py scan --video match.mp4 --roi kills

Then copy `config/profiles/ow2_default.json` to a profile of your own, adjust
the values and run with `OW_PROFILE=my_profile`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "owcore"))

from owcore.ffmpeg import extract_rois, probe  # noqa: E402
from owcore.profiles import load_profile  # noqa: E402
from owcore.vision import (  # noqa: E402
    border_mask,
    hsv_ratio,
    hsv_ratio_masked,
    iter_frames,
)

BOX_COLORS = [
    (80, 220, 255), (120, 255, 120), (255, 160, 90), (255, 120, 220),
]


# ------------------------------- preview ------------------------------------


def grab_frame(video: Path, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def cmd_preview(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = probe(Path(args.video))
    print(f"video: {info.width}x{info.height}, {info.duration_s:.1f}s")
    if abs(info.width / info.height - 16 / 9) > 0.02:
        print(
            "  warning: the default profile was made for 16:9. At a different "
            "aspect ratio the regions will be off -- use this preview to fix "
            "them."
        )

    names = [n for n in profile.data["rois"] if not profile.roi(n).fullscreen]
    for t in args.at:
        frame = grab_frame(Path(args.video), t)
        if frame is None:
            print(f"  could not read the frame at {t}s")
            continue
        h, w = frame.shape[:2]
        canvas = frame.copy()

        for i, name in enumerate(names):
            roi = profile.roi(name)
            x0, y0 = int(roi.x * w), int(roi.y * h)
            x1, y1 = int((roi.x + roi.w) * w), int((roi.y + roi.h) * h)
            color = BOX_COLORS[i % len(BOX_COLORS)]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(canvas, name, (x0 + 4, max(16, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            crop = frame[y0:y1, x0:x1]
            if crop.size:
                cv2.imwrite(str(out / f"roi_{name}_{t:g}s.png"), crop)

        cv2.imwrite(str(out / f"frame_{t:g}s.png"), canvas)
        print(f"  {t:g}s -> frame_{t:g}s.png + crops")

    print(f"\nimages in {out.resolve()}")
    print("If a rectangle does not cover the HUD element, adjust x/y/w/h "
          "(fractions from 0 to 1) in the profile.")
    return 0


# --------------------------------- scan -------------------------------------


def sparkline(values: list[float], width: int = 900, height: int = 260,
              threshold: float | None = None) -> np.ndarray:
    """Time-series chart drawn with OpenCV -- avoids pulling in matplotlib just
    for this."""
    img = np.full((height, width, 3), 26, np.uint8)
    if not values:
        return img
    top = max(values) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = int(i / max(1, len(values) - 1) * (width - 1))
        y = int(height - 20 - (v / top) * (height - 40))
        pts.append((x, y))
    if threshold is not None and top > 0:
        ty = int(height - 20 - (threshold / top) * (height - 40))
        cv2.line(img, (0, ty), (width, ty), (80, 80, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"threshold {threshold:.4f}", (8, max(12, ty - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 240), 1)
    cv2.polylines(img, [np.array(pts, np.int32)], False, (120, 230, 120), 1,
                  cv2.LINE_AA)
    cv2.putText(img, f"max {top:.4f}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def cmd_scan(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    roi_name = args.roi
    roi = profile.roi(roi_name)
    section = {"kills": "kills", "survival": "survival"}.get(roi_name, roi_name)
    cfg = profile.section(section)
    ranges = cfg.get("hsv_ranges", [])
    if not ranges:
        print(f"the profile section '{section}' has no hsv_ranges; nothing to measure")
        return 2

    print(f"cropping region '{roi_name}' ({roi.fps} fps)...")
    crops = extract_rois(Path(args.video), [roi], out / "_rois")
    path = crops[roi_name]

    values: list[float] = []
    times: list[float] = []
    mask = None
    for frame in iter_frames(path, fps_hint=roi.fps):
        if roi.fullscreen:
            if mask is None:
                mask = border_mask(frame.bgr.shape[:2],
                                   float(cfg.get("border_frac", 0.10)))
            values.append(hsv_ratio_masked(frame.bgr, ranges, mask))
        else:
            values.append(hsv_ratio(frame.bgr, ranges))
        times.append(frame.t)

    if not values:
        print("no frame read")
        return 2

    arr = np.array(values)
    p = {q: float(np.percentile(arr, q)) for q in (50, 90, 99, 99.9)}
    peak = float(arr.max())
    base = p[50]
    # the HUD element shows up for a small fraction of the time: the
    # "background" is the median and the "event" is the upper tail. Part way
    # between the two separates well without sticking to either side.
    suggested = base + 0.35 * (peak - base)
    release = base + 0.15 * (peak - base)

    print(f"\nframes measured   : {len(values)} ({times[-1]:.1f}s)")
    print(f"median (background): {base:.5f}")
    print(f"p90 / p99          : {p[90]:.5f} / {p[99]:.5f}")
    print(f"maximum (event)    : {peak:.5f}")
    if peak <= base * 1.5:
        print(
            "\n  The region never got much redder than usual.\n"
            "  Either the rectangle is in the wrong place (run preview mode),\n"
            "  or the HUD colour is different (colour-blind mode changes red)."
        )
        return 1

    print("\nsuggestion for the profile:")
    print(json.dumps(
        {section: {"min_ratio": round(suggested, 5),
                   "release_ratio": round(release, 5)}},
        indent=2,
    ))

    chart = out / f"series_{roi_name}.png"
    cv2.imwrite(str(chart), sparkline(values, threshold=suggested))
    print(f"\nseries chart: {chart.resolve()}")
    print("Each peak should be an event. If there are too many peaks, raise "
          "min_ratio; if some are missing, lower it.")
    return 0


# --------------------------------- main -------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("preview", help="draws the regions over real frames")
    pv.add_argument("--video", required=True)
    pv.add_argument("--at", type=float, nargs="+", default=[10.0, 60.0, 120.0],
                    help="instants (s) to sample")
    pv.add_argument("--profile", default=None)
    pv.add_argument("--out", default="data/calib")
    pv.set_defaults(func=cmd_preview)

    sc = sub.add_parser("scan", help="measures a region and suggests thresholds")
    sc.add_argument("--video", required=True)
    sc.add_argument("--roi", default="kills")
    sc.add_argument("--profile", default=None)
    sc.add_argument("--out", default="data/calib")
    sc.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
