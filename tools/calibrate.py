#!/usr/bin/env python
"""Calibra o perfil da HUD para a *sua* gravacao.

O sistema so acerta se souber onde a HUD do Overwatch fica na sua tela e de
que cor ela e. Isso muda com resolucao, proporcao, idioma, modo daltonico e
patch do jogo -- entao em vez de chutar constantes, esta ferramenta mostra o
que o detector esta enxergando.

Dois modos:

    # 1. Onde estao as regioes? Desenha os retangulos sobre quadros reais.
    python tools/calibrate.py preview --video partida.mp4 --at 30 90 150

    # 2. Que limiar usar? Mede a regiao quadro a quadro e sugere os numeros.
    python tools/calibrate.py scan --video partida.mp4 --roi kills

Depois copie `config/profiles/ow2_default.json` para um perfil seu, ajuste os
valores e rode com `OW_PROFILE=meu_perfil`.
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
            "  aviso: o perfil padrao foi feito para 16:9. Numa proporcao "
            "diferente as regioes vao sair do lugar -- use este preview para "
            "corrigi-las."
        )

    names = [n for n in profile.data["rois"] if not profile.roi(n).fullscreen]
    for t in args.at:
        frame = grab_frame(Path(args.video), t)
        if frame is None:
            print(f"  nao consegui ler o quadro em {t}s")
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

        cv2.imwrite(str(out / f"quadro_{t:g}s.png"), canvas)
        print(f"  {t:g}s -> quadro_{t:g}s.png + recortes")

    print(f"\nimagens em {out.resolve()}")
    print("Se um retangulo nao cobre o elemento da HUD, ajuste x/y/w/h "
          "(fracoes de 0 a 1) no perfil.")
    return 0


# --------------------------------- scan -------------------------------------


def sparkline(values: list[float], width: int = 900, height: int = 260,
              threshold: float | None = None) -> np.ndarray:
    """Grafico da serie temporal desenhado com OpenCV -- evita arrastar
    matplotlib so para isto."""
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
        cv2.putText(img, f"limiar {threshold:.4f}", (8, max(12, ty - 6)),
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
        print(f"a secao '{section}' do perfil nao tem hsv_ranges; nada a medir")
        return 2

    print(f"recortando a regiao '{roi_name}' ({roi.fps} fps)...")
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
        print("nenhum quadro lido")
        return 2

    arr = np.array(values)
    p = {q: float(np.percentile(arr, q)) for q in (50, 90, 99, 99.9)}
    peak = float(arr.max())
    base = p[50]
    # o elemento da HUD aparece numa fracao pequena do tempo: o "fundo" e a
    # mediana e o "evento" e a cauda de cima. Meio caminho entre os dois separa
    # bem sem grudar em nenhum dos lados.
    suggested = base + 0.35 * (peak - base)
    release = base + 0.15 * (peak - base)

    print(f"\nquadros medidos : {len(values)} ({times[-1]:.1f}s)")
    print(f"mediana (fundo) : {base:.5f}")
    print(f"p90 / p99       : {p[90]:.5f} / {p[99]:.5f}")
    print(f"maximo (evento) : {peak:.5f}")
    if peak <= base * 1.5:
        print(
            "\n  A regiao nunca ficou muito mais vermelha do que o normal.\n"
            "  Ou o retangulo esta no lugar errado (rode o modo preview),\n"
            "  ou a cor da HUD e outra (modo daltonico muda o vermelho)."
        )
        return 1

    print("\nsugestao para o perfil:")
    print(json.dumps(
        {section: {"min_ratio": round(suggested, 5),
                   "release_ratio": round(release, 5)}},
        indent=2,
    ))

    chart = out / f"serie_{roi_name}.png"
    cv2.imwrite(str(chart), sparkline(values, threshold=suggested))
    print(f"\ngrafico da serie: {chart.resolve()}")
    print("Cada pico deveria ser um evento. Se houver picos demais, suba o "
          "min_ratio; se faltarem, baixe.")
    return 0


# --------------------------------- main -------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("preview", help="desenha as regioes sobre quadros reais")
    pv.add_argument("--video", required=True)
    pv.add_argument("--at", type=float, nargs="+", default=[10.0, 60.0, 120.0],
                    help="instantes (s) para amostrar")
    pv.add_argument("--profile", default=None)
    pv.add_argument("--out", default="data/calib")
    pv.set_defaults(func=cmd_preview)

    sc = sub.add_parser("scan", help="mede uma regiao e sugere limiares")
    sc.add_argument("--video", required=True)
    sc.add_argument("--roi", default="kills")
    sc.add_argument("--profile", default=None)
    sc.add_argument("--out", default="data/calib")
    sc.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
